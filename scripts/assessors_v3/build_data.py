"""Build the training tables of the v3 pipeline from the converter's JSONL and the ADeLe v1 annotations.

    python scripts/assessors_v3/build_data.py [--benchmarks a b ...] [--workers 4] [--overwrite]

Only ADeLe-annotated items are kept (team decision, 2026-10-24): a benchmark is processed if it has an annotation
table in ADELE_DIR, and a response row only if its item_id is in that table. Everything else in the JSONL is skipped
without being parsed.

Rows. The converter writes one row per response run (trial layout: response_id, trial, n_runs) or, in the older
layout, one row per (subject, item) with the mean score and n_runs; both are read. All runs of the same subject on the
same item are aggregated into one pair, whatever their test condition (team decision, 2026-10-24: in many benchmarks
the condition only labels one subject's run): n (runs), s (sum of scores), q (sum of squared scores). The mean score is
s / n; Brier over the runs is (n p^2 - 2 p s + q) / n (simulate.py).

Instances. instance_id = item_id (the column is kept so that splitting an item into several instances later only
changes this script). Instances are the unit of the draws and splits; ADeLe and text features are joined on item_id.

Outputs in DATA (config.py), per benchmark so a rerun only does what is missing:
    pairs/<benchmark>.parquet     instance_id, item_id, subject_id, n, s, q
    items/<benchmark>.parquet     item_id, benchmark, text_hash (SHA-1 of the normalised text), I0 text statistics
    subjects/<benchmark>.parquet  subject_id and the subject's fields (model_name, normalized_name, ...)
    bench/<benchmark>.json        response counts, binary share, agentic, conditions, ...
and, merged at the end: items.parquet, subjects.parquet, benchmarks.csv, adele.parquet (item_id x 18 expected levels).
No item text is written. Memory: one row of four numbers per kept response while a file is read (array module),
aggregated with pandas at the end of the file; workers stop (LowMemory) when the server runs short.
"""

import argparse
import hashlib
import json
import re
import sys
from array import array
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LowMemory, check_start_ram, guard_memory, log, write_json, write_parquet  # noqa: E402
from config import ADELE_CODES, ADELE_DIR, DATA, JSONL_DIRS  # noqa: E402

SUBJECT_FIELDS = ["model_name", "normalized_name", "reasoning_effort", "thinking", "harness", "harness_version",
                  "access_date"]
PLACEHOLDER = re.compile(r"see image|see the (?:image|figure)|you are chatgpt|^date: ", re.I)


# ----------------------------------------------------------------------------- item features (I0)

def item_stats(text):
    """I0: cheap, domain-agnostic statistics of one item's text (the same as in scripts/assessors/features.py)."""
    c = text if isinstance(text, str) else ("" if text is None else json.dumps(text))
    n = max(len(c), 1)
    return {
        "log_chars": np.log1p(len(c)),
        "log_words": np.log1p(len(c.split())),
        "log_lines": np.log1p(c.count("\n")),
        "digit_share": len(re.findall(r"\d", c)) / n,
        "math_share": len(re.findall(r"[\\$^_{}=]", c)) / n,
        "code_share": len(re.findall(r"[`;(){}\[\]<>]", c)) / n,
        "upper_share": len(re.findall(r"[A-Z]", c)) / n,
        "n_urls": np.log1p(len(re.findall(r"https?://", c))),
        "n_questions": np.log1p(c.count("?")),
        "has_choices": float(bool(re.search(r"\(A\)|\bA\)|\bA\.", c))),
        "placeholder": float(bool(PLACEHOLDER.search(c))),
        "has_image": float("<image:" in c),
    }


ITEM_STAT_COLS = list(item_stats("x"))


def text_hash(text):
    """Identity of an item's text for the sibling rule (protocol D2): lower case, whitespace collapsed."""
    c = text if isinstance(text, str) else json.dumps(text, sort_keys=True)
    return hashlib.sha1(" ".join(c.lower().split()).encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------- fast row reading

_ID = b'"item_id":"'


def raw_item_id(line):
    """The item_id of a JSONL row by string search, without parsing the row (the question can be megabytes long).
    Returns the decoded id, or None. Escaped quotes inside the id are handled (an odd run of backslashes escapes)."""
    i = line.find(_ID)
    if i < 0:
        return None
    j = i + len(_ID)
    k = j
    while True:
        k = line.find(b'"', k)
        if k < 0:
            return None
        bs = 0
        while line[k - 1 - bs] == 0x5C:      # backslash
            bs += 1
        if bs % 2 == 0:
            break
        k += 1
    raw = line[j:k]
    return raw.decode("utf-8") if b"\\" not in raw else orjson.loads(b'"' + raw + b'"')


def parse_small(line):
    """The row without its two long text fields (question, ground_truth): the converter writes columns in a fixed
    order (scripts/preprocess/common.py, OBSERVATION_COLUMNS), so the text sits between ',"question":' and
    ',"eval_type":'. Those markers cannot occur inside a JSON string (its quotes are escaped there). Falls back to a
    full parse if the layout is different."""
    q = line.find(b',"question":')
    e = line.find(b',"eval_type":', q + 1) if q >= 0 else -1
    if q < 0 or e < 0:
        return orjson.loads(line)
    head = orjson.loads(line[:q] + b"}")
    head.update(orjson.loads(b"{" + line[e + 1:]))
    return head


# ----------------------------------------------------------------------------- one benchmark

def process(job):
    """Read one benchmark's JSONL; write its pairs, items, subjects and summary. Runs in a worker process."""
    bench, jsonl, adele_path, out = job
    out = Path(out)
    annotated = set(pd.read_parquet(adele_path, columns=["item_id"])["item_id"])
    item_idx, subj_idx, conditions = {}, {}, set()
    rows_item, rows_subj = array("i"), array("i")
    rows_n, rows_s, rows_q = array("d"), array("d"), array("d")
    items, subjects, info = [], {}, {"score_types": {}, "rows_read": 0, "rows_kept": 0, "ungraded": 0}
    with open(jsonl, "rb") as fh:
        for k, line in enumerate(fh):
            if k % 500_000 == 0:
                guard_memory()
            info["rows_read"] += 1
            iid = raw_item_id(line)
            if iid not in annotated:
                continue
            if iid not in item_idx:                         # first row of this item: full parse, text features
                r = orjson.loads(line)
                item_idx[iid] = len(item_idx)
                items.append({"item_id": iid, "benchmark": bench, "text_hash": text_hash(r.get("question")),
                              **item_stats(r.get("question"))})
                for key in ("agentic", "multi_turn", "modality", "domain"):
                    info.setdefault(key, r.get(key))
            else:
                r = parse_small(line)
            score = r.get("score")
            if score is None or not 0 <= score <= 1:       # ungraded or invalid: not a model outcome
                info["ungraded"] += 1
                continue
            st = r.get("score_type") or "unknown"
            info["score_types"][st] = info["score_types"].get(st, 0) + 1
            sid = r["subject_id"]
            if sid not in subj_idx:
                subj_idx[sid] = len(subj_idx)
                subjects[sid] = {f: r.get(f) for f in SUBJECT_FIELDS}
            conditions.add(r.get("test_condition") or "")   # counted for the summary only: conditions are pooled
            if "trial" in r:                                # trial layout: one row = one run
                n = 1.0
                s_, q_ = score, score * score
            else:                                           # aggregated layout: mean over n_runs runs
                n = float(r.get("n_runs") or 1)
                s_ = score * n
                q_ = s_ if st == "binary" else score * score * n   # binary runs: y^2 = y
            rows_item.append(item_idx[iid]); rows_subj.append(subj_idx[sid])
            rows_n.append(n); rows_s.append(s_); rows_q.append(q_)
            info["rows_kept"] += 1
    guard_memory()
    item_names = np.array(list(item_idx), dtype=object)
    subj_names = np.array(list(subj_idx), dtype=object)
    df = pd.DataFrame({"item": np.frombuffer(rows_item, np.int32), "subject": np.frombuffer(rows_subj, np.int32),
                       "n": np.frombuffer(rows_n),
                       "s": np.frombuffer(rows_s), "q": np.frombuffer(rows_q)})
    pairs = df.groupby(["subject", "item"], sort=False, as_index=False)[["n", "s", "q"]].sum()
    pairs["item_id"] = item_names[pairs["item"].to_numpy()]
    pairs["subject_id"] = subj_names[pairs["subject"].to_numpy()]
    pairs["instance_id"] = pairs["item_id"]
    pairs = pairs[["instance_id", "item_id", "subject_id", "n", "s", "q"]].astype(
        {"n": "float32", "s": "float32", "q": "float32"})
    for c in ("instance_id", "item_id", "subject_id"):
        pairs[c] = pairs[c].astype("category")
    write_parquet(pairs, out / "pairs" / f"{bench}.parquet")
    write_parquet(pd.DataFrame(items), out / "items" / f"{bench}.parquet")
    # subject fields as strings (None stays missing), so tables of different benchmarks concatenate cleanly
    write_parquet(pd.DataFrame([{"subject_id": s, **{k: None if v is None else str(v) for k, v in f.items()}}
                                for s, f in subjects.items()]), out / "subjects" / f"{bench}.parquet")
    total = sum(info["score_types"].values())
    info.update(benchmark=bench, pairs=len(pairs), instances=int(pairs["instance_id"].nunique()),
                items=len(item_idx), subjects=len(subj_idx), conditions=len(conditions),
                binary_share=info["score_types"].get("binary", 0) / total if total else 0.0,
                mean_score=float(pairs["s"].sum() / max(pairs["n"].sum(), 1)),
                annotated_items=len(annotated))
    write_json(info, out / "bench" / f"{bench}.json")
    return info


def merge(out):
    """Merge the per-benchmark tables into items.parquet, subjects.parquet, benchmarks.csv and adele.parquet."""
    out = Path(out)
    benches = sorted(p.stem for p in (out / "bench").glob("*.json"))
    infos = [json.loads((out / "bench" / f"{b}.json").read_text()) for b in benches]
    t = pd.DataFrame(infos).drop(columns=["score_types"])
    t.to_csv(out / "benchmarks.csv", index=False)
    write_parquet(pd.concat([pd.read_parquet(out / "items" / f"{b}.parquet") for b in benches], ignore_index=True),
                  out / "items.parquet")
    subj = pd.concat([pd.read_parquet(out / "subjects" / f"{b}.parquet") for b in benches], ignore_index=True)
    write_parquet(subj.drop_duplicates("subject_id"), out / "subjects.parquet")
    cols = ["item_id"] + [f"{c}_exp" for c in ADELE_CODES]
    tables = adele_tables()
    adele = pd.concat([pd.read_parquet(tables[b], columns=cols) for b in benches], ignore_index=True)
    write_parquet(adele.drop_duplicates("item_id"), out / "adele.parquet")
    log(f"merged {len(benches)} benchmarks: {int(t['pairs'].sum()):,} pairs, {int(t['items'].sum()):,} items, "
        f"{subj['subject_id'].nunique():,} subjects -> {out}")


def adele_tables():
    """benchmark -> its ADeLe table: ADELE_DIR/<run>/<benchmark>.parquet (or a flat ADELE_DIR/<benchmark>.parquet)."""
    return {p.stem: p for p in sorted(ADELE_DIR.glob("*.parquet")) + sorted(ADELE_DIR.glob("*/*.parquet"))}


def jsonl_files():
    """benchmark -> its JSONL, from the first of JSONL_DIRS that has it."""
    out = {}
    for d in JSONL_DIRS:
        for p in sorted(d.glob("*.jsonl")):
            out.setdefault(p.stem, p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmarks", nargs="*", help="default: every benchmark with an ADeLe table and a JSONL")
    ap.add_argument("--workers", type=int, default=4, help="parallel files (each holds one file's rows in memory)")
    ap.add_argument("--overwrite", action="store_true", help="rebuild benchmarks already built")
    a = ap.parse_args()
    check_start_ram()
    tables, files = adele_tables(), jsonl_files()
    annotated, available = set(tables), set(files)
    benches = sorted((set(a.benchmarks) if a.benchmarks else annotated) & annotated & available)
    missing = sorted(annotated - available)
    if missing:
        log(f"{len(missing)} annotated benchmarks have no JSONL in {':'.join(map(str, JSONL_DIRS))}: "
            f"{', '.join(missing[:10])}")
    todo = [b for b in benches if a.overwrite or not (DATA / "bench" / f"{b}.json").exists()]
    log(f"{len(benches)} benchmarks, {len(todo)} to build, {a.workers} workers -> {DATA}")
    # largest files first, so the longest jobs do not end up alone at the end
    todo.sort(key=lambda b: -files[b].stat().st_size)
    jobs = [(b, str(files[b]), str(tables[b]), str(DATA)) for b in todo]
    try:
        with Pool(a.workers, maxtasksperchild=1) as pool:
            for k, info in enumerate(pool.imap_unordered(process, jobs)):
                log(f"{k + 1}/{len(jobs)} {info['benchmark']}: {info['rows_kept']:,} of {info['rows_read']:,} rows "
                    f"kept -> {info['pairs']:,} pairs, {info['items']} items, {info['subjects']} subjects, "
                    f"binary {info['binary_share']:.0%}")
    except LowMemory as err:
        sys.exit(f"stopped: {err}. Finished benchmarks are kept; rerun to resume.")
    merge(DATA)


if __name__ == "__main__":
    main()
