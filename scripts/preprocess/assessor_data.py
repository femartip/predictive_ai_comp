"""Build the assessor dataset: one Parquet file per benchmark with everything needed to train an assessor, on every row.

    poetry run python scripts/preprocess/assessor_data.py [--workers 4] [--benchmarks a b ...] [--offline]
    poetry run python scripts/preprocess/assessor_data.py --resume data/datasets/assessor_data/RUN_ID

Inputs (data/datasets, the shared folder on vent):
  annotations/adele_v1/<run>/<benchmark>.parquet   ADeLe v1 levels of the annotated items (at most 600 per benchmark,
                                                    scripts/preprocess/annotations/); <run> is the conversion run the
                                                    items came from: no-assets or asset-image. Only these
                                                    items are kept.
  measurement_db/<run>/<RUN_ID>/<benchmark>.jsonl   the converter's responses (measurement_db.py): for each benchmark,
                                                    the newest conversion of its run whose manifest lists it as written
                                                    (a conversion that failed later still has its finished files), or
                                                    the folder given with --jsonl RUN=DIR
  the bank on Hugging Face (gated, at the conversion's pinned revision): each benchmark's subjects table, for every
  subject's provider, release_date and subject_features_extra, and its items table, for every item's item_features.
  The converter does not copy them into the JSONL, and CodaBench gives all four to predict(). Needs
  `poetry run hf auth login`.
  Epoch AI, models.dev, ReleasedModels and the Hugging Face Hub: model attributes (epoch_models_csv.model_table)

Output, one folder per build: data/datasets/assessor_data/<UTC-timestamp>_<id>/
  <benchmark>.parquet   one row per observation cell of an annotated item: (subject_id, item_id, test_condition,
                        interactors), the converter's own cell (its n_runs counts the cell's runs). Columns, in order:
    keys        benchmark, item_id, subject_id, test_condition, interactors
    outcome     score (mean over the cell's graded runs, in [0, 1], 1 = better), n_runs (graded runs in that mean),
                score_type, response_scale, normalization_policy, and the runs themselves: response_ids, trials,
                raw_scores (lists, one entry per graded run, in file order)
    item        question, ground_truth, eval_type, item_features (the bank's key=value text, e.g. "lang=c++")
    annotation  text_id (items with identical text share it), truncated_any, digit_mass_min, and the expected level
                <code>_exp of each of the 18 ADeLe v1 rubrics (AS AT CEc CEe CL KNa KNc KNf KNn KNs MCr MCt MCu MS QLl
                QLq SNs VO)
    benchmark   domain, modality, multi_turn, agentic, benchmark_release_date, programming_language, tools (as the
                converter writes them on every response)
    subject     model_name, normalized_name, model_key, model_key_source, provider, release_date, access_date,
                harness, harness_version, reasoning_effort, thinking, subject_features_extra (JSON text).
                normalized_name, provider and release_date are the bank's, as CodaBench gives them. model_key is the
                subject's model identity, also when the bank gives no normalized_name (model_key_source):
                  bank           its normalized_name
                  named_subject  its model_name matches (after normalisation, name_key) the model_name of a named
                                 subject, or a normalized_name without its provider word: that normalized_name
                  catalogue      its model_name matches an Epoch AI model or a models.dev listing: that name
                  derived        otherwise: its model_name without a benchmark prefix ("MMBench V1.1 / ..."), with
                                 "org__repo" written "org/repo"; model_names with the same name_key share it
                Matches are not reviewed: a few are wrong (e.g. a tabular model matching a catalogue name).
    model       model_* (by model_key): kind, provider, release_date, open_weights,
                publication_date, organization, country, accessibility, domain, base_model, parameters,
                active_parameters, training_compute_flop, training_dataset_size, finetune_compute_flop,
                post_training_compute_flop, training_hardware, frontier, architecture, moe, n_experts,
                multimodal_input, context_window, knowledge_cutoff, thinking_mode. Curated by epoch_models_csv.py:
                model_provider and model_release_date are filled from Epoch where the bank has none (for a derived
                Hugging Face repo id, the provider is its organisation). The Hugging Face Hub is queried for every
                model_key except derived ones that are not repo ids (thousands; local catalogues only).
  Dropped: responses whose score is missing or outside [0, 1] (ungraded: not a model outcome), and so cells without
  any graded run; both are counted in benchmarks.parquet. A benchmark left without cells gets no file.
  benchmarks.parquet  one row per benchmark: conversion, revision, rows read, runs kept, ungraded runs, cells, cells
                      dropped, items, subjects, binary share, mean score, and checks (fields that differ inside a cell,
                      subjects whose normalized_name differs between the JSONL and the bank)
  models.parquet      the full model table of epoch_models_csv.py, with the source of every value
  manifest.json       inputs, settings, counts, status and timestamps;  README.txt: this description
Every file has the same fixed schema (SCHEMA): a value that does not fit stops the build instead of reaching training.
Each benchmark is first read into <out>/parts/, so an interrupted build continues with --resume; parts are removed at
the end.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from uuid import uuid4

import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

import epoch_models_csv as E
from measurement_db_categories import REPOSITORY

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
DATASETS = ROOT / "data" / "datasets"
ADELE = DATASETS / "annotations" / "adele_v1"
BANK = DATASETS / "measurement_db"
OUTPUT = DATASETS / "assessor_data"
RUNS = ("no-assets", "asset-image")
MIN_FREE_RAM_GB = 5        # vent is shared: a worker stops (the build can resume) below this much available memory
ADELE_CODES = ["AS", "AT", "CEc", "CEe", "CL", "KNa", "KNc", "KNf", "KNn", "KNs", "MCr", "MCt", "MCu", "MS", "QLl",
               "QLq", "SNs", "VO"]

# --- the converter's JSONL columns, by role -----------------------------------------------------------------------
KEYS = ["benchmark", "item_id", "subject_id", "test_condition", "interactors"]
RUN_FIELDS = ["response_id", "trial", "score", "raw_score", "n_runs"]          # differ between the runs of a cell
CELL_FIELDS = ["score_type", "response_scale", "normalization_policy"]
ITEM_FIELDS = ["question", "ground_truth", "eval_type"]
BENCH_FIELDS = ["domain", "modality", "multi_turn", "agentic", "benchmark_release_date", "programming_language",
                "tools"]
SUBJECT_FIELDS = ["model_name", "normalized_name", "access_date", "harness", "harness_version", "reasoning_effort",
                  "thinking"]
JSONL_FIELDS = set(KEYS + RUN_FIELDS + CELL_FIELDS + ITEM_FIELDS + BENCH_FIELDS + SUBJECT_FIELDS)
LONG = ("question", "ground_truth")             # skipped by parse_small after an item's first row
HUB_FIELDS = ["provider", "release_date", "subject_features_extra"]           # from the bank's subjects tables
REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")      # a Hugging Face repo id: organisation/name
ANNOTATION = ["text_id", "truncated_any", "digit_mass_min", *[f"{c}_exp" for c in ADELE_CODES]]
# model table column -> row column
MODEL_COLUMNS = {"kind": "model_kind", "provider": "model_provider", "release_date": "model_release_date",
                 "open_weights": "model_open_weights", "publication_date": "model_publication_date",
                 "organization": "model_organization", "country": "model_country",
                 "model_accessibility": "model_accessibility", "epoch_domain": "model_domain",
                 "base_model": "model_base_model", "parameters": "model_parameters",
                 "active_parameters": "model_active_parameters", "training_compute_flop": "model_training_compute_flop",
                 "training_dataset_size": "model_training_dataset_size",
                 "finetune_compute_flop": "model_finetune_compute_flop",
                 "post_training_compute_flop": "model_post_training_compute_flop",
                 "training_hardware": "model_training_hardware", "frontier_model": "model_frontier",
                 "architecture": "model_architecture", "moe": "model_moe", "n_experts": "model_n_experts",
                 "multimodal_input": "model_multimodal_input", "context_window": "model_context_window",
                 "knowledge_cutoff": "model_knowledge_cutoff", "thinking_mode": "model_thinking_mode"}
MODEL_FLOAT = {"model_parameters", "model_active_parameters", "model_training_compute_flop",
               "model_training_dataset_size", "model_finetune_compute_flop", "model_post_training_compute_flop"}
MODEL_INT = {"model_n_experts", "model_context_window"}
MODEL_BOOL = {"model_open_weights", "model_moe", "model_multimodal_input", "model_thinking_mode"}

S, F, B, I = pa.string(), pa.float64(), pa.bool_(), pa.int64()
SCHEMA = pa.schema(
    [(c, S) for c in KEYS]
    + [("score", F), ("n_runs", I), ("score_type", S), ("response_scale", S), ("normalization_policy", S),
       ("response_ids", pa.list_(S)), ("trials", pa.list_(I)), ("raw_scores", pa.list_(F))]
    + [(c, S) for c in ITEM_FIELDS] + [("item_features", S)]
    + [("text_id", S), ("truncated_any", B), ("digit_mass_min", F)] + [(f"{c}_exp", F) for c in ADELE_CODES]
    + [("domain", S), ("modality", S), ("multi_turn", B), ("agentic", B), ("benchmark_release_date", S),
       ("programming_language", S), ("tools", S)]
    + [("model_name", S), ("normalized_name", S), ("model_key", S), ("model_key_source", S), ("provider", S),
       ("release_date", S), ("access_date", S),
       ("harness", S), ("harness_version", S), ("reasoning_effort", S), ("thinking", B),
       ("subject_features_extra", S)]
    + [(c, F if c in MODEL_FLOAT else I if c in MODEL_INT else B if c in MODEL_BOOL else S)
       for c in MODEL_COLUMNS.values()])


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def guard_memory():
    with open("/proc/meminfo") as fh:
        gb = next(int(line.split()[1]) for line in fh if line.startswith("MemAvailable:")) / 2 ** 20
    if gb < MIN_FREE_RAM_GB:
        raise MemoryError(f"only {gb:.1f} GB of memory available: stopping, rerun with --resume")


def write_json(obj, path):
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def missing(v):
    """None, NaN (pandas 3 stores a missing text value as NaN) or pandas' NA / NaT."""
    return v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and v != v)


def as_text(v):
    """Strings as they are; lists and dicts as JSON text; other scalars (dates, numbers) as str; None when missing."""
    if missing(v):
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, (list, dict)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    return str(v)


def as_bool(v):
    """True / False (Python, NumPy or the strings "true" / "false"); None when missing; anything else is an error."""
    if missing(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return v.lower() == "true"
    raise ValueError(f"not a boolean: {v!r}")


def as_int(v):
    """An integer (also from an integral float such as 128000.0); None when missing; anything else is an error."""
    if missing(v):
        return None
    if isinstance(v, float) and not v.is_integer():
        raise ValueError(f"not an integer: {v!r}")
    return int(v)


def to_table(df):
    """df as an Arrow table with exactly SCHEMA: every value converted to its column's type, or an error naming the
    column."""
    cols = {}
    for f in SCHEMA:
        v = df[f.name] if f.name in df else pd.Series([None] * len(df), index=df.index, dtype=object)
        if f.type == S:                              # on Python lists: pandas 3 would turn None back into NaN
            v = [as_text(x) for x in v.tolist()]
        elif f.type == B:
            v = [as_bool(x) for x in v.tolist()]
        elif f.type == F:
            v = [None if missing(x) else float(x) for x in pd.to_numeric(v, errors="raise").tolist()]
        elif f.type == I:
            v = [as_int(x) for x in pd.to_numeric(v, errors="raise").tolist()]
        else:                                        # lists
            v = v.tolist()
        try:
            cols[f.name] = pa.array(v, type=f.type)
        except (pa.ArrowException, ValueError) as err:
            bad = next((x for x in v if x is not None), None)
            raise ValueError(f"column {f.name} does not fit {f.type}: {err} (e.g. {bad!r})") from err
    return pa.table(cols, schema=SCHEMA)


def write_table(table, path):
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


# ----------------------------------------------------------------------------- reading the converter's JSONL quickly

_ID = b'"item_id":"'


def raw_item_id(line):
    """The item_id of a JSONL row by string search, without parsing the row (the question can be megabytes long)."""
    i = line.find(_ID)
    if i < 0:
        return None
    j = k = i + len(_ID)
    while True:
        k = line.find(b'"', k)
        if k < 0:
            return None
        bs = 0
        while line[k - 1 - bs] == 0x5C:      # an odd run of backslashes escapes the quote
            bs += 1
        if bs % 2 == 0:
            break
        k += 1
    raw = line[j:k]
    return raw.decode("utf-8") if b"\\" not in raw else orjson.loads(b'"' + raw + b'"')


def parse_small(line):
    """The row without its two long text fields: the converter writes its columns in a fixed order
    (common.OBSERVATION_COLUMNS), with the text between ',"question":' and ',"eval_type":'. Those markers cannot occur
    inside a JSON string (its quotes are escaped). Falls back to a full parse."""
    q = line.find(b',"question":')
    e = line.find(b',"eval_type":', q + 1) if q >= 0 else -1
    if q < 0 or e < 0:
        return orjson.loads(line)
    head = orjson.loads(line[:q] + b"}")
    head.update(orjson.loads(b"{" + line[e + 1:]))
    return head


# ----------------------------------------------------------------------------- pass 1: one benchmark (a worker)

def process(job):
    """Read one benchmark's JSONL; keep the annotated items' graded runs; one row per cell with the mean score and the
    runs as lists; join the item's text and annotation. Writes parts/<benchmark>.pkl and its info (last)."""
    bench, run, jsonl, adele_path, parts, revision = job
    adele = pd.read_parquet(adele_path)
    annotated = set(adele["item_id"])
    cells, items, ungraded_cells, mismatch = {}, {}, set(), {}
    info = {"benchmark": bench, "run": run, "jsonl": jsonl, "source_revision": revision,
            "annotated_items": len(annotated), "rows_read": 0, "runs_kept": 0, "ungraded_runs": 0}
    with open(jsonl, "rb") as fh:
        for k, line in enumerate(fh):
            if k % 500_000 == 0:
                guard_memory()
            info["rows_read"] += 1
            iid = raw_item_id(line)
            if iid not in annotated:
                continue
            if iid not in items:                       # first row of this item: full parse for its text
                r = orjson.loads(line)
                unknown = set(r) - JSONL_FIELDS
                if unknown:
                    raise ValueError(f"{bench}: JSONL columns this script does not know: {sorted(unknown)}")
                items[iid] = {f: r.get(f) for f in LONG}
            else:
                r = parse_small(line)
            key = (r["subject_id"], iid, r.get("test_condition"), as_text(r.get("interactors")))
            score = r.get("score")
            if score is None or not 0 <= score <= 1:   # ungraded or invalid: not a model outcome
                info["ungraded_runs"] += 1
                ungraded_cells.add(key)
                continue
            fields = {f: r.get(f) for f in CELL_FIELDS + ["eval_type"] + BENCH_FIELDS + SUBJECT_FIELDS}
            c = cells.get(key)
            if c is None:
                cells[key] = c = {"fields": fields, "ids": [], "trials": [], "raws": [], "sum": 0.0}
            elif fields != c["fields"]:                # should not happen: a cell's runs share these fields
                for f, v in fields.items():
                    if v != c["fields"][f]:
                        mismatch[f] = mismatch.get(f, 0) + 1
            c["ids"].append(r["response_id"])
            c["trials"].append(r.get("trial"))
            c["raws"].append(r.get("raw_score"))
            c["sum"] += score
            info["runs_kept"] += 1
    rows = []
    for (sid, iid, cond, inter), c in cells.items():
        n = len(c["ids"])
        rows.append({"benchmark": bench, "item_id": iid, "subject_id": sid, "test_condition": cond,
                     "interactors": inter, "score": c["sum"] / n, "n_runs": n, **c["fields"],
                     "response_ids": c["ids"], "trials": c["trials"], "raw_scores": c["raws"], **items[iid]})
    columns = ["benchmark", "item_id", "subject_id", "test_condition", "interactors", "score", "n_runs",
               *CELL_FIELDS, "eval_type", *BENCH_FIELDS, *SUBJECT_FIELDS, "response_ids", "trials", "raw_scores", *LONG]
    df = pd.DataFrame(rows, columns=columns)           # columns also when no cell is left (all runs ungraded)
    df = df.merge(adele[["item_id", *[a for a in ANNOTATION if a in adele]]], on="item_id", how="left")
    df.to_pickle(Path(parts) / f"{bench}.pkl")        # values kept as they are until the final, typed write
    info.update(cells=len(df), cells_without_graded_runs=len(ungraded_cells - set(cells)),
                cells_with_ungraded_runs=len(ungraded_cells & set(cells)),
                items=int(df["item_id"].nunique()) if len(df) else 0,
                subjects=int(df["subject_id"].nunique()) if len(df) else 0,
                binary_share=float((df["score_type"] == "binary").mean()) if len(df) else 0.0,
                mean_score=float((df["score"] * df["n_runs"]).sum() / df["n_runs"].sum()) if len(df) else None,
                score_types=json.dumps(df["score_type"].value_counts().to_dict() if len(df) else {}),
                fields_differing_in_a_cell=json.dumps(mismatch))
    write_json(info, Path(parts) / f"{bench}.json")   # last: marks the benchmark as read
    return info


# ----------------------------------------------------------------------------- inputs

def conversions(run, forced=None):
    """Conversions of a run (no-assets or asset-image), newest first: (folder, its manifest)."""
    dirs = [Path(forced)] if forced else sorted((BANK / run).iterdir(), reverse=True)
    return [(d, json.loads((d / "measurement_db.manifest.json").read_text())) for d in dirs
            if (d / "measurement_db.manifest.json").exists()]


def jobs(benchmarks, parts, forced):
    """(benchmark, run, jsonl, annotation table, parts folder, source revision) for every annotated benchmark: its
    JSONL from the newest conversion of its run that wrote it completely."""
    out, absent = [], []
    for run in RUNS:
        convs = conversions(run, forced.get(run))
        for table in sorted((ADELE / run).glob("*.parquet")):
            b = table.stem
            if benchmarks and b not in benchmarks:
                continue
            hit = next(((d, m) for d, m in convs if any(o["benchmark"] == b for o in m.get("outputs", []))), None)
            if hit:
                d, m = hit
                out.append((b, run, str(d / f"{b}.jsonl"), str(table), str(parts), m["source_revision"]))
            else:
                absent.append(b)
    if absent:
        log(f"{len(absent)} annotated benchmarks have no finished JSONL: {', '.join(absent)}")
    return out


def hub_subjects(bench, revision):
    """The bank's subjects table of one benchmark (gated): subject_id, its normalized_name there and HUB_FIELDS."""
    path = hf_hub_download(REPOSITORY, f"{bench}/formatted_tables/subjects.parquet", repo_type="dataset",
                           revision=revision)
    t = pd.read_parquet(path)
    out = pd.DataFrame({"subject_id": t["subject_id"],
                        "hub_normalized_name": t["normalized_name"].map(as_text) if "normalized_name" in t else None})
    for c in HUB_FIELDS:
        out[c] = t[c].map(as_text) if c in t else None
    return out


def hub_items(bench, revision):
    """item_id (ours: <benchmark>:<the bank's id>) and item_features from the bank's items table (gated)."""
    path = hf_hub_download(REPOSITORY, f"{bench}/formatted_tables/items.parquet", repo_type="dataset",
                           revision=revision)
    cols = [c for c in ("item_id", "item_features") if c in pq.read_schema(path).names]
    t = pd.read_parquet(path, columns=cols)
    return pd.DataFrame({"item_id": bench + ":" + t["item_id"].astype(str),
                         "item_features": t["item_features"].map(as_text) if "item_features" in t else None})


def name_key(name):
    """A model name reduced for matching: lower case; the part after the last " / " (benchmark prefixes) and after the
    last "/" or "__" (organisations); no trailing date; only letters, digits and dots."""
    n = str(name).lower().split(" / ")[-1]
    n = re.split(r"/|__", n)[-1]
    n = re.sub(r"[-_ ]?(20\d\d[-_]?\d\d[-_]?\d\d|\d{4})$", "", n)
    return re.sub(r"[^a-z0-9.]", "", n)


def clean_name(name):
    """model_name without a benchmark prefix, with "org__repo" written "org/repo"."""
    n = str(name).split(" / ")[-1].strip()
    return n.replace("__", "/", 1) if "/" not in n and "__" in n else n


def catalogue_names(offline):
    """name_key -> a model name in Epoch AI (first) or models.dev."""
    if not offline:
        E.download()
    out = {}
    for m in E.clean()["Model"]:
        for n in (m, re.sub(r"\s*\([^)]*\)", "", m)):
            out.setdefault(name_key(n), m)
    for rec in E.models_dev_records(offline):
        for n in (rec.get("name"), rec.get("id")):
            if isinstance(n, str):
                out.setdefault(name_key(n), rec.get("name") or n)
    out.pop("", None)
    return out


def model_keys(subjects, catalogue):
    """model_key and model_key_source of every subject row (see the module docstring)."""
    s = subjects
    named = s[s["normalized_name"].notna()]
    pairs = pd.concat([
        pd.DataFrame({"k": named["model_name"].map(name_key), "name": named["normalized_name"]}),
        pd.DataFrame({"k": named["normalized_name"].str.split(" ", n=1).str[-1].map(name_key),
                      "name": named["normalized_name"]})]).drop_duplicates()
    unique = pairs.groupby("k")["name"].nunique()
    from_named = pairs[pairs["k"].isin(unique[unique == 1].index) & (pairs["k"] != "")] \
        .drop_duplicates("k").set_index("k")["name"]
    k = s["model_name"].map(name_key)
    key = s["normalized_name"].astype(object).where(s["normalized_name"].notna(), None)
    source = pd.Series(["bank" if v is not None else None for v in key], index=s.index, dtype=object)
    for src, lut in (("named_subject", from_named), ("catalogue", pd.Series(catalogue, dtype=object))):
        hit = source.isna() & k.isin(lut.index)
        key[hit] = k[hit].map(lut)
        source[hit] = src
    rest = source.isna()
    cleaned = s.loc[rest, "model_name"].map(clean_name)
    group = k[rest].where(k[rest] != "", cleaned.str.lower())
    canonical = cleaned.groupby(group).agg(lambda v: v.value_counts().index[0])
    key[rest] = group.map(canonical)
    source[rest] = "derived"
    return key, source


def models_of(subjects):
    """One row per model_key with the columns epoch_models_csv.model_table expects (as its benchmark_models)."""
    s = subjects.assign(normalized_name=subjects["model_key"])
    first = lambda v: v.dropna().iloc[0] if v.notna().any() else None   # noqa: E731
    m = (s.groupby("normalized_name")
         .agg(provider=("provider", first), release_date=("release_date", first),
              benchmarks=("benchmark", lambda v: ";".join(sorted(set(v)))),
              display_names=("model_name", lambda v: ";".join(sorted(set(v.dropna())))),
              n_subjects=("subject_id", "size"),
              runs_thinking=("thinking", lambda v: True if v.eq(True).any()
                             else (False if v.eq(False).any() else None)),
              derived=("model_key_source", lambda v: (v == "derived").all()))
         .reset_index())
    repo = m["derived"] & m["normalized_name"].str.match(REPO_ID)
    m.loc[repo & m["provider"].isna(), "provider"] = m["normalized_name"].str.split("/").str[0]
    m["kind"] = [("agent system" if n.startswith("Agent: ") else E.NOT_A_MODEL.get(n, "model"))
                 for n in m["normalized_name"]]
    return m


# ----------------------------------------------------------------------------- main

README = __doc__.split("Output, one folder per build: ")[1].split("Every file has the same fixed schema")[0]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4, help="benchmarks read in parallel (each holds one file's cells)")
    ap.add_argument("--benchmarks", nargs="*", help="default: every annotated benchmark with a JSONL")
    ap.add_argument("--jsonl", nargs="*", default=[], metavar="RUN=DIR",
                    help="conversion folder per run (default: per benchmark, the newest that wrote it)")
    ap.add_argument("--offline", action="store_true", help="model sources from their local caches only")
    ap.add_argument("--output-root", type=Path, default=OUTPUT, help="parent of the build folder (tests)")
    ap.add_argument("--resume", type=Path, help="continue an interrupted build in this folder")
    return ap.parse_args()


def main():
    a = parse_args()
    if a.resume:
        out = a.resume
        manifest = json.loads((out / "manifest.json").read_text())
        forced, benchmarks = manifest["settings"]["jsonl"], manifest["settings"]["benchmarks"]
    else:
        forced, benchmarks = dict(x.split("=", 1) for x in a.jsonl), a.benchmarks
        out = a.output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        out.mkdir(parents=True)
        manifest = {"started_at_utc": utc_now(), "status": "running",
                    "inputs": {"adele": str(ADELE), "source_repository": REPOSITORY},
                    "settings": {"benchmarks": benchmarks, "jsonl": forced, "offline": a.offline}}
    try:
        HfApi().whoami()
    except Exception:  # noqa: BLE001 - any failure means no usable token
        sys.exit("no Hugging Face login: run `poetry run hf auth login` (provider and release_date are on the bank)")
    log(f"building -> {out}")
    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    todo = jobs(benchmarks, parts, forced)
    manifest["inputs"]["jsonl"] = {j[0]: j[2] for j in todo}
    manifest["inputs"]["source_revision"] = sorted({j[5] for j in todo})
    write_json(manifest, out / "manifest.json")

    # 1. read every benchmark: one row per cell with its item and annotation, one benchmark per worker
    left = [j for j in todo if not (parts / f"{j[0]}.json").exists()]
    left.sort(key=lambda j: -Path(j[2]).stat().st_size)          # largest first
    log(f"1. reading {len(left)} of {len(todo)} benchmarks, {a.workers} workers")
    try:
        with Pool(a.workers, maxtasksperchild=1) as pool:
            for k, info in enumerate(pool.imap_unordered(process, left)):
                log(f"  {k + 1}/{len(left)} {info['benchmark']}: {info['runs_kept']:,} of {info['rows_read']:,} "
                    f"rows kept -> {info['cells']:,} cells, {info['items']} items, {info['subjects']} subjects")
    except MemoryError as err:
        sys.exit(f"stopped: {err}")
    infos = [json.loads((parts / f"{j[0]}.json").read_text()) for j in todo]

    # 2. the bank's subject fields (provider, release_date, subject_features_extra) and item_features, per benchmark
    log("2. subject fields and item_features from the bank's subjects and items tables (Hugging Face)")
    hub = {j[0]: hub_subjects(j[0], j[5]) for j in todo}
    hub_it = {j[0]: hub_items(j[0], j[5]) for j in todo}

    # 3. model attributes, for every normalized_name of the build
    log("3. model attributes (epoch_models_csv.model_table)")
    cols = ["subject_id", *SUBJECT_FIELDS]
    subjects = pd.concat([pd.read_pickle(parts / f"{j[0]}.pkl")[cols].drop_duplicates("subject_id")
                          .merge(hub[j[0]], on="subject_id", how="left").assign(benchmark=j[0]) for j in todo],
                         ignore_index=True)
    subjects["model_key"], subjects["model_key_source"] = model_keys(subjects, catalogue_names(a.offline))
    log("  model_key sources (subjects): " + json.dumps(subjects["model_key_source"].value_counts().to_dict()))
    ms = models_of(subjects)
    hub_query = ~ms["derived"] | ms["normalized_name"].str.match(REPO_ID)
    models = pd.concat([E.model_table(ms[hub_query].drop(columns="derived"), offline=a.offline, hf_refresh=False),
                        E.model_table(ms[~hub_query].drop(columns="derived"), offline=True)], ignore_index=True)
    mrow = models.drop_duplicates("normalized_name").set_index("normalized_name")[list(MODEL_COLUMNS)] \
        .rename(columns=MODEL_COLUMNS)
    keys = subjects.set_index(["benchmark", "subject_id"])[["model_key", "model_key_source"]]

    # 4. one file per benchmark, every column on every row, written with the fixed schema
    log("4. writing one file per benchmark")
    counts = {"runs": 0, "cells": 0}
    for info in infos:
        b = info["benchmark"]
        if not info["cells"]:                          # nothing graded: listed in benchmarks.parquet, no file
            log(f"  {b}: no graded runs, no file")
            continue
        h = hub[b]
        df = pd.read_pickle(parts / f"{b}.pkl").merge(h, on="subject_id", how="left") \
            .merge(hub_it[b], on="item_id", how="left")
        info["items_missing_in_bank"] = int((~df["item_id"].drop_duplicates().isin(hub_it[b]["item_id"])).sum())
        differ = [as_text(x) != as_text(y) for x, y in zip(df["hub_normalized_name"], df["normalized_name"])]
        info["subjects_normalized_name_differs_from_bank"] = int(df.loc[differ, "subject_id"].nunique())
        info["subjects_missing_in_bank"] = int((~df["subject_id"].drop_duplicates().isin(h["subject_id"])).sum())
        df = df.join(keys.loc[b], on="subject_id").join(mrow, on="model_key")
        write_table(to_table(df), out / f"{b}.parquet")
        counts["runs"] += info["runs_kept"]
        counts["cells"] += len(df)
    bench = pd.DataFrame(infos)
    bench.to_parquet(out / "benchmarks.parquet", index=False)
    models.map(as_text).astype(object).to_parquet(out / "models.parquet", index=False)   # reference: all as text
    (out / "README.txt").write_text(f"Assessor dataset, build {out.name}\n\n{README}", encoding="utf-8")
    manifest.update(status="complete", finished_at_utc=utc_now(), counts={
        "benchmarks": len(bench), **counts, "items": int(bench["items"].sum()),
        "ungraded_runs_dropped": int(bench["ungraded_runs"].sum()),
        "cells_dropped": int(bench["cells_without_graded_runs"].sum()),
        "subjects": int(subjects["subject_id"].nunique()),
        "subjects_with_normalized_name": int(subjects.drop_duplicates("subject_id")["normalized_name"].notna().sum()),
        "model_key_sources": subjects["model_key_source"].value_counts().to_dict(),
        "models": len(models)})
    write_json(manifest, out / "manifest.json")
    for j in todo:                                               # the final files hold everything
        (parts / f"{j[0]}.pkl").unlink()
        (parts / f"{j[0]}.json").unlink()
    parts.rmdir()
    log(f"done: {json.dumps(manifest['counts'])} -> {out}")


if __name__ == "__main__":
    main()
