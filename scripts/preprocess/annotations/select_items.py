"""Choose the items of each benchmark to annotate with ADeLe and write them in the compact form the engines read.

    python select_items.py --jsonl-dir <converter run folder> --root <annotation root>
        (--benchmarks a b ... | --benchmark-file list.txt | --all) (--max-items N | --all-items | --item-list items.csv)
        [--seed 0] [--workers 8]

Runs where the converter's JSONL files are (vent), on the CPU. Input: <jsonl-dir>/<benchmark>.jsonl, one row per
response, item text in `question` (scripts/preprocess/measurement_db.py). Output per benchmark, in <root>/<benchmark>/:
items.jsonl (distinct texts), item_map.csv (benchmark, item_id, text_id) and selection.json; see annotate_common.py.

Selection, deterministic and identical on every machine: an item is chosen if its key, the SHA-1 of "<seed>:<item_id>",
is among the --max-items smallest of the benchmark (--all-items: every item; --item-list: exactly the listed ones).
Items with an empty text are skipped and counted. Identical texts are annotated once (text_id = SHA-1 of the text).

Memory: pass 1 streams the file once and keeps only the item ids, the offset of each item's first row and the chosen
keys; pass 2 seeks to the chosen rows and writes them one by one. No benchmark's texts are held in memory, whatever the
file size (the largest no-assets file is 146 GB). Output goes to temporary names and is renamed at the end, so a
benchmark folder either has a complete selection or none. Prints counts only, never item text.
"""

import argparse
import csv
import hashlib
import heapq
import json
import time
from multiprocessing import Pool
from pathlib import Path

import orjson

ITEM_KEY = b'"item_id":'


def _item_id(line):
    """The raw item_id of a JSONL row by string search (item_id precedes question in the converter's rows, and quotes
    inside strings are escaped, so the pattern cannot match inside a text); None if absent."""
    i = line.find(ITEM_KEY)
    if i < 0:
        return None
    j = line.find(b'"', i + len(ITEM_KEY))
    if j < 0:
        return None
    k = j
    while True:                                   # the closing quote: one not escaped by an odd run of backslashes
        k = line.find(b'"', k + 1)
        if k < 0:
            return None
        n_bs = 0
        while line[k - 1 - n_bs] == 0x5C:         # backslash
            n_bs += 1
        if n_bs % 2 == 0:
            return line[j + 1:k]


def _decode(raw):
    """The item_id as a string: the raw JSON bytes unescaped (pandas writes "/" as an escaped slash, for instance)."""
    try:
        return orjson.loads(b'"' + raw + b'"')
    except orjson.JSONDecodeError:
        return raw.decode("utf-8", "replace")


def _key(seed, item_id):
    return int.from_bytes(hashlib.sha1(f"{seed}:{item_id}".encode("utf-8")).digest()[:8], "big")


def scan(path, max_items, seed, wanted):
    """Pass 1: (distinct items, chosen [(raw_id, offset)]) for one file."""
    seen, heap, chosen = set(), [], []
    pos = 0
    with open(path, "rb") as fh:
        for line in fh:
            start = pos
            pos += len(line)
            raw = _item_id(line)
            if raw is None:                                   # unusual layout: parse to find the id
                r = orjson.loads(line)
                raw = orjson.dumps(r.get("item_id"))[1:-1]
            if raw in seen:
                continue
            seen.add(raw)
            if wanted is not None:
                if _decode(raw) in wanted:
                    chosen.append((raw, start))
            elif not max_items:
                chosen.append((raw, start))
            else:
                k = _key(seed, _decode(raw))
                if len(heap) < max_items:
                    heapq.heappush(heap, (-k, raw, start))
                elif -heap[0][0] > k:
                    heapq.heapreplace(heap, (-k, raw, start))
    if wanted is None and max_items:
        chosen = [(raw, start) for _, raw, start in heap]
    return len(seen), chosen


def select(job):
    """One benchmark: both passes and the output files. Returns its summary."""
    path, bdir, max_items, seed, wanted = job
    t0 = time.time()
    n_items, chosen = scan(path, max_items, seed, wanted)
    bdir = Path(bdir)
    bdir.mkdir(parents=True, exist_ok=True)
    tmp_items, tmp_map = bdir / ".items.jsonl.tmp", bdir / ".item_map.csv.tmp"
    texts, n_empty, n_mapped, chars, longest = set(), 0, 0, 0, 0
    with open(path, "rb") as src, open(tmp_items, "w", encoding="utf-8", newline="\n") as fi, \
            open(tmp_map, "w", encoding="utf-8", newline="\n") as fm:
        mapw = csv.writer(fm, lineterminator="\n")
        mapw.writerow(["benchmark", "item_id", "text_id"])
        for raw, start in sorted(chosen, key=lambda x: x[1]):
            src.seek(start)
            r = orjson.loads(src.readline())
            q = r.get("question") or ""
            if not q.strip():
                n_empty += 1
                continue
            tid = hashlib.sha1(q.encode("utf-8")).hexdigest()[:16]
            mapw.writerow([bdir.name, str(r["item_id"]), tid])
            n_mapped += 1
            if tid not in texts:
                texts.add(tid)
                fi.write(json.dumps({"text_id": tid, "content": q}, ensure_ascii=False) + "\n")
                chars += len(q)
                longest = max(longest, len(q))
    info = {"benchmark": bdir.name, "source": str(path), "source_bytes": Path(path).stat().st_size,
            "distinct_items": n_items, "chosen": len(chosen), "empty_skipped": n_empty, "items": n_mapped,
            "texts": len(texts), "text_chars": chars, "longest_text_chars": longest,
            "rule": ("item list" if wanted is not None else "all items" if not max_items
                     else f"{max_items} smallest SHA-1(seed:item_id)"), "seed": seed,
            "seconds": round(time.time() - t0, 1), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp_items.replace(bdir / "items.jsonl")
    tmp_map.replace(bdir / "item_map.csv")
    (bdir / "selection.json").write_text(json.dumps(info, indent=1))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-dir", required=True, help="converter run folder with <benchmark>.jsonl files")
    ap.add_argument("--root", required=True, help="annotation root to write to (never inside --jsonl-dir)")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--benchmarks", nargs="+")
    who.add_argument("--benchmark-file", help="one benchmark name per line")
    who.add_argument("--all", action="store_true", help="every <benchmark>.jsonl in --jsonl-dir")
    how = ap.add_mutually_exclusive_group(required=True)
    how.add_argument("--max-items", type=int, help="items per benchmark (deterministic hash selection)")
    how.add_argument("--all-items", action="store_true", help="every item (can be millions: check the EDA first)")
    how.add_argument("--item-list", help="CSV with benchmark,item_id: exactly these items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="benchmarks processed in parallel")
    ap.add_argument("--overwrite", action="store_true", help="redo benchmarks that already have a selection")
    a = ap.parse_args()

    src, root = Path(a.jsonl_dir).resolve(), Path(a.root).resolve()
    if root == src or src in root.parents:
        raise SystemExit("--root must not be inside --jsonl-dir (the shared data folder is read-only for this script)")
    if a.all:
        names = sorted(p.stem for p in src.glob("*.jsonl"))
    elif a.benchmark_file:
        names = [x.strip() for x in Path(a.benchmark_file).read_text().splitlines() if x.strip() and not x.startswith("#")]
    else:
        names = list(a.benchmarks)
    missing = [b for b in names if not (src / f"{b}.jsonl").exists()]
    if missing:
        raise SystemExit(f"not in {src}: {missing}")
    wanted = {}
    if a.item_list:
        import pandas as pd
        lst = pd.read_csv(a.item_list, usecols=["benchmark", "item_id"], dtype=str)
        wanted = lst.groupby("benchmark")["item_id"].apply(set).to_dict()
    jobs = []
    for b in names:
        if (root / b / "items.jsonl").exists() and not a.overwrite:
            print(f"  {b}: already selected, skipped (--overwrite to redo)")
            continue
        if (root / b / "parts").exists() and any((root / b / "parts").glob("part-*.csv")):
            raise SystemExit(f"{root / b} already has annotation parts: a new selection would not match them")
        jobs.append((str(src / f"{b}.jsonl"), str(root / b), 0 if a.all_items else (a.max_items or 0), a.seed,
                     wanted.get(b, set()) if a.item_list else None))
    jobs.sort(key=lambda j: -Path(j[0]).stat().st_size)          # biggest first
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(min(a.workers, max(1, len(jobs))), maxtasksperchild=1) as pool:
        for k, info in enumerate(pool.imap_unordered(select, jobs)):
            print(f"  [{time.time() - t0:5.0f}s] {k + 1}/{len(jobs)} {info['benchmark']}: {info['distinct_items']:,} "
                  f"items, {info['chosen']:,} chosen, {info['empty_skipped']} with empty text skipped, "
                  f"{info['items']:,} kept as {info['texts']:,} distinct texts, longest "
                  f"{info['longest_text_chars']:,} chars ({info['seconds']:.0f}s)", flush=True)
    print(f"done: {len(jobs)} benchmarks selected into {root}")


if __name__ == "__main__":
    main()
