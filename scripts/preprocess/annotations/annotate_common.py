"""Shared pieces of the ADeLe v1.0 annotation engines (annotate_vllm.py on vent, annotate_hf.py on the cluster).

Layout of an annotation root (one folder per benchmark, written by select_items.py, then by the engines):

    <root>/<benchmark>/items.jsonl        distinct texts to annotate: {"text_id", "content"} per line
    <root>/<benchmark>/item_map.csv       benchmark, item_id, text_id (several items can share one text)
    <root>/<benchmark>/selection.json     how the items were selected (counts, seed, source file)
    <root>/<benchmark>/parts/part-*.csv   annotations as they are produced, one row per (text_id, rubric)
    <root>/<benchmark>/adele_v1.parquet   final table, one row per item (written when every pair is scored)
    <root>/<benchmark>/done.json          summary of the finished benchmark
    <root>/<benchmark>/LOCK               present while an engine works on the benchmark

Safety rules both engines follow:
- texts are read from items.jsonl in chunks of at most --batch-chars characters, and prompts are built for one rubric
  at a time, so the prompts in memory never exceed one chunk × one rubric;
- before every batch, the free RAM of the machine (MemAvailable) is checked; below --min-free-ram-gb the engine stops
  cleanly, keeping everything written so far (rerun to resume);
- a benchmark is worked on by one process at a time (LOCK, created atomically); a lock left by a crashed run is
  removed with --steal-lock after checking that the process is gone;
- parts are written to a temporary name and renamed, so a crash never leaves a half-written part.
"""

import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

from adele_prompt import RUBRICS_V1, done_pairs, load_rubrics, prompt_ids, skeletons  # noqa: F401  (re-exported)

FINAL = "adele_v1.parquet"


class MemoryGuard(RuntimeError):
    """Free RAM fell below the configured floor: stop cleanly (the work so far is saved)."""


class Locked(RuntimeError):
    """Another process holds the benchmark: skip it and go on with the others."""


def add_common_args(ap):
    ap.add_argument("--root", required=True, help="annotation root (select_items.py output)")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--benchmarks", nargs="+", help="one or more benchmark names")
    who.add_argument("--benchmark-file", help="file with one benchmark name per line")
    who.add_argument("--all", action="store_true", help="every benchmark under --root that has items.jsonl")
    ap.add_argument("--rubrics", nargs="+", default=[str(RUBRICS_V1)], help="rubric folder(s); default the 18 v1")
    ap.add_argument("--max-len", type=int, default=32767, help="prompt window in tokens (same on both engines)")
    ap.add_argument("--tail-frac", type=float, default=0.75, help="share of a cut item's budget kept from its end")
    ap.add_argument("--batch-chars", type=int, default=20_000_000,
                    help="characters of item text per chunk (bounds the prompts held in memory)")
    ap.add_argument("--min-free-ram-gb", type=float, default=6.0,
                    help="stop cleanly if the machine's free RAM falls below this")
    ap.add_argument("--steal-lock", action="store_true",
                    help="remove a LOCK left by a process that no longer exists (same host only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="read, tokenise and batch everything without loading the model; reports prompts and tokens")


def require_parquet():
    """Fail at start, not after hours of scoring: finalize() writes Parquet, which needs pyarrow (missing in the
    cluster's distill-ft env on 2026-10-24: 20 tasks failed at their last step)."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit("pyarrow is not installed in this environment: the final tables are Parquet files. "
                         "Install it first (pip install pyarrow).")


def benchmarks_from(a):
    root = Path(a.root)
    if a.all:
        names = sorted(p.parent.name for p in root.glob("*/items.jsonl"))
    elif a.benchmark_file:
        names = [x.strip() for x in Path(a.benchmark_file).read_text().splitlines() if x.strip() and not x.startswith("#")]
    else:
        names = list(a.benchmarks)
    missing = [b for b in names if not (root / b / "items.jsonl").exists()]
    if missing:
        raise SystemExit(f"no items.jsonl under {root} for: {missing} (run select_items.py first)")
    return names


def mem_available_gb():
    """MemAvailable from /proc/meminfo in GB (None where it does not exist, e.g. Windows)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        return None
    return None


def check_ram(floor_gb):
    free = mem_available_gb()
    if free is not None and free < floor_gb:
        raise MemoryGuard(f"free RAM {free:.1f} GB is below the floor of {floor_gb:.1f} GB: stopping cleanly; "
                          "everything written so far is kept, rerun to resume")
    return free


def read_items(path):
    """Distinct texts of a benchmark, streamed: {"text_id", "content"} dicts."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def chunks(items, max_chars):
    """Consecutive lists of items whose texts total at most max_chars characters (a longer single text is alone)."""
    batch, size = [], 0
    for it in items:
        n = len(it["content"])
        if batch and size + n > max_chars:
            yield batch
            batch, size = [], 0
        batch.append(it)
        size += n
    if batch:
        yield batch


class Lock:
    """<benchmark>/LOCK created atomically; refuses to proceed if another live process holds it."""

    def __init__(self, bdir, steal=False):
        self.path = Path(bdir) / "LOCK"
        self.steal = steal
        self.held = False

    def __enter__(self):
        host = socket.gethostname()
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as fh:
                    json.dump({"host": host, "pid": os.getpid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh)
                self.held = True
                return self
            except FileExistsError:
                try:
                    info = json.loads(self.path.read_text() or "{}")
                except (OSError, json.JSONDecodeError):
                    info = {}                       # unreadable: treat as held by an unknown live process
                alive = not info or info.get("host") != host or _pid_alive(info.get("pid", -1))
                if self.steal and not alive:
                    self.path.unlink()
                    continue
                raise Locked(f"{self.path} is held by {info or 'an unknown process'} "
                             f"({'alive or on another host' if alive else 'dead: rerun with --steal-lock to remove it'})")
        raise Locked(f"could not acquire {self.path}")

    def __exit__(self, *exc):
        if self.held:
            self.path.unlink(missing_ok=True)
        return False


def _pid_alive(pid):
    """Whether a local process exists (signal 0 checks without signalling). Only on POSIX: on Windows, os.kill with 0
    would send Ctrl-C to the process, so there the lock is always treated as held."""
    if os.name != "posix":
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def write_part(parts_dir, rows):
    """Write rows to the next part-NNNNN.csv atomically (temporary name, then rename)."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    k = len(list(parts_dir.glob("part-*.csv")))
    tmp = parts_dir / f".part-{k:05d}.csv.tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(parts_dir / f"part-{k:05d}.csv")


def finalize(bdir, codes):
    """When every (text, rubric) pair has a score, write adele_v1.parquet (one row per item) and done.json; returns
    the number of missing pairs (0 = finished)."""
    bdir = Path(bdir)
    text_ids = [it["text_id"] for it in read_items(bdir / "items.jsonl")]
    parts = sorted((bdir / "parts").glob("part-*.csv"))
    long = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True) if parts else pd.DataFrame(
        columns=["text_id", "rubric"])
    long = long.drop_duplicates(["text_id", "rubric"], keep="last")
    have = set(zip(long["text_id"], long["rubric"]))
    missing = sum((t, c) not in have for t in text_ids for c in codes)
    if missing:
        return missing
    long = long[long["text_id"].isin(set(text_ids)) & long["rubric"].isin(codes)]
    P = long[[f"p{k}" for k in range(6)]].to_numpy(float)
    long["exp"] = P @ np.arange(6)
    long["argmax"] = P.argmax(axis=1)
    wide = long.pivot(index="text_id", columns="rubric", values=["exp", "argmax"] + [f"p{k}" for k in range(6)])
    wide.columns = [f"{rub}_{stat}" for stat, rub in wide.columns]
    order = [f"{c}_{s}" for c in codes for s in ["exp", "argmax"] + [f"p{k}" for k in range(6)]]
    g = long.groupby("text_id")
    wide = wide[order].join(pd.DataFrame({"n_tokens_max": g["n_tokens"].max(), "truncated_any": g["truncated"].any(),
                                          "digit_mass_min": g["digit_mass"].min()}))
    items = pd.read_csv(bdir / "item_map.csv").join(wide, on="text_id")
    tmp = bdir / f".{FINAL}.tmp"
    items.to_parquet(tmp, index=False)
    tmp.replace(bdir / FINAL)
    (bdir / "done.json").write_text(json.dumps({
        "benchmark": bdir.name, "items": len(items), "texts": len(text_ids), "rubrics": len(codes),
        "truncated_texts": int(wide["truncated_any"].sum()), "digit_mass_median": float(long["digit_mass"].median()),
        "digit_mass_p01": float(long["digit_mass"].quantile(0.01)), "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1))
    return 0


def run_benchmark(bdir, codes, tok, sk, a, score_fn, log=print):
    """Annotate one benchmark folder: chunks of texts × one rubric at a time; score_fn(rows) -> list of
    (p[6], digit_mass) for the rows (None in a dry run). Returns (prompts scored, prompt tokens)."""
    bdir = Path(bdir)
    done = done_pairs(bdir / "parts")
    n_prompts = n_tok = 0
    t0 = time.time()
    for ci, chunk in enumerate(chunks(read_items(bdir / "items.jsonl"), a.batch_chars)):
        for code in codes:
            free = check_ram(a.min_free_ram_gb)
            rows = [r for r in prompt_ids(tok, chunk, {code: sk[code]}, a.max_len, a.tail_frac)
                    if (r["text_id"], code) not in done]
            if not rows:
                continue
            n_prompts += len(rows)
            n_tok += sum(len(r["ids"]) for r in rows)
            if score_fn is not None:
                out = score_fn(rows)
                write_part(bdir / "parts", [
                    {"text_id": r["text_id"], "rubric": code, **{f"p{k}": p[k] for k in range(6)}, "digit_mass": m,
                     "n_tokens": len(r["ids"]), "truncated": r["truncated"]} for r, (p, m) in zip(rows, out)])
            el = time.time() - t0
            log(f"  {bdir.name} chunk {ci + 1}, {code}: {n_prompts:,} prompts, {n_tok / 1e6:.1f}M tokens, "
                f"{el / 60:.1f} min ({n_tok / max(el, 1e-9):,.0f} tokens/s); free RAM "
                f"{'n/a' if free is None else f'{free:.1f} GB'}")
            del rows
    return n_prompts, n_tok
