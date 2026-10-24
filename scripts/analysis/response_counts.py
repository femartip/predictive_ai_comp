"""How many training instances (response rows) an ADeLe annotation cap keeps: per benchmark, the responses whose item
is annotated under each cap, with the exact selection select_items.py uses.

    python scripts/analysis/response_counts.py --jsonl-dir <converter run folder> --out <dir>
        [--benchmarks-csv <token_lengths_benchmarks.csv> --excluded <excluded.txt>] [--caps 600 1000 2000] [--workers 8]

One streaming pass per JSONL (one row = one response): the item id of every row by string search, and the number of
responses per item. For each cap, the chosen items are the cap smallest SHA-1("<seed>:<item_id>") of the benchmark,
as in select_items.py, and the kept responses are theirs. Benchmarks with fewer than 80 items and the excluded ones are
left out (needs --benchmarks-csv for the item counts). Memory: one counter per item of a file. Writes
<out>/response_counts.csv, <out>/response_counts.png; prints the summary. No item text is read or written.
"""

import argparse
import heapq
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

for p in (Path(__file__).resolve().parents[1] / "preprocess" / "annotations", Path(__file__).resolve().parent / "code"):
    if (p / "select_items.py").exists():
        sys.path.insert(0, str(p))
        break
from select_items import _decode, _item_id, _key  # noqa: E402  (the annotation's own selection rule)


def count(job):
    path, caps, seed = job
    t0 = time.time()
    per = Counter()
    with open(path, "rb") as fh:
        for line in fh:
            raw = _item_id(line)
            if raw is not None:
                per[raw] += 1
    keys = {raw: _key(seed, _decode(raw)) for raw in per}
    out = {"benchmark": Path(path).stem, "items": len(per), "responses": sum(per.values())}
    for c in caps:
        chosen = heapq.nsmallest(c, keys, key=keys.get) if len(keys) > c else list(keys)
        out[f"kept_{c}"] = sum(per[r] for r in chosen)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def plot(t, caps, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter
    surface, ink, muted, grid = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e4e3de"
    colours = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    lo = max(1, min(t[[f"kept_{c}" for c in caps]].min().min(), t["responses"].min()))
    bins = np.logspace(np.log10(lo), np.log10(t["responses"].max() * 1.05), 36)
    ax.hist(t["responses"], bins=bins, color="#d9d8d3", edgecolor=surface, linewidth=1.0, label="all responses")
    for c, col in zip(caps, colours):
        ax.hist(t[f"kept_{c}"], bins=bins, histtype="step", linewidth=2, color=col, label=f"kept with a {c:,}-item cap")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Responses (training instances) per benchmark (log scale)", color=ink, fontsize=10)
    ax.set_ylabel("Benchmarks", color=ink, fontsize=10)
    ax.set_title(f"Training instances kept by each annotation cap ({len(t)} usable benchmarks)", color=ink,
                 fontsize=12, loc="left")
    total = t["responses"].sum()
    lines = [f"all responses: {total:>13,}"] + [
        f"cap {c:>5,}: {t[f'kept_{c}'].sum():>13,} ({t[f'kept_{c}'].sum() / total:.1%})" for c in caps]
    ax.text(0.015, 0.97, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=8, family="monospace",
            color=ink, bbox=dict(facecolor=surface, edgecolor=grid, pad=6), zorder=5)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=ink)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=9)
    fig.text(0.01, 0.012, "A response is one row of the converter's output: one subject's outcome on one item (repeated "
             "trials count separately).\nKept = responses whose item is among the cap items selected for annotation "
             "(select_items.py's deterministic rule).", color=muted, fontsize=7.2, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, facecolor=surface)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmarks-csv", default=None, help="token_lengths_benchmarks.csv (item counts, to filter)")
    ap.add_argument("--excluded", default=None)
    ap.add_argument("--caps", type=int, nargs="+", default=[600, 1000, 2000])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    names = sorted(p.stem for p in Path(a.jsonl_dir).glob("*.jsonl"))
    if a.benchmarks_csv:
        s = pd.read_csv(a.benchmarks_csv).set_index("benchmark")
        names = [b for b in names if b in s.index and s.loc[b, "items"] >= 80]
    if a.excluded:
        excl = {x.strip() for x in Path(a.excluded).read_text().splitlines() if x.strip() and not x.startswith("#")}
        names = [b for b in names if b not in excl]
    files = sorted((Path(a.jsonl_dir) / f"{b}.jsonl" for b in names), key=lambda f: -f.stat().st_size)
    rows, t0 = [], time.time()
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for k, r in enumerate(pool.imap_unordered(count, [(str(f), a.caps, a.seed) for f in files])):
            rows.append(r)
            print(f"  [{time.time() - t0:5.0f}s] {k + 1}/{len(files)} {r['benchmark']}: {r['items']:,} items, "
                  f"{r['responses']:,} responses ({r['seconds']:.0f}s)", flush=True)
    t = pd.DataFrame(rows).sort_values("responses", ascending=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "response_counts.csv", index=False)
    plot(t, a.caps, out / "response_counts.png")
    total = t["responses"].sum()
    print(f"\n{len(t)} benchmarks, {t['items'].sum():,} items, {total:,} responses")
    for c in a.caps:
        k = t[f"kept_{c}"]
        print(f"cap {c:>5,}: {k.sum():>13,} responses kept ({k.sum() / total:.1%}); benchmarks keeping all their "
              f"responses: {int((k == t['responses']).sum())}; median share kept per benchmark "
              f"{(k / t['responses']).median():.0%}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
