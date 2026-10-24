"""How many benchmarks an ADeLe annotation cap covers in full: histogram of distinct items per benchmark.

    python scripts/analysis/item_counts.py --benchmarks-csv <token_lengths_benchmarks.csv> --out <dir>
        [--excluded scripts/preprocess/annotations/shares/excluded.txt] [--caps 600 1000 2000]

Input: the per-benchmark table of scripts/analysis/token_lengths.py (column `items` = distinct items, counted on the
full files). Benchmarks with fewer than 80 items (too small to be evaluated under the competition's rule) and the
excluded ones are left out. For each cap: how many benchmarks are annotated in full (items <= cap) and how many items
the cap annotates in total. Writes <out>/item_counts.png and <out>/item_counts_caps.csv; prints the summary.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--excluded", default=None, help="file with one benchmark per line (# comments)")
    ap.add_argument("--caps", type=int, nargs="+", default=[600, 1000, 2000])
    a = ap.parse_args()

    s = pd.read_csv(a.benchmarks_csv)
    excl = set()
    if a.excluded:
        excl = {x.strip() for x in Path(a.excluded).read_text().splitlines() if x.strip() and not x.startswith("#")}
    s = s[(s["items"] >= 80) & ~s["benchmark"].isin(excl)]
    n = s["items"].to_numpy()
    rows = [{"cap": c, "benchmarks_in_full": int((n <= c).sum()), "benchmarks": len(n),
             "share_in_full": float((n <= c).mean()), "items_annotated": int(np.minimum(n, c).sum())} for c in a.caps]
    caps = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    caps.to_csv(out / "item_counts_caps.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter
    surface, ink, muted, grid, blue = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e4e3de", "#2a78d6"
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    bins = np.logspace(np.log10(80), np.log10(n.max() * 1.05), 36)
    ax.hist(n, bins=bins, color=blue, edgecolor=surface, linewidth=1.2)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    top = ax.get_ylim()[1]
    lines = []
    for r in caps.itertuples(index=False):
        ax.axvline(r.cap, color=ink, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
        ax.text(r.cap, top * 0.985, f"{r.cap:,}", color=ink, fontsize=8, ha="center", va="top",
                bbox=dict(facecolor=surface, edgecolor="none", pad=1.5), zorder=4)
        lines.append(f"cap {r.cap:>5,}: {r.benchmarks_in_full:>3} of {r.benchmarks} benchmarks in full "
                     f"({r.share_in_full:.0%}), {r.items_annotated:,} items")
    ax.text(0.985, 0.80, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=8,
            family="monospace", color=ink, bbox=dict(facecolor=surface, edgecolor=grid, pad=6), zorder=5)
    ax.set_xlabel("Distinct items per benchmark (log scale)", color=ink, fontsize=10)
    ax.set_ylabel("Benchmarks", color=ink, fontsize=10)
    ax.set_title(f"Benchmark sizes: {len(n)} usable benchmarks of the no-assets run", color=ink, fontsize=12, loc="left")
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=9)
    fig.text(0.01, 0.012, "Dashed lines: annotation caps. A benchmark at or left of a cap is annotated in full; a larger "
             "one gets a deterministic sample of that many items.\nLeft out: benchmarks with fewer than 80 items and the "
             "team's 7 exclusions.", color=muted, fontsize=7.2, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "item_counts.png", facecolor=surface)
    plt.close(fig)
    print(caps.to_string(index=False))
    print(f"items per benchmark: median {int(np.median(n)):,}, smallest {n.min():,}, largest {n.max():,}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
