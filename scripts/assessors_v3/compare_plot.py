"""Plot several evaluate.py runs side by side: per-budget Brier of their T2 predictor, and their Brier ALC gain over B3.

    python scripts/assessors_v3/compare_plot.py RUN [RUN ...] [--labels "A" "B" ...] [--regime primary] [--out FILE]

Left panel: Brier per budget (mean over draws of each draw's mean over pairs) of each run's prior | T2, with baseline
B3 (R1 | T1) and the oracle level from the first run (identical in every run on the same draws). Right panel: each run's
T2 minus B3 in Brier ALC on the same draws, with 95% cluster-bootstrap intervals over (repetition, cell) (protocol v3,
D6), for both regimes. Only draws present in every run are used.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BUDGETS, RESULTS  # noqa: E402
from report import cluster_bootstrap, load  # noqa: E402
from simulate import draw_scores  # noqa: E402

KEYS = ["rep", "regime", "cell", "draw"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]   # fixed order
SURFACE, INK, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#6b6a64", "#e4e3de", "#5f5e58"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="+", default=None, help="legend labels, one per run")
    ap.add_argument("--regime", default="primary", help="regime of the left panel")
    ap.add_argument("--out", default=str(RESULTS / "compare_approaches.png"))
    a = ap.parse_args()
    labels = a.labels or a.runs
    recs = {r: load(r) for r in a.runs}
    common = None
    for rec in recs.values():
        k = rec[KEYS].drop_duplicates()
        common = k if common is None else common.merge(k, on=KEYS)
    recs = {r: rec.merge(common, on=KEYS) for r, rec in recs.items()}

    def per_budget(rec, pred, regime):
        t = rec[(rec["predictor"] == pred) & (rec["regime"] == regime)]
        return t.groupby(KEYS + ["budget"])["brier"].mean().groupby("budget").mean().reindex(BUDGETS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=160, gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(BUDGETS))
    first = recs[a.runs[0]]
    ax1.plot(x, per_budget(first, "B3", a.regime), color=BASE, linewidth=1.6, linestyle=(0, (5, 3)),
             label="B3: R1 | T1 (no trained prior)")
    ax1.plot(x, per_budget(first, "oracle", a.regime), color=BASE, linewidth=1.4, linestyle=(0, (1, 2)),
             label="oracle level")
    for k, (r, lab) in enumerate(zip(a.runs, labels)):
        ax1.plot(x, per_budget(recs[r], "T2", a.regime), color=SERIES[k % len(SERIES)], linewidth=2, marker="o",
                 markersize=4, label=f"{lab} | T2")
    ax1.set_xticks(x, [str(b) for b in BUDGETS])
    ax1.set_xlabel("Labels revealed per pair (budget)", color=INK, fontsize=10)
    ax1.set_ylabel("Brier score", color=INK, fontsize=10)
    ax1.set_title(f"Brier per budget, {a.regime} regime", color=INK, fontsize=11, loc="left")
    ax1.legend(loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK)

    rows = []
    for k, (r, lab) in enumerate(zip(a.runs, labels)):
        s = draw_scores(recs[r])
        for regime, g in s.groupby("regime"):
            w = g.pivot_table(index=KEYS, columns="predictor", values="brier_alc")
            m, lo, hi = cluster_bootstrap((w["T2"] - w["B3"]).rename("d").reset_index())
            rows.append((k, lab, regime, m, lo, hi, g["rep"].nunique()))
    t = pd.DataFrame(rows, columns=["k", "label", "regime", "m", "lo", "hi", "reps"])
    regimes = sorted(t["regime"].unique())
    offsets = {rg: (j - (len(regimes) - 1) / 2) * 0.22 for j, rg in enumerate(regimes)}
    markers = {"primary": "o", "secondary": "s"}
    for _, row in t.iterrows():
        y = row.k + offsets[row.regime]
        col = SERIES[int(row.k) % len(SERIES)]
        ax2.plot([row.lo, row.hi], [y, y], color=col, linewidth=2, solid_capstyle="round")
        ax2.plot(row.m, y, marker=markers.get(row.regime, "o"), markersize=7, color=col,
                 markerfacecolor=col if row.regime == "primary" else SURFACE, markeredgewidth=1.6)
    ax2.axvline(0, color=BASE, linewidth=1, linestyle=(0, (5, 3)))
    ax2.set_yticks(range(len(labels)), labels, fontsize=8.5, color=INK)
    ax2.invert_yaxis()
    ax2.set_xlabel("Brier ALC: prior | T2 minus B3 (lower is better)", color=INK, fontsize=10)
    reps = int(t["reps"].max())
    ax2.set_title(f"Gain over B3, 95% cluster-bootstrap intervals ({reps} repetitions)", color=INK, fontsize=11,
                  loc="left")
    ax2.plot([], [], marker="o", color=MUTED, linestyle="", label="primary: unseen benchmark, unseen model")
    ax2.plot([], [], marker="s", color=MUTED, markerfacecolor=SURFACE, linestyle="", label="secondary: known model")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7.5, frameon=False, labelcolor=INK)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x" if ax is ax2 else "y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5)
    fig.text(0.01, 0.01, "Protocol v3 on the AI Measurement Data Bank (ADeLe v1 annotations). Left: mean over draws of "
             "each draw's mean over pairs. Right: paired differences on identical draws.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(a.out, facecolor=SURFACE)
    print(t.round(4).to_string(index=False))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
