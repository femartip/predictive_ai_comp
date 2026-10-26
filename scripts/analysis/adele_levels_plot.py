"""Polar heat map of the ADeLe v1 level distribution: one sector per rubric, one ring per level (0 inside, 5+ outside),
colour = number of annotated items at that level.

    python scripts/analysis/adele_levels_plot.py --src <adele_v1 folder> --out adele_v1_levels.png

--src: the published tables (<run>/<benchmark>.parquet, or flat). An item's level is its most likely level
(<code>_argmax) when the table has it, else its expected level rounded (the repo's slim copy has only <code>_exp).
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# clockwise from the top, as in the ADeLe papers' figures
ORDER = ["AS", "CEc", "CEe", "CL", "MCr", "MCt", "MCu", "MS", "QLl", "QLq", "SNs", "KNa", "KNc", "KNf", "KNn", "KNs",
         "AT", "VO"]
LEVELS = 6                                    # 0..5 (5 shown as "5+")
EDGES = np.array([0.0, 0.8, 1.8, 2.8, 3.8, 4.8, 5.8])   # ring radii (level 0 narrower, as in the reference figure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(Path(a.src).glob("*.parquet")) + sorted(Path(a.src).glob("*/*.parquet"))
    counts = np.zeros((len(ORDER), LEVELS), dtype=int)
    n_items = 0
    for f in files:
        t = pd.read_parquet(f)
        n_items += len(t)
        for i, c in enumerate(ORDER):
            lv = t[f"{c}_argmax"] if f"{c}_argmax" in t else t[f"{c}_exp"].round()
            counts[i] += np.bincount(np.clip(lv.to_numpy().astype(int), 0, LEVELS - 1), minlength=LEVELS)

    surface, ink, muted = "#fcfcfb", "#0b0b0b", "#6b6a64"
    cmap = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95",
                                                      "#0d366b"])
    vmax = counts.max()
    fig = plt.figure(figsize=(9.4, 8.2), dpi=170)
    fig.patch.set_facecolor(surface)
    ax = fig.add_axes([0.10, 0.13, 0.66, 0.75], projection="polar")
    ax.set_facecolor(surface)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    w = 2 * np.pi / len(ORDER)
    for i in range(len(ORDER)):
        for k in range(LEVELS):
            ax.bar(i * w + w / 2, EDGES[k + 1] - EDGES[k], width=w, bottom=EDGES[k], color=cmap(counts[i, k] / vmax),
                   edgecolor="white", linewidth=0.8)
    for r in EDGES[1:]:                                       # ring outlines
        ax.plot(np.linspace(0, 2 * np.pi, 400), np.full(400, r), color="#5f5e58", linewidth=0.8)
    for k in range(LEVELS):                                   # level labels along the top
        ax.text(0, (EDGES[k] + EDGES[k + 1]) / 2 if k else 0.15, "5+" if k == LEVELS - 1 else str(k),
                ha="center", va="center", fontsize=9, fontweight="bold", color=ink)
    for i, c in enumerate(ORDER):                             # rubric labels outside
        ax.text(i * w + w / 2, EDGES[-1] + 0.75, c, ha="center", va="center", fontsize=12, fontweight="bold",
                color="#4a4945")
    ax.set_ylim(0, EDGES[-1] + 0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    cax = fig.add_axes([0.89, 0.18, 0.022, 0.64])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=muted)
    cb.ax.set_title("Items", fontsize=9, fontweight="bold", color=ink, pad=8)
    fig.text(0.02, 0.015, f"ADeLe v1 levels of the AI Measurement Data Bank: {n_items:,} items from {len(files)} benchmarks "
             "(at most 600 per benchmark).\nLevel = the distilled annotator's most likely level.", fontsize=8,
             color=muted)
    fig.savefig(a.out, facecolor=surface)
    print(f"{len(files)} tables, {n_items:,} items -> {a.out}")
    print(pd.DataFrame(counts, index=ORDER, columns=[str(k) for k in range(LEVELS)]).to_string())


if __name__ == "__main__":
    main()
