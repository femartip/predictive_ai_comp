"""Summaries, plots, paired comparisons and routing diagnostics of evaluate_router.py's records.

    poetry run python scripts/assessors_router/report_router.py NAME      (evaluate_router.py runs it at the end)

Reads results/assessors_router/NAME/ and turns the per-draw records into tables and figures: the same outputs as
assessors_v3/report.py (whose scoring and cluster bootstrap it uses, protocol v3 sections 7 and 8),
plus the routing diagnostics. Writes RESULTS/NAME/report/:
    summary.csv        per regime and predictor: Brier ALC and ECE ALC (mean over draws), with each repetition's value
    per_budget.csv     per regime, predictor and budget: Brier and ECE (mean over draws of the mean over pairs), s.e.
    paired.csv         per regime: each predictor minus B3 and minus v3_assessor on the same draws, 95%
                       cluster-bootstrap interval over cells; a difference counts when its interval excludes zero
    budget0.csv        per regime and predictor, at budget 0: Brier = level + rest, level = mean over pairs of
                       (mean prediction - mean outcome)^2, rest = ranking and spread within the pair
    by_benchmark.csv   Brier ALC per benchmark and predictor
    by_draw_size.csv   Brier ALC by number of benchmarks in the draw
    by_route.csv       Brier ALC for agentic and non-agentic benchmarks
    routing.csv        per regime, known vs unknown test subjects: router accuracy against the oracle tier, mean
                       probability of the oracle tier, error and rank correlation of each measure as the submission
                       would know it (measured if known, else predicted)
    tiers.csv          router vs oracle tier of the unknown test subjects
    <regime>_brier.png, <regime>_ece.png   per-budget curves (as assessors_v3), every trained predictor
    compare.png        per-budget Brier (primary regime) and each assessor minus v3_assessor with 95% intervals, both
                       regimes (as assessors_v3/compare_plot.py)
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
import settings as S  # noqa: E402  (first: puts assessors_v3 on sys.path)
import report as V3  # noqa: E402
from config import BUDGETS  # noqa: E402
from simulate import draw_scores  # noqa: E402
from strength import MEASURES  # noqa: E402

# assessors_v3's palette and surface colours; one fixed colour per trained predictor, baselines in grey
SERIES = {"v3_assessor": "#2a78d6", "tier_router_assessor": "#eb6834", "tier_router_assessor_oracle": "#eda100",
          "general_assessor": "#1baf7a", "general_assessor_oracle": "#008300"}
LABELS = {"v3_assessor": "v3 assessor | T2", "tier_router_assessor": "tier router assessor | T2",
          "tier_router_assessor_oracle": "tier router assessor, true tier | T2",
          "general_assessor": "general assessor | T2", "general_assessor_oracle": "general assessor, true values | T2",
          "B3": "B3: R1 | T1 (no trained prior)",
          "oracle": "oracle level", "B2": "B2: running mean", "B1": "B1: 0.5"}
BASE_STYLE = {"B3": (0, (5, 3)), "oracle": (0, (1, 2))}
SURFACE, INK, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#6b6a64", "#e4e3de", "#5f5e58"
REGIME_TITLE = {"primary": "unseen benchmark, unseen model", "secondary": "unseen benchmark, known model"}


def load(name, pattern):
    files = sorted((S.RESULTS / name).glob(pattern))
    if not files:
        sys.exit(f"no {pattern} in {S.RESULTS / name}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def curves(ax, per, regime, metric):
    """Per-budget curves of one regime: baselines B3 and oracle in grey, the trained predictors in colour."""
    t = per[per["regime"] == regime]
    x = np.arange(len(BUDGETS))
    for pred, ls in BASE_STYLE.items():
        s = t[t["predictor"] == pred].set_index("budget").reindex(BUDGETS)
        if len(s.dropna(how="all")):
            ax.plot(x, s[metric].to_numpy(), color=BASE, linewidth=1.5, linestyle=ls, label=LABELS[pred])
    for pred, col in SERIES.items():
        s = t[t["predictor"] == pred].set_index("budget").reindex(BUDGETS)
        if not len(s.dropna(how="all")):
            continue
        m, se = s[metric].to_numpy(), s[f"{metric}_se"].to_numpy()
        ax.fill_between(x, m - se, m + se, color=col, alpha=0.15, linewidth=0)
        ax.plot(x, m, color=col, linewidth=2, marker="o", markersize=4, label=LABELS[pred])
    ax.set_xticks(x, [str(b) for b in BUDGETS])
    ax.set_xlabel("Labels revealed per pair (budget)", color=INK, fontsize=10)
    ax.set_ylabel("Brier score" if metric == "brier" else "ECE", color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK)


def plot_budget(per, regime, metric, path):
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    curves(ax, per, regime, metric)
    ax.set_title(f"{'Brier' if metric == 'brier' else 'ECE'} per budget, {regime} regime ({REGIME_TITLE[regime]})",
                 color=INK, fontsize=11, loc="left")
    style(ax)
    fig.text(0.01, 0.01, "Mean over draws of each draw's mean over pairs; band = +-1 s.e. over draws. "
             "B1 and B2 are in per_budget.csv.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_compare(per, paired, path):
    regime = "primary" if "primary" in set(per["regime"]) else per["regime"].iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=160, gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    curves(ax1, per, regime, "brier")
    ax1.set_title(f"Brier per budget, {regime} regime", color=INK, fontsize=11, loc="left")
    t = paired[(paired["minus"] == "v3_assessor") & paired["predictor"].isin(SERIES)]
    preds = [p for p in SERIES if p in set(t["predictor"])]
    regimes = sorted(t["regime"].unique())
    offsets = {rg: (j - (len(regimes) - 1) / 2) * 0.22 for j, rg in enumerate(regimes)}
    for _, row in t.iterrows():
        y = preds.index(row.predictor) + offsets[row.regime]
        col = SERIES[row.predictor]
        ax2.plot([row.ci_low, row.ci_high], [y, y], color=col, linewidth=2, solid_capstyle="round")
        ax2.plot(row.mean_diff, y, marker="o" if row.regime == "primary" else "s", markersize=7, color=col,
                 markerfacecolor=col if row.regime == "primary" else SURFACE, markeredgewidth=1.6)
    ax2.axvline(0, color=BASE, linewidth=1, linestyle=(0, (5, 3)))
    ax2.set_yticks(range(len(preds)), [LABELS[p].replace(" | T2", "") for p in preds], fontsize=8.5, color=INK)
    ax2.invert_yaxis()
    ax2.set_xlabel("Brier ALC: assessor minus v3 assessor (lower is better)", color=INK, fontsize=10)
    ax2.set_title("Gain over the v3 assessor, 95% cluster-bootstrap intervals", color=INK, fontsize=11, loc="left")
    ax2.plot([], [], marker="o", color=MUTED, linestyle="", label="primary: unseen benchmark, unseen model")
    ax2.plot([], [], marker="s", color=MUTED, markerfacecolor=SURFACE, linestyle="", label="secondary: known model")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7.5, frameon=False, labelcolor=INK)
    ax2.grid(axis="x", color=GRID, linewidth=0.8)
    for ax in (ax1, ax2):
        style(ax)
    fig.text(0.01, 0.01, "Protocol v3 on the AI Measurement Data Bank (ADeLe v1 annotations). Left: mean over draws of "
             "each draw's mean over pairs. Right: paired differences on identical draws.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def routing_table(routes):
    rows = []
    for (regime, known), g in routes.groupby(["regime", "known"]):
        p_true = g.apply(lambda r: r[f"p{int(r['tier_oracle'])}"], axis=1)
        row = {"regime": regime, "known": known, "subjects": len(g),
               "router_accuracy": (g["tier_router"] == g["tier_oracle"]).mean(), "p_oracle_tier": p_true.mean()}
        for m in MEASURES:
            row[f"{m}_mae"] = (g[f"{m}_used"] - g[f"{m}_oracle"]).abs().mean()
            row[f"{m}_spearman"] = g[f"{m}_used"].corr(g[f"{m}_oracle"], method="spearman")
        rows.append(row)
    return pd.DataFrame(rows)


def report(name):
    rec = load(name, "records_rep*_*.parquet")
    routes = load(name, "routes_rep*_*.parquet")
    out = S.RESULTS / name / "report"
    out.mkdir(exist_ok=True)
    scores = draw_scores(rec)

    summ = scores.groupby(["regime", "predictor"])[["brier_alc", "ece_alc"]].mean()
    per_rep = scores.groupby(["regime", "predictor", "rep"])["brier_alc"].mean().unstack("rep").add_prefix("rep")
    summ = summ.join(per_rep).reset_index().sort_values(["regime", "brier_alc"])
    summ.to_csv(out / "summary.csv", index=False)

    dm = rec.groupby(["regime", "rep", "cell", "draw", "predictor", "budget"])[["brier", "ece"]].mean()
    per = dm.groupby(["regime", "predictor", "budget"]).agg(
        brier=("brier", "mean"), brier_se=("brier", "sem"), ece=("ece", "mean"), ece_se=("ece", "sem")).reset_index()
    per.to_csv(out / "per_budget.csv", index=False)

    rows = []
    for ref in ("B3", "v3_assessor"):
        for pred in sorted(set(scores["predictor"]) - {ref}):
            rows += V3.paired(scores, pred, ref)
    paired = pd.DataFrame(rows)
    paired.to_csv(out / "paired.csv", index=False)

    r0 = rec[rec["budget"] == 0].assign(level=lambda t: (t["p_mean"] - t["y_mean"]) ** 2)
    b0 = r0.groupby(["regime", "predictor"])[["brier", "level"]].mean()
    b0["rest"] = b0["brier"] - b0["level"]
    b0 = b0.reset_index().sort_values(["regime", "brier"])
    b0.to_csv(out / "budget0.csv", index=False)

    pa = V3.pair_alc(rec)
    pa.groupby(["regime", "benchmark", "predictor"])["brier_alc"].mean().unstack("predictor") \
        .to_csv(out / "by_benchmark.csv")
    pa.groupby(["regime", "draw_benchmarks", "predictor"])["brier_alc"].mean().unstack("predictor") \
        .to_csv(out / "by_draw_size.csv")
    bt = pd.read_csv(S.RESULTS / name / "benchmarks.csv").set_index("benchmark")
    if "agentic" in bt:
        pa["route"] = pa["benchmark"].map(bt["agentic"]).map({True: "agentic", False: "non-agentic"}).fillna("unknown")
        pa.groupby(["regime", "route", "predictor"])["brier_alc"].mean().unstack("predictor") \
            .to_csv(out / "by_route.csv")

    rt = routing_table(routes)
    rt.to_csv(out / "routing.csv", index=False)
    unk = routes[~routes["known"]]
    tiers = pd.crosstab([unk["regime"], unk["tier_oracle"]], unk["tier_router"], rownames=["regime", "oracle tier"],
                        colnames=["router tier"])
    tiers.to_csv(out / "tiers.csv")

    for regime in per["regime"].unique():
        plot_budget(per, regime, "brier", out / f"{regime}_brier.png")
        plot_budget(per, regime, "ece", out / f"{regime}_ece.png")
    plot_compare(per, paired, out / "compare.png")

    pd.set_option("display.width", 200)
    print(summ.round(4).to_string(index=False))
    print("\npaired differences (negative = better)")
    print(paired.round(4).to_string(index=False))
    print("\nbudget 0: Brier = level + rest")
    print(b0.round(4).to_string(index=False))
    print("\nrouting")
    print(rt.round(3).to_string(index=False))
    print(f"\nrouter vs oracle tier, unknown test subjects\n{tiers.to_string()}")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name")
    report(ap.parse_args().name)


if __name__ == "__main__":
    main()
