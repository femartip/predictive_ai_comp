"""Summaries, plots and paired comparisons of evaluate.py's records (protocol v3, sections 7 and 8).

    python scripts/assessors_v3/report.py NAME [--ref B3] [--against OTHER_NAME] [--predictor T2]

Writes RESULTS/NAME/report/:
    summary.csv        per regime and predictor: Brier ALC and ECE ALC (mean over draws), with each repetition's value
    per_budget.csv     per regime, predictor and budget: Brier and ECE (mean over draws of the mean over pairs), s.e.
    paired.csv         per regime: each predictor minus --ref on the same draws, 95% cluster-bootstrap interval over
                       cells (D6); a difference counts when its interval excludes zero
    by_benchmark.csv   Brier ALC per benchmark (its pairs' ALC, averaged) and predictor
    by_draw_size.csv   Brier ALC by number of benchmarks in the draw
    by_route.csv       Brier ALC for agentic and non-agentic benchmarks
    <regime>_brier.png, <regime>_ece.png   per-budget curves with a +-1 s.e. band
--against OTHER_NAME compares --predictor between two runs on identical draws (same repetitions, cells and draws:
e.g. two priors), with the same bootstrap; written to against_<OTHER_NAME>.csv.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ALC_WEIGHTS, BOOTSTRAP, BUDGETS, DATA, RESULTS  # noqa: E402
from simulate import draw_scores  # noqa: E402

# Fixed colour per predictor (categorical palette order; a colour follows its predictor in every plot)
COLOURS = {"T2": "#2a78d6", "T1": "#eb6834", "T0": "#1baf7a", "B3": "#eda100", "B2": "#e87ba4", "oracle": "#008300",
           "B1": "#6250d6"}
LABELS = {"T2": "prior | T2", "T1": "prior | T1", "T0": "prior alone (T0)", "B3": "B3: R1 | T1",
          "B2": "B2: running mean", "oracle": "oracle level", "B1": "B1: 0.5"}


def load(name):
    files = sorted((RESULTS / name).glob("records_rep*_*.parquet"))
    if not files:
        sys.exit(f"no records in {RESULTS / name}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def cluster_bootstrap(diff, n=BOOTSTRAP, seed=0):
    """diff: per-draw differences with columns rep, cell, d. Resample whole (repetition, cell) clusters (draws of a
    cell share one prior and the same subjects, so they are not independent). Returns mean, 2.5% and 97.5%."""
    g = diff.groupby(["rep", "cell"])["d"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(g), size=(n, len(g)))
    boots = s[pick].sum(axis=1) / c[pick].sum(axis=1)
    return float(s.sum() / c.sum()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def paired(scores, a, b):
    """Per regime: predictor a minus predictor b (draw-level Brier ALC), with the bootstrap interval."""
    rows = []
    for regime, sc in scores.groupby("regime"):
        w = sc.pivot_table(index=["rep", "cell", "draw"], columns="predictor", values="brier_alc")
        if a not in w or b not in w:
            continue
        diff = (w[a] - w[b]).rename("d").reset_index()
        m, lo, hi = cluster_bootstrap(diff)
        rows.append({"regime": regime, "predictor": a, "minus": b, "mean_diff": m, "ci_low": lo, "ci_high": hi,
                     "counts": hi < 0 or lo > 0, "draws": len(diff)})
    return rows


def pair_alc(rec):
    """Brier ALC of every pair of every draw (for breakdowns)."""
    w = pd.Series(ALC_WEIGHTS, index=BUDGETS)
    r = rec.assign(wb=rec["budget"].map(w) * rec["brier"])
    keys = ["rep", "regime", "cell", "draw", "draw_benchmarks", "benchmark", "subject_id", "predictor"]
    return r.groupby(keys)["wb"].sum().rename("brier_alc").reset_index()


def plot(per, regime, metric, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#6b6a64", "#e4e3de"
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    t = per[per["regime"] == regime]
    x = np.arange(len(BUDGETS))
    for pred in [p for p in COLOURS if p in set(t["predictor"])]:
        s = t[t["predictor"] == pred].set_index("budget").reindex(BUDGETS)
        m, se = s[metric].to_numpy(), s[f"{metric}_se"].to_numpy()
        ax.fill_between(x, m - se, m + se, color=COLOURS[pred], alpha=0.15, linewidth=0)
        ax.plot(x, m, color=COLOURS[pred], linewidth=2, marker="o", markersize=4, label=LABELS[pred])
    if metric == "brier" and t["brier"].max() > 0.30:   # B2's one-label spike would squash the other curves
        lo = t["brier"].min()
        ax.set_ylim(lo - 0.1 * (0.30 - lo), 0.30)
        off = sorted(LABELS[p] for p in set(t.loc[t["brier"] > 0.30, "predictor"]))
        ax.text(0.01, 0.98, f"off scale above 0.30: {', '.join(off)}", transform=ax.transAxes, fontsize=7.5,
                color=muted, va="top")
    ax.set_xticks(x, [str(b) for b in BUDGETS])
    ax.set_xlabel("Labels revealed per pair (budget)", color=ink, fontsize=10)
    ax.set_ylabel("Brier score" if metric == "brier" else "ECE", color=ink, fontsize=10)
    ax.set_title(f"{'Brier' if metric == 'brier' else 'ECE'} per budget, {regime} regime "
                 f"({'unseen benchmark, unseen model' if regime == 'primary' else 'unseen benchmark, known model'})",
                 color=ink, fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=ink)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=9)
    fig.text(0.01, 0.01, "Mean over draws of each draw's mean over pairs; band = +-1 s.e. over draws.", color=muted,
             fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name")
    ap.add_argument("--ref", default="B3", help="reference predictor of the paired differences")
    ap.add_argument("--against", default=None, help="another results folder with the same draws")
    ap.add_argument("--predictor", default="T2", help="predictor compared with --against")
    a = ap.parse_args()
    rec = load(a.name)
    out = RESULTS / a.name / "report"
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
    for pred in sorted(set(scores["predictor"]) - {a.ref}):
        rows += paired(scores, pred, a.ref)
    pd.DataFrame(rows).to_csv(out / "paired.csv", index=False)

    pa = pair_alc(rec)
    pa.groupby(["regime", "benchmark", "predictor"])["brier_alc"].mean().unstack("predictor") \
        .to_csv(out / "by_benchmark.csv")
    pa.groupby(["regime", "draw_benchmarks", "predictor"])["brier_alc"].mean().unstack("predictor") \
        .to_csv(out / "by_draw_size.csv")
    bt = pd.read_csv(DATA / "benchmarks.csv").set_index("benchmark")
    if "agentic" in bt:
        pa["route"] = pa["benchmark"].map(bt["agentic"]).map({True: "agentic", False: "non-agentic"}).fillna("unknown")
        pa.groupby(["regime", "route", "predictor"])["brier_alc"].mean().unstack("predictor") \
            .to_csv(out / "by_route.csv")

    for regime in per["regime"].unique():
        plot(per, regime, "brier", out / f"{regime}_brier.png")
        plot(per, regime, "ece", out / f"{regime}_ece.png")

    if a.against:
        other = draw_scores(load(a.against))
        keys = ["rep", "regime", "cell", "draw"]
        j = scores[scores["predictor"] == a.predictor].merge(
            other[other["predictor"] == a.predictor], on=keys, suffixes=("", "_other"))
        res = []
        for regime, g in j.groupby("regime"):
            m, lo, hi = cluster_bootstrap(g.assign(d=g["brier_alc"] - g["brier_alc_other"]))
            res.append({"regime": regime, "predictor": a.predictor, "run": a.name, "minus": a.against,
                        "mean_diff": m, "ci_low": lo, "ci_high": hi, "counts": hi < 0 or lo > 0, "draws": len(g)})
        pd.DataFrame(res).to_csv(out / f"against_{a.against}.csv", index=False)
        print(pd.DataFrame(res).round(4).to_string(index=False))

    pd.set_option("display.width", 160)
    print(summ.round(4).to_string(index=False))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
