"""Compare several evaluate.py runs on their common draws: one table per regime for the overnight / decision reports.

    python scripts/assessors_v3/compare_runs.py RUN [RUN ...] [--ref-run RUN] [--predictor T2] [--out compare.csv]

For each run: Brier ALC and ECE ALC of --predictor (mean over the draws present in every run), its budget-0 Brier
split into level and rest (when the records have p_mean), and paired differences with 95% cluster-bootstrap intervals
over (repetition, cell) (protocol v3, D6) against baseline B3 and against --ref-run's --predictor. A difference counts
when its interval excludes zero. Also lists the baselines (B1, B3, oracle) once, and the repetitions each run has.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import cluster_bootstrap, load  # noqa: E402
from config import RESULTS  # noqa: E402
from simulate import draw_scores  # noqa: E402

KEYS = ["rep", "regime", "cell", "draw"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref-run", default=None, help="run whose --predictor the others are compared with")
    ap.add_argument("--predictor", default="T2")
    ap.add_argument("--out", default=None, help="CSV (default RESULTS/compare_<predictor>.csv)")
    a = ap.parse_args()
    recs = {r: load(r) for r in a.runs}
    scores = {r: draw_scores(rec) for r, rec in recs.items()}
    common = None                                            # draws present in every run
    for s in scores.values():
        k = s[KEYS].drop_duplicates()
        common = k if common is None else common.merge(k, on=KEYS)
    scores = {r: s.merge(common, on=KEYS) for r, s in scores.items()}
    ref = a.ref_run or a.runs[0]
    rows = []
    for r in a.runs:
        s, rec = scores[r], recs[r].merge(common, on=KEYS)
        for regime, g in s.groupby("regime"):
            w = g.pivot_table(index=KEYS, columns="predictor", values="brier_alc")
            row = {"run": r, "regime": regime, "reps": int(g["rep"].nunique()), "draws": len(w),
                   f"{a.predictor}_alc": w[a.predictor].mean(),
                   "ece_alc": g[g["predictor"] == a.predictor]["ece_alc"].mean()}
            r0 = rec[(rec["regime"] == regime) & (rec["budget"] == 0) & (rec["predictor"] == a.predictor)]
            row["budget0"] = r0["brier"].mean()
            if "p_mean" in r0:
                row["budget0_level"] = ((r0["p_mean"] - r0["y_mean"]) ** 2).mean()
            m, lo, hi = cluster_bootstrap((w[a.predictor] - w["B3"]).rename("d").reset_index())
            row.update(vs_B3=m, vs_B3_lo=lo, vs_B3_hi=hi)
            if r != ref:
                o = scores[ref][(scores[ref]["regime"] == regime) & (scores[ref]["predictor"] == a.predictor)]
                j = g[g["predictor"] == a.predictor].merge(o, on=KEYS, suffixes=("", "_ref"))
                m, lo, hi = cluster_bootstrap(j.assign(d=j["brier_alc"] - j["brier_alc_ref"]))
                row.update(vs_ref=m, vs_ref_lo=lo, vs_ref_hi=hi)
            rows.append(row)
    t = pd.DataFrame(rows)
    base = scores[a.runs[0]]
    b = base[base["predictor"].isin(["B1", "B3", "oracle"])].groupby(["regime", "predictor"])["brier_alc"].mean()
    out = Path(a.out) if a.out else RESULTS / f"compare_{a.predictor}.csv"
    t.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    for regime, g in t.groupby("regime"):
        print(f"\n== {regime} regime ({a.predictor}; reference run for vs_ref: {ref}); baselines: "
              + ", ".join(f"{p} {v:.4f}" for p, v in b[regime].items()))
        print(g.drop(columns="regime").sort_values(f"{a.predictor}_alc").round(4).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
