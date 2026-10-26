"""Quick experiment: predict a subject's outcome on an item from model metafeatures and ADeLe demands only.

    python scripts/assessors_v3/quick_meta_adele.py [--per-benchmark 30000] [--n-jobs 16]

No benchmark identity, no base rates (R1), no model strength (R2), no test-time adaptation: LightGBM on
  - model metafeatures: S1 (components.system_table: the 11 variables from the subject's competition fields),
  - ADeLe demands: the 18 expected levels of the item (I2),
  - both,
compared with a constant (the training mean). Rows: (subject, item) pairs of build_data.py, at most --per-benchmark
per benchmark (so the largest benchmarks do not dominate); target = the pair's mean score.
Two 70/30 splits: random pairs (the item and the model are usually seen in training under other pairings), and whole
benchmarks (closer to the competition: the benchmark is new). Metrics on the 30%: Brier, log loss, AUC (outcome >= 0.5).
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log  # noqa: E402
from components import system_table  # noqa: E402
from config import LGBM_PARAMS  # noqa: E402
from data import Data  # noqa: E402
from prior import sample_rows  # noqa: E402


def auc(y, p):
    """ROC AUC by the rank-sum formula (ties get average ranks)."""
    y = np.asarray(y, bool)
    r = pd.Series(p).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def scores(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "auc": auc(y >= 0.5, p)}


def fit_predict(X_tr, y_tr, X_te, n_jobs):
    params = {k: v for k, v in LGBM_PARAMS.items() if k != "n_estimators"}
    params.update(num_threads=n_jobs, seed=0)
    cats = [c for c in X_tr.columns if X_tr[c].dtype.name == "category"]
    b = lgb.train(params, lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats),
                  num_boost_round=LGBM_PARAMS["n_estimators"])
    imp = pd.Series(b.feature_importance("gain"), index=b.feature_name())
    return b.predict(X_te), imp / imp.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-benchmark", type=int, default=30_000)
    ap.add_argument("--n-jobs", type=int, default=16)
    a = ap.parse_args()
    d = Data()
    rng = np.random.default_rng(0)
    rows = sample_rows(d.P, np.ones(len(d.P), bool), a.per_benchmark, rng)
    P = d.P.iloc[rows]
    y = (P["s"] / P["n"]).to_numpy(float)
    S = system_table(d).iloc[P["subj"].to_numpy()].reset_index(drop=True)
    S["provider"] = S["provider"].astype("category")
    A = d.adele.iloc[P["item"].to_numpy()].reset_index(drop=True)
    sets = {"model metafeatures": S, "ADeLe demands": A, "both": pd.concat([S, A], axis=1)}
    log(f"{len(P):,} pairs from {P['bench'].nunique()} benchmarks, {P['subj'].nunique():,} subjects; "
        f"mean outcome {y.mean():.3f}")

    bench = P["bench"].to_numpy()
    ub = np.unique(bench)
    test_bench = set(rng.choice(ub, size=int(round(0.3 * len(ub))), replace=False).tolist())
    splits = {"random pairs": rng.random(len(P)) < 0.3,
              "whole benchmarks": np.isin(bench, list(test_bench))}
    res, imps = [], {}
    for split, te in splits.items():
        tr = ~te
        res.append({"split": split, "features": "constant (training mean)",
                    **scores(y[te], np.full(te.sum(), y[tr].mean()))})
        for name, X in sets.items():
            p, imp = fit_predict(X[tr], y[tr], X[te], a.n_jobs)
            res.append({"split": split, "features": name, **scores(y[te], p)})
            imps[(split, name)] = imp
            log(f"{split} / {name}: done")
    r = pd.DataFrame(res)
    pd.set_option("display.width", 160)
    print(f"\ntest = 30% ({int(splits['random pairs'].sum()):,} random pairs; {len(test_bench)} of {len(ub)} benchmarks, "
          f"{int(splits['whole benchmarks'].sum()):,} pairs)")
    print(r.round(4).to_string(index=False))
    for split in splits:
        print(f"\ntop features, 'both', {split}:",
              ", ".join(f"{k} {v:.0%}" for k, v in imps[(split, 'both')].nlargest(8).items()))


if __name__ == "__main__":
    main()
