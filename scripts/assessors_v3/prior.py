"""The prior (assessor, offline part): LightGBM started from R1, on the chosen input components (components.py).

    z(subject, item) = start + trees(features)            (log-odds; prior = sigmoid(z))

- Training rows: the cell's training pairs (protocol.Repetition.train_mask), at most TRAIN_PAIRS_PER_BENCHMARK per
  benchmark. Label: the pair's mean score (soft label, cross-entropy objective; non-binary training benchmarks keep
  their partial credit). Weight: min(runs, 4), rescaled so every training benchmark has the same total weight, then to
  mean 1 (LightGBM's leaf minimums assume weights of order 1).
- R1: each training row starts from its own benchmark's base rate (init score), so the trees learn deviations from
  the benchmark's level, not the level itself; a hidden benchmark starts from `start`, the mean base rate of the
  training benchmarks (equal weight per benchmark). Without R1 the trees learn the level too.
- Without feature components ("R1" alone) the prior is `start` for every row (baseline B3's prior).
Devices: "cpu", "gpu" (LightGBM's OpenCL learner) or "cuda" (CUDA learner); "auto" probes cuda, then gpu, then cpu.
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log, logit  # noqa: E402
from components import CATEGORICAL, FEATURES, parse  # noqa: E402
from config import LGBM_PARAMS, R2_PAIRS_PER_BENCHMARK, TRAIN_PAIRS_PER_BENCHMARK  # noqa: E402


class Context:
    """What the feature components may use: the data, a seed, the sibling groups (R2's inner folds), a cache for
    per-subject tables, and R2's own row sample."""

    def __init__(self, d, seed, siblings):
        self.d, self.seed, self.siblings = d, seed, siblings
        self.cache, self.rasch_rows = {}, None


def sample_rows(P, mask, per_bench, rng):
    """Indices of at most per_bench random rows per benchmark among the rows where mask is True."""
    idx = np.flatnonzero(mask)
    r = pd.Series(rng.random(len(idx)))
    keep = r.groupby(P["bench"].to_numpy()[idx]).rank(method="first").to_numpy() <= per_bench
    return idx[keep]


def benchmark_weights(P, rows):
    """min(runs, 4) per row, equal total weight per benchmark, mean 1."""
    w = pd.Series(np.minimum(P["n"].to_numpy()[rows], 4.0))
    b = P["bench"].to_numpy()[rows]
    w = w / w.groupby(b).transform("sum")
    return (w / w.mean()).to_numpy()


def lgb_device(requested, n_jobs):
    """The LightGBM device to use: the requested one if it works here; 'auto' tries cuda, gpu, then cpu."""
    if requested == "cpu":
        return "cpu"
    x = np.random.default_rng(0).normal(size=(2000, 5))
    for dev in (["cuda", "gpu"] if requested == "auto" else [requested]):
        try:
            lgb.train({"objective": "binary", "device_type": dev, "verbose": -1, "num_threads": n_jobs},
                      lgb.Dataset(x, label=(x[:, 0] > 0).astype(float)), num_boost_round=3)
            return dev
        except Exception as err:  # noqa: BLE001 - any failure means this LightGBM build cannot use that device
            log(f"LightGBM device '{dev}' not usable ({type(err).__name__}: {str(err)[:80]})")
    if requested != "auto":
        sys.exit(f"LightGBM cannot use device '{requested}' here (use --device cpu or auto)")
    return "cpu"


def fit_prior(d, rep, cell, spec, test_rows, n_jobs=4, device="cpu", seed=0):
    """Train the prior for one cell; return its logits on test_rows (indices into d.P), the start (R1 on a hidden
    benchmark, logit) and the feature importances (gain)."""
    P = d.P
    use_r1, comps = parse(spec)
    rng = np.random.default_rng([seed, rep.rep, cell[0], 99 if cell[1] is None else cell[1]])
    mask = rep.train_mask(cell)
    rate = d.bench_mean(mask)                                  # base rate of every training benchmark
    start = float(logit(rate.mean()))
    if not comps:
        z = np.full(len(test_rows), start if use_r1 else float(logit(P["s"][mask].sum() / P["n"][mask].sum())))
        return {"z": z, "start": start, "importance": pd.Series(dtype=float), "train_rows": int(mask.sum())}

    tr = sample_rows(P, mask, TRAIN_PAIRS_PER_BENCHMARK, rng)
    ctx = Context(d, seed, rep.siblings)
    if "R2" in comps:
        ctx.rasch_rows = sample_rows(P, mask, R2_PAIRS_PER_BENCHMARK, rng)
    parts_tr, parts_te = [], []
    for c in comps:
        a, b = FEATURES[c](ctx, tr, test_rows)
        parts_tr.append(a)
        parts_te.append(b)
    X_tr = pd.concat(parts_tr, axis=1)
    X_te = pd.concat(parts_te, axis=1)
    cats = [c for c in X_tr.columns if c in CATEGORICAL]
    for c in X_tr.columns:
        if c in cats:                                          # same categories in both tables
            levels = sorted(X_tr[c].dropna().unique())
            X_tr[c] = pd.Categorical(X_tr[c], categories=levels)
            X_te[c] = pd.Categorical(X_te[c], categories=levels)
        else:
            X_tr[c] = X_tr[c].astype(np.float32)
            X_te[c] = X_te[c].astype(np.float32)

    y = (P["s"].to_numpy()[tr] / P["n"].to_numpy()[tr]).astype(float)
    init = logit(rate.reindex(P["bench"].to_numpy()[tr]).to_numpy()) if use_r1 else None
    params = {k: v for k, v in LGBM_PARAMS.items() if k != "n_estimators"}
    params.update({"seed": seed, "num_threads": n_jobs, "device_type": device})
    data = lgb.Dataset(X_tr, label=y, weight=benchmark_weights(P, tr), init_score=init, categorical_feature=cats)
    booster = lgb.train(params, data, num_boost_round=LGBM_PARAMS["n_estimators"])
    z = booster.predict(X_te, raw_score=True) + (start if use_r1 else 0.0)
    imp = pd.Series(booster.feature_importance(importance_type="gain"), index=booster.feature_name())
    return {"z": z, "start": start, "importance": imp, "train_rows": len(tr)}
