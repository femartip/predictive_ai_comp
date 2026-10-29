"""The assessors (priors) of a cell, all built with assessors_v3's fit_prior so they share the v3 prior's settings:

    v3_assessor              the v3 recommended prior (settings.PRIOR) on all training rows: the bar to beat
    tier_assessors           one assessor per performance tier: the same prior, trained only on the rows of the tier's
                             models, on the training benchmarks where the tier has at least MIN_TIER_PAIRS pairs. R1
                             then starts each one from its own tier's mean base rate.
    general_assessor         one assessor on all training rows: the same prior plus four model columns (component
                             M1): strength and mean score over the training benchmarks (strength.py), model_known, and
                             completeness (how many of provider, parameters and release date are known, 0-3).
                             Test rows: the model's values on the training rows if it is known (model_known = 1), else
                             values predicted from its fields (model_known = 0). Training rows get theirs the same way,
                             by cross-validation over the training benchmarks (strength.train_values), so the assessor
                             learns from both cases, as it meets them at test time.
    general_assessor_oracle  trained exactly like general_assessor; test rows get the true values instead (measured on
                             all the data, the test benchmark included). A ceiling, not a submission.
B0 (the hidden benchmark's mean ADeLe levels) is trained and predicted as the mean over all its annotated items;
evidence_logits re-predicts general_assessor's test rows with B0 as the submission builds it (general_assessor_
inference in streaming.py).
Each returns the logits of the cell's test rows (in the given order), the start of those rows and the feature
importances.
"""

import numpy as np
import pandas as pd

import settings as S
import components
from prior import fit_prior
from strength import MEASURES

PARTS = ("v3_assessor", "tier_assessors", "general_assessor", "general_assessor_oracle")


class _Masked:
    """A Repetition whose training mask is replaced (fit_prior only reads rep, siblings and train_mask)."""

    def __init__(self, R, mask):
        self.rep, self.siblings, self._mask = R.rep, R.siblings, mask

    def train_mask(self, cell):
        return self._mask


def _fit(d, R, cell, spec, test_rows, n_jobs, device):
    pr = fit_prior(d, R, cell, spec, test_rows, n_jobs=n_jobs, device=device, seed=R.rep)
    return {"z": pr["z"].astype(np.float32), "start": float(pr["start"][0]), "importance": pr["importance"],
            "train_rows": pr["train_rows"]}


def fit_v3(d, R, cell, test_rows, n_jobs, device):
    return _fit(d, R, cell, S.PRIOR, test_rows, n_jobs, device)


def fit_tiers(d, R, cell, routing, test_rows, n_jobs, device):
    """One fit per tier; None for a tier left without training rows (the v3 assessor replaces it)."""
    P = d.P
    mask = R.train_mask(cell)
    tier = routing["model_tier"].reindex(P["model"].to_numpy()).to_numpy()
    bench = P["bench"].to_numpy()
    out = []
    for k in range(len(routing["cuts"]) + 1):
        m = mask & (tier == k)
        m &= np.bincount(bench[m], minlength=len(d.bench_names))[bench] >= S.MIN_TIER_PAIRS
        out.append(_fit(d, _Masked(R, m), cell, S.PRIOR, test_rows, n_jobs, device) if m.any() else None)
    return out


def fit_general(d, R, cell, routing, test_rows, n_jobs, device, values="used"):
    """The prior plus M1, the model columns. values: the test rows' values, "used" (general_assessor) or "oracle"
    (see the module docstring). The "used" fit keeps its trees and its test rows' features (fit_prior_kept), so
    evidence_logits can predict them with B0 as predict() builds it."""
    P = d.P
    model, bench, subj = P["model"].to_numpy(), P["bench"].to_numpy(), P["subj"].to_numpy()
    cols = [*MEASURES, "model_known"]

    def m1(ctx, tr, te):
        a = routing["train_values"].reindex(pd.MultiIndex.from_arrays([bench[tr], subj[tr]]))[cols]
        a = a.reset_index(drop=True)
        if values == "oracle":
            b = routing["oracle"].reindex(model[te])[list(MEASURES)].reset_index(drop=True)
        else:
            b = pd.DataFrame({m: routing["test_value"][m][subj[te]] for m in MEASURES})
        b["model_known"] = routing["test_known"][subj[te]]
        a["completeness"] = routing["completeness"][subj[tr]]
        b["completeness"] = routing["completeness"][subj[te]]
        return a, b

    components.FEATURES["M1"] = m1               # registered for this fit only
    try:
        if values == "oracle":
            return _fit(d, R, cell, S.PRIOR + "+M1", test_rows, n_jobs, device)
        fit = fit_prior_kept(d, R, cell, S.PRIOR + "+M1", test_rows, n_jobs, device, seed=R.rep)
    finally:
        del components.FEATURES["M1"]
    return {"z": fit["z"].astype(np.float32), "start": float(fit["start"]), "importance": fit["importance"],
            "train_rows": fit["train_rows"], "model": fit["model"], "X": fit["X"]}


def fit_prior_kept(d, R, cell, spec, test_rows, n_jobs, device, seed):
    """prior.fit_prior for a spec started from R1 (same rows, features, weights and LightGBM settings, the same rng
    draws), also returning the booster (model, as text) and the test rows' features (X, float32). Also used by
    submission/fit_tables.py for the final assessor, where it is checked against fit_prior itself."""
    import lightgbm as lgb
    from config import LGBM_PARAMS, R2_PAIRS_PER_BENCHMARK, TRAIN_PAIRS_PER_BENCHMARK
    from common import logit
    from prior import Context, benchmark_weights, sample_rows
    P = d.P
    start_mode, comps = components.parse(spec)
    assert start_mode == "R1" and comps, "fit_prior_kept: a spec with R1 and features"
    rng = np.random.default_rng([seed, R.rep, cell[0], 99 if cell[1] is None else cell[1]])
    mask = R.train_mask(cell)
    rate = d.bench_mean(mask)
    r1 = float(logit(rate.mean()))
    tr = sample_rows(P, mask, TRAIN_PAIRS_PER_BENCHMARK, rng)
    ctx = Context(d, seed, R.siblings)
    ctx.rate = rate
    if "R2" in comps:
        ctx.rasch_rows = sample_rows(P, mask, R2_PAIRS_PER_BENCHMARK, rng)
    parts = [components.FEATURES[c](ctx, tr, test_rows) for c in comps]
    X_tr = pd.concat([a for a, _ in parts], axis=1)
    X_te = pd.concat([b for _, b in parts], axis=1)
    cats = [c for c in X_tr.columns if c in components.CATEGORICAL]
    for c in X_tr.columns:
        if c in cats:
            levels = sorted(X_tr[c].dropna().unique())
            X_tr[c] = pd.Categorical(X_tr[c], categories=levels)
            X_te[c] = pd.Categorical(X_te[c], categories=levels)
        else:
            X_tr[c] = X_tr[c].astype(np.float32)
            X_te[c] = X_te[c].astype(np.float32)
    y = (P["s"].to_numpy()[tr] / P["n"].to_numpy()[tr]).astype(float)
    init = logit(rate.reindex(P["bench"].to_numpy()[tr]).to_numpy())
    params = {k: v for k, v in LGBM_PARAMS.items() if k != "n_estimators"}
    params.update({"seed": seed, "num_threads": n_jobs, "device_type": device})
    data = lgb.Dataset(X_tr, label=y, weight=benchmark_weights(P, tr), init_score=init, categorical_feature=cats)
    booster = lgb.train(params, data, num_boost_round=LGBM_PARAMS["n_estimators"])
    z = booster.predict(X_te, raw_score=True) + r1
    imp = pd.Series(booster.feature_importance(importance_type="gain"), index=booster.feature_name())
    return {"z": z, "start": r1, "importance": imp, "train_rows": len(tr), "model": booster.model_to_string(),
            "X": X_te, "booster": booster}


def evidence_logits(d, fit, pos, draw, pairs, budgets):
    """The general assessor's logits on one draw with B0 as predict() builds it from its call's evidence (the
    submission's assessor_input): at budget k, the labeled items of a hidden benchmark are the distinct items revealed
    so far across the draw's pairs on it (each pair's first min(k, acquisition items)). A revealed (labeled) row gets
    the mean levels of those items; an evaluation row the mean over them and its own item (at budget 0: its own
    levels). NaN levels are skipped, as in components.bench_profiles. pos: test row -> row of fit["X"].
    Returns {pair: {budget: (acquisition logits of the revealed rows, evaluation logits)}}."""
    import lightgbm as lgb
    if "booster" not in fit:
        fit["booster"] = lgb.Booster(model_str=fit["model"])
        fit["Xn"] = fit["X"].to_numpy(np.float32)
    X, jb = fit["Xn"], [fit["X"].columns.get_loc(c) for c in fit["X"].columns if c.startswith("bench_")]
    A = d.adele.to_numpy(float)
    item_of = d.P["item"].to_numpy()
    rows, b0s, index = [], [], []
    for k in budgets:
        revealed = {}
        for g in draw:
            p = pairs[g]
            revealed.setdefault(p["bench"], set()).update(item_of[p["acq_rows"][:min(k, len(p["acq_rows"]))]].tolist())
        for g in draw:
            p = pairs[g]
            items = sorted(revealed[p["bench"]])
            V = A[items] if items else np.empty((0, A.shape[1]))
            s, n = np.nansum(V, 0), np.sum(~np.isnan(V), 0)
            acq, ev = p["acq_rows"][:min(k, len(p["acq_rows"]))], p["eval_rows"]
            own = A[item_of[ev]]
            new = ~np.isin(item_of[ev], items)[:, None]          # an evaluation item not already revealed
            ev_s = s + np.where(new, np.nan_to_num(own), 0.0)
            ev_n = n + (new & ~np.isnan(own))
            with np.errstate(invalid="ignore", divide="ignore"):
                ev_b0 = np.where(ev_n > 0, ev_s / ev_n, np.nan)
                acq_b0 = np.tile(np.where(n > 0, s / n, np.nan), (len(acq), 1))
            for r, b0, kind in ((acq, acq_b0, "acq"), (ev, ev_b0, "eval")):
                rows.append(pos.get_indexer(r))
                b0s.append(b0)
                index.append((g, k, kind, len(r)))
    Xd = X[np.concatenate(rows)]
    Xd[:, jb] = np.concatenate(b0s).astype(np.float32)
    z = fit["booster"].predict(Xd, raw_score=True) + fit["start"]
    out, i = {}, 0
    for g, k, kind, n in index:
        out.setdefault(g, {}).setdefault(k, {})[kind] = z[i:i + n].astype(np.float32)
        i += n
    return {g: {k: (v["acq"], v["eval"]) for k, v in by_k.items()} for g, by_k in out.items()}
