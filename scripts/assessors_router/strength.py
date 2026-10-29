"""Model performance measures, performance tiers and the router (README sections 2 and 4).

Two measures of a model (normalized_name; a subject without one is its own model), from a set of pair rows:
    strength    benchmark-adjusted: rel[model, bench] = logit((successes + 1) / (items + 2)) - logit(base rate of
                bench) (as G1, per model); strength = sum over its benchmarks of rel / (benchmarks + STRENGTH_SHRINK).
                A model is not rated by which benchmarks it happened to answer, and each benchmark counts once.
    mean_score  its plain mean score over all its rows (every benchmark, every run).
Tiers: K_TIERS quantiles of the training models' strengths (each model once); tier 0 is the weakest.

Router: from the subject's competition fields only (S1 of assessors_v3: provider, dates, effort, reasoning, harness,
size, tiers from the name), a small LightGBM classifier gives tier probabilities and one regressor per measure a
predicted value. Trained on the training subjects, each model weighing the same in total (a model can have many
subjects).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

import settings as S
from common import logit
from components import CATEGORICAL, system_table

MEASURES = ("strength", "mean_score")


# ----------------------------------------------------------------------------- measures

def _tables(d, rows, rate):
    """One row per (model, bench) seen in `rows`: rel (see the module docstring), s and n (score and run sums)."""
    P = d.P
    y = P["s"].to_numpy()[rows] / P["n"].to_numpy()[rows]
    df = pd.DataFrame({"model": P["model"].to_numpy()[rows], "bench": P["bench"].to_numpy()[rows], "y": y,
                       "s": P["s"].to_numpy()[rows], "n": P["n"].to_numpy()[rows]})
    g = df.groupby(["model", "bench"]).agg(ysum=("y", "sum"), items=("y", "size"), s=("s", "sum"), n=("n", "sum"))
    g = g.reset_index()
    r = (g["ysum"] + 1) / (g["items"] + 2)
    br = np.clip(rate.reindex(g["bench"]).to_numpy(), 1e-3, 1 - 1e-3)
    g["rel"] = logit(r) - logit(br)
    return g


def measures(t):
    """Model code -> each measure (DataFrame, one column per measure)."""
    g = t.groupby("model").agg(rel=("rel", "sum"), k=("rel", "size"), s=("s", "sum"), n=("n", "sum"))
    return pd.DataFrame({"strength": g["rel"] / (g["k"] + S.STRENGTH_SHRINK), "mean_score": g["s"] / g["n"]})


def oracle_measures(d):
    """Measures of every model on all the data, hidden benchmarks included: only for the oracle predictors."""
    return measures(_tables(d, np.arange(len(d.P)), d.bench_mean()))


def tier_cuts(strength, k=S.K_TIERS):
    return np.quantile(np.asarray(strength, float), np.arange(1, k) / k)


def assign(values, cuts):
    """Tier of each strength value (0 = weakest)."""
    return np.searchsorted(cuts, np.asarray(values, float), side="right")


# ----------------------------------------------------------------------------- router

class Router:
    """Small LightGBM models on the subjects' fields (one row per subject code)."""

    def __init__(self, d, n_jobs=4, seed=0):
        self.X = system_table(d)
        self.cats = [c for c in self.X.columns if c in CATEGORICAL]
        self.n_jobs, self.seed = n_jobs, seed

    def _frame(self, subj, levels):
        X = self.X.iloc[subj].reset_index(drop=True).copy()
        for c in X.columns:
            X[c] = pd.Categorical(X[c], categories=levels[c]) if c in self.cats else X[c].astype(np.float32)
        return X

    def fit(self, subj, target, weight, k=None):
        """A classifier into k classes (tiers) if k is given, else a regressor."""
        levels = {c: sorted(self.X[c].iloc[subj].dropna().unique()) for c in self.cats}
        params = {**S.ROUTER_PARAMS, "objective": "multiclass" if k else "regression", "seed": self.seed,
                  "num_threads": self.n_jobs}
        if k:
            params["num_class"] = k
        data = lgb.Dataset(self._frame(subj, levels), label=target, weight=weight, categorical_feature=self.cats)
        self.model = lgb.train(params, data, num_boost_round=S.ROUTER_ROUNDS), levels
        return self

    def predict(self, subj):
        booster, levels = self.model
        return booster.predict(self._frame(subj, levels))


def model_weights(d, subj):
    """Weight of each subject so that every model weighs 1 in total."""
    m = pd.Series(d.subj_model[subj])
    return 1.0 / m.map(m.value_counts()).to_numpy()


# ----------------------------------------------------------------------------- one cell

def route(d, R, cell, test_subj, oracle, k=S.K_TIERS, n_jobs=4):
    """Everything a cell needs about models, from its training rows only (no leakage), plus the oracle values.
    Returns a dict:
        r1            logit of the mean training base rate (R1's start, baseline B3)
        model_tier    training model code -> tier;  cuts: tier boundaries
        train_values  (bench, subj) of every training pair -> the general assessor's model columns for its rows: each
                      measure and model_known (see train_values)
        test_value    measure -> per subject code, its value for a test subject: measured on the training rows if its
                      model is known, else predicted from its fields;  test_known: per subject code, 1 if known
        completeness  per subject code, how many of provider, parameters and release date are known (0-3)
        oracle        the oracle measures (all the data, the hidden benchmarks included), per model code
        test          one row per test subject: known, tier probabilities p0..p{k-1} (one-hot for a known model), router
                      and oracle tiers, and for each measure its oracle, predicted and used value"""
    P = d.P
    mask = R.train_mask(cell)
    rows = np.flatnonzero(mask)
    rate = d.bench_mean(mask)
    meas = measures(_tables(d, rows, rate))
    cuts = tier_cuts(meas["strength"], k)
    model_tier = pd.Series(assign(meas["strength"], cuts), index=meas.index)

    tr_subj = np.unique(P["subj"].to_numpy()[rows])
    tr_model = d.subj_model[tr_subj]
    w = model_weights(d, tr_subj)
    clf = Router(d, n_jobs, seed=R.rep).fit(tr_subj, model_tier.reindex(tr_model).to_numpy(), w, k)

    test_subj = np.unique(np.asarray(test_subj))
    te_model = d.subj_model[test_subj]
    known = np.isin(te_model, meas.index)
    probs = clf.predict(test_subj).reshape(len(test_subj), k)
    probs[known] = np.eye(k)[model_tier.reindex(te_model[known]).to_numpy()]
    o_te = oracle.reindex(te_model)
    test = {"subj": test_subj, "model": te_model, "known": known, **{f"p{j}": probs[:, j] for j in range(k)},
            "tier_router": probs.argmax(1), "tier_oracle": assign(o_te["strength"], cuts)}

    # test subjects: measured on the training rows if the model is known, else predicted from the fields
    test_value = {}
    test_known = np.full(len(d.subj_names), np.nan)
    test_known[test_subj] = known.astype(float)
    for m in MEASURES:
        pred = Router(d, n_jobs, seed=R.rep).fit(tr_subj, meas[m].reindex(tr_model).to_numpy(), w).predict(test_subj)
        used = np.where(known, meas[m].reindex(te_model).to_numpy(), pred)
        tv = np.full(len(d.subj_names), np.nan)
        tv[test_subj] = used
        test_value[m] = tv
        test.update({f"{m}_oracle": o_te[m].to_numpy(), f"{m}_pred": pred, f"{m}_used": used})
    return {"r1": float(logit(rate.mean())), "model_tier": model_tier, "cuts": cuts,
            "train_values": train_values(d, R, rows, rate, n_jobs), "test_value": test_value,
            "test_known": test_known, "completeness": completeness(d), "oracle": oracle, "test": pd.DataFrame(test)}


def completeness(d):
    """Per subject code: how many of provider, parameters and release date are known (0-3), from S1's table. Tells the
    general assessor how much a predicted strength rests on: with nothing known it is close to a guess."""
    X = system_table(d)
    return ((X["provider"] != "(unknown)").astype(int) + np.isfinite(X["log_params"]).astype(int)
            + X["release_years"].notna().astype(int)).to_numpy(float)


def train_values(d, R, rows, rate, n_jobs):
    """The general assessor's model columns for its training rows, made the way a test subject gets them: cross-
    validation over the training benchmarks (INNER_FOLDS folds, sibling groups kept together, as a hidden benchmark).
    A pair of a fold gets, for its model, the measures on the other folds' rows if the model appears there
    (model_known = 1), else the measures predicted from its fields by regressors trained on the other folds' models
    (model_known = 0). Returns a DataFrame indexed by (bench, subj): MEASURES and model_known."""
    P = d.P
    bench = P["bench"].to_numpy()[rows]
    groups = R.siblings[bench]
    ug = np.unique(groups)
    fold_of = dict(zip(ug, np.random.default_rng([R.rep, 31]).permutation(len(ug)) % S.INNER_FOLDS))
    fold = np.array([fold_of[g] for g in groups])
    parts = []
    for f in range(S.INNER_FOLDS):
        inside, other = rows[fold == f], rows[fold != f]
        if not len(inside) or not len(other):
            continue
        mo = measures(_tables(d, other, rate))
        pairs = pd.DataFrame({"bench": P["bench"].to_numpy()[inside], "subj": P["subj"].to_numpy()[inside]}) \
            .drop_duplicates().reset_index(drop=True)
        model = d.subj_model[pairs["subj"].to_numpy()]
        kn = np.isin(model, mo.index)
        vals = mo.reindex(model).reset_index(drop=True).astype(float)
        if (~kn).any():
            o_subj = np.unique(P["subj"].to_numpy()[other])
            o_model, o_w = d.subj_model[o_subj], model_weights(d, o_subj)
            unknown = pairs["subj"].to_numpy()[~kn]
            for m in MEASURES:
                vals.loc[~kn, m] = Router(d, n_jobs, seed=R.rep).fit(o_subj, mo[m].reindex(o_model).to_numpy(),
                                                                     o_w).predict(unknown)
        parts.append(pd.concat([pairs, vals], axis=1).assign(model_known=kn.astype(float)))
    return pd.concat(parts, ignore_index=True).set_index(["bench", "subj"])
