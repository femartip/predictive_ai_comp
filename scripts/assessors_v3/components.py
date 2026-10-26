"""The prior's input components (notation of approaches.md), each one isolated so any combination can be tested.

A prior is written as components joined by '+', e.g. "R1+R2+S1+I0+I2" (the default, the basic assessor):

    R1  start: each training benchmark's base rate is the LightGBM init score of its rows; on an unseen benchmark the
        prior starts from the mean base rate of the training benchmarks (equal weight per benchmark)
    R2  model strength: a Rasch-style fit across benchmarks (logit p = c + difficulty[instance] + strength[model])
    S1  system metadata from the competition's subject fields (11 variables of approaches.md), optionally completed
        with model attributes from models.csv (ASSESSOR_MODELS_CSV)
    I0  item text statistics (build_data.item_stats)
    I2  ADeLe v1 demand levels: the 18 expected levels
Routing components (2026-10-25), aimed at the budget-0 error, which is mostly an error of level:
    K1  start routed by task (replaces R1's global start; training rows still start from their own base rate): a
        hidden benchmark starts from the base rates of its K1_NEIGHBOURS most similar training benchmarks, by
        benchmark profile (inverse-distance weighted mean of their logit base rates)
    B0  benchmark profile as features: the mean ADeLe demands of the row's benchmark, so the trees can learn level by
        task type (e.g. "math-heavy benchmarks are hard for small models")
    G1  model-type routing: how much the row's model group (provider x size tier x reasoning, from S1) beats or trails
        each benchmark's base rate, averaged over the training benchmarks, leaving the row's own benchmark out; works
        for unseen models (unlike R2), since it only uses the group
The benchmark profile is the mean of the 18 expected levels over the benchmark's annotated items; for a hidden
benchmark it uses its items' demands only (no labels), which predict() can compute from the items it is asked about.

Every feature component is a function  f(ctx, train_rows, test_rows) -> (X_train, X_test)  returning one DataFrame of
features per row set (row indices into Data.P). Features may be missing (NaN): LightGBM learns a direction for them.
To add a component, write such a function and register it in FEATURES. R1 is not a feature (prior.py uses it as the
start), so it is handled by prior.py, not here.

Input discipline (protocol v3, section 6): only what predict() sees on the platform: the subject's fields
(normalized_name, provider, release_date, access_date, harness, reasoning_effort, harness_version) and the item's
content. No subject_id, no benchmark name.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_data import ITEM_STAT_COLS  # noqa: E402
from config import K1_NEIGHBOURS, MODELS_CSV, R2_DROPOUT, R2_INNER_FOLDS  # noqa: E402

ORIGIN = pd.Timestamp("2023-01-01")
EFFORT = {"none": 0, "minimal": 0.5, "low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}
REASONING = re.compile(r"think|reason|\br1\b|-r1|\bo[134]\b|o[134]-|qwq|deepseek-r|magistral", re.I)
SCAFFOLD = re.compile(r"agent|scaffold|hands|aider|codex|claude.code|cursor|swe-|mini-swe", re.I)
SMALL = re.compile(r"\b(?:mini|flash|haiku|nano|small|lite|tiny|micro)\b", re.I)
LARGE = re.compile(r"\b(?:pro|opus|ultra|large|max|plus|turbo)\b", re.I)
PARAMS = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")


def _years(values):
    t = pd.to_datetime(pd.Series(values, dtype=object), errors="coerce", format="mixed", utc=True).dt.tz_localize(None)
    return ((t - ORIGIN).dt.days / 365.25).to_numpy()


# ----------------------------------------------------------------------------- S1: system metadata

def system_table(d):
    """S1 per subject code (computed once). The 11 variables of approaches.md:
    provider; release date and access date (years since 2023); reasoning effort (0 if not given, 1 = low ... 5 = max);
    reasoning model (name or effort or thinking run); uses a harness; agent or scaffold harness; size in parameters
    when the name says it (e.g. "32B"); small tier ("mini", "flash", ...); large tier ("pro", "opus", ...); model name
    missing. provider and release_date come from models.csv when given (the bank's JSONL does not carry them; the
    platform gives both), else provider is the first word of normalized_name and release date is missing."""
    s = d.subjects
    name = s["normalized_name"].fillna("")
    label = (name + " " + s["model_name"].fillna("")).str.strip()
    effort = s["reasoning_effort"].fillna("").str.lower().map(EFFORT).fillna(0.0)
    harness = s["harness"].fillna("")
    out = pd.DataFrame({
        "provider": name.str.split().str[0].str.lower().fillna("(unknown)"),
        "release_years": np.nan,
        "access_years": _years(s["access_date"]),
        "effort": effort.to_numpy(float),
        "reasoning": (label.str.contains(REASONING) | (effort > 0)
                      | s["thinking"].fillna("").str.lower().eq("true")).astype(float),
        "has_harness": (harness != "").astype(float),
        "scaffold": harness.str.contains(SCAFFOLD).astype(float),
        "log_params": np.log10(pd.to_numeric(label.str.extract(PARAMS)[0], errors="coerce") * 1e9),
        "small_tier": label.str.contains(SMALL).astype(float),
        "large_tier": label.str.contains(LARGE).astype(float),
        "name_missing": (name == "").astype(float),
    })
    if MODELS_CSV is not None and MODELS_CSV.exists():
        m = pd.read_csv(MODELS_CSV).drop_duplicates("normalized_name").set_index("normalized_name").reindex(name)
        if "provider" in m:
            prov = m["provider"].astype(object).where(m["provider"].notna()).str.lower().to_numpy()
            out["provider"] = np.where(pd.isna(prov), out["provider"], prov)
        if "release_date" in m:
            out["release_years"] = _years(m["release_date"].to_numpy())
        if "parameters" in m:
            p = np.log10(pd.to_numeric(m["parameters"], errors="coerce")).to_numpy()
            out["log_params"] = np.where(np.isnan(p), out["log_params"], p)
    return out


def S1(ctx, tr, te):
    if "S1" not in ctx.cache:
        ctx.cache["S1"] = system_table(ctx.d)
    t = ctx.cache["S1"]
    subj = ctx.d.P["subj"].to_numpy()
    return t.iloc[subj[tr]].reset_index(drop=True), t.iloc[subj[te]].reset_index(drop=True)


# ----------------------------------------------------------------------------- item components

def I0(ctx, tr, te):
    t = ctx.d.items[ITEM_STAT_COLS]
    item = ctx.d.P["item"].to_numpy()
    return t.iloc[item[tr]].reset_index(drop=True), t.iloc[item[te]].reset_index(drop=True)


def I2(ctx, tr, te):
    t = ctx.d.adele
    item = ctx.d.P["item"].to_numpy()
    return t.iloc[item[tr]].reset_index(drop=True), t.iloc[item[te]].reset_index(drop=True)


# ----------------------------------------------------------------------------- R2: model strength

def fit_rasch(inst, model, y, w, lam_item=1.0, lam_model=1.0):
    """logit P(success) = c + difficulty[instance] + strength[model], weighted soft-label logistic fit with L2
    penalties. Instances absorb each benchmark's difficulty, so strengths are comparable across benchmarks as long as
    models overlap. Returns strength per model code (pd.Series)."""
    ii, items = pd.factorize(inst)
    mm, models = pd.factorize(model)
    n_i, n_m = len(items), len(models)

    def loss(theta):
        c, a, b = theta[0], theta[1:1 + n_i], theta[1 + n_i:]
        p = 1 / (1 + np.exp(-(c + a[ii] + b[mm])))
        nll = -np.sum(w * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
        g = w * (p - y)
        grad = np.concatenate([[g.sum()], np.bincount(ii, g, n_i) + lam_item * a, np.bincount(mm, g, n_m) + lam_model * b])
        return nll + 0.5 * lam_item * a @ a + 0.5 * lam_model * b @ b, grad

    res = minimize(loss, np.zeros(1 + n_i + n_m), jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return pd.Series(res.x[1 + n_i:], index=models)


def R2(ctx, tr, te):
    """Strength of the row's model (normalized_name).
    - Training rows: fitted on the training rows outside the row's own inner benchmark fold (R2_INNER_FOLDS folds of
      sibling groups), so a row's strength never uses its own benchmark, as for a hidden benchmark.
    - Test rows: fitted on all training rows; missing for models never seen in training. In the primary regime (unseen
      models) that is every test row: R2 then only matters through what the trees learned for a missing strength.
    - So that the trees learn that case, the strength of a random R2_DROPOUT share of the training models is hidden.
    The fits use ctx.rasch_rows (a smaller per-benchmark sample of the training rows)."""
    P, d = ctx.d.P, ctx.d
    rng = np.random.default_rng([ctx.seed, 23])
    fit_rows = ctx.rasch_rows
    bench = P["bench"].to_numpy()
    groups = ctx.siblings[bench]
    ug = np.unique(groups[fit_rows])
    inner = dict(zip(ug, rng.permutation(len(ug)) % R2_INNER_FOLDS))
    fold_fit = np.array([inner[g] for g in groups[fit_rows]])
    fold_tr = np.array([inner.get(g, -1) for g in groups[tr]])
    y = (P["s"].to_numpy() / P["n"].to_numpy())
    w = np.minimum(P["n"].to_numpy(), 4.0)
    model = P["model"].to_numpy()
    feat_tr = np.full(len(tr), np.nan)
    for f in range(R2_INNER_FOLDS):
        keep = fit_rows[fold_fit != f]
        st = fit_rasch(P["inst"].to_numpy()[keep], model[keep], y[keep], w[keep])
        rows = fold_tr == f
        feat_tr[rows] = pd.Series(model[tr][rows]).map(st).to_numpy()
    st_all = fit_rasch(P["inst"].to_numpy()[fit_rows], model[fit_rows], y[fit_rows], w[fit_rows])
    hidden = rng.random(len(d.model_names)) < R2_DROPOUT
    feat_tr[hidden[model[tr]]] = np.nan
    feat_te = pd.Series(model[te]).map(st_all).to_numpy()
    return pd.DataFrame({"strength": feat_tr}), pd.DataFrame({"strength": feat_te})


# ----------------------------------------------------------------------------- routing (K1, B0, G1)

def bench_profiles(d):
    """Benchmark code x 18: mean expected ADeLe level over the benchmark's annotated items."""
    item_bench = d.inst_bench[np.unique(d.inst_item, return_index=True)[1]]   # benchmark of each item code
    return d.adele.groupby(item_bench).mean().reindex(range(len(d.bench_names)))


def routed_start(d, rate, benches):
    """K1: logit start of each benchmark in `benches` from its K1_NEIGHBOURS nearest training benchmarks (`rate`:
    training benchmark code -> base rate), by Euclidean distance between profiles standardised over the training
    benchmarks; inverse-distance weights. Returns {benchmark code: logit start}."""
    prof = bench_profiles(d)
    train = np.array(rate.index)
    Pt = prof.loc[train]
    mu, sd = Pt.mean(), Pt.std().replace(0, 1)
    Zt = ((Pt - mu) / sd).to_numpy()
    lr = np.log(np.clip(rate.to_numpy(), 1e-3, 1 - 1e-3) / (1 - np.clip(rate.to_numpy(), 1e-3, 1 - 1e-3)))
    out = {}
    for b in np.unique(benches):
        z = ((prof.loc[b] - mu) / sd).to_numpy()
        dist = np.sqrt(np.nansum((Zt - z) ** 2, axis=1))
        k = np.argsort(dist)[:K1_NEIGHBOURS]
        w = 1 / (dist[k] + 1e-3)
        out[int(b)] = float(np.sum(w * lr[k]) / w.sum())
    return out


def B0(ctx, tr, te):
    if "B0" not in ctx.cache:
        ctx.cache["B0"] = bench_profiles(ctx.d).add_prefix("bench_")
    t = ctx.cache["B0"]
    bench = ctx.d.P["bench"].to_numpy()
    return t.iloc[bench[tr]].reset_index(drop=True), t.iloc[bench[te]].reset_index(drop=True)


def model_groups(d):
    """Model group of each subject code: provider x size tier (small, large, other) x reasoning, from S1."""
    s = system_table(d)
    tier = np.where(s["small_tier"] > 0, "small", np.where(s["large_tier"] > 0, "large", "mid"))
    return (s["provider"].astype(str) + "|" + tier + "|" + s["reasoning"].astype(int).astype(str)).to_numpy()


def G1(ctx, tr, te):
    """Model-type routing. On the training rows: each (subject, benchmark) pair's smoothed success rate, its logit
    minus the benchmark's logit base rate (rel), averaged per (model group, benchmark), then over benchmarks with equal
    weight. A training row gets its group's mean over the *other* training benchmarks (as for a hidden benchmark); a
    test row the mean over all training benchmarks. Missing for groups seen on no (other) benchmark."""
    d, P = ctx.d, ctx.d.P
    if "groups" not in ctx.cache:
        ctx.cache["groups"] = model_groups(d)
    grp = ctx.cache["groups"]
    subj, bench = P["subj"].to_numpy(), P["bench"].to_numpy()
    y = P["s"].to_numpy() / P["n"].to_numpy()
    df = pd.DataFrame({"subj": subj[tr], "bench": bench[tr], "y": y[tr]})
    pair = df.groupby(["subj", "bench"])["y"].agg(["sum", "count"]).reset_index()
    rate = (pair["sum"] + 1) / (pair["count"] + 2)
    br = ctx.rate.reindex(pair["bench"]).to_numpy()
    pair["rel"] = np.log(rate / (1 - rate)) - np.log(br / (1 - br))
    pair["grp"] = grp[pair["subj"].to_numpy()]
    gb = pair.groupby(["grp", "bench"])["rel"].mean()
    S = gb.groupby(level="grp").sum()
    N = gb.groupby(level="grp").size()
    g_tr, b_tr = grp[subj[tr]], bench[tr]
    own = gb.reindex(pd.MultiIndex.from_arrays([g_tr, b_tr])).to_numpy()
    s_g, n_g = S.reindex(g_tr).to_numpy(), N.reindex(g_tr).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_tr = np.where(n_g > 1, (s_g - np.nan_to_num(own)) / (n_g - ~np.isnan(own)), np.nan)
    g_te = grp[subj[te]]
    f_te = (S / N).reindex(g_te).to_numpy()
    return pd.DataFrame({"group_rel": f_tr}), pd.DataFrame({"group_rel": f_te})


# Registry: name -> feature function. R1 and K1 are starts (prior.py), not features.
FEATURES = {"R2": R2, "S1": S1, "I0": I0, "I2": I2, "B0": B0, "G1": G1}
STARTS = ("R1", "K1")
CATEGORICAL = {"provider"}


def parse(spec):
    """'R1+R2+S1+I0+I2' -> (start: 'R1', 'K1' or None, [feature components in registry order])."""
    parts = [p.strip().upper() for p in spec.split("+") if p.strip()]
    unknown = set(parts) - set(FEATURES) - set(STARTS)
    if unknown:
        raise SystemExit(f"unknown prior components {sorted(unknown)}; known: {', '.join(STARTS + tuple(FEATURES))}")
    starts = [s for s in STARTS if s in parts]
    if len(starts) > 1:
        raise SystemExit("use one start: R1 or K1")
    return (starts[0] if starts else None), [c for c in FEATURES if c in parts]
