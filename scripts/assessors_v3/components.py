"""The prior's input components (notation of approaches.md), each one isolated so any combination can be tested.

A prior is written as components joined by '+', e.g. "R1+R2+S1+I0+I2" (the default, the basic assessor):

    R1  start: each training benchmark's base rate is the LightGBM init score of its rows; on an unseen benchmark the
        prior starts from the mean base rate of the training benchmarks (equal weight per benchmark)
    R2  model strength: a Rasch-style fit across benchmarks (logit p = c + difficulty[instance] + strength[model])
    S1  system metadata from the competition's subject fields (11 variables of approaches.md), optionally completed
        with model attributes from models.csv (ASSESSOR_MODELS_CSV)
    I0  item text statistics (build_data.item_stats)
    I2  ADeLe v1 demand levels: the 18 expected levels

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
from config import MODELS_CSV, R2_DROPOUT, R2_INNER_FOLDS  # noqa: E402

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


# Registry: name -> feature function. R1 is the start (prior.py), not a feature.
FEATURES = {"R2": R2, "S1": S1, "I0": I0, "I2": I2}
CATEGORICAL = {"provider"}


def parse(spec):
    """'R1+R2+S1+I0+I2' -> (uses R1, [feature components in registry order])."""
    parts = [p.strip().upper() for p in spec.split("+") if p.strip()]
    unknown = set(parts) - set(FEATURES) - {"R1"}
    if unknown:
        raise SystemExit(f"unknown prior components {sorted(unknown)}; known: R1, {', '.join(FEATURES)}")
    return "R1" in parts, [c for c in FEATURES if c in parts]
