"""Test-time models T1 and T2 (approaches.md), fitted by MAP on the labels revealed so far in the draw.

    T1:  logit p = prior                     + b[bench] + u[subject, bench] + v[instance]
    T2:  logit p = start + lam[bench] * dev  + b[bench] + u[subject, bench] + v[instance],   dev = prior - start

- b: is this hidden benchmark easier or harder than expected? u: does this subject do better or worse than expected on
  it? v: how did other subjects of the draw do on this instance? Each ~ N(0, tau^2), so with few labels the
  corrections shrink to zero (tau["bench"], tau["subject"], tau["item"]).
- T2's lam[bench] ~ N(alpha, tau["slope"]^2) says how much to trust the prior on that benchmark: lam = 1 is the prior
  as it is, lam = 0 ignores it (T1 started from R1). With no labels, T2 = start + alpha * dev (= the prior if
  alpha = 1).
- Labels from every pair of the draw enter the same fit (the platform's hook sees all labels bought so far).
Slim version of night_experiments/tt.py (which also has T3 item-feature slopes, kNN, empirical Bayes, ...).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TERMS = ("bench", "subject", "item")


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def keys(bench, subj, inst):
    """Group keys of the random effects for rows (bench, subject, instance codes)."""
    return {"bench": np.asarray(bench), "subject": np.asarray(bench) * 1_000_003 + np.asarray(subj),
            "item": np.asarray(inst)}


def fit(y, base, dev, groups, tau, slope):
    """MAP fit. y: revealed labels (n,); base: prior (T1) or start (T2) logits (n,); dev: prior - start (n,), used if
    slope; groups: keys(...) of the labelled rows; tau: scales. Returns the fitted model (a dict)."""
    model = {"tau": tau, "slope": slope, "eff": {}, "lam": {}}
    if len(y) == 0:
        return model
    idx, levels, sizes = [], [], []
    for t in TERMS:
        codes, uniq = pd.factorize(groups[t])
        idx.append(codes)
        levels.append(uniq)
        sizes.append(len(uniq))
    off = np.cumsum([0] + sizes)
    prec = np.concatenate([np.full(s, 1 / tau[t] ** 2) for t, s in zip(TERMS, sizes)])
    bi, nb = idx[0], sizes[0]                       # slope per benchmark: indexed like the bench effect
    n_re = off[-1]
    alpha, tau_s = tau["alpha"], tau["slope"]

    def logits(th):
        z = base.copy()
        for k in range(len(TERMS)):
            z += th[off[k] + idx[k]]
        if slope:
            z += th[n_re + bi] * dev
        return z

    def loss(th):
        p = _sigmoid(logits(th))
        g = p - y
        re = th[:n_re]
        val = -np.sum(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)) + 0.5 * np.sum(prec * re ** 2)
        grad = np.zeros_like(th)
        for k in range(len(TERMS)):
            grad[off[k]:off[k + 1]] = np.bincount(idx[k], g, sizes[k])
        grad[:n_re] += prec * re
        if slope:
            lam = th[n_re:]
            val += 0.5 * np.sum((lam - alpha) ** 2) / tau_s ** 2
            grad[n_re:] = np.bincount(bi, g * dev, nb) + (lam - alpha) / tau_s ** 2
        return val, grad

    x0 = np.zeros(n_re + (nb if slope else 0))
    if slope:
        x0[n_re:] = alpha
    th = minimize(loss, x0, jac=True, method="L-BFGS-B").x
    model["eff"] = {t: dict(zip(levels[k], th[off[k]:off[k + 1]])) for k, t in enumerate(TERMS)}
    if slope:
        model["lam"] = dict(zip(levels[0], th[n_re:]))
    return model


def predict(model, base, dev, groups):
    """Probabilities for rows with the given base, dev and group keys. Levels never seen in the labels get no
    correction; a benchmark without labels gets lam = alpha."""
    z = base.copy()
    for t, e in model["eff"].items():
        z += np.array([e.get(k, 0.0) for k in groups[t]])
    if model["slope"]:
        lam = model["lam"]
        z += np.array([lam.get(k, model["tau"]["alpha"]) for k in groups["bench"]]) * dev
    return _sigmoid(z)
