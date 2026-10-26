"""Fast path of protocol v3 (section 6): score predictors on draws with the platform's streaming semantics.

Acquisition here is the basic rule: buy every candidate. With a fixed random order per pair, the labels a pair holds
at budget k are then its first k acquisition instances (fewer if it has fewer), and every pair reaches a checkpoint
before any pair gets more labels, as on the platform. At each budget the test-time model is refitted on every label of
the draw (all pairs) and the evaluation instances are scored.

Predictors (name: prior | test-time, approaches.md):
    B1      0.5 always
    B2      the running mean of the pair's revealed labels (0.5 before the first label): the official baseline
    B3      R1 | T1 (R1's global start + test-time corrections; no trained prior)
    S1T1    start | T1: the prior's own start (= B3 with R1; the routed start with K1) + T1, no trees: routing alone
    oracle  each pair's true mean score on its evaluation instances: a floor for any level-only method
    T0      prior | T0 (the prior alone)
    T1      prior | T1
    T2      prior | T2
Scores per (draw, pair, budget, predictor): Brier over every response run of the evaluation instances,
sum(n p^2 - 2 p s + q) / sum(n), and ECE (ECE_BINS equal-width bins, weighted by runs, within the pair); plus the
pair's mean prediction and mean outcome (p_mean, y_mean, weighted by runs), so report.py can split the Brier into a
level part, (p_mean - y_mean)^2, and the rest (ranking and spread within the pair).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import testtime as TT  # noqa: E402
from config import ALC_WEIGHTS, BUDGETS, ECE_BINS  # noqa: E402

PREDICTORS = ("B1", "B2", "B3", "S1T1", "oracle", "T0", "T1", "T2")


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def pair_scores(p, n, s, q, bins=ECE_BINS):
    """Brier and ECE of predictions p on one pair's evaluation instances (n runs, s sum, q sum of squares)."""
    brier = float(np.sum(n * p ** 2 - 2 * p * s + q) / np.sum(n))
    b = np.minimum((p * bins).astype(int), bins - 1)
    gap = np.abs(np.bincount(b, n * p, bins) - np.bincount(b, s, bins)).sum()
    return brier, float(gap / np.sum(n))


def run_draw(draw, pairs, predictors, tau1, tau2, budgets=BUDGETS):
    """Score one draw. draw: list of (bench, subj); pairs: (bench, subj) -> dict with acq_inst, revealed, acq_z
    (prior logits), eval_inst, eval_z, n, s, q, start (the prior's start for this pair's benchmark, logit) and r1
    (R1's global start). Returns a list of records (bench, subj, budget, predictor, brier, ece, labels, p_mean,
    y_mean)."""
    P = [pairs[g] for g in draw]
    ev_z = np.concatenate([p["eval_z"] for p in P])
    ev_st = np.concatenate([np.full(len(p["eval_inst"]), p["start"]) for p in P])
    ev_r1 = np.concatenate([np.full(len(p["eval_inst"]), p["r1"]) for p in P])
    ev_g = TT.keys(np.concatenate([np.full(len(p["eval_inst"]), p["bench"]) for p in P]),
                   np.concatenate([np.full(len(p["eval_inst"]), p["subj"]) for p in P]),
                   np.concatenate([p["eval_inst"] for p in P]))
    cut = np.cumsum([0] + [len(p["eval_inst"]) for p in P])
    out = []
    for k in budgets:
        m = [min(k, len(p["acq_inst"])) for p in P]
        y = np.concatenate([p["revealed"][:mm] for p, mm in zip(P, m)])
        z = np.concatenate([p["acq_z"][:mm] for p, mm in zip(P, m)])
        st = np.concatenate([np.full(mm, p["start"]) for p, mm in zip(P, m)])
        r1 = np.concatenate([np.full(mm, p["r1"]) for p, mm in zip(P, m)])
        g = TT.keys(np.concatenate([np.full(mm, p["bench"]) for p, mm in zip(P, m)]),
                    np.concatenate([np.full(mm, p["subj"]) for p, mm in zip(P, m)]),
                    np.concatenate([p["acq_inst"][:mm] for p, mm in zip(P, m)]))
        preds = {}
        for name in predictors:
            if name == "B1":
                preds[name] = np.full(len(ev_z), 0.5)
            elif name == "B2":
                preds[name] = np.concatenate([np.full(len(p["eval_inst"]), p["revealed"][:mm].mean() if mm else 0.5)
                                              for p, mm in zip(P, m)])
            elif name == "oracle":
                preds[name] = np.concatenate([np.full(len(p["eval_inst"]), p["s"].sum() / p["n"].sum()) for p in P])
            elif name == "T0":
                preds[name] = _sigmoid(ev_z)
            elif name == "B3":
                preds[name] = TT.predict(TT.fit(y, r1, None, g, tau1, False), ev_r1, None, ev_g)
            elif name == "S1T1":
                preds[name] = TT.predict(TT.fit(y, st, None, g, tau1, False), ev_st, None, ev_g)
            elif name == "T1":
                preds[name] = TT.predict(TT.fit(y, z, None, g, tau1, False), ev_z, None, ev_g)
            elif name == "T2":
                preds[name] = TT.predict(TT.fit(y, st, z - st, g, tau2, True), ev_st, ev_z - ev_st, ev_g)
        for i, p in enumerate(P):
            sl = slice(cut[i], cut[i + 1])
            y_mean = float(p["s"].sum() / p["n"].sum())
            for name, pr in preds.items():
                b, e = pair_scores(pr[sl], p["n"], p["s"], p["q"])
                out.append((p["bench"], p["subj"], k, name, b, e, m[i], float(np.average(pr[sl], weights=p["n"])),
                            y_mean))
    return out


def run_draws(draws, pairs, predictors, tau1, tau2, first_index=0):
    """Score several draws; a DataFrame with one row per (draw, pair, budget, predictor)."""
    rows = []
    for t, draw in enumerate(draws):
        for r in run_draw(draw, pairs, predictors, tau1, tau2):
            rows.append((first_index + t, len({b for b, _ in draw}), *r))
    return pd.DataFrame(rows, columns=["draw", "draw_benchmarks", "bench", "subj", "budget", "predictor", "brier",
                                       "ece", "labels", "p_mean", "y_mean"])


# ----------------------------------------------------------------------------- scores

def draw_scores(rec, keys=("rep", "regime", "cell", "draw")):
    """Per draw and predictor: the mean over pairs at each budget (the leaderboard averages pairs), then ALC =
    the weighted mean over budgets, for Brier and ECE. Returns one row per (keys..., predictor)."""
    keys = [k for k in keys if k in rec.columns]
    m = rec.groupby(keys + ["predictor", "budget"])[["brier", "ece"]].mean()
    w = pd.Series(ALC_WEIGHTS, index=BUDGETS, name="w")
    m = m.join(w, on="budget")
    alc = (m[["brier", "ece"]].mul(m["w"], axis=0)).groupby(level=keys + ["predictor"]).sum()
    return alc.rename(columns={"brier": "brier_alc", "ece": "ece_alc"}).reset_index()
