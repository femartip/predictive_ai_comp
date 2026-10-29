"""Streaming simulation of the routed assessors: assessors_v3's simulate.py with this folder's predictors.

Same semantics as v3: every candidate is bought in the pair's fixed order, every pair reaches a checkpoint before any
pair gets more labels, and at each budget the test-time model is refitted on every label of the draw. Every assessor
gets T2 on top (README section 3), each with its own test-time scales (taus[name]).

Predictors:
    B1, B2, B3, oracle              the v3 baselines (B3 = R1 | T1; oracle = each pair's true level)
    v3_assessor                     the v3 recommended prior | T2: the bar to beat
    tier_router_assessor            the router picks the subject's tier from its fields (most likely tier); that
                                    tier's assessor | T2. A known model goes to its own training tier.
    tier_router_assessor_oracle     the same with the subject's true tier: the ceiling of routing
    general_assessor                one assessor with the model columns: measured if the model is known, else
                                    predicted from its fields | T2
    general_assessor_oracle         the same with the true values (test benchmark included): its ceiling
    general_assessor_inference      general_assessor with the hidden benchmark's B0 as predict() builds it from its
                                    call's evidence: the item and the draw's labeled items of its benchmark so far
                                    (assessors.evidence_logits, per draw and budget)
Records: the columns of v3's run_draws, so its scoring (draw_scores) and bootstrap apply unchanged.
"""

import numpy as np
import pandas as pd

import settings  # noqa: F401  (puts assessors_v3 on sys.path)
import testtime as TT
from config import BUDGETS
from simulate import pair_scores

BASELINES = ("B1", "B2", "B3", "oracle")
# the fitted part each assessor needs
PART = {"v3_assessor": "v3_assessor", "tier_router_assessor": "tier_assessors",
        "tier_router_assessor_oracle": "tier_assessors", "general_assessor": "general_assessor",
        "general_assessor_oracle": "general_assessor_oracle", "general_assessor_inference": "general_assessor"}
PREDICTORS = BASELINES + tuple(PART)


def _logits(name, p, budget, evidence):
    """(acquisition logits, evaluation logits, start) of an assessor on one pair at a budget. evidence: this pair's
    {budget: (acquisition, evaluation logits)} for general_assessor_inference."""
    if name == "general_assessor_inference":
        acq, ev = evidence[budget]
        return acq, ev, p["general_assessor"]["start"]
    if PART[name] != "tier_assessors":
        x = p[PART[name]]
        return x["acq"], x["eval"], x["start"]
    k = p["tier_oracle" if name == "tier_router_assessor_oracle" else "tier_router"]
    x = p["tier_assessors"]
    return x["acq"][k], x["eval"][k], x["starts"][k]


def run_draw(draw, pairs, predictors, taus, budgets=BUDGETS, evidence=None):
    """Score one draw (see v3 simulate.run_draw). pairs: (bench, subj) -> dict built by evaluate_router.py; evidence:
    this draw's assessors.evidence_logits (needed by general_assessor_inference)."""
    P = [pairs[g] for g in draw]
    ne = [len(p["eval_inst"]) for p in P]
    ev_g = TT.keys(np.repeat([p["bench"] for p in P], ne), np.repeat([p["subj"] for p in P], ne),
                   np.concatenate([p["eval_inst"] for p in P]))
    ev_r1 = np.repeat([p["r1"] for p in P], ne)
    cut = np.cumsum([0] + ne)
    out = []
    for k in budgets:
        m = [min(k, len(p["acq_inst"])) for p in P]
        y = np.concatenate([p["revealed"][:mm] for p, mm in zip(P, m)])
        g = TT.keys(np.repeat([p["bench"] for p in P], m), np.repeat([p["subj"] for p in P], m),
                    np.concatenate([p["acq_inst"][:mm] for p, mm in zip(P, m)]))
        preds = {}
        for name in predictors:
            if name == "B1":
                preds[name] = np.full(cut[-1], 0.5)
            elif name == "B2":
                preds[name] = np.repeat([p["revealed"][:mm].mean() if mm else 0.5 for p, mm in zip(P, m)], ne)
            elif name == "oracle":
                preds[name] = np.repeat([p["s"].sum() / p["n"].sum() for p in P], ne)
            elif name == "B3":
                r1 = np.repeat([p["r1"] for p in P], m)
                preds[name] = TT.predict(TT.fit(y, r1, None, g, taus["B3"], False), ev_r1, None, ev_g)
            else:
                parts = [_logits(name, p, k, evidence and evidence[g]) for g, p in zip(draw, P)]
                za = np.concatenate([a[:mm] for (a, _, _), mm in zip(parts, m)]).astype(float)
                sa = np.repeat([s for _, _, s in parts], m)
                ze = np.concatenate([e for _, e, _ in parts]).astype(float)
                se = np.repeat([s for _, _, s in parts], ne)
                fit = TT.fit(y, sa, za - sa, g, taus[name], True)
                preds[name] = TT.predict(fit, se, ze - se, ev_g)
        for i, p in enumerate(P):
            sl = slice(cut[i], cut[i + 1])
            y_mean = float(p["s"].sum() / p["n"].sum())
            for name, pr in preds.items():
                b, e = pair_scores(pr[sl], p["n"], p["s"], p["q"])
                out.append((p["bench"], p["subj"], k, name, b, e, m[i], float(np.average(pr[sl], weights=p["n"])),
                            y_mean))
    return out


def run_draws(draws, pairs, predictors, taus, first_index=0, evidence=None):
    rows = []
    for t, draw in enumerate(draws):
        for r in run_draw(draw, pairs, predictors, taus, evidence=evidence and evidence[t]):
            rows.append((first_index + t, len({b for b, _ in draw}), *r))
    return pd.DataFrame(rows, columns=["draw", "draw_benchmarks", "bench", "subj", "budget", "predictor", "brier",
                                       "ece", "labels", "p_mean", "y_mean"])
