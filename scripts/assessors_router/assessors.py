"""The assessors (priors) of a cell, all built with assessors_v3's fit_prior so they share the v3 prior's settings:

    v3_assessor              the v3 recommended prior (settings.PRIOR) on all training rows: the bar to beat
    tier_assessors           one assessor per performance tier: the same prior, trained only on the rows of the tier's
                             models, on the training benchmarks where the tier has at least MIN_TIER_PAIRS pairs. R1
                             then starts each one from its own tier's mean base rate.
    general_assessor         one assessor on all training rows: the same prior plus three model columns (component
                             M1): strength and mean score over the training benchmarks (strength.py), and model_known.
                             Test rows: the model's values on the training rows if it is known (model_known = 1), else
                             values predicted from its fields (model_known = 0). Training rows get theirs the same way,
                             by cross-validation over the training benchmarks (strength.train_values), so the assessor
                             learns from both cases, as it meets them at test time.
    general_assessor_oracle  trained exactly like general_assessor; test rows get the true values instead (measured on
                             all the data, the test benchmark included). A ceiling, not a submission.
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


def fit_general(d, R, cell, routing, test_rows, n_jobs, device, oracle):
    """The prior plus M1, the model columns (see the module docstring for the two versions)."""
    P = d.P
    model, bench, subj = P["model"].to_numpy(), P["bench"].to_numpy(), P["subj"].to_numpy()
    cols = [*MEASURES, "model_known"]

    def m1(ctx, tr, te):
        a = routing["train_values"].reindex(pd.MultiIndex.from_arrays([bench[tr], subj[tr]]))[cols]
        if oracle:
            b = routing["oracle"].reindex(model[te])[list(MEASURES)].reset_index(drop=True)
        else:
            b = pd.DataFrame({m: routing["test_value"][m][subj[te]] for m in MEASURES})
        b["model_known"] = routing["test_known"][subj[te]]
        return a.reset_index(drop=True), b

    components.FEATURES["M1"] = m1               # registered for this fit only
    try:
        return _fit(d, R, cell, S.PRIOR + "+M1", test_rows, n_jobs, device)
    finally:
        del components.FEATURES["M1"]
