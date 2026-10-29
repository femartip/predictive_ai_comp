"""Fit what the submission's assessor_input.py and assessor.py need, on all the training data, as
scripts/assessors_router computes it for a test cell (strength.route, components.G1, assessors.fit_general), with
every benchmark and model in training:

    submission/assessor/meta.json          r1 (R1's start: logit of the mean base rate of the benchmarks), the feature
                                           columns in the assessor's order, the router's columns and categories,
                                           T2's scales, the build, the seed and the training-row sample
    submission/assessor/models.parquet     model_key -> strength, mean_score (strength.measures on all rows)
    submission/assessor/router_<m>.txt     strength.Router regressors (LightGBM) of each measure, on every subject
    submission/assessor/groups.parquet     model group -> group_rel (components.G1, test-row formula)
    submission/assessor/assessor.txt       the general assessor (LightGBM, prior R1+B0+G1+I2+M1, fit_assessor)

G1 and the assessor use the rows fit_prior samples (training_rows: at most TRAIN_PAIRS_PER_BENCHMARK per benchmark,
fit_prior's seed for seed SEED, repetition 0, CELL). The assessor's training rows get their model columns by
cross-validation over the benchmarks (strength.train_values), as in the evaluation: one model, the folds only give
the values. Its fit is checked against prior.fit_prior itself, run on the same data.
--check compares, for every training subject, assessor_input's subject features with the training ones.

    poetry run python submission/fit_tables.py [--data BUILD] [--check]
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/assessors_router'))
sys.path.insert(0, str(ROOT / 'submission/submission'))
import settings as S  # noqa: E402  (first: puts assessors_v3 on sys.path)
import assessors as A  # noqa: E402
import components  # noqa: E402
from bank import BankData, builds  # noqa: E402
from common import log, logit  # noqa: E402
from config import ADELE_CODES, LGBM_PARAMS, TAU, TRAIN_PAIRS_PER_BENCHMARK  # noqa: E402
from prior import Context, fit_prior, sample_rows  # noqa: E402
from protocol import sibling_groups  # noqa: E402
from strength import (MEASURES, Router, _tables, completeness, measures, model_weights,  # noqa: E402
                      train_values)

OUT = ROOT / 'submission/submission/assessor'
SEED = 0
CELL = (0, None)        # fit_prior's cell argument: only its seed reads it (AllTraining holds no benchmark out)


class AllTraining:
    """A protocol Repetition with every row in training (what fit_prior and train_values read of one)."""

    def __init__(self, d):
        self.n, self.rep, self.siblings = len(d.P), 0, sibling_groups(d)

    def train_mask(self, cell):
        return np.ones(self.n, bool)


def training_rows(d):
    """The rows the final prior is trained on: fit_prior's sample (its seed for seed SEED, repetition 0, CELL)."""
    rng = np.random.default_rng([SEED, 0, CELL[0], 99 if CELL[1] is None else CELL[1]])
    return sample_rows(d.P, np.ones(len(d.P), bool), TRAIN_PAIRS_PER_BENCHMARK, rng)


def group_table(d, tr, rate):
    """components.G1's per-group value for a test row: per (subject, benchmark) the smoothed rate's logit minus the
    benchmark's logit base rate, averaged per (group, benchmark), then over benchmarks."""
    P = d.P
    grp = components.model_groups(d)
    subj, bench = P['subj'].to_numpy()[tr], P['bench'].to_numpy()[tr]
    y = P['s'].to_numpy()[tr] / P['n'].to_numpy()[tr]
    pair = pd.DataFrame({'subj': subj, 'bench': bench, 'y': y}).groupby(['subj', 'bench'])['y'] \
        .agg(['sum', 'count']).reset_index()
    r = (pair['sum'] + 1) / (pair['count'] + 2)
    br = rate.reindex(pair['bench']).to_numpy()
    pair['rel'] = np.log(r / (1 - r)) - np.log(br / (1 - br))
    pair['grp'] = grp[pair['subj'].to_numpy()]
    gb = pair.groupby(['grp', 'bench'])['rel'].mean()
    table = (gb.groupby(level='grp').sum() / gb.groupby(level='grp').size()).rename('group_rel')
    ctx = Context(d, SEED, None)                     # check against components.G1 itself on a few rows
    ctx.rate = rate
    probe = tr[:: max(1, len(tr) // 2000)]
    _, te = components.G1(ctx, tr, probe)
    expected = table.reindex(grp[P['subj'].to_numpy()[probe]]).to_numpy()
    assert np.allclose(te['group_rel'].to_numpy(), expected, equal_nan=True), 'group table differs from G1'
    return table.rename_axis('group').reset_index()


def fit_assessor(d, rate, meas, n_jobs=S.N_JOBS):
    """The general assessor on all the data: assessors.fit_general's features for its training rows (M1 from
    train_values), trained by assessors.fit_prior_kept (fit_prior, keeping the booster). Checked against fit_prior
    itself on a probe of rows. Returns the booster's feature columns."""
    P = d.P
    R = AllTraining(d)
    tv = train_values(d, R, np.arange(len(P)), rate, n_jobs)
    comp = completeness(d)
    model, bench, subj = P['model'].to_numpy(), P['bench'].to_numpy(), P['subj'].to_numpy()

    def m1(ctx, tr, te):                         # assessors.fit_general's M1; on the probe every model is known
        a = tv.reindex(pd.MultiIndex.from_arrays([bench[tr], subj[tr]]))[[*MEASURES, 'model_known']]
        a = a.reset_index(drop=True)
        b = meas.reindex(model[te])[list(MEASURES)].reset_index(drop=True).assign(model_known=1.0)
        a['completeness'], b['completeness'] = comp[subj[tr]], comp[subj[te]]
        return a, b

    spec = S.PRIOR + '+M1'
    components.FEATURES['M1'] = m1               # registered for this fit only
    try:
        tr = training_rows(d)
        probe = tr[:: max(1, len(tr) // 5000)]
        fit = A.fit_prior_kept(d, R, CELL, spec, probe, n_jobs, 'cpu', seed=SEED)
        assert fit['train_rows'] == len(tr), 'fit_prior_kept trained on other rows than training_rows'
        ref = fit_prior(d, R, CELL, spec, probe, n_jobs=n_jobs, device='cpu', seed=SEED)
        diff = float(np.abs(fit['z'] - ref['z']).max())
        assert diff < 1e-6, f'the final fit differs from fit_prior on {len(probe)} rows (max {diff})'
    finally:
        del components.FEATURES['M1']
    fit['booster'].save_model(str(OUT / 'assessor.txt'))
    log(f'assessor: {len(tr):,} training rows, {fit["X"].shape[1]} features; the same as fit_prior on {len(probe)} '
        f'rows (max difference {diff:.1e})')
    return list(fit['X'].columns)


def fit(d, build):
    P = d.P
    rows = np.arange(len(P))
    rate = d.bench_mean()
    tr = training_rows(d)
    meas = measures(_tables(d, rows, rate))
    subj = np.unique(P['subj'].to_numpy())
    model, w = d.subj_model[subj], model_weights(d, subj)
    OUT.mkdir(parents=True, exist_ok=True)
    for m in MEASURES:
        r = Router(d, n_jobs=S.N_JOBS, seed=SEED).fit(subj, meas[m].reindex(model).to_numpy(), w)
        booster, levels = r.model
        booster.save_model(str(OUT / f'router_{m}.txt'))
        router_levels = {c: [str(v) for v in lv] for c, lv in levels.items()}
        router_columns = list(r.X.columns)
    group_table(d, tr, rate).to_parquet(OUT / 'groups.parquet', index=False)
    levels = [f'{c}_exp' for c in ADELE_CODES]
    columns = levels + [f'bench_{c}' for c in levels] + ['group_rel', *MEASURES, 'model_known', 'completeness']
    assert fit_assessor(d, rate, meas) == columns, 'the assessor was trained on other columns'
    table = meas.copy()
    table.index = d.model_names[table.index.to_numpy()]
    table.rename_axis('model_key').reset_index().to_parquet(OUT / 'models.parquet', index=False)
    meta = {'r1': float(logit(rate.mean())), 'columns': columns, 'router_columns': router_columns,
            'router_levels': router_levels, 'tau': TAU, 'build': str(build), 'seed': SEED,
            'train_pairs_per_benchmark': TRAIN_PAIRS_PER_BENCHMARK, 'training_rows': int(len(tr)),
            'prior': S.PRIOR + '+M1', 'lgbm': LGBM_PARAMS, 'benchmarks': int(len(rate)), 'models': int(len(meas))}
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    log(f'wrote {OUT}: r1 {meta["r1"]:.4f}, {len(meas)} models, {meta["training_rows"]:,} training rows')


def check(d, build):
    """assessor_input's S1 and completeness for every training subject, from its competition fields (normalized_name =
    model_key, as training reads it) and the model attributes the build carries, against the training tables."""
    import pyarrow.parquet as pq
    import assessor_input as AI
    cols = ['subject_id', 'model_key', 'provider', 'release_date', 'access_date', 'harness', 'reasoning_effort',
            'model_provider', 'model_release_date', 'model_parameters']
    subj = pd.concat([pq.read_table(f, columns=cols).to_pandas().drop_duplicates('subject_id')
                      for f in sorted(Path(build).glob('*.parquet')) if f.stem not in ('models', 'benchmarks')])
    subj = subj.drop_duplicates('subject_id').set_index('subject_id').reindex(d.subj_names)
    train = components.system_table(d)
    inf = pd.DataFrame([AI.system_row({**r._asdict(), 'normalized_name': r.model_key})
                        for r in subj.reset_index(drop=True).itertuples(index=False)])
    bad = {}
    for c in train.columns:
        a, b = train[c].to_numpy(), inf[c].to_numpy()
        same = (a == b) if c == 'provider' else np.isclose(a.astype(float), b.astype(float), equal_nan=True)
        bad[c] = int((~same).sum())
    comp = ((inf['provider'] != '(unknown)').astype(int) + np.isfinite(inf['log_params']).astype(int)
            + inf['release_years'].notna().astype(int)).to_numpy(float)
    bad['completeness'] = int((comp != completeness(d)).sum())
    log(f'check, {len(train)} subjects, differences per column: {bad}')
    for c, n in bad.items():
        if n and c != 'completeness':
            i = np.flatnonzero(~((train[c] == inf[c]) if c == 'provider' else
                                 np.isclose(train[c].astype(float), inf[c].astype(float), equal_nan=True)))[:5]
            print(c, pd.DataFrame({'train': train[c].to_numpy()[i], 'inference': inf[c].to_numpy()[i],
                                   'model_key': subj['model_key'].to_numpy()[i]}).to_string())
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, default=None, help='assessor_data build (default: the newest)')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    build = a.data or builds()[-1]
    with tempfile.TemporaryDirectory() as tmp:       # S1's model table, only needed while fitting
        d = BankData(build, Path(tmp) / 'models.csv')
        if a.check:
            check(d, build)
        else:
            fit(d, build)


if __name__ == '__main__':
    main()
