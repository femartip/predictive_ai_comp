"""The general assessor at prediction time: its prior on the input rows (assessor_input.build), then T2 on the labeled
pairs, as the evaluation applies them (scripts/assessors_router/streaming.py, testtime.py):

    prior    z = start + trees(features)          assessor/assessor.txt (raw score, by lgb_numpy: the platform has no
                                                  lightgbm), start = R1
    T2       logit p = start + lam[bench] * (z - start) + b[bench] + u[subject, bench] + v[item], fitted by MAP on the
             labeled pairs' labels with the scales of assessor/meta.json (tau); with no labels, p = sigmoid(start +
             alpha * (z - start))

    from assessor import predict
    p = predict(row, labeled_rows, labels)
"""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from lgb_numpy import Model
import testtime as TT

ASSESSOR = Path(__file__).with_name('assessor')


@lru_cache(maxsize=1)
def model():
    meta = json.loads((ASSESSOR / 'meta.json').read_text(encoding='utf-8'))
    booster = Model(ASSESSOR / 'assessor.txt')
    assert booster.feature_names == meta['columns'], 'assessor.txt was trained on other columns than meta.json'
    return booster, meta


def prior(rows):
    """Prior logits z of input rows."""
    return model()[0].raw_score([r['features'] for r in rows]) + np.array([r['start'] for r in rows])


def predict(row, labeled_rows=None, labels=None) -> float:
    """P(success) of the target row, with T2 fitted on the labeled rows and their labels."""
    tau = model()[1]['tau']
    rows = [row, *(labeled_rows or [])]
    z = prior(rows)
    start = np.array([r['start'] for r in rows])
    keys = {t: np.array([r['keys'][t] for r in rows], dtype=object) for t in TT.TERMS}
    y = np.asarray(labels if labels is not None else [], float)
    assert len(y) == len(rows) - 1, 'one label per labeled row'
    fit = TT.fit(y, start[1:], z[1:] - start[1:], {t: k[1:] for t, k in keys.items()}, tau, True)
    p = TT.predict(fit, start[:1], z[:1] - start[:1], {t: k[:1] for t, k in keys.items()})
    return float(p[0])
