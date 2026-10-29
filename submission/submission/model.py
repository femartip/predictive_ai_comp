"""The submission: item processing, subject processing, ADeLe annotation, the assessor's input rows, then the general
assessor with T2 on the labeled pairs.

A failed prediction stops the whole run on the platform, so a call that raises falls back, with the error logged: first
to the assessor's prior without T2, then to its start (R1, the mean base rate of the training benchmarks), then to 0.5.
"""

import logging
import math

from data_processing import preprocess
from subject_processing import process
from item_annotation import annotate
from assessor_input import build
import assessor

logger = logging.getLogger(__name__)


def _probability(p):
    p = float(p)
    if not math.isfinite(p):
        raise ValueError(f'not a finite probability: {p}')
    return min(1.0, max(0.0, p))


def predict(input: list, labeled: list | None = None) -> float:
    row = None
    try:
        processed_input, processed_labeled = annotate(*process(*preprocess(input, labeled)))
        row, labeled_rows = build(processed_input, processed_labeled)
        return _probability(assessor.predict(row, labeled_rows, [label for _, label in labeled or []]))
    except Exception:
        logger.exception('predict failed; falling back')
    try:
        if row is not None:
            return _probability(assessor.predict(row))           # the prior, no T2
        return _probability(1 / (1 + math.exp(-assessor.model()[1]['r1'])))
    except Exception:
        logger.exception('fallback failed; returning 0.5')
        return 0.5
