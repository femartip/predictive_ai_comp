"""The submission: item processing, subject processing, ADeLe annotation, the assessor's input rows, then the general
assessor with T2 on the labeled pairs."""

from data_processing import preprocess
from subject_processing import process
from item_annotation import annotate
from assessor_input import build
import assessor


def predict(input: list, labeled: list | None = None) -> float:
    processed_input, processed_labeled = annotate(*process(*preprocess(input, labeled)))
    row, labeled_rows = build(processed_input, processed_labeled)
    return assessor.predict(row, labeled_rows, [label for _, label in labeled or []])
