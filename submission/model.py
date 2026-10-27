"""Integration example only: replace the constant baseline with the team's predictor."""

from data_processing import preprocess


def predict(input: list, labeled: list | None = None) -> float:
    processed_input, processed_labeled = preprocess(input, labeled)
    # TEAM INTEGRATION: pass these to model processing / the trained assessor.
    # return float(assessor.predict(processed_input, labeled=processed_labeled))
    return 0.5
