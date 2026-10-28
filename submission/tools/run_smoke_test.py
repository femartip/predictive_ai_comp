#!/usr/bin/env python3
"""Run a small local smoke test against an unpacked submission directory."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import math
import numbers
import os
import sys
import tempfile
import time
from pathlib import Path

try:
    from streaming_ingestion import AcquisitionHook
except ModuleNotFoundError as exc:
    if exc.name != "streaming_ingestion":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "competition_contract"))
    from streaming_ingestion import AcquisitionHook


try:
    from prediction_workers import prediction_concurrency, prediction_input_key, write_parallel_predictions
except ModuleNotFoundError as exc:
    if exc.name != "prediction_workers":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "competition_contract"))
    from prediction_workers import prediction_concurrency, prediction_input_key, write_parallel_predictions


KIT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_TEST_DIR = KIT_DIR / "sample_data" / "test"
SAMPLE_REF_DIR = KIT_DIR / "sample_data" / "ref"
LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--concurrency", type=int, choices=range(1, 17), default=1,
                        help="Test isolated prediction workers; use 4 to match the default CPU runtime.")
    args = parser.parse_args()

    try:
        result = run_smoke_test(args.submission_dir, concurrency=args.concurrency)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "OK: smoke test passed "
        f"({int(result['prediction_count'])} predictions, brier={result['brier']:.6f})."
    )
    return 0


def run_smoke_test(submission_dir: Path, *, concurrency: int = 1) -> dict[str, float]:
    submission_dir = submission_dir.resolve()
    concurrency = prediction_concurrency(submission_dir, limit=concurrency)
    model_path = submission_dir / "model.py"
    if not model_path.exists():
        raise ValueError(f"model.py not found at {model_path}")

    test_rows = _load_csv(SAMPLE_TEST_DIR / "test_items.csv", key="item_id")
    model_rows = _load_csv(SAMPLE_TEST_DIR / "models.csv", key="model_id")
    pairs = _load_csv_rows(SAMPLE_TEST_DIR / "test_pairs.csv")
    labels = {
        (row["model_id"], row["item_id"]): int(row["label"])
        for row in _load_csv_rows(SAMPLE_REF_DIR / "ground_truth.csv")
    }

    inputs = [_build_input(pair, test_rows, model_rows) for pair in pairs]

    labeling_path = submission_dir / "labeling.py"
    labeled = []
    if labeling_path.exists():
        labeling = _load_module(labeling_path, "smoke_labeling", submission_dir)
        acquisition = getattr(labeling, "acquisition_function", None)
        if not callable(acquisition):
            raise ValueError("labeling.py must define callable acquisition_function(input, prediction, labeled, context).")
        for input_row in inputs:
            decision = AcquisitionHook(acquisition)(_copy_input(input_row), 0.5, [], {
                "subject_id": "subject_001", "benchmark_id": "benchmark_001",
                "labels_remaining": 31, "labels_acquired": 0, "max_labels": 31,
                "items_remaining": 40,
            })
    if inputs:
        labeled = [
            [_copy_input(inputs[0]), labels[(pairs[0]["model_id"], pairs[0]["item_id"])]]
        ]

    model = _load_module(model_path, "smoke_model", submission_dir)
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise ValueError("model.py must define callable predict(input, labeled=None).")

    if concurrency > 1:
        with tempfile.TemporaryDirectory(prefix="paiec_smoke_") as directory:
            output = Path(directory) / "predictions.csv"
            write_parallel_predictions(
                model_path, pairs, inputs, labeled, output, concurrency=concurrency,
                deadline=time.monotonic() + 300, per_call_timeout=120, log=print,
                validate_score=lambda value: _assert_probability(value, "predict()"),
                worker_env={LOCAL_SMOKE_TEST_ENV: "1"},
            )
            rows = _load_csv_rows(output)
            predictions = [float(row["predicted_probability"]) for row in rows]
        truth = [labels[(pair["model_id"], pair["item_id"])] for pair in pairs]
        return {"prediction_count": len(predictions), "brier": _brier(truth, predictions)}

    predictions: list[float] = []
    truth: list[int] = []
    predictions_by_input = {}
    for pair, input_row in zip(pairs, inputs):
        key = prediction_input_key(input_row)
        if key not in predictions_by_input:
            try:
                value = predict(_copy_input(input_row), labeled=copy.deepcopy(labeled))
            except TypeError as exc:
                raise ValueError("predict() must accept the labeled keyword argument.") from exc
            predictions_by_input[key] = _assert_probability(value, "predict()")
        score = predictions_by_input[key]
        predictions.append(score)
        truth.append(labels[(pair["model_id"], pair["item_id"])])

    return {
        "prediction_count": len(predictions),
        "brier": _brier(truth, predictions),
    }


def _load_csv(path: Path, *, key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in _load_csv_rows(path) if row.get(key)}


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"Sample data file missing: {path}")
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


SUBJECT_FIELDS = (
    "normalized_name",
    "provider",
    "release_date",
    "access_date",
    "harness",
    "reasoning_effort",
    "harness_version",
    "subject_features_extra",
)

ITEM_FIELDS = ("item_content", "item_features", "interactors", "benchmark_id")


def _build_input(
    pair: dict[str, str],
    item_rows: dict[str, dict[str, str]],
    model_rows: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    item = item_rows.get(pair["item_id"], {})
    model = model_rows.get(pair["model_id"], {})
    return [
        {key: (model.get(key) or "").strip() for key in SUBJECT_FIELDS},
        {
            "item_content": item.get("item_content") or "",
            "item_features": (item.get("item_features") or "").strip(),
            "interactors": (item.get("interactors") or "").strip(),
            "benchmark_id": (item.get("benchmark_id") or "").strip(),
        },
    ]


def _copy_input(input_row: list[dict[str, str]]) -> list[dict[str, str]]:
    return [dict(input_row[0]), dict(input_row[1])]


def _load_module(path: Path, module_name: str, submission_dir: Path):
    previous_path = list(sys.path)
    previous_env = os.environ.get(LOCAL_SMOKE_TEST_ENV)
    sys.path.insert(0, str(submission_dir))
    os.environ[LOCAL_SMOKE_TEST_ENV] = "1"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not import {path.name}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_env is None:
            os.environ.pop(LOCAL_SMOKE_TEST_ENV, None)
        else:
            os.environ[LOCAL_SMOKE_TEST_ENV] = previous_env
        sys.path[:] = previous_path


def _assert_probability(value, label: str) -> float:
    number = _assert_finite_number(value, label)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must return a probability in [0, 1].")
    return number


def _assert_finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must return a finite numeric value.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must return a finite numeric value.")
    return number


def _brier(labels: list[int], predictions: list[float]) -> float:
    total = sum((prediction - label) ** 2 for label, prediction in zip(labels, predictions))
    return total / max(len(labels), 1)


if __name__ == "__main__":
    raise SystemExit(main())
