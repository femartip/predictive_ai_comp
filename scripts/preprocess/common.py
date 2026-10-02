"""Shared layout for the per-benchmark assessor files (JSON Lines): one row per (subject, item)."""

from pathlib import Path

import pandas as pd
import yaml

ROOT = next(p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents] if (p / "pyproject.toml").exists())
DATA = ROOT / "data"
OUT = DATA / "processed"
MODELS_FILE = Path(__file__).with_name("models.yaml")

COLUMNS = [
    # IDs
    "item_id", "benchmark", "subject_id",
    # Target
    "score", "n_runs", "score_type",
    # Item
    "question", "ground_truth", "eval_type", "domain", "modality", "multi_turn", "programming_language",
    "benchmark_release_date",
    # Model
    "provider", "model_family", "model_name", "params_total_b", "params_active_b", "architecture", "model_type",
    "thinking", "reasoning_effort", "open_weights", "release_date", "knowledge_cutoff", "context_window",
    "multimodal_input",
    # Setup
    "agentic", "harness", "harness_version", "tools", "access_date",
]

MODEL_FIELDS = ["model_family", "params_total_b", "params_active_b", "architecture", "model_type", "thinking",
                "open_weights", "knowledge_cutoff", "context_window", "multimodal_input"]


def load_metadata(benchmark):
    return yaml.safe_load((DATA / benchmark / "metadata.yaml").read_text())["benchmark"]


def benchmark_columns(meta):
    """Item and setup columns that are constant within a benchmark, from metadata.yaml."""
    return {
        "score_type": meta["response_type"],
        "domain": ";".join(meta["domain"]),
        "modality": ";".join(meta["modality"]),
        "multi_turn": meta["multi_single_turn"] == "multi_turn",
        "benchmark_release_date": meta["release_date"],
        "agentic": meta["subject_type"] == "agent",
    }


def subject_columns(subjects):
    """Model and setup columns per subject_id: build fields plus the curated models.yaml entry."""
    curated = yaml.safe_load(MODELS_FILE.read_text()) or {}
    missing = sorted(set(subjects.normalized_name) - set(curated))
    if missing:
        print(f"warning: no curated entry in {MODELS_FILE.name} for {missing}")
    rows = []
    for s in subjects.itertuples():
        model = curated.get(s.normalized_name, {})
        row = {"subject_id": s.subject_id, "provider": s.provider, "model_name": s.display_name,
               "reasoning_effort": s.reasoning_effort, "release_date": s.release_date, "harness": s.harness,
               "harness_version": s.harness_version, "access_date": s.access_date}
        row |= {f: model.get(f) for f in MODEL_FIELDS}
        # A model without a thinking mode has no reasoning effort to set.
        if pd.isna(row["reasoning_effort"]) and model.get("thinking") is False:
            row["reasoning_effort"] = "none"
        rows.append(row)
    return pd.DataFrame(rows)


def write(df, benchmark):
    missing = set(COLUMNS) - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"
    assert df.item_id.str.startswith(f"{benchmark}:").all()
    assert not df.duplicated(["item_id", "subject_id"]).any(), "duplicate (item, subject) rows"
    assert df.score.between(0, 1).all(), "score outside [0, 1]"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{benchmark}.jsonl"
    df[COLUMNS].to_json(path, orient="records", lines=True, force_ascii=False)
    return path
