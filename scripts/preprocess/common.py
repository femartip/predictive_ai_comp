"""Shared layouts and writers for the assessor JSON Lines files."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import HfApi, hf_hub_download

ROOT = next(p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents] if (p / "pyproject.toml").exists())
DATA = ROOT / "data"
OUT = DATA / "processed"

COLUMNS = [
    # IDs
    "item_id", "benchmark", "subject_id",
    # Target
    "score", "n_runs", "score_type",
    # Item
    "question", "ground_truth", "eval_type", "domain", "modality", "multi_turn", "programming_language",
    "benchmark_release_date",
    # Model: identity only. Model attributes live in data/processed/models.csv (scripts/preprocess/epoch_models_csv.py);
    # join on normalized_name. thinking and reasoning_effort describe the run, not the model.
    "model_name", "normalized_name", "thinking", "reasoning_effort",
    # Setup
    "agentic", "harness", "harness_version", "tools", "access_date",
]

# Observation-level layout used by the generic Measurement DB converter. Unlike
# the older aggregate layout above, this preserves every released trial.
OBSERVATION_COLUMNS = [
    # IDs and observation key
    "response_id", "item_id", "benchmark", "subject_id", "trial", "test_condition", "interactors",
    # Target. score is normalized; raw_score is the released value.
    "score", "raw_score", "response_scale", "n_runs", "score_type",
    # Released model output
    "trace",
    # Item
    "question", "ground_truth", "eval_type", "domain", "modality", "multi_turn", "programming_language",
    "benchmark_release_date",
    # Model identity
    "model_name", "normalized_name", "thinking", "reasoning_effort",
    # Setup
    "agentic", "harness", "harness_version", "tools", "access_date",
]



def load_metadata(benchmark, repository=None, revision=None):
    """Load benchmark metadata locally or from a Hugging Face dataset revision."""
    if repository is None:
        path = DATA / benchmark / "metadata.yaml"
    else:
        path = Path(hf_hub_download(
            repo_id=repository,
            filename=f"{benchmark}/metadata.yaml",
            repo_type="dataset",
            revision=revision,
        ))
    return yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]


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
    """Model identity and run setup per subject_id, from the build's subjects table. thinking (did this run
    think) is unknown here; benchmark scripts that can tell set it."""
    return pd.DataFrame({
        "subject_id": subjects.subject_id, "model_name": subjects.display_name,
        "normalized_name": subjects.normalized_name, "thinking": None, "reasoning_effort": subjects.reasoning_effort,
        "harness": subjects.harness, "harness_version": subjects.harness_version, "access_date": subjects.access_date,
    }).reset_index(drop=True)


def huggingface_provenance(repository, source_tables, revision=None, loaded_at_utc=None):
    """Capture the exact Hugging Face dataset version used by a conversion."""
    revision = revision or HfApi().dataset_info(repository).sha
    return {
        "loaded_at_utc": loaded_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_repository": repository,
        "source_revision": revision,
        "source_tables": list(source_tables),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(df, benchmark, provenance=None):
    missing = set(COLUMNS) - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"
    assert df.item_id.str.startswith(f"{benchmark}:").all()
    assert not df.duplicated(["item_id", "subject_id"]).any(), "duplicate (item, subject) rows"
    assert df.score.between(0, 1).all(), "score outside [0, 1]"
    assert df.normalized_name.notna().all(), "rows without normalized_name"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{benchmark}.jsonl"
    df[COLUMNS].to_json(path, orient="records", lines=True, force_ascii=False)

    if provenance is not None:
        manifest = {
            "benchmark": benchmark,
            **provenance,
            "output_file": path.name,
            "output_rows": len(df),
            "output_sha256": _sha256(path),
        }
        manifest_path = OUT / f"{benchmark}.manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return path


def _validate_observations(df, benchmark):
    missing = set(OBSERVATION_COLUMNS) - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"
    assert df.item_id.str.startswith(f"{benchmark}:").all()
    assert df.response_id.notna().all(), "rows without response_id"
    assert not df.response_id.duplicated().any(), "duplicate response_id rows"
    assert df.trial.notna().all() and df.trial.ge(1).all(), "trial must be a positive integer"
    graded = df.score.notna()
    assert df.loc[graded, "score"].between(0, 1).all(), "normalized score outside [0, 1]"
    assert df.loc[df.raw_score.notna(), "response_scale"].notna().all(), "graded rows without response_scale"


def write_observation_batches(batches, benchmark, provenance=None, output_dir=None):
    """Write observation frames incrementally so large benchmarks stay out of memory."""
    destination = Path(output_dir) if output_dir is not None else OUT
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{benchmark}.jsonl"
    temporary_path = destination / f".{benchmark}.jsonl.tmp"
    output_rows = 0
    with temporary_path.open("w", encoding="utf-8") as file:
        for df in batches:
            _validate_observations(df, benchmark)
            text = df[OBSERVATION_COLUMNS].to_json(
                orient="records", lines=True, force_ascii=False,
            )
            file.write(text)
            if text and not text.endswith("\n"):
                file.write("\n")
            output_rows += len(df)
    temporary_path.replace(path)

    if provenance is not None:
        manifest = {
            "benchmark": benchmark,
            **provenance,
            "output_file": path.name,
            "output_rows": output_rows,
            "output_sha256": _sha256(path),
        }
        manifest_path = destination / f"{benchmark}.manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return path


def write_observations(df, benchmark, provenance=None, output_dir=None):
    """Write one in-memory observation frame; primarily useful for small data and tests."""
    assert not df.response_id.duplicated().any(), "duplicate response_id rows"
    return write_observation_batches([df], benchmark, provenance, output_dir)
