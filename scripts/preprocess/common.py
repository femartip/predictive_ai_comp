"""Shared layout for the per-benchmark assessor files (JSON Lines): one row per (subject, item)."""

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
    "provider", "model_family", "model_name", "normalized_name", "params_total_b", "params_active_b", "architecture", "model_type",
    "thinking", "reasoning_effort", "open_weights", "release_date", "knowledge_cutoff", "context_window",
    "multimodal_input",
    # Setup
    "agentic", "harness", "harness_version", "tools", "access_date",
]

MODEL_FIELDS = ["model_family", "params_total_b", "params_active_b", "architecture", "model_type", "thinking",
                "open_weights", "knowledge_cutoff", "context_window", "multimodal_input"]


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


def load_curated():
    """Curated model attributes from models.yaml, keyed by normalized_name; empty if the file is absent."""
    return (yaml.safe_load(MODELS_FILE.read_text(encoding="utf-8")) or {}) if MODELS_FILE.exists() else {}


def subject_columns(subjects):
    """Model and setup columns per subject_id: build fields plus the curated models.yaml entry, if any."""
    curated = load_curated()
    missing = sorted(set(subjects.normalized_name) - set(curated))
    if missing and curated:
        print(f"warning: no curated entry in {MODELS_FILE.name} for {missing}")
    rows = []
    for s in subjects.itertuples():
        model = curated.get(s.normalized_name, {})
        row = {"subject_id": s.subject_id, "provider": s.provider, "model_name": s.display_name,
               "normalized_name": s.normalized_name,
               "reasoning_effort": s.reasoning_effort, "release_date": s.release_date, "harness": s.harness,
               "harness_version": s.harness_version, "access_date": s.access_date}
        row |= {f: model.get(f) for f in MODEL_FIELDS}
        # A model without a thinking mode has no reasoning effort to set.
        if pd.isna(row["reasoning_effort"]) and model.get("thinking") is False:
            row["reasoning_effort"] = "none"
        rows.append(row)
    return pd.DataFrame(rows)


def huggingface_provenance(repository, source_tables):
    """Capture the exact Hugging Face dataset version used by a conversion."""
    revision = HfApi().dataset_info(repository).sha
    return {
        "loaded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
