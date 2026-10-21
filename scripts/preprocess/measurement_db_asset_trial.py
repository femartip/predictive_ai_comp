"""Run a fixed, small end-to-end trial of every Measurement DB asset handler.

The 20 source items below cover all 26 currently supported MIME subtypes. Only
the explicitly listed asset references are retained from each source manifest;
this prevents asset-heavy items from turning the validation run into a full
benchmark download. One deterministic released response is written per item.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from assets import AssetProcessingContext, SUPPORTED_MEDIA_TYPES, recipe_provenance
from assets.pipeline import add_asset_descriptions, parse_manifest
from common import OUT, huggingface_provenance, load_metadata, write_observations
from measurement_db import REPOSITORY, build_output, utc_now


TRIAL_OUT = OUT / "asset_trial"
REMOTE_ATTEMPTS = 5


def _asset(asset_id: str, media_type: str, ordinal: int, path: str, role: str) -> dict:
    return {
        "asset_id": asset_id,
        "media_type": media_type,
        "ordinal": ordinal,
        "path": path,
        "role": role,
    }


# Deliberately fixed, not sampled at runtime. These IDs were selected from
# Measurement DB revision 54074453da7b9911be0f48e4c560be547bc84d21.
TRIAL_ITEMS = [
    {
        "benchmark": "ehrflowbench",
        "item_id": "0418bf42feca50c6",
        "assets": [
            _asset("2e7f67e1212714e4f4165568979634c67f00e14f8e6d9eea1775457172df5ea3", "application/json", 2, "benchmarks/ehrflowbench/processed/mimic_iv_demo/split_metadata.json", "input"),
            _asset("3c56616de8decc8c10ff01235c6538137f06657037f3d139444483f5376fa550", "application/vnd.apache.parquet", 1, "benchmarks/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_formatted_ehr.parquet", "input"),
            _asset("b5bd01885c0f2546b1b6437d553015ed3f88c16719a5fa3934727d6228c1d544", "text/markdown", 3, "benchmarks/ehrflowbench/processed/mimic_iv_demo/mimic_iv_demo_value_reference.md", "input"),
        ],
    },
    {
        "benchmark": "intercode",
        "item_id": "2cab2582201a188e",
        "assets": [
            _asset("20f9eee38396055a939019bdfd65a3e0730fa66539d9d4a63eb5daf834baa1f8", "application/octet-stream", 2, "data/ctf/task_assets/6/flag.txt.en", "input"),
            _asset("ebc5968dcfdb73479e52384b727439d596768bdda1b85eee278703688d5353f5", "text/plain", 3, "data/ctf/task_assets/6/pw.txt", "input"),
            _asset("bc1ac8e1ae5bb52885440cb853f5a69e4dd903759936f84c6d1d147cc0d0c295", "text/x-python", 1, "data/ctf/task_assets/6/ende.py", "input"),
        ],
    },
    {
        "benchmark": "rclicks",
        "item_id": "001b3d54df2d453d",
        "assets": [
            _asset("49513a577b45e6419ab884736bba258c7d2d61c65301069dfc0b550568f88721", "image/bmp", 2, "images/49513a577b45e6419ab884736bba258c7d2d61c65301069dfc0b550568f88721.bmp", "input"),
            _asset("5f9d7dbb32bc871450ce1384e5fbca5df7cfd5edbafb4464aed140194d41329e", "image/jpeg", 1, "images/5f9d7dbb32bc871450ce1384e5fbca5df7cfd5edbafb4464aed140194d41329e.jpg", "input"),
            _asset("0cb2906fcea4abca9d2283177519c494ac1d0c9758b614bb8ba4fe7342ae2294", "image/png", 3, "images/0cb2906fcea4abca9d2283177519c494ac1d0c9758b614bb8ba4fe7342ae2294.png", "input"),
        ],
    },
    {
        "benchmark": "dave",
        "item_id": "000f0a9ee6909281",
        "assets": [
            _asset("0051206b37786373d28b7d87577861743fcd0375f113d021be4fe8d9d804dbf9", "audio/mpeg", 2, "audio.mp3", "input_audio"),
            _asset("e7209dc566f4041cdc25cb2aca11fbfe97af6c442c5df8375493c3e17e3c09bd", "video/mp4", 1, "video.mp4", "input_video"),
        ],
    },
    {
        "benchmark": "stanford_orb",
        "item_id": "8424097eebacd5fb",
        "assets": [
            _asset("0503bb4cc35a1f7f7c9b7a3cea6e95343ba02754db41250531dfc53bd20e6cf9", "image/x-exr", 60, "blender_HDR/pitcher_scene001/train/0041.exr", "capture_observation"),
            _asset("2332fafddb0478f2d192b32d164c9ca9b6c86ffdb24a0d9dd1b6448e428f4b47", "model/obj", 273, "ground_truth/pitcher_scene001/mesh_blender/mesh.obj", "reference"),
        ],
    },
    {
        "benchmark": "brace",
        "item_id": "000b000b7017c1cf",
        "assets": [
            _asset("1b28ef10ba9c5742d4a2d72f73c549a08d1a6398483e451f5f847c625cce315f", "audio/wav", 1, "benchmark_hallucination/Clotho/audio/St Pauls Bells 1.wav", "input"),
        ],
    },
    {
        "benchmark": "hle",
        "item_id": "4dce1cca9181b766",
        "assets": [
            _asset("9b3b3356970befd6897f0548a97d3f3b2a2e0d8f9847a04f8d627eddc7dd5ced", "image/gif", 1, "images/9b3b3356970befd6897f0548a97d3f3b2a2e0d8f9847a04f8d627eddc7dd5ced", "input"),
        ],
    },
    {
        "benchmark": "hle",
        "item_id": "633a9b8f871e70ab",
        "assets": [
            _asset("a4c8412f1bc0ff66f8732fa1abd27db0e381c2230155b808ccd6218f9b381a97", "image/webp", 1, "images/a4c8412f1bc0ff66f8732fa1abd27db0e381c2230155b808ccd6218f9b381a97", "input"),
        ],
    },
    {
        "benchmark": "intercode",
        "item_id": "0025a0a69e908baf",
        "assets": [
            _asset("61f6e8000cb5bb696f123fa55e111acdcd8da7eca64f452a36fd1a6e5fb1781a", "application/sql", 1, "data/sql/spider/ic_spider_dbs.sql", "input"),
        ],
    },
    {
        "benchmark": "intercode",
        "item_id": "0235d9c735696eb7",
        "assets": [
            _asset("98cbf221f2c190c6058ed215718659a7a9f655b36264345c21c1cbec2c1239d1", "application/pdf", 1, "data/ctf/task_assets/67/Financial_Report_for_ABC_Labs.pdf", "input"),
        ],
    },
    {
        "benchmark": "intercode",
        "item_id": "81e3d392da3d7153",
        "assets": [
            _asset("29bb706180b8bdfc452a1400d9d2d0dda4cd3faa105ce090e6d8bf651fc7dac2", "application/gzip", 1, "data/ctf/task_assets/65/dds1-alpine.flag.img.gz", "input"),
        ],
    },
    {
        "benchmark": "intercode",
        "item_id": "f7a454a71627555e",
        "assets": [
            _asset("6630839c01313ef902f3bd264ee637c7cec535d6ca8080838eda182fb0077822", "application/zip", 1, "data/ctf/task_assets/37/files.zip", "input"),
        ],
    },
    {
        "benchmark": "legal_rag_bench",
        "item_id": "0027ab2bd2d1135e",
        "assets": [
            _asset("3a3565bc5429f6cead90548e81f87352b449927e4be1cdd804c28d766bb9c246", "application/x-ndjson", 1, "corpus.jsonl", "retrieval_corpus"),
        ],
    },
    {
        "benchmark": "nis3d",
        "item_id": "081f66d465bda34b",
        "assets": [
            _asset("d5d0a158f3031e28205e5b3f5e630d2cdb93fa2c6bb35d230ad46429a30a8e8c", "image/tiff", 1, "Zebrafish_1/data.tif", "microscopy_volume"),
        ],
    },
    {
        "benchmark": "qatch",
        "item_id": "0005948352d98282",
        "assets": [
            _asset("1b2414f44c04f84bbe30b4dee2eac4c0f24eb39829d694a40e779e8eb069cd01", "application/vnd.sqlite3", 1, "database.sqlite", "input"),
        ],
    },
    {
        "benchmark": "intercode",
        "item_id": "00ec264f47409bea",
        "assets": [
            _asset("c5fb550aa1578fe2454e8ab06221165df90311231cb71d3d9b0ce036a8235274", "text/x-shellscript", 1, "docker/bash_scripts/setup_nl2b_fs_4.sh", "input"),
        ],
    },
    {
        "benchmark": "dataclawbench",
        "item_id": "0053765b0bed4dae",
        "assets": [
            _asset("016e39c51abb7f940506d2855bbc0419a6ba50dbbe14e62684285d6c7985d569", "text/csv", 4, "database/enterprise/company_operation_status_detail.csv", "input"),
        ],
    },
    {
        "benchmark": "stanford_orb",
        "item_id": "89a81edce6e11666",
        "assets": [
            _asset("09bddd2b816fa282615ee4f7f584234c5f8430c14bc31a2064ceecf07e4a1567", "application/x-npy", 274, "ground_truth/pitcher_scene001/z_depth/0061.npy", "reference"),
        ],
    },
    {
        "benchmark": "brace",
        "item_id": "000fd2e371fbccbe",
        "assets": [
            _asset("713d96ea4e236b6fcbc2bbfb5dcf28321fb1b3ff8259b70bc68537ac1df398be", "audio/wav", 1, "benchmark/Clotho/audio/fs_brokenMixer302-2.wav", "input"),
        ],
    },
    {
        "benchmark": "brace",
        "item_id": "0010f48f89083014",
        "assets": [
            _asset("f065db4ffeb2363314981ca03260721ed225029a6617a77ece7d25be6a4f374d", "audio/wav", 1, "benchmark_hallucination/AudioCaps/audio/107755.wav", "input"),
        ],
    },
]


def validate_fixed_selection() -> None:
    keys = [(sample["benchmark"], sample["item_id"]) for sample in TRIAL_ITEMS]
    if len(TRIAL_ITEMS) != 20 or len(set(keys)) != 20:
        raise ValueError("TRIAL_ITEMS must contain exactly 20 distinct benchmark/item pairs")
    media_types = {
        asset["media_type"]
        for sample in TRIAL_ITEMS
        for asset in sample["assets"]
    }
    if media_types != SUPPORTED_MEDIA_TYPES:
        missing = sorted(SUPPORTED_MEDIA_TYPES - media_types)
        extra = sorted(media_types - SUPPORTED_MEDIA_TYPES)
        raise ValueError(f"fixed trial MIME coverage mismatch; missing={missing}, extra={extra}")
    asset_ids = [
        asset["asset_id"]
        for sample in TRIAL_ITEMS
        for asset in sample["assets"]
    ]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("the fixed trial repeats an asset_id")


def validate_multimodal_runtime() -> None:
    """Fail before remote reads if native/model dependencies cannot be imported."""
    try:
        import av
        import torch
        import torchvision
        from transformers import (
            Qwen2AudioForConditionalGeneration,
            Qwen2_5OmniThinkerForConditionalGeneration,
            Qwen2_5_VLForConditionalGeneration,
        )
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "multimodal dependencies could not be loaded; run `poetry install` and verify "
            "the pinned torch/torchvision versions from poetry.lock"
        ) from error
    # Keep lazy imports alive through this check without loading model weights.
    assert Qwen2AudioForConditionalGeneration
    assert Qwen2_5OmniThinkerForConditionalGeneration
    assert Qwen2_5_VLForConditionalGeneration
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(
        f"Runtime: torch {torch.__version__}, torchvision {torchvision.__version__}, "
        f"PyAV {av.__version__}, device={device}",
        flush=True,
    )


def remote_parquet(
    revision: str,
    filename: str,
    columns: list[str],
    filters: list[tuple] | None = None,
) -> pa.Table:
    """Read only required Parquet columns/row groups, retrying transient HTTP failures."""
    uri = f"hf://datasets/{REPOSITORY}@{revision}/{filename}"
    for attempt in range(1, REMOTE_ATTEMPTS + 1):
        try:
            with HfFileSystem().open(uri, "rb") as source:
                return pq.read_table(
                    source,
                    columns=columns,
                    filters=filters,
                    pre_buffer=False,
                )
        except Exception as error:
            if attempt == REMOTE_ATTEMPTS:
                raise RuntimeError(f"failed to read {filename} after {attempt} attempts") from error
            delay = 2 ** (attempt - 1)
            print(f"  remote read failed ({type(error).__name__}); retrying in {delay}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _require_rows(frame: pd.DataFrame, column: str, expected: set[str], label: str) -> None:
    actual = set(frame[column].astype(str))
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f"{label} row mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}")


def _validate_and_filter_manifests(items: pd.DataFrame, samples: list[dict]) -> pd.DataFrame:
    selected_by_item = {sample["item_id"]: sample["assets"] for sample in samples}
    output = items.copy()
    for index, row in output.iterrows():
        released = {
            (str(asset["asset_id"]), str(asset["media_type"])): asset
            for asset in parse_manifest(row.asset_manifest, str(row.item_id))
        }
        selected = selected_by_item[str(row.item_id)]
        for asset in selected:
            key = (asset["asset_id"], asset["media_type"])
            if key not in released:
                raise ValueError(f"{row.item_id}: fixed asset {key} is no longer in the source manifest")
        output.at[index, "asset_manifest"] = json.dumps(
            sorted(selected, key=lambda asset: asset["ordinal"]),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return output


def _one_response_per_item(responses: pd.DataFrame, item_ids: list[str]) -> tuple[pd.DataFrame, dict]:
    _require_rows(responses, "item_id", set(item_ids), "responses")
    observation_key = ["subject_id", "item_id", "test_condition", "interactors"]
    responses = responses.copy()
    responses["source_n_runs"] = responses.groupby(
        observation_key,
        dropna=False,
    ).response_id.transform("size")
    selected = (
        responses.sort_values(["item_id", "trial", "subject_id", "response_id"])
        .groupby("item_id", sort=False, as_index=False)
        .head(1)
        .copy()
    )
    if len(selected) != len(item_ids):
        raise ValueError("could not choose exactly one released response per fixed item")
    n_runs = dict(zip(selected.response_id, selected.source_n_runs))
    return selected.drop(columns="source_n_runs"), n_runs


def process_trial_benchmark(
    benchmark: str,
    samples: list[dict],
    api: HfApi,
    revision: str,
    loaded_at_utc: str,
    asset_context: AssetProcessingContext,
) -> dict:
    prefix = f"{benchmark}/formatted_tables"
    item_ids = [sample["item_id"] for sample in samples]
    selected_assets = [asset for sample in samples for asset in sample["assets"]]
    asset_ids = [asset["asset_id"] for asset in selected_assets]
    item_filter = [("item_id", "in", item_ids)]

    items = remote_parquet(
        revision,
        f"{prefix}/items.parquet",
        ["item_id", "content", "grading_criterion", "verifier", "asset_manifest"],
        item_filter,
    ).to_pandas()
    _require_rows(items, "item_id", set(item_ids), f"{benchmark} items")
    items = _validate_and_filter_manifests(items, samples)

    assets = remote_parquet(
        revision,
        f"{prefix}/assets.parquet",
        ["asset_id", "data"],
        [("asset_id", "in", asset_ids)],
    )
    asset_frame = assets.select(["asset_id"]).to_pandas()
    _require_rows(asset_frame, "asset_id", set(asset_ids), f"{benchmark} assets")

    with tempfile.TemporaryDirectory(prefix="measurement-db-asset-trial-") as directory:
        assets_path = Path(directory) / "assets.parquet"
        pq.write_table(assets, assets_path)
        items, asset_stats = add_asset_descriptions(items, assets_path, asset_context)

    response_columns = [
        "response_id", "subject_id", "item_id", "trial",
        "test_condition", "interactors", "response",
    ]
    responses = remote_parquet(
        revision,
        f"{prefix}/responses.parquet",
        response_columns,
        item_filter,
    ).to_pandas()
    selected_responses, source_n_runs = _one_response_per_item(responses, item_ids)

    subject_ids = selected_responses.subject_id.astype(str).unique().tolist()
    subjects = remote_parquet(
        revision,
        f"{prefix}/subjects.parquet",
        [
            "subject_id", "display_name", "normalized_name", "reasoning_effort",
            "harness", "harness_version", "access_date",
        ],
        [("subject_id", "in", subject_ids)],
    ).to_pandas()
    _require_rows(subjects, "subject_id", set(subject_ids), f"{benchmark} subjects")

    traces = None
    trace_name = f"{prefix}/traces.parquet"
    source_tables = [
        f"{benchmark}/metadata.yaml",
        f"{prefix}/benchmarks.parquet",
        f"{prefix}/items.parquet",
        f"{prefix}/subjects.parquet",
        f"{prefix}/responses.parquet",
        f"{prefix}/assets.parquet",
    ]
    if api.file_exists(REPOSITORY, trace_name, repo_type="dataset", revision=revision):
        response_ids = selected_responses.response_id.astype(str).tolist()
        traces = remote_parquet(
            revision,
            trace_name,
            ["response_id", "trace"],
            [("response_id", "in", response_ids)],
        ).to_pandas()
        source_tables.append(trace_name)

    benchmark_table = remote_parquet(
        revision,
        f"{prefix}/benchmarks.parquet",
        ["benchmark_id"],
    ).to_pandas()
    metadata = load_metadata(benchmark, REPOSITORY, revision)
    output = build_output(
        benchmark,
        metadata,
        benchmark_table,
        items,
        subjects,
        selected_responses,
        traces,
    )
    output["n_runs"] = output.response_id.map(source_n_runs).astype("int64")
    item_order = {item_id: position for position, item_id in enumerate(item_ids)}
    output["_trial_order"] = output.item_id.str.removeprefix(f"{benchmark}:").map(item_order)
    output = output.sort_values("_trial_order").drop(columns="_trial_order").reset_index(drop=True)

    provenance = huggingface_provenance(
        REPOSITORY,
        source_tables,
        revision=revision,
        loaded_at_utc=loaded_at_utc,
    )
    provenance["asset_trial"] = {
        "selection": "20 source items fixed in scripts/preprocess/measurement_db_asset_trial.py",
        "one_response_per_item": "lowest (trial, subject_id, response_id)",
        "source_manifests_filtered": True,
        "selected_items": [
            {"item_id": sample["item_id"], "assets": sample["assets"]}
            for sample in samples
        ],
        "remote_parquet_row_filtering": True,
    }
    provenance["asset_processing"] = {
        **asset_stats,
        **asset_context.provenance(),
        "recipes": recipe_provenance(asset_stats["media_types"], asset_context),
        "descriptions_replace_assets": True,
        "manifest_order": "ordinal",
    }
    path = write_observations(output, benchmark, provenance, output_dir=TRIAL_OUT)
    return {
        "benchmark": benchmark,
        "items": len(output),
        "assets": len(selected_assets),
        "media_types": sorted(asset_stats["media_types"]),
        "output_file": path.name,
    }


def main() -> None:
    validate_fixed_selection()
    validate_multimodal_runtime()
    api = HfApi()
    revision = api.dataset_info(REPOSITORY).sha
    loaded_at_utc = utc_now()
    grouped = defaultdict(list)
    for sample in TRIAL_ITEMS:
        grouped[sample["benchmark"]].append(sample)

    print(
        f"Asset trial: {len(TRIAL_ITEMS)} fixed items, {len(SUPPORTED_MEDIA_TYPES)} MIME types, "
        f"{len(grouped)} benchmarks at {revision}",
        flush=True,
    )
    outputs = []
    asset_context = AssetProcessingContext(OUT / "asset_descriptions.sqlite3")
    try:
        for position, (benchmark, samples) in enumerate(sorted(grouped.items()), 1):
            print(f"[{position}/{len(grouped)}] {benchmark}: {len(samples)} item(s)", flush=True)
            result = process_trial_benchmark(
                benchmark,
                samples,
                api,
                revision,
                loaded_at_utc,
                asset_context,
            )
            outputs.append(result)
            print(
                f"  wrote {result['items']} rows using {result['assets']} selected assets",
                flush=True,
            )
    finally:
        asset_context.close()

    manifest = {
        "loaded_at_utc": loaded_at_utc,
        "source_repository": REPOSITORY,
        "source_revision": revision,
        "selection": "fixed 20-item asset trial",
        "items": len(TRIAL_ITEMS),
        "covered_media_types": sorted(SUPPORTED_MEDIA_TYPES),
        "outputs": outputs,
    }
    TRIAL_OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = TRIAL_OUT / "asset_trial.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Finished -> {TRIAL_OUT} ({manifest_path.name})", flush=True)


if __name__ == "__main__":
    main()
