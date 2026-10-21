"""Download and convert every benchmark in the AIMS Measurement DB.

The repository revision is pinned once per run. Each output row is one released
response, so repeated trials and traces remain attached to the correct attempt.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFolder

from assets import AssetProcessingContext, handler_for, recipe_provenance
from assets.pipeline import add_asset_descriptions, manifest_references
from common import OUT, benchmark_columns, huggingface_provenance, subject_columns, write_observation_batches


REPOSITORY = "aims-foundations/measurement-db"
TABLES = ("benchmarks", "items", "subjects", "responses")
NORMALIZATION_POLICY = {
    "bounded": "linear min-max; reversed when lower_is_better",
    "one_sided_better_at_bound": "1 / (1 + distance_from_finite_bound)",
    "one_sided_better_at_infinity": "distance_from_finite_bound / (1 + distance_from_finite_bound)",
    "single_discrete_value": "1",
    "unsupported": "fully unbounded, unordered, or directionless scales are skipped",
}


class UnsupportedScaleError(ValueError):
    """The released scale cannot be mapped meaningfully to a utility in [0, 1]."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_object(value, field: str) -> dict:
    """Return a JSON object stored either as a dictionary or a JSON string."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"{field} is not a JSON object: {value!r}")


def compact_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _finite_bound(value, name: str):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise UnsupportedScaleError(f"non-finite {name} bound: {value}")
    return value


def normalization_mode(scale: dict) -> tuple[str, float | None, float | None, str]:
    """Validate a scale and describe the deterministic normalization to apply."""
    kind = scale.get("kind")
    direction = scale.get("direction")
    if direction not in {"higher_is_better", "lower_is_better"}:
        raise UnsupportedScaleError(f"scale direction is {direction!r}")

    if kind == "discrete":
        values = [float(value) for value in scale.get("values", [])]
        if not values or not all(math.isfinite(value) for value in values):
            raise UnsupportedScaleError(f"invalid discrete values: {values!r}")
        return "discrete", min(values), max(values), direction

    if kind == "interval":
        minimum = _finite_bound(scale.get("min"), "minimum")
        maximum = _finite_bound(scale.get("max"), "maximum")
        if minimum is None and maximum is None:
            raise UnsupportedScaleError("fully unbounded interval")
        if minimum is not None and maximum is not None and maximum < minimum:
            raise UnsupportedScaleError(f"interval maximum {maximum} is below minimum {minimum}")
        return "interval", minimum, maximum, direction

    raise UnsupportedScaleError(f"unsupported scale kind {kind!r}")


def normalize_scores(raw_score: pd.Series, response_scale: pd.Series) -> pd.Series:
    """Normalize released scores to [0, 1], with 1 always representing the preferred outcome."""
    normalized = pd.Series(float("nan"), index=raw_score.index, dtype="float64")
    graded = raw_score.notna()
    if response_scale[graded].isna().any():
        raise UnsupportedScaleError("a graded response has no concrete response scale")

    for encoded_scale in response_scale[graded].unique():
        scale = json.loads(encoded_scale)
        kind, minimum, maximum, direction = normalization_mode(scale)
        mask = graded & response_scale.eq(encoded_scale)
        values = raw_score.loc[mask].astype(float)

        if minimum == maximum:
            result = pd.Series(1.0, index=values.index)
        elif minimum is not None and maximum is not None:
            result = (values - minimum) / (maximum - minimum)
            if direction == "lower_is_better":
                result = 1 - result
        elif minimum is not None:
            distance = (values - minimum).clip(lower=0)
            result = (1 / (1 + distance) if direction == "lower_is_better"
                      else distance / (1 + distance))
        else:
            distance = (maximum - values).clip(lower=0)
            result = (1 / (1 + distance) if direction == "higher_is_better"
                      else distance / (1 + distance))

        normalized.loc[mask] = result.clip(0, 1)

    return normalized


def item_columns(items: pd.DataFrame, benchmark: str, benchmark_scale: dict) -> pd.DataFrame:
    """Extract generic item fields and resolve mixed scales at item level."""
    criteria = items.grading_criterion.map(lambda value: json_object(value, "grading_criterion"))
    verifiers = items.verifier.map(lambda value: json_object(value, "verifier"))

    if benchmark_scale.get("kind") == "mixed":
        scales = criteria.map(lambda criterion: criterion.get("response_scale"))
        if scales.isna().any():
            missing = items.loc[scales.isna(), "item_id"].head(5).tolist()
            raise UnsupportedScaleError(f"mixed benchmark items without response_scale: {missing}")
    else:
        scales = pd.Series([benchmark_scale] * len(items), index=items.index)

    encoded_scales = scales.map(compact_json)
    for encoded in encoded_scales.unique():
        normalization_mode(json.loads(encoded))

    reference = criteria.map(lambda criterion: criterion.get("reference_answer"))
    rule = criteria.map(lambda criterion: criterion.get("rule"))
    return pd.DataFrame({
        "source_item_id": items.item_id,
        "item_id": benchmark + ":" + items.item_id,
        "question": items.content,
        "ground_truth": reference.where(reference.notna(), rule),
        "eval_type": verifiers.map(lambda verifier: verifier.get("class") or verifier.get("kind")),
        "response_scale": encoded_scales,
    })


def build_output(
    benchmark: str,
    metadata: dict,
    benchmark_table: pd.DataFrame,
    items: pd.DataFrame,
    subjects: pd.DataFrame,
    responses: pd.DataFrame,
    traces: pd.DataFrame | None,
) -> pd.DataFrame:
    """Convert standardized Measurement DB tables to the common observation layout."""
    if len(benchmark_table) != 1 or benchmark_table.iloc[0].benchmark_id != benchmark:
        raise ValueError(f"{benchmark}: invalid benchmarks table")
    if responses.response_id.duplicated().any():
        raise ValueError(f"{benchmark}: duplicate response_id values")

    benchmark_scale = metadata["response_scale"]
    item_data = item_columns(items, benchmark, benchmark_scale)
    subject_data = subject_columns(subjects)

    observation_key = ["subject_id", "item_id", "test_condition", "interactors"]
    response_data = responses.copy()
    response_data["n_runs"] = (
        response_data.groupby(observation_key, dropna=False).response_id.transform("size")
    )
    response_data = response_data.rename(columns={"item_id": "source_item_id", "response": "raw_score"})

    output = response_data.merge(
        item_data,
        on="source_item_id",
        how="left",
        validate="many_to_one",
        indicator="_item_join",
    )
    if not output._item_join.eq("both").all():
        raise ValueError(f"{benchmark}: responses reference missing items")
    output = output.drop(columns="_item_join")

    output = output.merge(
        subject_data,
        on="subject_id",
        how="left",
        validate="many_to_one",
        indicator="_subject_join",
    )
    if not output._subject_join.eq("both").all():
        raise ValueError(f"{benchmark}: responses reference missing subjects")
    output = output.drop(columns="_subject_join")

    if traces is None:
        output["trace"] = None
    else:
        if traces.response_id.duplicated().any():
            raise ValueError(f"{benchmark}: duplicate trace response_id values")
        output = output.merge(
            traces[["response_id", "trace"]],
            on="response_id",
            how="left",
            validate="one_to_one",
        )

    output["score"] = normalize_scores(output.raw_score, output.response_scale)
    output["benchmark"] = benchmark
    output["programming_language"] = None
    output["tools"] = None
    for column, value in benchmark_columns(metadata).items():
        output[column] = value

    if len(output) != len(responses):
        raise ValueError(f"{benchmark}: row count changed during joins")
    return output


def discover_benchmarks(api: HfApi, revision: str) -> list[str]:
    """Return every top-level directory; there is intentionally no allow/deny list."""
    entries = api.list_repo_tree(REPOSITORY, repo_type="dataset", revision=revision, recursive=False)
    return sorted(entry.path for entry in entries if isinstance(entry, RepoFolder))


def source_files(api: HfApi, benchmark: str, revision: str) -> list[str]:
    files = [f"{benchmark}/formatted_tables/{name}.parquet" for name in TABLES]
    files.append(f"{benchmark}/metadata.yaml")
    trace = f"{benchmark}/formatted_tables/traces.parquet"
    if api.file_exists(REPOSITORY, trace, repo_type="dataset", revision=revision):
        files.append(trace)
    assets = f"{benchmark}/formatted_tables/assets.parquet"
    if api.file_exists(REPOSITORY, assets, repo_type="dataset", revision=revision):
        files.append(assets)
    return files


def download_files(filenames: list[str], revision: str) -> dict[str, Path]:
    return {
        filename: Path(hf_hub_download(
            repo_id=REPOSITORY,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        ))
        for filename in filenames
    }


def process_benchmark(
    benchmark: str,
    api: HfApi,
    revision: str,
    loaded_at_utc: str,
    output_dir: Path | None = None,
    asset_context: AssetProcessingContext | None = None,
) -> dict:
    filenames = source_files(api, benchmark, revision)
    asset_name = f"{benchmark}/formatted_tables/assets.parquet"
    # Inspect the much smaller items table before downloading a potentially
    # enormous assets.parquet. This also makes an unsupported MIME fail early.
    initial_filenames = [name for name in filenames if name != asset_name]
    paths = download_files(initial_filenames, revision)
    prefix = f"{benchmark}/formatted_tables/"
    table = lambda name: pd.read_parquet(paths[f"{prefix}{name}.parquet"])
    metadata = yaml.safe_load(paths[f"{benchmark}/metadata.yaml"].read_text(encoding="utf-8"))["benchmark"]
    trace_name = f"{prefix}traces.parquet"
    benchmark_table = table("benchmarks")
    if len(benchmark_table) != 1 or benchmark_table.iloc[0].benchmark_id != benchmark:
        raise ValueError(f"{benchmark}: invalid benchmarks table")
    items = table("items")
    asset_stats = None
    owned_asset_context = None
    if asset_name in filenames:
        references = manifest_references(items)
        for media_type in references.media_type.unique():
            handler_for(media_type)
        paths.update(download_files([asset_name], revision))
        if asset_context is None:
            owned_asset_context = AssetProcessingContext(OUT / "asset_descriptions.sqlite3")
            asset_context = owned_asset_context
        try:
            items, asset_stats = add_asset_descriptions(items, paths[asset_name], asset_context)
        finally:
            if owned_asset_context is not None:
                owned_asset_context.close()
    item_data = item_columns(items, benchmark, metadata["response_scale"])
    subject_data = subject_columns(table("subjects"))

    connection = duckdb.connect()
    connection.register("item_data", item_data)
    connection.register("subject_data", subject_data)
    connection.from_parquet(str(paths[f"{prefix}responses.parquet"])).create_view("responses")
    trace_select = "NULL::VARCHAR AS trace"
    trace_join = ""
    if trace_name in paths:
        connection.from_parquet(str(paths[trace_name])).project("response_id, trace").create_view("traces")
        trace_select = "t.trace"
        trace_join = "LEFT JOIN traces t ON r.response_id = t.response_id"

    query = f"""
        SELECT
            r.response_id,
            i.item_id,
            r.subject_id,
            r.trial,
            r.test_condition,
            r.interactors,
            r.response AS raw_score,
            count(*) OVER (
                PARTITION BY r.subject_id, r.item_id, r.test_condition, r.interactors
            ) AS n_runs,
            {trace_select},
            i.question,
            i.ground_truth,
            i.eval_type,
            i.response_scale,
            s.model_name,
            s.normalized_name,
            s.thinking,
            s.reasoning_effort,
            s.harness,
            s.harness_version,
            s.access_date
        FROM responses r
        JOIN item_data i ON r.item_id = i.source_item_id
        JOIN subject_data s ON r.subject_id = s.subject_id
        {trace_join}
    """
    source_rows = connection.execute("SELECT count(*) FROM responses").fetchone()[0]
    reader = connection.execute(query).fetch_record_batch(rows_per_batch=50_000)
    constants = benchmark_columns(metadata)
    stats = {"rows": 0, "graded_rows": 0, "trace_rows": 0}

    def output_batches():
        for batch in reader:
            output = batch.to_pandas()
            output["score"] = normalize_scores(output.raw_score, output.response_scale)
            output["benchmark"] = benchmark
            output["programming_language"] = None
            output["tools"] = None
            for column, value in constants.items():
                output[column] = value
            stats["rows"] += len(output)
            stats["graded_rows"] += int(output.score.notna().sum())
            stats["trace_rows"] += int(output.trace.notna().sum())
            yield output

    provenance = huggingface_provenance(
        REPOSITORY,
        filenames,
        revision=revision,
        loaded_at_utc=loaded_at_utc,
    )
    provenance["normalization"] = NORMALIZATION_POLICY
    if asset_stats is not None:
        provenance["asset_processing"] = {
            **asset_stats,
            **asset_context.provenance(),
            "recipes": recipe_provenance(asset_stats["media_types"], asset_context),
            "descriptions_replace_assets": True,
            "manifest_order": "ordinal",
        }
    try:
        path = write_observation_batches(output_batches(), benchmark, provenance, output_dir=output_dir)
    finally:
        connection.close()
    if stats["rows"] != source_rows:
        raise ValueError(f"{benchmark}: wrote {stats['rows']} of {source_rows} response rows")
    return {
        "benchmark": benchmark,
        **stats,
        "asset_processing": asset_stats,
        "output_file": path.name,
    }


def write_collection_manifest(revision: str, loaded_at_utc: str, discovered: list[str], outputs: list[dict], skipped: list[dict]):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "loaded_at_utc": loaded_at_utc,
        "source_repository": REPOSITORY,
        "source_revision": revision,
        "discovery": "every top-level repository directory",
        "normalization": NORMALIZATION_POLICY,
        "benchmarks_discovered": len(discovered),
        "benchmarks_written": len(outputs),
        "benchmarks_skipped": len(skipped),
        "outputs": outputs,
        "skipped": skipped,
    }
    path = OUT / "measurement_db.manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    api = HfApi()
    revision = api.dataset_info(REPOSITORY).sha
    loaded_at_utc = utc_now()
    benchmarks = discover_benchmarks(api, revision)
    print(f"Found {len(benchmarks)} benchmark directories at {revision}", flush=True)

    outputs = []
    skipped = []
    asset_context = AssetProcessingContext(OUT / "asset_descriptions.sqlite3")
    try:
        for position, benchmark in enumerate(benchmarks, 1):
            print(f"[{position}/{len(benchmarks)}] {benchmark}", flush=True)
            try:
                result = process_benchmark(
                    benchmark,
                    api,
                    revision,
                    loaded_at_utc,
                    asset_context=asset_context,
                )
            except UnsupportedScaleError as error:
                skipped.append({"benchmark": benchmark, "reason": str(error)})
                print(f"  skipped: {error}", flush=True)
                continue
            outputs.append(result)
            print(
                f"  wrote {result['rows']:,} rows ({result['trace_rows']:,} with traces) to {result['output_file']}",
                flush=True,
            )
    finally:
        asset_context.close()

    manifest = write_collection_manifest(revision, loaded_at_utc, benchmarks, outputs, skipped)
    print(f"Finished: {len(outputs)} written, {len(skipped)} skipped -> {manifest}")


if __name__ == "__main__":
    main()
