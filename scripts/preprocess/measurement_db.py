"""Download and convert selected benchmarks in the AIMS Measurement DB.

The local category catalog pins the repository revision. Each output row is one released
response, so repeated trials remain distinct. Model traces are not downloaded or exported.
"""

import argparse
import fcntl
import json
import logging
import math
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from shlex import quote
from uuid import uuid4

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi, hf_hub_download

from assets import AssetProcessingContext, handler_for
from assets.pipeline import add_asset_descriptions, manifest_references
from common import OUT, benchmark_columns, subject_columns, write_observation_batches
from measurement_db_categories import CATEGORY_FILE, CATEGORIES, REPOSITORY, build_catalog, load_catalog, select_benchmarks


TABLES = ("benchmarks", "items", "subjects", "responses")
NORMALIZATION_METADATA = {
    "per_item_field": "normalization_policy",
    "description": "Each observation records only the name of its applied normalization policy.",
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
        if minimum is not None and maximum is not None and maximum < minimum:
            raise UnsupportedScaleError(f"interval maximum {maximum} is below minimum {minimum}")
        return "interval", minimum, maximum, direction

    raise UnsupportedScaleError(f"unsupported scale kind {kind!r}")



def applied_normalization_policy(scale: dict) -> str:
    """Return only the name of the transform applied to this item's scale."""
    _, minimum, maximum, direction = normalization_mode(scale)
    if minimum is None and maximum is None:
        return "logistic"
    if minimum == maximum:
        return "constant"
    if minimum is not None and maximum is not None:
        return "linear_min_max"
    better_at_bound = ((minimum is not None and direction == "lower_is_better")
                       or (maximum is not None and direction == "higher_is_better"))
    return "reciprocal_distance" if better_at_bound else "distance_ratio"


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

        if minimum is None and maximum is None:
            signed = values if direction == "higher_is_better" else -values
            # Evaluate sigmoid without exponentiating a positive number, so
            # extreme scores cannot overflow. Fixed parameters avoid dependence
            # on the response batch or the set of subjects being compared.
            exp_negative_abs = (-signed.abs()).map(math.exp)
            result = (1 / (1 + exp_negative_abs)).where(
                signed >= 0, exp_negative_abs / (1 + exp_negative_abs),
            )
        elif minimum == maximum:
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

    output["score"] = normalize_scores(output.raw_score, output.response_scale)
    policies = {
        encoded: applied_normalization_policy(json.loads(encoded))
        for encoded in output.response_scale.dropna().unique()
    }
    output["normalization_policy"] = output.response_scale.map(policies)
    output["benchmark"] = benchmark
    output["programming_language"] = None
    output["tools"] = None
    for column, value in benchmark_columns(metadata).items():
        output[column] = value

    if len(output) != len(responses):
        raise ValueError(f"{benchmark}: row count changed during joins")
    return output


def source_files(api: HfApi, benchmark: str, revision: str) -> list[str]:
    files = [f"{benchmark}/formatted_tables/{name}.parquet" for name in TABLES]
    files.append(f"{benchmark}/metadata.yaml")
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
    batch_size: int = 256,
    logger: logging.Logger | None = None,
    memory_limit: str = '4GB',
    threads: int = 4,
) -> dict:
    if batch_size < 1 or threads < 1:
        raise ValueError("batch_size and threads must be positive")
    logger = logger or logging.getLogger(__name__)
    logger.info("%s: reading source tables", benchmark)
    filenames = source_files(api, benchmark, revision)
    asset_name = f"{benchmark}/formatted_tables/assets.parquet"
    # Inspect the much smaller items table before downloading a potentially
    # enormous assets.parquet. This also makes an unsupported MIME fail early.
    initial_filenames = [name for name in filenames if name != asset_name]
    paths = download_files(initial_filenames, revision)
    prefix = f"{benchmark}/formatted_tables/"
    table = lambda name: pd.read_parquet(paths[f"{prefix}{name}.parquet"])
    metadata = yaml.safe_load(paths[f"{benchmark}/metadata.yaml"].read_text(encoding="utf-8"))["benchmark"]
    benchmark_table = table("benchmarks")
    if len(benchmark_table) != 1 or benchmark_table.iloc[0].benchmark_id != benchmark:
        raise ValueError(f"{benchmark}: invalid benchmarks table")
    items_path = paths[f"{prefix}items.parquet"]
    metadata_columns = ['item_id', 'grading_criterion', 'verifier']
    if 'asset_manifest' in pq.read_schema(items_path).names:
        metadata_columns.append('asset_manifest')
    # Large question columns stay in Parquet unless asset expansion needs them.
    items = pd.read_parquet(items_path, columns=metadata_columns)
    owned_asset_context = None
    references = manifest_references(items)
    stream_questions = references.empty
    if stream_questions:
        filenames = [name for name in filenames if name != asset_name]
        items['content'] = None
        logger.info("%s: question text will stream directly from Parquet", benchmark)
    else:
        content = pd.read_parquet(items_path, columns=['item_id', 'content'])
        items = items.merge(content, on='item_id', validate='one_to_one')
        if asset_name not in filenames:
            raise ValueError(f"{benchmark}: item manifests reference assets but assets.parquet is missing")
        for media_type in references.media_type.unique():
            handler_for(media_type)
        paths.update(download_files([asset_name], revision))
        if asset_context is None:
            owned_asset_context = AssetProcessingContext(
                OUT / "asset_descriptions.sqlite3", artifact_root=(output_dir or OUT) / "assets",
            )
            asset_context = owned_asset_context
        try:
            items, _ = add_asset_descriptions(items, paths[asset_name], asset_context)
        finally:
            if owned_asset_context is not None:
                owned_asset_context.close()
    logger.info("%s: preparing joins and score normalization", benchmark)
    item_data = item_columns(items, benchmark, metadata["response_scale"])
    subject_data = subject_columns(table("subjects"))

    logger.info("%s: DuckDB memory limit %s, threads %s", benchmark, memory_limit, threads)
    connection = duckdb.connect(config={
        'memory_limit': memory_limit,
        'threads': threads,
        'preserve_insertion_order': False,
    })
    # Keep large question text on the probe side; hash tables contain narrow metadata.
    connection.execute("SET disabled_optimizers = 'join_order,build_side_probe_side'")
    connection.register("item_data", item_data)
    connection.register("subject_data", subject_data)
    if stream_questions:
        connection.from_parquet(str(items_path)).project('item_id, content').create_view('item_questions')
        question_select = 'q.content'
        item_source = 'item_questions q JOIN item_data i ON q.item_id = i.source_item_id'
    else:
        question_select = 'i.question'
        item_source = 'item_data i'
    connection.from_parquet(str(paths[f"{prefix}responses.parquet"])).create_view("responses")
    query = f"""
        WITH response_runs AS MATERIALIZED (
            SELECT *, count(*) OVER (
                PARTITION BY subject_id, item_id, test_condition, interactors
            ) AS n_runs
            FROM responses
        )
        SELECT
            r.response_id,
            i.item_id,
            r.subject_id,
            r.trial,
            r.test_condition,
            r.interactors,
            r.response AS raw_score,
            r.n_runs,
            {question_select} AS question,
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
        FROM {item_source}
        JOIN response_runs r ON r.item_id = i.source_item_id
        JOIN subject_data s ON r.subject_id = s.subject_id
    """
    source_rows = connection.execute("SELECT count(*) FROM responses").fetchone()[0]
    logger.info("%s: streaming %s responses in batches of %s", benchmark, f"{source_rows:,}", batch_size)
    reader = connection.execute(query).to_arrow_reader(batch_size=batch_size)
    constants = benchmark_columns(metadata)
    stats = {"rows": 0, "graded_rows": 0}

    def output_batches():
        last_progress = 0.0
        for batch in reader:
            output = batch.to_pandas()
            output["score"] = normalize_scores(output.raw_score, output.response_scale)
            policies = {
                encoded: applied_normalization_policy(json.loads(encoded))
                for encoded in output.response_scale.dropna().unique()
            }
            output["normalization_policy"] = output.response_scale.map(policies)
            output["benchmark"] = benchmark
            output["programming_language"] = None
            output["tools"] = None
            for column, value in constants.items():
                output[column] = value
            stats["rows"] += len(output)
            stats["graded_rows"] += int(output.score.notna().sum())
            now = time.monotonic()
            if now - last_progress >= 5:
                logger.info("%s: decoded %s/%s response rows", benchmark, f"{stats['rows']:,}", f"{source_rows:,}")
                last_progress = now
            yield output

    try:
        path = write_observation_batches(output_batches(), benchmark, output_dir=output_dir)
    finally:
        connection.close()
    if stats["rows"] != source_rows:
        raise ValueError(f"{benchmark}: wrote {stats['rows']} of {source_rows} response rows")
    return {
        "benchmark": benchmark,
        **stats,
        "output_file": path.name,
    }


def create_run_directory(output_root: Path, categories: list[str], started_at: str) -> Path:
    """Allocate an independent folder; repeated or concurrent runs never overwrite."""
    selected = sorted(set(categories))
    label = 'all' if set(selected) == set(CATEGORIES) else '+'.join(selected)
    timestamp = datetime.fromisoformat(started_at.replace('Z', '+00:00')).astimezone(timezone.utc)
    run_id = timestamp.strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    directory = output_root / label / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_collection_manifest(
    revision: str, loaded_at_utc: str, discovered: list[str], outputs: list[dict], skipped: list[dict],
    *, selection: dict | None = None, output_dir: Path | None = None,
    status: str = 'complete', error: dict | None = None,
    current_benchmark: str | None = None,
):
    destination = output_dir if output_dir is not None else OUT
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": destination.name,
        "status": status,
        "current_benchmark": current_benchmark,
        "loaded_at_utc": loaded_at_utc,
        "finished_at_utc": None if status == 'running' else utc_now(),
        "source_repository": REPOSITORY,
        "source_revision": revision,
        "selection": {
            key: value for key, value in (selection or {}).items()
            if key in {'requested_categories', 'match', 'benchmarks_selected', 'benchmarks_excluded', 'catalog_snapshot', 'excluded_benchmarks'}
        },
        "benchmarks_discovered": len(discovered),
        "benchmarks_written": len(outputs),
        "benchmarks_skipped": len(skipped),
        "outputs": [
            {key: entry[key] for key in ('benchmark', 'output_file', 'rows', 'categories') if key in entry}
            for entry in outputs
        ],
        "skipped": skipped,
    }
    if error is not None:
        manifest['error'] = error
    path = destination / "measurement_db.manifest.json"
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


@contextmanager
def run_lock(run_dir: Path):
    """Prevent two processes from updating the same run, even after an OS kill."""
    with (run_dir / '.run.lock').open('a+', encoding='utf-8') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Run folder is already active: {run_dir}') from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()) + '\n')
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load_resume_state(run_dir: Path):
    """Restore the pinned catalog and trust only completed, present JSONL files."""
    path = run_dir / 'measurement_db.manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('source_repository') != REPOSITORY:
        raise ValueError(f'{path}: unexpected source repository')
    snapshot = manifest.get('selection', {}).get('catalog_snapshot', 'measurement_db_categories.json')
    snapshot_path = (run_dir / snapshot).resolve()
    if not snapshot_path.is_relative_to(run_dir.resolve()):
        raise ValueError('Catalog snapshot must be inside the resumed run folder')
    catalog = load_catalog(snapshot_path)
    if catalog['source_revision'] != manifest.get('source_revision'):
        raise ValueError('Run manifest and catalog snapshot have different source revisions')
    requested = manifest.get('selection', {}).get('requested_categories')
    if not requested or set(requested) - set(CATEGORIES):
        raise ValueError('Run manifest has no valid category selection')
    if not isinstance(manifest.get('loaded_at_utc'), str):
        raise ValueError('Run manifest has no original start timestamp')
    outputs, missing, seen = [], [], set()
    for entry in manifest.get('outputs', []):
        name = entry['benchmark']
        if name not in catalog['benchmarks'] or name in seen:
            raise ValueError(f'Invalid or repeated completed benchmark: {name}')
        seen.add(name)
        if entry.get('output_file') != f'{name}.jsonl':
            raise ValueError(f'{name}: unexpected output filename in resume manifest')
        output = run_dir / entry['output_file']
        if output.is_file() and (output.stat().st_size > 0 or entry.get('rows') == 0):
            outputs.append(entry)
        else:
            missing.append(name)
    return catalog, manifest, outputs, missing


@contextmanager
def run_logging(run_dir: Path):
    """Flush progress and full Python tracebacks to both stdout and one run log."""
    logger = logging.Logger(f'measurement_db.{run_dir.name}', level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%dT%H:%M:%SZ')
    formatter.converter = time.gmtime
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / 'run.log', encoding='utf-8')]
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    try:
        yield logger
    finally:
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for category in CATEGORIES:
        parser.add_argument(
            '--' + category, dest='categories', action='append_const', const=category,
            help=f'Include benchmarks categorized as {category}; flags combine with OR.',
        )
    parser.add_argument('--resume', type=Path, metavar='RUN_FOLDER',
                        help='Continue an existing run, keeping its catalog, revision, and completed outputs.')
    parser.add_argument('--category-file', type=Path,
                        help='Local benchmark category catalog (also pins the dataset revision).')
    parser.add_argument('--output-root', type=Path,
                        help='Parent for category/run folders (default: data/processed/measurement_db).')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Maximum response rows per output batch (default: 256); reduce for large questions.')
    parser.add_argument('--memory-limit', default='4GB',
                        help='DuckDB working-memory budget, leaving room for Python/Arrow (default: 4GB).')
    parser.add_argument('--threads', type=int, default=4,
                        help='DuckDB worker threads (default: 4).')
    parser.add_argument('--refresh-categories', action='store_true',
                        help='Rebuild the category catalog from the latest dataset revision and exit.')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error('--batch-size and --threads must be positive')
    if args.refresh_categories and args.categories:
        parser.error('--refresh-categories cannot be combined with category selection flags')
    if args.resume and (args.categories or args.output_root or args.category_file or args.refresh_categories):
        parser.error('--resume restores its saved selection; do not combine with category, catalog, output-root, or refresh options')
    return args


def execute_run(args, run_dir: Path, catalog: dict, *, previous=None, outputs=None, missing=None):
    """Run only unfinished benchmarks; callers hold the run-folder lock."""
    revision = catalog['source_revision']
    loaded_at_utc = previous['loaded_at_utc'] if previous is not None else args.started_at
    requested = previous['selection']['requested_categories'] if previous is not None else args.categories
    discovered = sorted(catalog['benchmarks'])
    benchmarks = select_benchmarks(catalog, requested)
    selection = {
        'requested_categories': sorted(set(requested or CATEGORIES)),
        'match': 'any',
        'benchmarks_selected': len(benchmarks),
        'benchmarks_excluded': len(discovered) - len(benchmarks),
        'excluded_benchmarks': catalog['excluded_benchmarks'],
        'catalog_snapshot': previous['selection'].get('catalog_snapshot', 'measurement_db_categories.json') if previous else 'measurement_db_categories.json',
    }
    if previous is None:
        (run_dir / 'measurement_db_categories.json').write_text(
            json.dumps(catalog, indent=2, ensure_ascii=False) + '\n', encoding='utf-8',
        )
        (run_dir / 'README.txt').write_text(
            'Measurement DB preprocessing run\n'
            f'Run ID: {run_dir.name}\n'
            f'Started (UTC): {loaded_at_utc}\n'
            f'Selected categories: {", ".join(selection["requested_categories"])}\n'
            f'Dataset revision: {revision}\n'
            f'Excluded benchmarks: {", ".join(catalog["excluded_benchmarks"]) or "none"}\n\n'
            'A benchmark is selected when ANY requested category matches. Each selected\n'
            'benchmark includes all of its items and assets, even if it has other categories.\n'
            'The run manifest records actual categories per benchmark, output files, and run status.\n'
            'Each benchmark has its own JSONL; one compact manifest describes this entire run.\n'
            'Model traces are not downloaded or included in the JSONLs.\n'
            'The catalog snapshot records category\n'
            'membership at this dataset revision. Exported binary artifacts are under assets/.\n'
            'Model weights and shared processing caches are outside this folder.\n\n'
            'Progress and Python tracebacks are saved in run.log and printed in the console.\n'
            'If the OS kills Python, the log and current_benchmark show where progress stopped;\n'
            'a SIGKILL cannot be caught by Python or saved as a traceback.\n\n'
            'To resume from the repository directory:\n'
            f'poetry run python scripts/preprocess/measurement_db.py --resume {quote(str(run_dir.resolve()))}\n'
            'Completed JSONLs and recorded skips are retained; the unfinished benchmark restarts.\n\n'
            'To share: compress this entire run folder, including its manifests and assets/.\n',
            encoding='utf-8',
        )
    outputs = list(outputs or [])
    skipped = [entry for entry in previous.get('skipped', []) if entry['benchmark'] in benchmarks] if previous else []
    missing = missing or []
    current_benchmark = None

    def checkpoint(status='running', error=None):
        return write_collection_manifest(
            revision, loaded_at_utc, discovered, outputs, skipped, selection=selection,
            output_dir=run_dir, status=status, error=error,
            current_benchmark=current_benchmark if status == 'running' else None,
        )

    with run_logging(run_dir) as logger:
        done = {entry['benchmark'] for entry in outputs}
        already_skipped = {entry['benchmark'] for entry in skipped}
        remaining = [name for name in benchmarks if name not in done | already_skipped]
        if previous is not None:
            logger.info('Resuming: %s completed, %s skipped, %s remaining', len(done), len(already_skipped), len(remaining))
            for name in missing:
                logger.warning('%s: completed output missing or empty; it will be rebuilt if selected', name)
            if not remaining and not missing and previous.get('status') == 'complete':
                logger.info('Nothing to resume; this run is already complete')
                return
        logger.info(
            'Selected %s/%s benchmarks at %s: %s', len(benchmarks), len(discovered),
            revision, ', '.join(selection['requested_categories']),
        )
        if catalog['excluded_benchmarks']:
            logger.info('Excluded by policy: %s', ', '.join(catalog['excluded_benchmarks']))
        logger.info('Run output: %s', run_dir.resolve())
        logger.info('Output batch size: %s rows; DuckDB memory limit: %s; threads: %s',
                    args.batch_size, args.memory_limit, args.threads)
        api = HfApi()
        asset_context = None
        try:
            checkpoint()
            asset_context = AssetProcessingContext(
                OUT / 'asset_descriptions.sqlite3', artifact_root=run_dir / 'assets',
            )
            for position, benchmark in enumerate(benchmarks, 1):
                if benchmark in done or benchmark in already_skipped:
                    continue
                current_benchmark = benchmark
                checkpoint()
                logger.info('[%s/%s] %s', position, len(benchmarks), benchmark)
                actual_categories = catalog['benchmarks'][benchmark]['categories']
                try:
                    result = process_benchmark(
                        benchmark,
                        api,
                        revision,
                        loaded_at_utc,
                        output_dir=run_dir,
                        asset_context=asset_context,
                        batch_size=args.batch_size,
                        logger=logger,
                        memory_limit=args.memory_limit,
                        threads=args.threads,
                    )
                except UnsupportedScaleError as error:
                    skipped.append({
                        'benchmark': benchmark, 'categories': actual_categories, 'reason': str(error),
                    })
                    logger.warning('%s: skipped: %s', benchmark, error)
                else:
                    result['categories'] = actual_categories
                    outputs.append(result)
                    logger.info(
                        '%s: wrote %s rows to %s', benchmark,
                        f"{result['rows']:,}", result['output_file'],
                    )
                checkpoint()
        except BaseException as error:
            status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            logger.exception('%s at %s: %s: %s', status.upper(), current_benchmark, type(error).__name__, error)
            try:
                checkpoint(
                    status=status,
                    error={'benchmark': current_benchmark, 'type': type(error).__name__, 'message': str(error)},
                )
            except Exception:
                logger.exception('Could not save the failure manifest; original error is shown above')
            raise
        finally:
            if asset_context is not None:
                asset_context.close()

        manifest = checkpoint(status='complete')
        logger.info('Finished: %s written, %s skipped -> %s', len(outputs), len(skipped), manifest)
        archive = run_dir.parent / f'{run_dir.parent.name}__{run_dir.name}.tar.gz'
        logger.info(
            'To compress this run: tar -czf %s -C %s %s',
            quote(str(archive.resolve())), quote(str(run_dir.parent.resolve())), quote(run_dir.name),
        )


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.refresh_categories:
        build_catalog(args.category_file or CATEGORY_FILE)
        return
    if args.resume:
        run_dir = args.resume.resolve()
        if not run_dir.is_dir():
            raise ValueError(f'Resume folder does not exist: {run_dir}')
        with run_lock(run_dir):
            catalog, previous, outputs, missing = load_resume_state(run_dir)
            execute_run(args, run_dir, catalog, previous=previous, outputs=outputs, missing=missing)
    else:
        catalog = load_catalog(args.category_file or CATEGORY_FILE)
        args.started_at = utc_now()
        run_dir = create_run_directory(
            args.output_root or OUT / 'measurement_db', list(args.categories or CATEGORIES), args.started_at,
        )
        with run_lock(run_dir):
            execute_run(args, run_dir, catalog)


if __name__ == "__main__":
    main()
