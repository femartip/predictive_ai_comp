"""Build and read the local, revision-pinned Measurement DB category catalog."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from huggingface_hub.hf_api import RepoFolder

REPOSITORY = 'aims-foundations/measurement-db'
CATEGORY_FILE = Path(__file__).with_suffix('.json')
DEFAULT_EXCLUDED_BENCHMARKS = {
    'alignment_faking': 'Behavior-presence labels have no declared utility ordering.',
    'annotating_errors_wcf': 'Excluded as requested: mixed criteria include unordered directness labels.',
}
CATEGORIES = (
    'no-assets', 'asset-image', 'asset-audio', 'asset-model',
    'asset-text', 'asset-video', 'asset-application',
)


def categories_for_media_types(media_types):
    """Classify source MIME types; models and PDFs keep their source categories."""
    categories = set()
    for media_type in media_types:
        major, separator, subtype = media_type.partition('/')
        category = 'asset-' + major
        if not separator or not subtype or category not in CATEGORIES:
            raise ValueError(f'Cannot categorize asset MIME type {media_type!r}')
        categories.add(category)
    return sorted(categories or {'no-assets'})


def load_catalog(path: Path = CATEGORY_FILE):
    try:
        catalog = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as error:
        raise ValueError(
            f'Category catalog missing: {path}. Run measurement_db.py --refresh-categories '
            f'--category-file {path} first.'
        ) from error
    if catalog.get('schema_version') != 1 or catalog.get('source_repository') != REPOSITORY:
        raise ValueError(f'{path}: unsupported category catalog schema or source repository')
    if not re.fullmatch(r'[0-9a-f]{40}', catalog.get('source_revision', '')):
        raise ValueError(f'{path}: source_revision must be a pinned dataset commit')
    if not isinstance(catalog.get('benchmarks'), dict) or not catalog['benchmarks']:
        raise ValueError(f'{path}: benchmark category mapping is empty or invalid')
    for benchmark, entry in catalog['benchmarks'].items():
        categories = entry.get('categories')
        if not isinstance(categories, list) or not categories or set(categories) - set(CATEGORIES):
            raise ValueError(f'{benchmark}: invalid categories {categories!r}')
        if 'no-assets' in categories and categories != ['no-assets']:
            raise ValueError(f'{benchmark}: no-assets cannot be combined with asset categories')
        if categories != categories_for_media_types(entry['media_types']):
            raise ValueError(f'{benchmark}: categories disagree with catalog MIME types')
    configured = catalog.get('excluded_benchmarks', {})
    if not isinstance(configured, dict) or any(
        name not in catalog['benchmarks'] or not isinstance(reason, str) or not reason
        for name, reason in configured.items()
    ):
        raise ValueError(f'{path}: invalid excluded_benchmarks mapping')
    catalog['excluded_benchmarks'] = {
        **{name: reason for name, reason in DEFAULT_EXCLUDED_BENCHMARKS.items() if name in catalog['benchmarks']},
        **configured,
    }
    return catalog


def select_benchmarks(catalog, categories=None, *, exclusive=False):
    """Match requested categories, optionally excluding every unrequested asset category."""
    requested = set(categories or CATEGORIES)
    if requested - set(CATEGORIES):
        raise ValueError(f'Unknown benchmark categories: {sorted(requested - set(CATEGORIES))}')
    excluded = set(DEFAULT_EXCLUDED_BENCHMARKS) | set(catalog.get('excluded_benchmarks', {}))
    return sorted(
        name for name, entry in catalog['benchmarks'].items()
        if name not in excluded and requested.intersection(entry['categories'])
        and (not exclusive or set(entry['categories']).issubset(requested))
    )


def inspect_benchmark(benchmark, revision):
    """Read only the manifest column, never asset payloads or response tables."""
    uri = f'datasets/{REPOSITORY}@{revision}/{benchmark}/formatted_tables/items.parquet'
    for attempt in range(3):
        try:
            media_types = set()
            with HfFileSystem().open(uri, 'rb', block_size=256 * 1024) as source:
                parquet = pq.ParquetFile(source)
                if 'asset_manifest' in parquet.schema_arrow.names:
                    for batch in parquet.iter_batches(batch_size=10000, columns=['asset_manifest']):
                        for encoded in pc.unique(batch.column(0)).to_pylist():
                            if encoded is None:
                                continue
                            manifest = json.loads(encoded) if isinstance(encoded, str) else encoded
                            if not isinstance(manifest, list):
                                raise ValueError(f'{benchmark}: asset_manifest is not a list')
                            media_types.update(asset['media_type'] for asset in manifest)
                item_count = parquet.metadata.num_rows
            return {
                'categories': categories_for_media_types(media_types),
                'media_types': sorted(media_types),
                'items': item_count,
            }
        except (OSError, ConnectionError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def build_catalog(path: Path = CATEGORY_FILE, *, revision=None, workers=6):
    """Explicit refresh; normal preprocessing only reads the saved JSON file."""
    configured_exclusions = {}
    if path.exists():
        configured_exclusions = load_catalog(path)['excluded_benchmarks']
    api = HfApi()
    revision = api.dataset_info(REPOSITORY, revision=revision).sha
    benchmarks = sorted(
        entry.path for entry in api.list_repo_tree(
            REPOSITORY, repo_type='dataset', revision=revision, recursive=False,
        ) if isinstance(entry, RepoFolder)
    )
    entries = {}
    print(f'Building category catalog: {len(benchmarks)} benchmarks at {revision}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(inspect_benchmark, name, revision): name for name in benchmarks}
        for future in as_completed(futures):
            benchmark = futures[future]
            entries[benchmark] = future.result()
            print(f'  [{len(entries)}/{len(benchmarks)}] {benchmark}: {", ".join(entries[benchmark]["categories"])}', flush=True)
    catalog = {
        'schema_version': 1,
        'source_repository': REPOSITORY,
        'source_revision': revision,
        'generated_at_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'classification': 'MIME major types referenced in items.asset_manifest; any-match benchmark selection',
        'excluded_benchmarks': {
            name: reason for name, reason in {**DEFAULT_EXCLUDED_BENCHMARKS, **configured_exclusions}.items()
            if name in entries
        },
        'benchmarks': {name: entries[name] for name in benchmarks},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    load_catalog(temporary)
    temporary.replace(path)
    counts = Counter(category for entry in entries.values() for category in entry['categories'])
    print(f'Category catalog saved: {path}\nCategory counts (mixed benchmarks count more than once): {dict(sorted(counts.items()))}', flush=True)
    return catalog
