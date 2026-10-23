"""Join item manifests to binary assets and append processed descriptions."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd

from . import AssetProcessingContext, cached_asset, handler_for, process_asset
from .base import ProcessedAsset


REFERENCE_COLUMNS = ["item_id", "asset_id", "media_type", "ordinal"]


def parse_manifest(value: object, item_id: str) -> list[dict]:
    if value is None or (not isinstance(value, (str, list)) and pd.isna(value)):
        return []
    manifest = json.loads(value) if isinstance(value, str) else value
    if not isinstance(manifest, list):
        raise ValueError(f"{item_id}: asset_manifest is not a list")
    for entry in manifest:
        if not isinstance(entry, dict):
            raise ValueError(f"{item_id}: asset manifest entry is not an object")
        missing = {"asset_id", "media_type", "ordinal"} - set(entry)
        if missing:
            raise ValueError(f"{item_id}: asset manifest entry is missing {sorted(missing)}")
    return sorted(manifest, key=lambda entry: entry["ordinal"])


def manifest_references(items: pd.DataFrame) -> pd.DataFrame:
    """Flatten manifests while preserving their released per-item order."""
    if "asset_manifest" not in items:
        return pd.DataFrame(columns=REFERENCE_COLUMNS)
    rows = []
    for item in items[["item_id", "asset_manifest"]].itertuples(index=False):
        for entry in parse_manifest(item.asset_manifest, item.item_id):
            rows.append({
                "item_id": item.item_id,
                "asset_id": str(entry["asset_id"]),
                "media_type": str(entry["media_type"]),
                "ordinal": int(entry["ordinal"]),
            })
    references = pd.DataFrame(rows, columns=REFERENCE_COLUMNS)
    if references.empty:
        return references
    conflict = references.groupby("asset_id").media_type.nunique().gt(1)
    if conflict.any():
        asset_ids = conflict[conflict].index[:5].tolist()
        raise ValueError(f"asset IDs referenced with conflicting media types: {asset_ids}")
    return references


def _unique_assets(references: pd.DataFrame) -> pd.DataFrame:
    return references[["asset_id", "media_type"]].drop_duplicates().sort_values(
        ["media_type", "asset_id"]
    ).reset_index(drop=True)


def _uncached_assets(
    unique_assets: pd.DataFrame,
    context: AssetProcessingContext,
) -> tuple[dict[tuple[str, str], ProcessedAsset], pd.DataFrame]:
    results = {}
    missing = []
    for row in unique_assets.itertuples(index=False):
        # Resolve every type before opening the very large asset table. Unknown
        # types therefore fail explicitly without downloading model weights.
        handler_for(row.media_type)
        result = cached_asset(row.asset_id, row.media_type, context)
        if result is None:
            missing.append({"asset_id": row.asset_id, "media_type": row.media_type})
        else:
            results[(row.asset_id, row.media_type)] = result
    return results, pd.DataFrame(missing, columns=["asset_id", "media_type"])


def _generate_missing(
    assets_path: Path,
    missing: pd.DataFrame,
    context: AssetProcessingContext,
    results: dict[tuple[str, str], ProcessedAsset],
) -> None:
    if missing.empty:
        return
    connection = duckdb.connect()
    try:
        connection.register("missing_assets", missing)
        connection.from_parquet(str(assets_path)).create_view("assets")
        columns = {row[0] for row in connection.execute("DESCRIBE assets").fetchall()}
        required = {"asset_id", "data"}
        if not required.issubset(columns):
            raise ValueError(f"{assets_path.name} is missing columns {sorted(required - columns)}")
        duplicate = connection.execute(
            """
            SELECT a.asset_id
            FROM assets a JOIN missing_assets m USING (asset_id)
            GROUP BY a.asset_id HAVING count(*) != 1
            LIMIT 5
            """
        ).fetchall()
        if duplicate:
            raise ValueError(f"assets table has duplicate asset IDs: {[row[0] for row in duplicate]}")
        reader = connection.execute(
            """
            SELECT a.asset_id, m.media_type, a.data
            FROM assets a JOIN missing_assets m USING (asset_id)
            ORDER BY m.media_type, a.asset_id
            """
        ).to_arrow_reader(batch_size=16)
        found = set()
        for batch in reader:
            for row in batch.to_pandas().itertuples(index=False):
                key = (row.asset_id, row.media_type)
                if key in found:
                    raise ValueError(f"duplicate binary asset {row.asset_id}")
                found.add(key)
                results[key] = process_asset(row.asset_id, row.media_type, bytes(row.data), context)
        expected = set(map(tuple, missing[["asset_id", "media_type"]].itertuples(index=False, name=None)))
        absent = sorted(expected - found)
        if absent:
            raise ValueError(f"asset manifest references missing binary assets: {absent[:5]}")
    finally:
        connection.close()


def add_asset_descriptions(
    items: pd.DataFrame,
    assets_path: Path,
    context: AssetProcessingContext,
) -> tuple[pd.DataFrame, dict]:
    """Append descriptions in manifest ordinal order and leave asset-free items unchanged."""
    references = manifest_references(items)
    if references.empty:
        return items, {
            "items_with_assets": 0,
            "manifest_references": 0,
            "unique_assets": 0,
            "cache_hits": 0,
            "generated_assets": 0,
            "media_types": [],
        }

    unique_assets = _unique_assets(references)
    results, missing = _uncached_assets(unique_assets, context)
    cache_hits = len(results)
    _generate_missing(assets_path, missing, context, results)

    fragments = defaultdict(list)
    for row in references.sort_values(["item_id", "ordinal"]).itertuples(index=False):
        result = results[(row.asset_id, row.media_type)]
        fragments[row.item_id].append(result.question_fragment)

    output = items.copy()
    additions = output.item_id.map(
        lambda item_id: "\n\n" + "\n\n".join(fragments[item_id]) if item_id in fragments else ""
    )
    output["content"] = output.content.fillna("").astype(str) + additions
    stats = {
        "items_with_assets": len(fragments),
        "manifest_references": len(references),
        "unique_assets": len(unique_assets),
        "cache_hits": cache_hits,
        "generated_assets": len(missing),
        "media_types": sorted(unique_assets.media_type.unique().tolist()),
    }
    return output, stats


__all__ = ["add_asset_descriptions", "manifest_references", "parse_manifest"]
