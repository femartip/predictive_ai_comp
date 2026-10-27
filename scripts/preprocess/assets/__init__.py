"""Exact MIME dispatch for Measurement DB asset preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .application import (
    process_gzip,
    process_json,
    process_ndjson,
    process_npy,
    process_octet_stream,
    process_parquet,
    process_pdf,
    process_sql,
    process_sqlite,
    process_zip,
)
from .audio import process_mpeg, process_wav
from .base import AssetLimits, AssetProcessingContext, ProcessedAsset
from .image import (
    process_bmp,
    process_exr,
    process_gif,
    process_jpeg,
    process_png,
    process_tiff,
    process_webp,
)
from .model import process_obj
from .text import process_csv, process_markdown, process_plain, process_python, process_shellscript
from .video import process_mp4


class UnsupportedAssetTypeError(ValueError):
    """No explicit policy has been implemented for this exact MIME type."""


@dataclass(frozen=True)
class Handler:
    function: Callable[..., ProcessedAsset]
    family: str
    prompt_version: str
    cacheable: bool = True


PDF_PROCESSING_POLICY = {'mode': 'text-only', 'recipe_version': 'pdf-text-only-v2'}


HANDLERS = {
    "application/gzip": Handler(process_gzip, "application", "gzip-recursive-v3", False),
    "application/json": Handler(process_json, "application", "json-verbatim-valid-v1"),
    "application/octet-stream": Handler(
        process_octet_stream, "application", "octet-verified-media-or-preserve-v2", False
    ),
    "application/pdf": Handler(process_pdf, "application", PDF_PROCESSING_POLICY['recipe_version']),
    "application/sql": Handler(process_sql, "application", "sql-verbatim-v1"),
    "application/vnd.apache.parquet": Handler(
        process_parquet, "application", "parquet-schema-rows-v1"
    ),
    "application/vnd.sqlite3": Handler(process_sqlite, "application", "sqlite-readonly-utf8-preserved-v2"),
    "application/x-ndjson": Handler(process_ndjson, "application", "ndjson-valid-lines-v1"),
    "application/x-npy": Handler(process_npy, "application", "npy-restricted-object-data-v2"),
    "application/zip": Handler(process_zip, "application", "zip-verified-members-v2", False),
    "audio/mpeg": Handler(process_mpeg, "audio", "audio-timestamped-description-v1"),
    "audio/wav": Handler(process_wav, "audio", "audio-timestamped-description-v1"),
    "image/bmp": Handler(process_bmp, "image", "bmp-lossless-png-description-v1"),
    "image/gif": Handler(process_gif, "image", "gif-eight-frame-description-v1"),
    "image/jpeg": Handler(process_jpeg, "image", "jpeg-description-v1"),
    "image/png": Handler(process_png, "image", "png-transparency-description-v1"),
    "image/tiff": Handler(process_tiff, "image", "tiff-eight-page-description-v1"),
    "image/webp": Handler(process_webp, "image", "webp-eight-frame-description-v1"),
    "image/x-exr": Handler(process_exr, "image", "exr-tonemap-all-channels-description-v1"),
    "model/obj": Handler(process_obj, "image", "obj-six-view-render-description-v1"),
    "text/csv": Handler(process_csv, "text", "csv-head-tail-record-limit-v2"),
    "text/markdown": Handler(process_markdown, "text", "markdown-head-tail-line-limit-v2"),
    "text/plain": Handler(process_plain, "text", "plain-head-tail-line-limit-v2"),
    "text/x-python": Handler(process_python, "text", "python-head-tail-never-execute-v2"),
    "text/x-shellscript": Handler(
        process_shellscript, "text", "shell-head-tail-never-execute-v2"
    ),
    "video/mp4": Handler(process_mp4, "video", "mp4-sixteen-frame-audio-description-v1"),
}

SUPPORTED_MEDIA_TYPES = frozenset(HANDLERS)


def handler_for(media_type: str) -> Handler:
    try:
        return HANDLERS[media_type]
    except KeyError as error:
        raise UnsupportedAssetTypeError(
            f"no asset processor for {media_type!r}; implement its exact subtype before preprocessing it"
        ) from error


def cached_asset(
    asset_id: str,
    media_type: str,
    context: AssetProcessingContext,
) -> ProcessedAsset | None:
    handler = handler_for(media_type)
    if not handler.cacheable:
        return None
    version = context.recipe_version(handler.prompt_version)
    result = context.cached(asset_id, media_type, handler.family, version)
    # Artifact paths belong to a particular output folder and must be regenerated
    # there rather than reused from a previous run's cache entry.
    return result if result is not None and result.relative_path is None else None


def process_asset(
    asset_id: str,
    media_type: str,
    data: bytes,
    context: AssetProcessingContext,
    *,
    filename: str | None = None,
    archive_depth: int = 0,
) -> ProcessedAsset:
    """Use an exact handler and cache the result against its complete recipe."""
    handler = handler_for(media_type)
    version = context.recipe_version(handler.prompt_version)
    if handler.cacheable:
        cached = context.cached(asset_id, media_type, handler.family, version)
        if cached is not None and cached.relative_path is None:
            return cached
    result = handler.function(
        asset_id,
        data,
        context,
        filename=filename,
        archive_depth=archive_depth,
    )
    if result.media_type != media_type:
        raise ValueError(f"{media_type} processor returned {result.media_type}")
    if handler.cacheable and result.relative_path is None:
        context.store(asset_id, handler.family, version, result)
    return result


def recipe_provenance(media_types, context: AssetProcessingContext | None = None) -> dict:
    """Record the independently versioned handler used for each exact MIME type."""
    output = {}
    for media_type in sorted(media_types):
        handler = handler_for(media_type)
        version = handler.prompt_version
        if context is not None:
            version = context.recipe_version(version)
        output[media_type] = {
            "family": handler.family,
            "recipe_version": version,
            "cacheable": handler.cacheable,
        }
    return output


__all__ = [
    "AssetLimits",
    "AssetProcessingContext",
    "ProcessedAsset",
    "PDF_PROCESSING_POLICY",
    "SUPPORTED_MEDIA_TYPES",
    "UnsupportedAssetTypeError",
    "cached_asset",
    "handler_for",
    "process_asset",
    "recipe_provenance",
]
