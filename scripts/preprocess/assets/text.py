"""Verbatim, non-executing processors for every supported text MIME type."""

from __future__ import annotations

import csv
from io import StringIO

from .base import AssetProcessingContext, ProcessedAsset, embedded_result


def decode_text(data: bytes) -> str:
    """Prefer Unicode standards and reject highly uncertain legacy decoding."""
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    try:
        from charset_normalizer import from_bytes
    except ImportError as error:
        raise RuntimeError("Legacy text decoding requires charset-normalizer. Run `poetry install`.") from error
    match = from_bytes(data).best()
    if match is None or match.encoding is None or match.percent_chaos > 20:
        raise UnicodeError("text encoding could not be determined safely")
    return str(match)


def _truncation_marker(kind: str, original_bytes: int, omitted: int, unit: str) -> str:
    return (
        f'<asset_truncation kind="{kind}" original_bytes="{original_bytes}" '
        f'omitted_{unit}="{omitted}" />'
    )


def _truncate_complete_lines(text: str, limit: int | None, kind: str) -> tuple[str, str | None]:
    encoded_size = len(text.encode("utf-8"))
    if limit is None or encoded_size <= limit:
        return text, None
    lines = text.splitlines(keepends=True)
    kept = []
    used = 0
    for line in lines:
        size = len(line.encode("utf-8"))
        if used + size > limit:
            break
        kept.append(line)
        used += size
    marker = _truncation_marker(kind, encoded_size, len(lines) - len(kept), "lines")
    return "".join(kept), marker


def _csv_record_chunks(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    try:
        dialect = csv.Sniffer().sniff(text[:65_536])
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(StringIO(text, newline=""), dialect=dialect)
    ends = []
    for _ in reader:
        ends.append(reader.line_num)
    chunks = []
    start = 0
    for end in ends:
        chunks.append("".join(lines[start:end]))
        start = end
    if start < len(lines):
        chunks.append("".join(lines[start:]))
    return chunks


def _truncate_csv(text: str, limit: int | None) -> tuple[str, str | None]:
    encoded_size = len(text.encode("utf-8"))
    if limit is None or encoded_size <= limit:
        return text, None
    chunks = _csv_record_chunks(text)
    kept = []
    used = 0
    for chunk in chunks:
        size = len(chunk.encode("utf-8"))
        if used + size > limit:
            break
        kept.append(chunk)
        used += size
    marker = _truncation_marker("csv", encoded_size, len(chunks) - len(kept), "records")
    return "".join(kept), marker


def _result(content: str, marker: str | None, media_type: str, tag: str) -> ProcessedAsset:
    result = embedded_result(content, media_type, tag)
    if marker is None:
        return result
    return ProcessedAsset(
        question_fragment=result.question_fragment + "\n" + marker,
        relative_path=None,
        media_type=media_type,
    )


def process_csv(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    content, marker = _truncate_csv(decode_text(data), context.limits.max_csv_bytes)
    return _result(content, marker, "text/csv", "csv_asset")


def process_markdown(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    content, marker = _truncate_complete_lines(
        decode_text(data), context.limits.max_markdown_bytes, "markdown"
    )
    return _result(content, marker, "text/markdown", "markdown_asset")


def process_plain(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    content, marker = _truncate_complete_lines(
        decode_text(data), context.limits.max_plain_text_bytes, "plain_text"
    )
    return _result(content, marker, "text/plain", "text_asset")


def process_python(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    content, marker = _truncate_complete_lines(
        decode_text(data), context.limits.max_python_bytes, "python"
    )
    return _result(content, marker, "text/x-python", "python_asset")


def process_shellscript(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    content, marker = _truncate_complete_lines(
        decode_text(data), context.limits.max_shellscript_bytes, "shellscript"
    )
    return _result(content, marker, "text/x-shellscript", "shellscript_asset")
