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
        f'<asset_truncation kind="{kind}" mode="head_tail" original_bytes="{original_bytes}" '
        f'omitted_{unit}="{omitted}" />'
    )


def _truncate_chunks(text: str, chunks: list[str], limit: int, kind: str, unit: str) -> str:
    """Keep complete lines/CSV records at both ends when they fit the byte limit."""
    sizes = [len(chunk.encode("utf-8")) for chunk in chunks]
    first, last, used = 0, len(chunks), 0
    while first < last and used + sizes[first] <= limit // 2:
        used += sizes[first]
        first += 1
    while last > first and used + sizes[last - 1] <= limit:
        last -= 1
        used += sizes[last]
    original = len(text.encode("utf-8"))
    if first and last < len(chunks):
        head, tail = "".join(chunks[:first]), "".join(chunks[last:])
        marker = _truncation_marker(kind, original, last - first, unit)
    else:
        # A single enormous line/record must still expose its beginning and end.
        encoded = text.encode("utf-8")
        head_bytes = (limit + 1) // 2
        tail_bytes = limit - head_bytes
        head = encoded[:head_bytes].decode("utf-8", errors="ignore")
        tail = encoded[-tail_bytes:].decode("utf-8", errors="ignore") if tail_bytes else ""
        omitted = original - len(head.encode("utf-8")) - len(tail.encode("utf-8"))
        marker = _truncation_marker(kind, original, omitted, "bytes")
    return head + "\n" + marker + "\n" + tail


def _truncate_complete_lines(text: str, limit: int | None, kind: str) -> str:
    encoded_size = len(text.encode("utf-8"))
    if limit is None or encoded_size <= limit:
        return text
    return _truncate_chunks(text, text.splitlines(keepends=True), limit, kind, "lines")


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


def _truncate_csv(text: str, limit: int | None) -> str:
    encoded_size = len(text.encode("utf-8"))
    if limit is None or encoded_size <= limit:
        return text
    return _truncate_chunks(text, _csv_record_chunks(text), limit, "csv", "records")


def process_csv(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    content = _truncate_csv(decode_text(data), context.limits.max_csv_bytes)
    return embedded_result(content, "text/csv", "csv_asset")


def process_markdown(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    content = _truncate_complete_lines(
        decode_text(data), context.limits.max_markdown_bytes, "markdown"
    )
    return embedded_result(content, "text/markdown", "markdown_asset")


def process_plain(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    content = _truncate_complete_lines(
        decode_text(data), context.limits.max_plain_text_bytes, "plain_text"
    )
    return embedded_result(content, "text/plain", "text_asset")


def process_python(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    content = _truncate_complete_lines(
        decode_text(data), context.limits.max_python_bytes, "python"
    )
    return embedded_result(content, "text/x-python", "python_asset")


def process_shellscript(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    content = _truncate_complete_lines(
        decode_text(data), context.limits.max_shellscript_bytes, "shellscript"
    )
    return embedded_result(content, "text/x-shellscript", "shellscript_asset")
