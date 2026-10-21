"""Safe processors for structured, document, archive, and binary application assets."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import re
import sqlite3
import stat
import struct
import tarfile
import tempfile
import zipfile
from datetime import date, datetime, time
from decimal import Decimal
from io import BytesIO
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .base import (
    AssetProcessingContext,
    ProcessedAsset,
    embedded_result,
    evenly_spaced_indices,
)
from .text import decode_text


TRUNCATION_KEY = "__measurement_db_truncation__"

EXTENSION_TYPES = {
    ".bmp": "image/bmp",
    ".csv": "text/csv",
    ".db": "application/vnd.sqlite3",
    ".exr": "image/x-exr",
    ".gif": "image/gif",
    ".gz": "application/gzip",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".ndjson": "application/x-ndjson",
    ".npy": "application/x-npy",
    ".obj": "model/obj",
    ".parquet": "application/vnd.apache.parquet",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".py": "text/x-python",
    ".sh": "text/x-shellscript",
    ".sql": "application/sql",
    ".sqlite": "application/vnd.sqlite3",
    ".sqlite3": "application/vnd.sqlite3",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".txt": "text/plain",
    ".wav": "audio/wav",
    ".webp": "image/webp",
    ".zip": "application/zip",
}


def _json_compatible(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "NaN"}
        if math.isinf(value):
            return {"__float__": "Infinity" if value > 0 else "-Infinity"}
        return value
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, bytes):
        return {"__base64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    if isinstance(value, (datetime, date, time)):
        return {"__iso8601__": value.isoformat()}
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, np.ndarray):
        return [_json_compatible(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return {"__repr__": repr(value), "__type__": type(value).__name__}


def _json_dump(value, *, indent=None) -> str:
    return json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )


def _with_marker(result: ProcessedAsset, marker: str | None) -> ProcessedAsset:
    if marker is None:
        return result
    return ProcessedAsset(result.question_fragment + "\n" + marker, result.relative_path, result.media_type)


def _truncate_json_value(value, maximum: int, original_bytes: int):
    metadata = {"original_bytes": original_bytes}
    if isinstance(value, dict):
        key = TRUNCATION_KEY
        while key in value:
            key = "_" + key
        output = {}
        entries = list(value.items())
        for position, (name, item) in enumerate(entries):
            marker = {**metadata, "omitted_members": len(entries) - position}
            candidate = {**output, str(name): item, key: marker}
            if len(_json_dump(candidate).encode("utf-8")) > maximum:
                break
            output[str(name)] = item
        kept = len(output)
        output[key] = {**metadata, "omitted_members": len(entries) - kept}
        return output
    if isinstance(value, list):
        output = []
        for position, item in enumerate(value):
            marker = {TRUNCATION_KEY: {**metadata, "omitted_elements": len(value) - position}}
            if len(_json_dump([*output, item, marker]).encode("utf-8")) > maximum:
                break
            output.append(item)
        kept = len(output)
        output.append({TRUNCATION_KEY: {**metadata, "omitted_elements": len(value) - kept}})
        return output
    prefix = str(value)
    wrapper = {"value_prefix": prefix, TRUNCATION_KEY: metadata}
    while prefix and len(_json_dump(wrapper).encode("utf-8")) > maximum:
        prefix = prefix[: max(0, len(prefix) // 2)]
        wrapper["value_prefix"] = prefix
    return wrapper


def process_json(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    text = decode_text(data)
    value = json.loads(text)
    maximum = context.limits.max_json_bytes
    if maximum is not None and len(text.encode("utf-8")) > maximum:
        text = _json_dump(_truncate_json_value(value, maximum, len(data)), indent=2)
    return embedded_result(text, "application/json", "json_asset")


def process_ndjson(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    text = decode_text(data)
    lines = text.splitlines(keepends=True)
    record_lines = [line for line in lines if line.strip()]
    for number, line in enumerate(record_lines, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid NDJSON record {number}: {error}") from error
    maximum = context.limits.max_ndjson_bytes
    marker = None
    if maximum is not None and len(text.encode("utf-8")) > maximum:
        kept = []
        used = 0
        for line in lines:
            size = len(line.encode("utf-8"))
            if used + size > maximum:
                break
            kept.append(line)
            used += size
        omitted = sum(bool(line.strip()) for line in lines[len(kept):])
        text = "".join(kept)
        marker = (
            f'<asset_truncation kind="ndjson" original_bytes="{len(data)}" '
            f'omitted_records="{omitted}" />'
        )
    return _with_marker(embedded_result(text, "application/x-ndjson", "ndjson_asset"), marker)


def process_sql(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    text = decode_text(data)
    maximum = context.limits.max_sql_bytes
    marker = None
    if maximum is not None and len(text.encode("utf-8")) > maximum:
        try:
            import sqlparse
        except ImportError as error:
            raise RuntimeError("Limited SQL processing requires sqlparse. Run `poetry install`.") from error
        statements = sqlparse.split(text, strip_semicolon=False)
        kept = []
        used = 0
        for statement in statements:
            chunk = statement + "\n"
            size = len(chunk.encode("utf-8"))
            if used + size > maximum:
                break
            kept.append(chunk)
            used += size
        text = "".join(kept)
        marker = (
            f'<asset_truncation kind="sql" original_bytes="{len(data)}" '
            f'omitted_statements="{len(statements) - len(kept)}" />'
        )
    return _with_marker(embedded_result(text, "application/sql", "sql_asset"), marker)


def process_pdf(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError("PDF processing requires PyMuPDF. Run `poetry install`.") from error
    document = fitz.open(stream=data, filetype="pdf")
    try:
        total = len(document)
        if total == 0:
            raise ValueError("PDF has no pages")
        maximum = context.limits.max_pdf_pages
        indices = list(range(total)) if maximum is None else evenly_spaced_indices(total, maximum)
        sections = []
        for index in indices:
            page = document[index]
            extracted_text = page.get_text("text", sort=False)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            prompt = (
                f"Describe page {index + 1} of {total} objectively and exhaustively, focusing on layout, "
                "figures, diagrams, handwriting, tables, and visible text that a PDF text layer may miss. "
                "Do not infer or answer the benchmark question."
            )
            visual = context.describe_images([image], prompt)
            sections.append(
                f'<pdf_page number="{index + 1}" total="{total}">\n'
                f"<extracted_text>\n{extracted_text}\n</extracted_text>\n"
                f"<visual_description>\n{visual}\n</visual_description>\n"
                "</pdf_page>"
            )
        omitted = total - len(indices)
        if omitted:
            sections.append(
                f'<asset_truncation kind="pdf" omitted_pages="{omitted}" '
                f'included_pages="{",".join(str(index + 1) for index in indices)}" />'
            )
        return embedded_result("\n".join(sections), "application/pdf", "pdf_asset")
    finally:
        document.close()


def process_parquet(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Parquet processing requires pyarrow. Run `poetry install`.") from error
    table = pq.read_table(BytesIO(data))
    total = table.num_rows
    maximum = context.limits.max_parquet_rows
    included = total if maximum is None else min(total, maximum)
    metadata = {
        (key.decode("utf-8", "replace")): base64.b64encode(value).decode("ascii")
        for key, value in (table.schema.metadata or {}).items()
    }
    header = {
        "rows": total,
        "columns": table.num_columns,
        "schema": str(table.schema),
        "schema_metadata_base64": metadata,
    }
    rows = table.slice(0, included).to_pylist()
    content = "metadata=" + _json_dump(header) + "\nrows_jsonl:\n"
    content += "\n".join(_json_dump(row) for row in rows)
    if included < total:
        content += (
            f'\n<asset_truncation kind="parquet" omitted_rows="{total - included}" '
            f'original_rows="{total}" />'
        )
    return embedded_result(content, "application/vnd.apache.parquet", "parquet_asset")


def _quote_sqlite_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def process_sqlite(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    temporary_name = None
    connection = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as temporary:
            temporary.write(data)
            temporary_name = temporary.name
        uri = Path(temporary_name).as_uri() + "?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        schemas = connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
        sections = ["schema_jsonl:"]
        sections.extend(
            _json_dump({"type": row[0], "name": row[1], "table": row[2], "sql": row[3]})
            for row in schemas
        )
        ordinary_tables = [
            row[1] for row in schemas
            if row[0] == "table" and row[3] and not row[3].lstrip().upper().startswith("CREATE VIRTUAL TABLE")
        ]
        maximum = context.limits.max_sqlite_rows_per_table
        for table_name in ordinary_tables:
            quoted = _quote_sqlite_identifier(table_name)
            total = connection.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]
            columns = connection.execute(f"PRAGMA table_info({quoted})").fetchall()
            primary_key = [row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]]
            order = ", ".join(_quote_sqlite_identifier(name) for name in primary_key)
            if not order:
                try:
                    connection.execute(f"SELECT rowid FROM {quoted} LIMIT 0")
                    order = "rowid"
                except sqlite3.DatabaseError:
                    order = ""
            query = f"SELECT * FROM {quoted}"
            if order:
                query += f" ORDER BY {order}"
            if maximum is not None:
                query += f" LIMIT {int(maximum)}"
            cursor = connection.execute(query)
            names = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            sections.append(f'<sqlite_table name={json.dumps(table_name)} rows="{total}">')
            sections.extend(_json_dump(dict(zip(names, row))) for row in rows)
            if len(rows) < total:
                sections.append(
                    f'<asset_truncation kind="sqlite_table" omitted_rows="{total - len(rows)}" />'
                )
            sections.append("</sqlite_table>")
        return embedded_result("\n".join(sections), "application/vnd.sqlite3", "sqlite_asset")
    finally:
        if connection is not None:
            connection.close()
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def process_npy(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    array = np.load(BytesIO(data), allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise ValueError("application/x-npy payload is not a NumPy array")
    order = "F" if array.flags.f_contiguous and not array.flags.c_contiguous else "C"
    metadata = {
        "dtype": str(array.dtype),
        "dtype_fields": array.dtype.descr if array.dtype.fields else None,
        "shape": list(array.shape),
        "memory_order": order,
        "elements": int(array.size),
    }
    maximum = context.limits.max_npy_elements
    if maximum is None or array.size <= maximum:
        values = _json_compatible(array)
        body = {"metadata": metadata, "values": values}
    else:
        flat = array.reshape(-1, order="C")
        indices = evenly_spaced_indices(array.size, maximum)
        body = {
            "metadata": metadata,
            "sampled_flat_values": [
                {"index": index, "value": _json_compatible(flat[index])} for index in indices
            ],
            TRUNCATION_KEY: {"omitted_elements": int(array.size - len(indices))},
        }
    return embedded_result(_json_dump(body, indent=2), "application/x-npy", "npy_asset")


def _safe_member_path(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or any(part == "" for part in path.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return path.as_posix()


def detect_media_type(data: bytes, filename: str | None = None) -> str | None:
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"PK\x03\x04"):
        return "application/zip"
    if data.startswith(b"\x1f\x8b"):
        return "application/gzip"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if data.startswith(b"v/1\x01"):
        return "image/x-exr"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "audio/wav"
    if data.startswith(b"ID3") or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return "audio/mpeg"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "video/mp4"
    if data.startswith(b"SQLite format 3\x00"):
        return "application/vnd.sqlite3"
    if data.startswith(b"\x93NUMPY"):
        return "application/x-npy"
    if data.startswith(b"PAR1") and data.endswith(b"PAR1"):
        return "application/vnd.apache.parquet"
    suffix = Path(filename).suffix.lower() if filename else ""
    if suffix in EXTENSION_TYPES:
        return EXTENSION_TYPES[suffix]
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            json.loads(text)
            return "application/json"
        except json.JSONDecodeError:
            pass
    if re.search(r"(?m)^\s*v\s+[-+0-9.]", text) and re.search(r"(?m)^\s*f\s+\d", text):
        return "model/obj"
    return "text/plain" if text else None


def _nested_result(
    path: str,
    declared: str,
    detected: str,
    result: ProcessedAsset,
) -> str:
    return (
        f'<archive_member path={json.dumps(path)} declared_media_type={json.dumps(declared)} '
        f'detected_media_type={json.dumps(detected)}>\n{result.question_fragment}\n</archive_member>'
    )


def _process_members(
    members: list[tuple[str, int, int, str, bytes | None]],
    context: AssetProcessingContext,
    archive_depth: int,
) -> tuple[list[dict], list[str], int]:
    from . import process_asset

    manifest = []
    fragments = []
    processed = 0
    for path, compressed_size, uncompressed_size, checksum, payload in members:
        manifest.append({
            "path": path,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "checksum": checksum,
        })
        if payload is None:
            continue
        detected = detect_media_type(payload, path)
        if detected is None:
            detected = "application/octet-stream"
        nested_id = hashlib.sha256(payload).hexdigest()
        result = process_asset(
            nested_id,
            detected,
            payload,
            context,
            filename=path,
            archive_depth=archive_depth + 1,
        )
        fragments.append(_nested_result(path, "archive member", detected, result))
        processed += 1
    return manifest, fragments, len(members) - processed


def _include_archive_member(
    context: AssetProcessingContext,
    processed: int,
    used_bytes: int,
    uncompressed_size: int,
) -> bool:
    maximum_entries = context.limits.max_archive_entries
    maximum_bytes = context.limits.max_archive_uncompressed_bytes
    if maximum_entries is not None and processed >= maximum_entries:
        return False
    return maximum_bytes is None or used_bytes + uncompressed_size <= maximum_bytes


def process_zip(
    asset_id: str,
    data: bytes,
    context: AssetProcessingContext,
    *,
    archive_depth: int = 0,
    **_,
) -> ProcessedAsset:
    if archive_depth >= context.limits.max_archive_depth:
        raise ValueError(f"archive nesting exceeds max_archive_depth={context.limits.max_archive_depth}")
    members = []
    with zipfile.ZipFile(BytesIO(data)) as archive:
        processed = 0
        used_bytes = 0
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            path = _safe_member_path(info.filename)
            if info.is_dir():
                continue
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted ZIP member is not supported: {path}")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError(f"ZIP symbolic link is not supported: {path}")
            include = _include_archive_member(context, processed, used_bytes, info.file_size)
            payload = archive.read(info) if include else None
            if include:
                processed += 1
                used_bytes += info.file_size
            members.append((path, info.compress_size, info.file_size, f"crc32:{info.CRC:08x}", payload))
    manifest, fragments, omitted = _process_members(members, context, archive_depth)
    content = "manifest=" + _json_dump(manifest) + "\n" + "\n".join(fragments)
    if omitted:
        content += f'\n<asset_truncation kind="zip" omitted_members="{omitted}" />'
    return embedded_result(content, "application/zip", "zip_asset")


def _gzip_header(data: bytes) -> dict:
    if len(data) < 10 or data[:2] != b"\x1f\x8b":
        raise ValueError("invalid GZIP header")
    flags = data[3]
    metadata = {"mtime": struct.unpack("<I", data[4:8])[0]}
    position = 10
    if flags & 0x04:
        length = struct.unpack("<H", data[position:position + 2])[0]
        position += 2 + length
    if flags & 0x08:
        end = data.find(b"\x00", position)
        if end < 0:
            raise ValueError("invalid GZIP original filename")
        metadata["original_filename"] = data[position:end].decode("latin-1")
    return metadata


def _decompress_gzip(data: bytes, maximum: int | None) -> bytes:
    if maximum is None:
        return gzip.decompress(data)
    with gzip.GzipFile(fileobj=BytesIO(data)) as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError(f"GZIP payload exceeds max_gzip_uncompressed_bytes={maximum}")
    return payload


def _process_tar(data: bytes, context: AssetProcessingContext, archive_depth: int) -> tuple[str, int]:
    members = []
    with tarfile.open(fileobj=BytesIO(data), mode="r:") as archive:
        processed = 0
        used_bytes = 0
        for info in sorted(archive.getmembers(), key=lambda item: item.name):
            path = _safe_member_path(info.name)
            if info.isdir():
                continue
            if info.issym() or info.islnk():
                raise ValueError(f"TAR link is not supported: {path}")
            if not info.isfile():
                raise ValueError(f"unsupported TAR member type: {path}")
            include = _include_archive_member(context, processed, used_bytes, info.size)
            source = archive.extractfile(info)
            if source is None:
                raise ValueError(f"could not read TAR member: {path}")
            if include:
                payload = source.read()
                digest = hashlib.sha256(payload).hexdigest()
                processed += 1
                used_bytes += info.size
            else:
                payload = None
                hasher = hashlib.sha256()
                while chunk := source.read(1024 * 1024):
                    hasher.update(chunk)
                digest = hasher.hexdigest()
            checksum = f"sha256:{digest}"
            members.append((path, info.size, info.size, checksum, payload))
    manifest, fragments, omitted = _process_members(members, context, archive_depth)
    content = "tar_manifest=" + _json_dump(manifest) + "\n" + "\n".join(fragments)
    return content, omitted


def process_gzip(
    asset_id: str,
    data: bytes,
    context: AssetProcessingContext,
    *,
    archive_depth: int = 0,
    filename: str | None = None,
    **_,
) -> ProcessedAsset:
    if archive_depth >= context.limits.max_archive_depth:
        raise ValueError(f"archive nesting exceeds max_archive_depth={context.limits.max_archive_depth}")
    metadata = _gzip_header(data)
    payload = _decompress_gzip(data, context.limits.max_gzip_uncompressed_bytes)
    stream = BytesIO(payload)
    try:
        with tarfile.open(fileobj=stream, mode="r:"):
            is_tar = True
    except tarfile.TarError:
        is_tar = False
    if is_tar:
        nested, omitted = _process_tar(payload, context, archive_depth + 1)
        content = "gzip_header=" + _json_dump(metadata) + "\n" + nested
        if omitted:
            content += f'\n<asset_truncation kind="tar" omitted_members="{omitted}" />'
    else:
        from . import process_asset

        original_name = metadata.get("original_filename") or (Path(filename).stem if filename else None)
        detected = detect_media_type(payload, original_name) or "application/octet-stream"
        nested_id = hashlib.sha256(payload).hexdigest()
        result = process_asset(
            nested_id,
            detected,
            payload,
            context,
            filename=original_name,
            archive_depth=archive_depth + 1,
        )
        content = (
            "gzip_header=" + _json_dump(metadata) + "\n"
            + _nested_result(original_name or "<gzip-payload>", "compressed payload", detected, result)
        )
    return embedded_result(content, "application/gzip", "gzip_asset")


def process_octet_stream(
    asset_id: str,
    data: bytes,
    context: AssetProcessingContext,
    *,
    filename: str | None = None,
    archive_depth: int = 0,
    **_,
) -> ProcessedAsset:
    maximum = context.limits.max_octet_stream_bytes
    if maximum is not None and len(data) > maximum:
        raise ValueError(f"unknown binary exceeds max_octet_stream_bytes={maximum}")
    detected = detect_media_type(data, filename)
    if detected is not None and detected != "application/octet-stream":
        from . import process_asset

        nested_id = hashlib.sha256(data).hexdigest()
        inner = process_asset(
            nested_id,
            detected,
            data,
            context,
            filename=filename,
            archive_depth=archive_depth,
        )
        fragment = (
            f'<detected_binary_asset declared_media_type="application/octet-stream" '
            f'detected_media_type="{detected}">\n{inner.question_fragment}\n</detected_binary_asset>'
        )
        return ProcessedAsset(fragment, inner.relative_path, "application/octet-stream")

    relative_path = context.write_binary_artifact(data)
    strings = [
        match.decode("ascii")
        for match in re.findall(rb"[\x20-\x7e]{4,}", data)
    ][:context.limits.max_printable_strings]
    metadata = {
        "relative_path": relative_path,
        "byte_size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "printable_strings": strings,
    }
    result = embedded_result(_json_dump(metadata, indent=2), "application/octet-stream", "unknown_binary_asset")
    return ProcessedAsset(result.question_fragment, relative_path, result.media_type)
