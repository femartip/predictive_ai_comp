"""Shared types, limits, cache, and lazy Hugging Face model runners."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Literal, Sequence

from huggingface_hub import HfApi


AssetFamily = Literal["image", "audio", "video", "text", "application"]

DEFAULT_IMAGE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_AUDIO_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
DEFAULT_VIDEO_MODEL = "Qwen/Qwen2.5-Omni-7B"
LOCAL_PROCESSOR_REVISION = "asset-processors-v1"


def decode_audio_waveform(data: bytes, sampling_rate: int):
    """Decode the first MP3/WAV audio stream to mono float32 samples in memory."""
    try:
        import av
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Audio processing requires PyAV. Run `poetry install`.") from error

    chunks = []
    with av.open(BytesIO(data)) as container:
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            raise ValueError("audio asset contains no decodable audio stream")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sampling_rate)
        for frame in container.decode(audio_streams[0]):
            chunks.extend(resampler.resample(frame))
        chunks.extend(resampler.resample(None))
    if not chunks:
        raise ValueError("audio asset decoded to no samples")
    return np.concatenate([
        frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
        for frame in chunks
    ])


@dataclass(frozen=True)
class AssetLimits:
    """Optional semantic limits; ``None`` always means unlimited."""

    max_csv_bytes: int | None = None
    max_markdown_bytes: int | None = None
    max_plain_text_bytes: int | None = None
    max_python_bytes: int | None = None
    max_shellscript_bytes: int | None = None
    max_json_bytes: int | None = None
    max_ndjson_bytes: int | None = None
    max_sql_bytes: int | None = None
    max_pdf_pages: int | None = None
    max_parquet_rows: int | None = None
    max_sqlite_rows_per_table: int | None = None
    max_npy_elements: int | None = None
    max_archive_entries: int | None = None
    max_archive_uncompressed_bytes: int | None = None
    max_gzip_uncompressed_bytes: int | None = None
    max_octet_stream_bytes: int | None = None
    max_archive_depth: int = 3
    max_video_frames: int = 16
    max_obj_render_faces: int = 50_000
    max_printable_strings: int = 200

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive or None")


@dataclass(frozen=True)
class ProcessedAsset:
    """The piece of an item produced from one asset."""

    question_fragment: str
    relative_path: str | None
    media_type: str


class AssetDescriptionCache:
    """Persistent cache keyed by the source asset and exact processing recipe."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS asset_descriptions (
                asset_id TEXT NOT NULL,
                media_type TEXT NOT NULL,
                family TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                question_fragment TEXT NOT NULL,
                relative_path TEXT,
                created_at_utc TEXT NOT NULL,
                PRIMARY KEY (
                    asset_id, media_type, family, model_id,
                    model_revision, prompt_version
                )
            )
            """
        )
        self.connection.commit()

    def get(
        self,
        asset_id: str,
        media_type: str,
        family: AssetFamily,
        model_id: str,
        model_revision: str,
        prompt_version: str,
    ) -> ProcessedAsset | None:
        row = self.connection.execute(
            """
            SELECT question_fragment, relative_path, media_type
            FROM asset_descriptions
            WHERE asset_id = ? AND media_type = ? AND family = ?
              AND model_id = ? AND model_revision = ? AND prompt_version = ?
            """,
            (asset_id, media_type, family, model_id, model_revision, prompt_version),
        ).fetchone()
        return ProcessedAsset(*row) if row is not None else None

    def put(
        self,
        asset_id: str,
        family: AssetFamily,
        model_id: str,
        model_revision: str,
        prompt_version: str,
        result: ProcessedAsset,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO asset_descriptions (
                asset_id, media_type, family, model_id, model_revision,
                prompt_version, question_fragment, relative_path, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                result.media_type,
                family,
                model_id,
                model_revision,
                prompt_version,
                result.question_fragment,
                result.relative_path,
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class AssetProcessingContext:
    """Own configuration, artifacts, cache, and at most one loaded 7B model."""

    def __init__(
        self,
        cache_path: Path,
        artifact_root: Path | None = None,
        limits: AssetLimits | None = None,
        image_model_id: str | None = None,
        audio_model_id: str | None = None,
        video_model_id: str | None = None,
        image_model_revision: str | None = None,
        audio_model_revision: str | None = None,
        video_model_revision: str | None = None,
    ):
        self.cache = AssetDescriptionCache(cache_path)
        self.artifact_root = artifact_root or cache_path.parent / "assets"
        self.limits = limits or AssetLimits()
        self.model_ids: dict[AssetFamily, str] = {
            "image": image_model_id or os.getenv("AIMS_IMAGE_MODEL", DEFAULT_IMAGE_MODEL),
            "audio": audio_model_id or os.getenv("AIMS_AUDIO_MODEL", DEFAULT_AUDIO_MODEL),
            "video": video_model_id or os.getenv("AIMS_VIDEO_MODEL", DEFAULT_VIDEO_MODEL),
            "text": "local/text-asset-processors",
            "application": "local/application-asset-processors",
        }
        self.requested_revisions: dict[AssetFamily, str | None] = {
            "image": image_model_revision or os.getenv("AIMS_IMAGE_MODEL_REVISION"),
            "audio": audio_model_revision or os.getenv("AIMS_AUDIO_MODEL_REVISION"),
            "video": video_model_revision or os.getenv("AIMS_VIDEO_MODEL_REVISION"),
            "text": LOCAL_PROCESSOR_REVISION,
            "application": LOCAL_PROCESSOR_REVISION,
        }
        self.resolved_revisions: dict[AssetFamily, str] = {}
        self._models: dict[AssetFamily, object] = {}
        self._processors: dict[AssetFamily, object] = {}

    def model_revision(self, family: AssetFamily) -> str:
        """Resolve mutable Hugging Face names; local processors use a code-policy revision."""
        if family not in self.resolved_revisions:
            requested = self.requested_revisions[family]
            if family in {"text", "application"}:
                self.resolved_revisions[family] = requested or LOCAL_PROCESSOR_REVISION
            else:
                info = HfApi().model_info(self.model_ids[family], revision=requested)
                if not info.sha:
                    raise RuntimeError(f"Hugging Face returned no revision for {self.model_ids[family]}")
                self.resolved_revisions[family] = info.sha
        return self.resolved_revisions[family]

    def recipe_version(self, version: str) -> str:
        """Include all configured limits in cache identity."""
        encoded = json.dumps(asdict(self.limits), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        return f"{version};limits={digest}"

    def cached(
        self,
        asset_id: str,
        media_type: str,
        family: AssetFamily,
        recipe_version: str,
    ) -> ProcessedAsset | None:
        return self.cache.get(
            asset_id,
            media_type,
            family,
            self.model_ids[family],
            self.model_revision(family),
            recipe_version,
        )

    def store(
        self,
        asset_id: str,
        family: AssetFamily,
        recipe_version: str,
        result: ProcessedAsset,
    ) -> None:
        self.cache.put(
            asset_id,
            family,
            self.model_ids[family],
            self.model_revision(family),
            recipe_version,
            result,
        )

    def write_binary_artifact(self, data: bytes) -> str:
        """Persist an unknown binary by content hash and return its output-relative path."""
        digest = hashlib.sha256(data).hexdigest()
        directory = self.artifact_root / "binary"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{digest}.bin"
        if not path.exists():
            temporary = path.with_suffix(".bin.tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
        return path.relative_to(self.artifact_root.parent).as_posix()

    def _release_other_family(self, keep: AssetFamily) -> None:
        """Avoid keeping multiple large multimodal models in memory together."""
        for family in list(self._models):
            if family != keep:
                self._models.pop(family, None)
                self._processors.pop(family, None)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _image_components(self):
        if "image" not in self._models:
            self._release_other_family("image")
            try:
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            except ImportError as error:
                raise RuntimeError(
                    "Image processing requires transformers, torch, accelerate, and Pillow. "
                    "Run `poetry install`."
                ) from error
            revision = self.model_revision("image")
            model_id = self.model_ids["image"]
            self._processors["image"] = AutoProcessor.from_pretrained(model_id, revision=revision)
            self._models["image"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype="auto",
                device_map="auto",
            ).eval()
        return self._processors["image"], self._models["image"]

    def _audio_components(self):
        if "audio" not in self._models:
            self._release_other_family("audio")
            try:
                from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
            except ImportError as error:
                raise RuntimeError(
                    "Audio processing requires transformers, torch, accelerate, and PyAV. "
                    "Run `poetry install`."
                ) from error
            revision = self.model_revision("audio")
            model_id = self.model_ids["audio"]
            self._processors["audio"] = AutoProcessor.from_pretrained(model_id, revision=revision)
            self._models["audio"] = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype="auto",
                device_map="auto",
            ).eval()
        return self._processors["audio"], self._models["audio"]

    def _video_components(self):
        if "video" not in self._models:
            self._release_other_family("video")
            try:
                from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
            except ImportError as error:
                raise RuntimeError(
                    "Video processing requires transformers, torch, accelerate, torchvision, and PyAV. "
                    "Run `poetry install`."
                ) from error
            revision = self.model_revision("video")
            model_id = self.model_ids["video"]
            self._processors["video"] = Qwen2_5OmniProcessor.from_pretrained(model_id, revision=revision)
            self._models["video"] = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype="auto",
                device_map="auto",
            ).eval()
        return self._processors["video"], self._models["video"]

    def describe_images(self, images: Sequence[object], prompt: str) -> str:
        """Generate text from one image or an ordered sequence of frames/pages."""
        processor, model = self._image_components()
        content = []
        for position, _ in enumerate(images, 1):
            if len(images) > 1:
                content.append({"type": "text", "text": f"Image {position}, in sequence:"})
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=list(images), padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        return processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def describe_audio(self, data: bytes, prompt: str) -> str:
        """Decode MP3 or WAV in memory and ask the local audio model for text."""
        processor, model = self._audio_components()
        sampling_rate = processor.feature_extractor.sampling_rate
        waveform = decode_audio_waveform(data, sampling_rate)
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "in-memory-audio"},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=text,
            audio=[waveform],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        generated = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        generated = generated[:, inputs.input_ids.size(1):]
        return processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def describe_video(self, data: bytes, prompt: str) -> str:
        """Process MP4 video and its soundtrack jointly, using only a temporary source file."""
        processor, model = self._video_components()
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary:
                temporary.write(data)
                temporary_name = temporary.name
            messages = [{"role": "user", "content": [
                {
                    "type": "video",
                    "video": Path(temporary_name).as_uri(),
                    "nframes": self.limits.max_video_frames,
                },
                {"type": "text", "text": prompt},
            ]}]
            inputs = processor.apply_chat_template(
                messages,
                load_audio_from_video=True,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            ).to(model.device)
            generated = model.generate(
                **inputs,
                use_audio_in_video=True,
                max_new_tokens=2048,
                do_sample=False,
            )
            generated = generated[:, inputs.input_ids.size(1):]
            return processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)

    def provenance(self) -> dict:
        return {
            "models": {
                family: {
                    "model_id": self.model_ids[family],
                    "model_revision": revision,
                }
                for family, revision in sorted(self.resolved_revisions.items())
            },
            "limits": asdict(self.limits),
        }

    def close(self) -> None:
        self._models.clear()
        self._processors.clear()
        self.cache.close()


def description_result(description: str, media_type: str) -> ProcessedAsset:
    """Wrap generated text in an explicit family tag for the item question."""
    family = media_type.split("/", 1)[0]
    fragment = f'<{family}_description media_type="{media_type}">\n{description.strip()}\n</{family}_description>'
    return ProcessedAsset(question_fragment=fragment, relative_path=None, media_type=media_type)


def embedded_result(content: str, media_type: str, tag: str) -> ProcessedAsset:
    fragment = f'<{tag} media_type="{media_type}">\n{content}\n</{tag}>'
    return ProcessedAsset(question_fragment=fragment, relative_path=None, media_type=media_type)


def evenly_spaced_indices(total: int, maximum: int = 8) -> list[int]:
    """Select at most ``maximum`` ordered indices, always including both ends."""
    if total < 1:
        raise ValueError("a sequence must have at least one element")
    count = min(total, maximum)
    if count == 1:
        return [0]
    return [round(position * (total - 1) / (count - 1)) for position in range(count)]
