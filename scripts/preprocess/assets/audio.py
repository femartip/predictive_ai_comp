"""One independently editable processor for every supported audio MIME type."""

from __future__ import annotations

from .base import AssetProcessingContext, ProcessedAsset, description_result


AUDIO_PROMPT = """Describe this audio exhaustively and objectively for a reader who cannot hear it.
Provide a timestamped transcript of all intelligible speech and identify speaker changes when possible.
Also describe relevant non-speech sounds, music, ambience, silence, and their approximate timestamps.
Do not infer the benchmark question and do not answer any task."""


def process_mpeg(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    return description_result(context.describe_audio(data, AUDIO_PROMPT), "audio/mpeg")


def process_wav(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    return description_result(context.describe_audio(data, AUDIO_PROMPT), "audio/wav")
