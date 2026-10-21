"""Processors for supported video MIME types."""

from __future__ import annotations

from .base import AssetProcessingContext, ProcessedAsset, description_result


VIDEO_PROMPT = """Describe this video exhaustively and objectively for a reader who cannot see or hear it.
Use a timestamped timeline. Cover scenes, actions, people, objects, camera changes, visible text, speech,
speaker changes, music, ambience, silence, and other relevant sounds. The supplied visual stream contains
up to {max_frames} evenly sampled chronological frames including the beginning and end. Do not infer the benchmark
question and do not answer any task."""


def process_mp4(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    prompt = VIDEO_PROMPT.format(max_frames=context.limits.max_video_frames)
    return description_result(context.describe_video(data, prompt), "video/mp4")
