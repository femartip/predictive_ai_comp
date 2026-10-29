"""One independently editable processor for every supported image MIME type."""

from __future__ import annotations

import os
import tempfile
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .base import AssetProcessingContext, ProcessedAsset, description_result, evenly_spaced_indices


BASE_IMAGE_PROMPT = """Describe the supplied image exhaustively and objectively for a reader who cannot see it.
Include all visible text verbatim, objects, people, spatial relationships, diagrams, axes, legends, colors,
symbols, and details that could be relevant to interpreting the image. Do not infer the benchmark question,
do not answer any task, and do not omit a detail merely because it seems unimportant."""


def _opened_image(data: bytes) -> Image.Image:
    with Image.open(BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).copy()
    return image


def _checkerboard(image: Image.Image, square: int = 16) -> Image.Image:
    """Make transparency visible instead of silently flattening it to one color."""
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (238, 238, 238, 255))
    draw = ImageDraw.Draw(background)
    for y in range(0, rgba.height, square):
        for x in range(0, rgba.width, square):
            if (x // square + y // square) % 2:
                draw.rectangle((x, y, x + square - 1, y + square - 1), fill=(190, 190, 190, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def _rgb(image: Image.Image, preserve_transparency: bool = False) -> Image.Image:
    has_alpha = image.mode in {"RGBA", "LA"} or "transparency" in image.info
    if preserve_transparency and has_alpha:
        return _checkerboard(image)
    return image.convert("RGB")


def _png_roundtrip(image: Image.Image) -> Image.Image:
    output = BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    with Image.open(output) as decoded:
        return decoded.convert("RGB").copy()


def _sequence_frames(data: bytes, preserve_transparency: bool) -> tuple[list[Image.Image], int]:
    with Image.open(BytesIO(data)) as source:
        total = getattr(source, "n_frames", 1)
        frames = []
        for index in evenly_spaced_indices(total):
            source.seek(index)
            frame = ImageOps.exif_transpose(source.convert("RGBA")).copy()
            frames.append(_rgb(frame, preserve_transparency=preserve_transparency))
    return frames, total


def process_jpeg(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    image = _rgb(_opened_image(data))
    return description_result(context.describe_images([image], BASE_IMAGE_PROMPT), "image/jpeg")


def process_png(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    image = _rgb(_opened_image(data), preserve_transparency=True)
    prompt = BASE_IMAGE_PROMPT + "\nA checkerboard indicates transparent or partially transparent regions; describe them."
    return description_result(context.describe_images([image], prompt), "image/png")


def process_bmp(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    # The PNG exists only in memory; the final dataset does not retain a converted file.
    image = _png_roundtrip(_rgb(_opened_image(data)))
    prompt = BASE_IMAGE_PROMPT + "\nThe source was a BMP losslessly decoded through an in-memory PNG."
    return description_result(context.describe_images([image], prompt), "image/bmp")


def process_gif(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    frames, total = _sequence_frames(data, preserve_transparency=True)
    if total == 1:
        prompt = BASE_IMAGE_PROMPT + "\nThe source is a static GIF; a checkerboard indicates transparency."
    else:
        prompt = (
            BASE_IMAGE_PROMPT
            + f"\nThe {len(frames)} supplied images are evenly sampled, chronological frames from an animated "
            f"GIF containing {total} frames. Describe both the content and the motion or changes over time. "
            "A checkerboard indicates transparency."
        )
    return description_result(context.describe_images(frames, prompt), "image/gif")


def process_tiff(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    pages, total = _sequence_frames(data, preserve_transparency=True)
    if total == 1:
        prompt = BASE_IMAGE_PROMPT + "\nThe source is a single-page TIFF."
    else:
        prompt = (
            BASE_IMAGE_PROMPT
            + f"\nThe {len(pages)} supplied images are evenly sampled, ordered pages or slices from a "
            f"{total}-page TIFF. Describe each visible structure and how it changes between pages or slices."
        )
    return description_result(context.describe_images(pages, prompt), "image/tiff")


def process_webp(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    frames, total = _sequence_frames(data, preserve_transparency=True)
    if total == 1:
        prompt = BASE_IMAGE_PROMPT + "\nThe source is a static WebP; a checkerboard indicates transparency."
    else:
        prompt = (
            BASE_IMAGE_PROMPT
            + f"\nThe {len(frames)} supplied images are evenly sampled, chronological frames from an animated "
            f"WebP containing {total} frames. Describe both content and motion. A checkerboard indicates transparency."
        )
    return description_result(context.describe_images(frames, prompt), "image/webp")


def _channel_pixels(channel) -> np.ndarray:
    pixels = getattr(channel, "pixels", channel)
    return np.asarray(pixels, dtype=np.float32)


def _tone_map_rgb(rgb: np.ndarray) -> Image.Image:
    rgb = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    rgb = np.clip(rgb, 0, None)
    positive = rgb[rgb > 0]
    scale = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    scale = max(scale, np.finfo(np.float32).eps)
    mapped = rgb / (rgb + scale)
    mapped = np.power(np.clip(mapped, 0, 1), 1 / 2.2)
    return Image.fromarray(np.rint(mapped * 255).astype(np.uint8), mode="RGB")


def _grayscale_preview(values: np.ndarray, label: str) -> Image.Image:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = values[np.isfinite(values)]
    if finite.size:
        low, high = np.percentile(finite, [1, 99])
    else:
        low, high = 0.0, 1.0
    if high <= low:
        normalized = np.zeros(values.shape, dtype=np.uint8)
    else:
        normalized = np.rint(np.clip((values - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(normalized, mode="L").convert("RGB")
    banner = Image.new("RGB", (image.width, image.height + 24), "white")
    banner.paste(image, (0, 24))
    ImageDraw.Draw(banner).text((4, 4), f"EXR channel: {label}", fill="black")
    return banner


def _exr_previews(data: bytes) -> tuple[list[Image.Image], list[str]]:
    try:
        import OpenEXR
    except ImportError as error:
        raise RuntimeError("EXR processing requires OpenEXR. Run `poetry install`.") from error

    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as temporary:
            temporary.write(data)
            temporary_name = temporary.name
        with OpenEXR.File(temporary_name) as exr:
            channels = exr.channels()
            if "RGB" in channels:
                rgb = _channel_pixels(channels["RGB"])
            elif all(name in channels for name in ("R", "G", "B")):
                rgb = np.stack([_channel_pixels(channels[name]) for name in ("R", "G", "B")], axis=-1)
            else:
                raise ValueError("EXR has no RGB or separate R/G/B channels")

            if rgb.ndim != 3 or rgb.shape[-1] < 3:
                raise ValueError(f"unexpected EXR RGB shape: {rgb.shape}")
            previews = [_png_roundtrip(_tone_map_rgb(rgb[..., :3]))]
            excluded = {"R", "G", "B", "A", "RGB", "RGBA"}
            source_extra_names = sorted(name for name in channels if name.upper() not in excluded)
            extra_names = []
            for name in source_extra_names:
                pixels = _channel_pixels(channels[name])
                if pixels.ndim == 2:
                    previews.append(_png_roundtrip(_grayscale_preview(pixels, name)))
                    extra_names.append(name)
                elif pixels.ndim == 3:
                    for component in range(pixels.shape[-1]):
                        label = f"{name}[{component}]"
                        previews.append(_png_roundtrip(_grayscale_preview(pixels[..., component], label)))
                        extra_names.append(label)
        return previews, extra_names
    finally:
        if temporary_name is not None:
            os.unlink(temporary_name)


def process_exr(asset_id: str, data: bytes, context: AssetProcessingContext, **_) -> ProcessedAsset:
    previews, extra_names = _exr_previews(data)
    extras = ", ".join(extra_names) if extra_names else "none"
    prompt = (
        BASE_IMAGE_PROMPT
        + "\nThe first image is a deterministic 8-bit tone-mapped preview of the source HDR EXR. "
        f"Subsequent labelled images show every scalar non-color channel. Extra channels: {extras}. "
        "Describe the RGB content and the information carried by the additional channels separately."
    )
    return description_result(context.describe_images(previews, prompt), "image/x-exr")
