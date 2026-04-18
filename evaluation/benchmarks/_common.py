"""Shared helpers for external-benchmark loaders.

Kept internal (leading underscore) because this is an implementation detail
of the benchmark-loader package, not a stable public API.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def bbox_to_center_xyxy(
    bbox: tuple[float, float, float, float],
    image_width: float,
    image_height: float,
) -> tuple[float, float]:
    """Convert an ``[x1, y1, x2, y2]`` absolute-pixel bbox to a normalized center.

    Returns ``(cx, cy)`` in ``[0, 1]^2``, using the image extents for
    normalization. The VLM assistant message expects this form as
    ``<point>cx cy</point>``.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / image_width
    cy = (y1 + y2) / 2.0 / image_height
    return (max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)))


def bbox_to_center_xywh(
    bbox: tuple[float, float, float, float],
    image_width: float,
    image_height: float,
) -> tuple[float, float]:
    """Convert an ``[x, y, w, h]`` absolute-pixel bbox to a normalized center."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / image_width
    cy = (y + h / 2.0) / image_height
    return (max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)))


def make_vlm_sample(
    image_path: str,
    instruction: str,
    center_xy: tuple[float, float],
    *,
    component_type: str | None = None,
) -> dict[str, Any]:
    """Build a single VLM-SFT messages dict matching grounding_to_vlm's output.

    When *component_type* is given, it's folded into the user instruction so
    ``extract_component_type`` in ``grounding_eval.py`` can pick it up for the
    per-component breakdown.
    """
    text = instruction.strip()
    if component_type and component_type.lower() not in text.lower():
        # Prepend the component capitalization grounding_eval regex looks for.
        text = f"{component_type}: {text}"

    cx, cy = center_xy
    uri = image_path
    if not uri.startswith("file:///") and not uri.startswith("http"):
        # Normalize local path to file:// URI (grounding_eval strips the scheme).
        uri = f"file:///{Path(image_path).as_posix().lstrip('/')}"

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": uri},
                    {"type": "text", "text": text},
                ],
            },
            {
                "role": "assistant",
                "content": f"<point>{cx:.4f} {cy:.4f}</point>",
            },
        ]
    }


def write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    """Write *samples* to *path* one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for obj in samples:
            fh.write(json.dumps(obj, separators=(",", ":")))
            fh.write("\n")
    logger.info("Wrote %d samples to %s", len(samples), path)


def cached_jsonl_path(cache_dir: Path, benchmark_name: str) -> Path:
    """Return the canonical cached-jsonl location for a benchmark."""
    return cache_dir / benchmark_name / "test.jsonl"


def save_image_from_record(
    image_field: Any,
    cache_images_dir: Path,
    fallback_name: str,
) -> Path:
    """Persist an HF image field to disk and return the absolute path.

    *image_field* is typically a ``PIL.Image.Image`` (when ``datasets`` is used
    with image feature decoding on) or a ``dict`` with ``path``/``bytes`` keys.
    """
    cache_images_dir.mkdir(parents=True, exist_ok=True)

    # Already a path on disk
    if isinstance(image_field, (str, Path)):
        p = Path(image_field)
        if p.exists():
            return p.resolve()

    # dict form (common for HF load_dataset without image decoding)
    if isinstance(image_field, dict):
        if image_field.get("path") and Path(image_field["path"]).exists():
            return Path(image_field["path"]).resolve()
        if image_field.get("bytes"):
            out = cache_images_dir / f"{fallback_name}.png"
            out.write_bytes(image_field["bytes"])
            return out.resolve()

    # PIL.Image.Image
    save = getattr(image_field, "save", None)
    if callable(save):
        out = cache_images_dir / f"{fallback_name}.png"
        save(out)
        return out.resolve()

    raise ValueError(f"Cannot persist image field of type {type(image_field).__name__}")
