"""ScreenSpot-Pro loader.

Source
------
HuggingFace dataset: ``likaixin/ScreenSpot-Pro``

ScreenSpot-Pro is the high-resolution, professional-software follow-up to
ScreenSpot-v2. It covers CAD, IDE, scientific, and creative applications
at 4K+ resolution, which stresses grounding models differently from v2.

Revision pin
------------
``REVISION`` tracks main by default. Override with
``QONTINUI_SCREENSPOT_PRO_REV`` for reproducibility.

Schema (as observed on main, 2025-Q1)
-------------------------------------
Each split row has roughly::

    {
        "image": <PIL.Image.Image>,
        "bbox": [x1, y1, x2, y2],          # absolute pixel xyxy
        "instruction": "<instruction>",
        "application": "photoshop" | "blender" | ...,
        "group": "CAD" | "IDE" | "Scientific" | "Creative" | "Office" | ...,
        "platform": "windows" | "macos" | ...,
        "ui_type": "icon" | "text",
    }

Bbox convention used here
-------------------------
Input:  ``[x1, y1, x2, y2]`` absolute pixels (xyxy).
Output: normalized ``(cx, cy)`` via ``bbox_to_center_xyxy``.

We map ``ui_type`` to a coarse component label (``Icon`` / ``Label``)
so the per-component breakdown in ``grounding_eval.py`` works. The
``group`` field is not currently used because the breakdown regex
only recognizes component-level names.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ._common import (
    bbox_to_center_xyxy,
    cached_jsonl_path,
    make_vlm_sample,
    save_image_from_record,
    write_jsonl,
)

logger = logging.getLogger(__name__)

BENCHMARK_NAME = "screenspot_pro"
HF_DATASET_ID = "likaixin/ScreenSpot-Pro"
REVISION: str | None = None
SPLIT = "test"


_COMPONENT_MAP = {
    "icon": "Icon",
    "text": "Label",
}


def _normalize_row(row: dict, cache_images_dir: Path, idx: int) -> dict | None:
    bbox = row.get("bbox")
    instruction = row.get("instruction") or ""
    image_field = row.get("image")

    if bbox is None or image_field is None or not instruction:
        return None

    try:
        image_path = save_image_from_record(
            image_field,
            cache_images_dir,
            fallback_name=f"screenspot_pro_{idx:06d}",
        )
    except Exception as exc:
        logger.warning("Skipping row %d: image save failed: %s", idx, exc)
        return None

    try:
        from PIL import Image

        with Image.open(image_path) as im:
            w, h = im.size
    except Exception as exc:
        logger.warning("Skipping row %d: image open failed: %s", idx, exc)
        return None

    cx, cy = bbox_to_center_xyxy(tuple(bbox), float(w), float(h))
    comp = _COMPONENT_MAP.get(str(row.get("ui_type", "")).lower())
    return make_vlm_sample(
        image_path=str(image_path),
        instruction=instruction,
        center_xy=(cx, cy),
        component_type=comp,
    )


def load(cache_dir: Path) -> Path:
    """Download (or load from HF cache) ScreenSpot-Pro and emit a VLM jsonl."""
    out_path = cached_jsonl_path(cache_dir, BENCHMARK_NAME)
    if out_path.exists():
        logger.info("Using cached %s at %s", BENCHMARK_NAME, out_path)
        return out_path

    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            f"Loading external benchmark '{BENCHMARK_NAME}' requires the "
            "'datasets' package. Install with: pip install datasets"
        ) from exc

    revision = os.environ.get("QONTINUI_SCREENSPOT_PRO_REV", REVISION)
    logger.info(
        "Loading HF dataset %s (revision=%s, split=%s)",
        HF_DATASET_ID,
        revision or "main",
        SPLIT,
    )
    ds = load_dataset(HF_DATASET_ID, revision=revision, split=SPLIT)

    cache_images_dir = out_path.parent / "images"
    samples: list[dict] = []
    for idx, row in enumerate(ds):
        norm = _normalize_row(row, cache_images_dir, idx)
        if norm is not None:
            samples.append(norm)

    if not samples:
        raise RuntimeError(f"No usable samples from {HF_DATASET_ID}")

    write_jsonl(out_path, samples)
    return out_path
