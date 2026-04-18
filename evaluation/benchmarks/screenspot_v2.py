"""ScreenSpot-v2 loader.

Source
------
HuggingFace dataset: ``rootsautomation/ScreenSpot``

This is the canonical copy of the ScreenSpot-v2 benchmark used by most
published GUI grounding papers (UI-TARS, SeeClick, OS-Atlas, etc.).
There are forks with slightly different splits — we pin to
``rootsautomation/ScreenSpot`` because its schema is well-documented and
widely referenced.

Revision pin
------------
``REVISION`` below is set to ``None`` to track main; callers that need
reproducibility should override by setting ``QONTINUI_SCREENSPOT_V2_REV``
in the environment to a commit SHA from the HF dataset card.

Schema (as observed on main, 2025-Q1)
-------------------------------------
Each split row has roughly::

    {
        "file_name": "<image_filename>.png",
        "image": <PIL.Image.Image>,
        "bbox": [x1, y1, x2, y2],         # absolute pixel xyxy
        "instruction": "<natural-language instruction>",
        "data_type": "icon" | "text",     # coarse component group
        "data_source": "mobile" | "desktop" | "web",
    }

We normalize to VLM-messages:
- bbox center in normalized ``[0,1]^2`` as the assistant ``<point>`` output
- ``data_type`` is mapped to a coarse component-type label
  (``Icon`` or ``Label``) so grounding_eval's per-component breakdown
  lights up.

Bbox convention used here
-------------------------
Input:  ``[x1, y1, x2, y2]`` absolute pixels (xyxy).
Output: normalized ``(cx, cy)`` via ``bbox_to_center_xyxy``.
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

BENCHMARK_NAME = "screenspot_v2"
HF_DATASET_ID = "rootsautomation/ScreenSpot"
REVISION: str | None = None  # override via QONTINUI_SCREENSPOT_V2_REV env var
SPLIT = "test"


_COMPONENT_MAP = {
    "icon": "Icon",
    "text": "Label",
}


def _normalize_row(row: dict, cache_images_dir: Path, idx: int) -> dict | None:
    """Convert a single HF row to a VLM-messages dict. Returns None on skip."""
    bbox = row.get("bbox")
    instruction = row.get("instruction") or ""
    image_field = row.get("image")

    if bbox is None or image_field is None or not instruction:
        return None

    try:
        image_path = save_image_from_record(
            image_field,
            cache_images_dir,
            fallback_name=f"screenspot_v2_{idx:06d}",
        )
    except Exception as exc:
        logger.warning("Skipping row %d: image save failed: %s", idx, exc)
        return None

    # Pull image dims from the saved file (works for PIL.Image too if lazy).
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            w, h = im.size
    except Exception as exc:
        logger.warning("Skipping row %d: image open failed: %s", idx, exc)
        return None

    cx, cy = bbox_to_center_xyxy(tuple(bbox), float(w), float(h))
    comp = _COMPONENT_MAP.get(str(row.get("data_type", "")).lower())
    return make_vlm_sample(
        image_path=str(image_path),
        instruction=instruction,
        center_xy=(cx, cy),
        component_type=comp,
    )


def load(cache_dir: Path) -> Path:
    """Download (or load from HF cache) ScreenSpot-v2 and emit a VLM jsonl.

    Parameters
    ----------
    cache_dir:
        Root directory for this repo's benchmark cache.

    Returns
    -------
    Path to the cached ``test.jsonl`` for this benchmark.
    """
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

    revision = os.environ.get("QONTINUI_SCREENSPOT_V2_REV", REVISION)
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
