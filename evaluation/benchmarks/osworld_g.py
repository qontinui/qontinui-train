"""OSWorld-G grounding split loader.

Source
------
HuggingFace dataset: ``xlangai/OSWorld-G``

OSWorld-G is the grounding benchmark derived from OSWorld (the agent-benchmark
for real desktop environments). The grounding split tests whether a model
can locate a target element given a free-form instruction on realistic
desktop screenshots.

Revision pin
------------
``REVISION`` tracks main; override via ``QONTINUI_OSWORLD_G_REV``.
Because the OSWorld-G dataset card has moved between the ``xlangai`` and
community mirrors over time, we prefer ``xlangai/OSWorld-G`` and document
this choice here rather than accepting silent drift.

Schema (as observed on main, 2025-Q1)
-------------------------------------
Each split row has roughly::

    {
        "image": <PIL.Image.Image>,
        "instruction": "<instruction>",
        "bbox": [x1, y1, x2, y2]  OR  [cx, cy, w, h]  depending on revision,
        "target_type": "button" | "textbox" | ... (optional),
    }

The bbox layout in OSWorld-G has shifted between revisions. We detect
``xyxy`` vs ``cxcywh`` heuristically: if width/height derived from the
first-two / last-two pair is positive and fits within the image, we treat
as ``xyxy``; otherwise we fall back to ``xywh`` (x, y, w, h top-left).

Bbox convention used here
-------------------------
Normalized to ``(cx, cy)`` in ``[0, 1]^2``. We try xyxy first (canonical),
then xywh. ``cxcywh`` is not supported until a concrete revision uses it;
revisit if a loader sample is obviously off-center.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ._common import (
    bbox_to_center_xywh,
    bbox_to_center_xyxy,
    cached_jsonl_path,
    make_vlm_sample,
    save_image_from_record,
    write_jsonl,
)

logger = logging.getLogger(__name__)

BENCHMARK_NAME = "osworld_g"
HF_DATASET_ID = "xlangai/OSWorld-G"
REVISION: str | None = None
SPLIT = "test"


def _center_from_bbox(
    bbox: tuple[float, float, float, float],
    image_width: float,
    image_height: float,
) -> tuple[float, float]:
    """Best-effort bbox→center, detecting xyxy vs xywh."""
    a, b, c, d = bbox
    # xyxy sanity: x2 > x1 and y2 > y1 and within image
    if c > a and d > b and c <= image_width * 1.01 and d <= image_height * 1.01:
        return bbox_to_center_xyxy(bbox, image_width, image_height)
    # fall back to xywh
    return bbox_to_center_xywh(bbox, image_width, image_height)


def _normalize_row(row: dict, cache_images_dir: Path, idx: int) -> dict | None:
    bbox = row.get("bbox")
    instruction = row.get("instruction") or row.get("query") or ""
    image_field = row.get("image")

    if bbox is None or image_field is None or not instruction:
        return None

    try:
        image_path = save_image_from_record(
            image_field,
            cache_images_dir,
            fallback_name=f"osworld_g_{idx:06d}",
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

    cx, cy = _center_from_bbox(tuple(bbox), float(w), float(h))

    target_type = row.get("target_type") or row.get("element_type")
    comp = None
    if isinstance(target_type, str) and target_type:
        # Capitalize so extract_component_type can match.
        comp = target_type.strip().capitalize()

    return make_vlm_sample(
        image_path=str(image_path),
        instruction=instruction,
        center_xy=(cx, cy),
        component_type=comp,
    )


def load(cache_dir: Path) -> Path:
    """Download (or load from HF cache) OSWorld-G and emit a VLM jsonl."""
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

    revision = os.environ.get("QONTINUI_OSWORLD_G_REV", REVISION)
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
