"""Per-domain held-out test split loader.

Reads the VGA correction log at ``corrections_dir/corrections.jsonl``
and returns, for each ``target_process`` (domain), a list of evaluation
samples in the same shape that ``grounding_eval.load_test_samples``
produces — i.e. ``{"messages": [user, assistant]}`` dicts with a
``<point>x y</point>`` assistant turn.

Selection rule (plan §13 v6 retrain trigger)
--------------------------------------------
Only entries that meet ALL of:

1. ``source == "builder"`` (i.e. the user confirmed/corrected during
   state-machine building, not from a runtime drift event); OR
   ``test_reserved is True`` (explicitly marked for holdout).
2. ``private`` is False (or respected via ``include_private=True``).
3. Image file on disk.

The strict default filter is there so runtime-correction noise does not
leak into the held-out set. Runtime corrections are retraining fuel; the
held-out set is the regression gate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """VLM-SFT-shaped sample bundled with its originating domain.

    The ``messages`` field is what ``grounding_eval.evaluate_model``
    consumes. The other fields are attached for diagnostics — the eval
    loop ignores them.
    """

    messages: list[dict[str, Any]]
    target_process: str
    image_path: Path
    image_sha: str
    prompt: str

    def to_vlm_dict(self) -> dict[str, Any]:
        """Return the bare ``{"messages": [...]}`` dict the eval loop needs."""
        return {"messages": self.messages}


# ---------------------------------------------------------------------------
# Prompt + coordinate conversion — mirror the exporter exactly.
# ---------------------------------------------------------------------------

_GROUND_PROMPT_TEMPLATE = (
    "Locate the following element in the screenshot and output its "
    "position as <point>x y</point> where x and y are integers between "
    "0 and 1000 (normalized coordinates).\n\nElement: {prompt}"
)


def _read_image_dimensions(image_path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        logger.error("Pillow required for image dimensions: %s", exc)
        return None

    try:
        with Image.open(image_path) as im:
            return im.size
    except (OSError, ValueError) as exc:
        logger.warning("Cannot open image %s: %s", image_path, exc)
        return None


def _normalized_thousand(
    bbox: dict[str, Any], image_w: int, image_h: int
) -> tuple[int, int]:
    cx = bbox["x"] + bbox["w"] / 2.0
    cy = bbox["y"] + bbox["h"] / 2.0
    if image_w <= 0 or image_h <= 0:
        return 0, 0
    nx = int(round(cx / image_w * 1000))
    ny = int(round(cy / image_h * 1000))
    return max(0, min(1000, nx)), max(0, min(1000, ny))


def _entry_to_eval_sample(entry: dict[str, Any]) -> EvalSample | None:
    image_path_str = entry.get("image_path")
    prompt = entry.get("prompt")
    bbox = entry.get("corrected_bbox")
    target_process = entry.get("target_process")
    image_sha = entry.get("image_sha", "")

    if not (
        image_path_str
        and isinstance(prompt, str)
        and isinstance(bbox, dict)
        and isinstance(target_process, str)
    ):
        return None

    image_path = Path(image_path_str)
    if not image_path.exists():
        logger.warning("Image not found, skipping: %s", image_path)
        return None

    dims = _read_image_dimensions(image_path)
    if dims is None:
        return None
    image_w, image_h = dims
    nx, ny = _normalized_thousand(bbox, image_w, image_h)

    abs_posix = image_path.resolve().as_posix()
    uri = f"file:///{abs_posix.lstrip('/')}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": uri},
                {
                    "type": "text",
                    "text": _GROUND_PROMPT_TEMPLATE.format(prompt=prompt),
                },
            ],
        },
        {
            "role": "assistant",
            "content": f"<point>{nx} {ny}</point>",
        },
    ]

    return EvalSample(
        messages=messages,
        target_process=target_process,
        image_path=image_path,
        image_sha=str(image_sha),
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_per_domain_splits(
    corrections_dir: Path,
    *,
    include_private: bool = False,
    sources: tuple[str, ...] | None = None,
) -> dict[str, list[EvalSample]]:
    """Load VGA corrections grouped by ``target_process``.

    Args:
        corrections_dir: Directory containing ``corrections.jsonl`` (the
            same dir ``CorrectionLogger`` writes to).
        include_private: Pass True to include entries flagged private.
            Default False.
        sources: Tuple of acceptable ``source`` values. Defaults to
            ``("builder",)`` — runtime corrections are excluded from the
            regression gate. Entries with ``test_reserved=true`` are
            always included regardless of source.

    Returns:
        Mapping ``{target_process: [EvalSample, ...]}``. Domains with
        zero eligible samples are omitted.
    """
    if sources is None:
        sources = ("builder",)

    corrections_jsonl = corrections_dir / "corrections.jsonl"
    if not corrections_jsonl.exists():
        logger.warning("Correction log not found: %s", corrections_jsonl)
        return {}

    per_domain: dict[str, list[EvalSample]] = {}

    with corrections_jsonl.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d: invalid JSON (%s), skipping", lineno, exc)
                continue

            # Privacy filter.
            if not include_private:
                if entry.get("private", True):
                    continue
                ip = entry.get("image_path")
                if ip and Path(f"{ip}.private").exists():
                    continue

            # Source / reservation filter.
            is_reserved = bool(entry.get("test_reserved", False))
            if not is_reserved and entry.get("source") not in sources:
                continue

            sample = _entry_to_eval_sample(entry)
            if sample is None:
                continue

            per_domain.setdefault(sample.target_process, []).append(sample)

    for domain, samples in per_domain.items():
        logger.info("external_app_splits: domain=%s samples=%d", domain, len(samples))
    return per_domain
