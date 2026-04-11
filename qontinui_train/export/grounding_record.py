"""Grounding data record schema and JSONL writer.

Defines a shared schema for grounding-model fine-tuning data produced by
two sources:

- **Static**: component-level screenshots from qontinui-web with DOM-extracted
  bounding boxes (source="static", action=None).
- **Dynamic**: native-app trajectory logs captured during workflow execution
  (source="dynamic", action describes the performed GUI action).

Both sources write to the same ``grounding.jsonl`` file via
:class:`GroundingJSONLWriter`, which handles SHA256 exact dedup and optional
perceptual-hash near-dedup.

The grounding schema is intentionally different from the existing
``manifest.jsonl`` used by :class:`TrainingDataExporter`: it inlines element
bounding boxes alongside the image reference because grounding models need
``(image, elements)`` pairs in a single record.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

logger = logging.getLogger(__name__)

# Optional perceptual hashing --------------------------------------------------
try:
    import imagehash

    _IMAGEHASH_AVAILABLE = True
except ImportError:
    _IMAGEHASH_AVAILABLE = False

# Perceptual hash hamming-distance threshold for near-dedup.
_PHASH_HAMMING_THRESHOLD = 4

# Maximum grounding.jsonl size before rotation (bytes).
_MAX_JSONL_BYTES = 100 * 1024 * 1024  # 100 MB


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GroundingElement:
    """A single UI element detected in the screenshot."""

    role: str
    """Semantic role: ``"button"``, ``"textbox"``, ``"menuitem"``, etc."""

    text: str | None
    """Visible text label, if any."""

    bbox: tuple[int, int, int, int]
    """Bounding box as ``(x, y, width, height)`` in pixels."""

    interactable: bool | None = None
    """Optional OmniParser interactability flag."""


@dataclass
class GroundingAction:
    """The GUI action that was performed (dynamic records only)."""

    type: str
    """Action type: ``"click"``, ``"type"``, ``"scroll"``, etc."""

    target_bbox: tuple[int, int, int, int] | None = None
    """Bounding box around the action target, if known."""

    typed_text: str | None = None
    """Text that was typed (for ``"type"`` actions)."""

    success: bool | None = None
    """Whether the action succeeded."""

    success_source: str | None = None
    """How success was determined: ``"wsm"``, ``"pixel_diff"``, ``"record_flag"``."""


@dataclass
class GroundingRecord:
    """One grounding data record — the unit written to ``grounding.jsonl``."""

    image_hash: str
    """SHA256[:16] of the PNG bytes (dedup key)."""

    image_path: str
    """Relative path inside the output directory, e.g. ``"images/<hash>.png"``."""

    viewport_width: int
    viewport_height: int

    elements: list[GroundingElement] = field(default_factory=list)
    """UI elements detected in the screenshot."""

    action: GroundingAction | None = None
    """The action performed (``None`` for static records)."""

    source: Literal["static", "dynamic"] = "static"

    timestamp: str = ""
    """ISO-8601 UTC timestamp."""

    session_id: str | None = None

    phash: str | None = None
    """Perceptual hash for near-dedup (hex string)."""

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict, omitting ``None`` values."""
        d = asdict(self)
        # Strip None leaves for compactness
        d["elements"] = [_strip_none(e) for e in d["elements"]]
        if d["action"] is not None:
            d["action"] = _strip_none(d["action"])
        return _strip_none(d)


def _strip_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class GroundingJSONLWriter:
    """Append-only writer for ``grounding.jsonl`` with dedup and rotation.

    Parameters
    ----------
    output_dir:
        Root directory. ``grounding.jsonl`` and ``images/`` live here.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.grounding_path = self.output_dir / "grounding.jsonl"

        # Dedup sets
        self._seen_hashes: set[str] = set()
        self._seen_phashes: set[str] = set()

        # Bootstrap directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)

        # Load existing image hashes (same pattern as TrainingDataExporter)
        for img_path in self.images_dir.glob("*.png"):
            self._seen_hashes.add(img_path.stem)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(
        self,
        record: GroundingRecord,
        image_bytes: bytes | None = None,
    ) -> bool:
        """Write one grounding record.

        Parameters
        ----------
        record:
            The record to write. ``image_hash`` and ``image_path`` are
            overwritten when *image_bytes* is provided.
        image_bytes:
            Raw PNG bytes of the screenshot. If ``None``, the record must
            already have valid ``image_hash`` and ``image_path``.

        Returns
        -------
        bool
            ``True`` if the record was written, ``False`` if it was
            deduplicated away.
        """
        if image_bytes is not None:
            sha_hash = self._sha256(image_bytes)

            # Exact dedup
            if sha_hash in self._seen_hashes:
                logger.debug("Exact dedup: %s already seen", sha_hash)
                return False

            # Perceptual near-dedup (optional)
            phash_hex = self._compute_phash(image_bytes)
            if phash_hex and self._is_phash_duplicate(phash_hex):
                logger.debug("Perceptual dedup: phash %s too close", phash_hex)
                return False

            # Persist image
            image_rel = f"images/{sha_hash}.png"
            image_abs = self.output_dir / image_rel
            if not image_abs.exists():
                image_abs.write_bytes(image_bytes)

            self._seen_hashes.add(sha_hash)
            if phash_hex:
                self._seen_phashes.add(phash_hex)

            record.image_hash = sha_hash
            record.image_path = image_rel
            record.phash = phash_hex

        # Rotate if needed
        self._maybe_rotate()

        # Append JSONL line
        line = json.dumps(record.to_dict(), separators=(",", ":"))
        with open(self.grounding_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        return True

    # ------------------------------------------------------------------
    # Hashing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(data: bytes) -> str:
        """SHA256[:16] — same truncation as TrainingDataExporter._hash_image."""
        return hashlib.sha256(data).hexdigest()[:16]

    @staticmethod
    def _compute_phash(image_bytes: bytes) -> str | None:
        """Compute perceptual hash if imagehash is installed."""
        if not _IMAGEHASH_AVAILABLE:
            return None
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return str(imagehash.average_hash(img))
        except Exception:
            logger.debug("Perceptual hash failed", exc_info=True)
            return None

    def _is_phash_duplicate(self, phash_hex: str) -> bool:
        """Check if *phash_hex* is within hamming distance of any seen phash."""
        if not _IMAGEHASH_AVAILABLE:
            return False
        try:
            candidate = imagehash.hex_to_hash(phash_hex)
            for seen_hex in self._seen_phashes:
                seen = imagehash.hex_to_hash(seen_hex)
                if candidate - seen <= _PHASH_HAMMING_THRESHOLD:
                    return True
        except Exception:
            logger.debug("Perceptual dedup check failed", exc_info=True)
        return False

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def _maybe_rotate(self) -> None:
        """Rotate ``grounding.jsonl`` when it exceeds the size threshold."""
        if not self.grounding_path.exists():
            return
        try:
            size = self.grounding_path.stat().st_size
        except OSError:
            return
        if size < _MAX_JSONL_BYTES:
            return

        # Find next available rotation index
        idx = 1
        while True:
            rotated = self.grounding_path.parent / f"grounding.{idx}.jsonl"
            if not rotated.exists():
                break
            idx += 1

        self.grounding_path.rename(rotated)
        logger.info("Rotated grounding.jsonl → %s", rotated.name)
