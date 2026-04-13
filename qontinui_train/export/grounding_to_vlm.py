"""Convert GroundingRecord JSONL → Qwen2.5-VL supervised fine-tuning chat format.

Phase 3 of the component-render synthetic grounding data pipeline.

Usage (CLI)::

    python -m qontinui_train.export.grounding_to_vlm \\
        --input-dir dataset/ \\
        --output-dir dataset/vlm_sft/ \\
        --seed 42

Output files written to ``output-dir``:

- ``vlm_train.jsonl``   – 80 % training split
- ``vlm_val.jsonl``     – 10 % validation split
- ``vlm_test.jsonl``    – 10 % test split (+ all pure-static records)
- ``split_stats.json``  – counts per split and per component type
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATES = [
    "In the screenshot, find the element: {desc}",
    "Locate the following UI element: {desc}",
    "Find and point to: {desc}",
    "Where is the {desc}?",
    "Point to the {desc} in this screenshot.",
]


# ---------------------------------------------------------------------------
# Description generation
# ---------------------------------------------------------------------------


def _build_description(element: dict[str, Any], metadata: dict[str, Any] | None) -> str:
    """Return a natural-language description of *element*.

    When *metadata* is available the description uses component-level
    vocabulary (component, variant, state).  Otherwise it falls back to the
    element's semantic role and text label.
    """
    text: str | None = element.get("text") or None

    if metadata:
        component: str = metadata.get("component") or element.get("role", "element")
        variant: str | None = metadata.get("variant")
        state: str | None = metadata.get("state")

        # Build adjective phrase: "disabled destructive"
        adjectives: list[str] = []
        if state:
            adjectives.append(state)
        if variant:
            adjectives.append(variant)

        adj_str = " ".join(adjectives)
        base = f"{adj_str} {component}".strip() if adj_str else component

        if text:
            return f"{base} labeled '{text}'"
        return base

    # No metadata — plain role / text
    role: str = element.get("role", "element")
    if text:
        return f"{role} labeled '{text}'"
    return role


def _pick_prompt(desc: str, rng: random.Random) -> str:
    """Pick a random instruction prefix and fill in *desc*."""
    template = rng.choice(_PROMPT_TEMPLATES)
    return template.format(desc=desc)


# ---------------------------------------------------------------------------
# Coordinate normalisation
# ---------------------------------------------------------------------------


def _normalise_center(
    bbox: list[int],
    viewport_width: int,
    viewport_height: int,
) -> tuple[float, float]:
    """Return the (x_center, y_center) of *bbox* normalised to [0, 1].

    *bbox* is ``[x, y, width, height]`` in pixels.
    """
    x, y, w, h = bbox
    x_center = (x + w / 2) / viewport_width
    y_center = (y + h / 2) / viewport_height
    return x_center, y_center


# ---------------------------------------------------------------------------
# Per-record sample conversion
# ---------------------------------------------------------------------------


def record_to_samples(
    record: dict[str, Any],
    input_dir: Path,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Convert one GroundingRecord dict into zero or more VLM SFT samples.

    Returns an empty list when the record has no elements or the image file
    cannot be resolved.
    """
    elements: list[dict[str, Any]] = record.get("elements", [])
    if not elements:
        return []

    image_path_rel: str = record.get("image_path", "")
    if not image_path_rel:
        return []

    # Resolve to absolute path
    image_abs = (input_dir / image_path_rel).resolve()
    image_uri = image_abs.as_uri()  # file:///...

    viewport_width: int = record.get("viewport_width", 1)
    viewport_height: int = record.get("viewport_height", 1)
    metadata: dict[str, Any] | None = record.get("metadata")

    samples: list[dict[str, Any]] = []

    for element in elements:
        bbox = element.get("bbox")
        if not bbox or len(bbox) < 4:
            continue

        desc = _build_description(element, metadata)
        prompt_text = _pick_prompt(desc, rng)

        x_c, y_c = _normalise_center(bbox, viewport_width, viewport_height)
        # Clamp to [0, 1] in case of rounding edge cases
        x_c = max(0.0, min(1.0, x_c))
        y_c = max(0.0, min(1.0, y_c))

        sample: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_uri},
                        {"type": "text", "text": prompt_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": f"<point>{x_c:.2f} {y_c:.2f}</point>",
                },
            ]
        }
        samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------


def _stratum_key(record: dict[str, Any]) -> str:
    """Return the stratum label for stratified splitting."""
    metadata: dict[str, Any] | None = record.get("metadata")
    if metadata and metadata.get("component"):
        return str(metadata["component"])
    elements: list[dict[str, Any]] = record.get("elements", [])
    if elements:
        return str(elements[0].get("role", "unknown"))
    return "unknown"


def _is_pure_static(record: dict[str, Any]) -> bool:
    """Return True for records that belong exclusively to the test set.

    Pure-static records are those from the original static gallery:
    ``source == "static"`` AND ``metadata is None``.
    """
    return record.get("source") == "static" and record.get("metadata") is None


def split_records(
    records: list[dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split *records* into (train, val, test) lists.

    Pure-static records (source=="static" and no metadata) go exclusively to
    the test set.  The remainder is split 80/10/10 stratified by component
    type.

    Parameters
    ----------
    records:
        All GroundingRecord dicts loaded from JSONL files.
    seed:
        RNG seed for reproducibility.
    train_ratio, val_ratio:
        Fractions for the non-test partition (``test_ratio = 1 - train - val``).

    Returns
    -------
    (train, val, test)
        Three lists of GroundingRecord dicts.
    """
    rng = random.Random(seed)

    pure_static: list[dict[str, Any]] = []
    stratified: list[dict[str, Any]] = []

    for rec in records:
        if _is_pure_static(rec):
            pure_static.append(rec)
        else:
            stratified.append(rec)

    # Group by stratum
    by_stratum: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for rec in stratified:
        by_stratum[_stratum_key(rec)].append(rec)

    train_out: list[dict[str, Any]] = []
    val_out: list[dict[str, Any]] = []
    test_out: list[dict[str, Any]] = list(pure_static)

    for stratum_records in by_stratum.values():
        shuffled = list(stratum_records)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, round(n * train_ratio)) if n >= 3 else n
        n_val = max(0, round(n * val_ratio)) if n >= 3 else 0

        train_out.extend(shuffled[:n_train])
        val_out.extend(shuffled[n_train : n_train + n_val])
        test_out.extend(shuffled[n_train + n_val :])

    # Shuffle final lists so strata are interleaved
    rng.shuffle(train_out)
    rng.shuffle(val_out)
    rng.shuffle(test_out)

    return train_out, val_out, test_out


# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------


def load_records(input_dir: Path) -> list[dict[str, Any]]:
    """Load all grounding records from *input_dir*.

    Reads ``grounding.jsonl`` and any ``grounding.*.jsonl`` rotation files.
    Invalid / unparseable lines are skipped with a warning.
    """
    pattern_files: list[Path] = sorted(input_dir.glob("grounding.jsonl")) + sorted(
        input_dir.glob("grounding.*.jsonl")
    )

    records: list[dict[str, Any]] = []
    for jsonl_path in pattern_files:
        logger.info("Loading %s", jsonl_path)
        with open(jsonl_path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "%s:%d – skipping invalid JSON: %s",
                        jsonl_path.name,
                        lineno,
                        exc,
                    )

    logger.info("Loaded %d records from %d file(s)", len(records), len(pattern_files))
    return records


def _write_samples(samples: list[dict[str, Any]], path: Path) -> None:
    """Write *samples* as JSONL to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, separators=(",", ":")) + "\n")
    logger.info("Wrote %d samples → %s", len(samples), path)


# ---------------------------------------------------------------------------
# High-level conversion pipeline
# ---------------------------------------------------------------------------


def convert(
    input_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full conversion pipeline.

    Parameters
    ----------
    input_dir:
        Directory containing ``grounding.jsonl`` (and rotated variants) plus
        the ``images/`` sub-directory.
    output_dir:
        Destination for ``vlm_train.jsonl``, ``vlm_val.jsonl``,
        ``vlm_test.jsonl``, and ``split_stats.json``.
    seed:
        Random seed used for prompt-template selection AND stratified splitting.

    Returns
    -------
    dict
        The same statistics written to ``split_stats.json``.
    """
    rng = random.Random(seed)

    records = load_records(input_dir)
    if not records:
        logger.warning("No records found in %s – nothing to do.", input_dir)
        return {"train": 0, "val": 0, "test": 0, "by_component": {}}

    train_recs, val_recs, test_recs = split_records(records, seed=seed)

    # Convert each split
    def _recs_to_samples(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rec in recs:
            out.extend(record_to_samples(rec, input_dir, rng))
        return out

    train_samples = _recs_to_samples(train_recs)
    val_samples = _recs_to_samples(val_recs)
    test_samples = _recs_to_samples(test_recs)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_samples(train_samples, output_dir / "vlm_train.jsonl")
    _write_samples(val_samples, output_dir / "vlm_val.jsonl")
    _write_samples(test_samples, output_dir / "vlm_test.jsonl")

    # Collect per-component counts across all records
    by_component: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"train": 0, "val": 0, "test": 0}
    )
    for rec in train_recs:
        by_component[_stratum_key(rec)]["train"] += 1
    for rec in val_recs:
        by_component[_stratum_key(rec)]["val"] += 1
    for rec in test_recs:
        by_component[_stratum_key(rec)]["test"] += 1

    stats: dict[str, Any] = {
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
        "total_records": len(records),
        "pure_static_test_records": sum(1 for r in test_recs if _is_pure_static(r)),
        "by_component": {k: dict(v) for k, v in sorted(by_component.items())},
    }

    stats_path = output_dir / "split_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Split stats → %s", stats_path)

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert GroundingRecord JSONL to Qwen2.5-VL SFT chat format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing grounding.jsonl and images/",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Destination for vlm_*.jsonl and split_stats.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits and prompt selection",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s – %(message)s",
    )

    stats = convert(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
    )

    print("\nSplit summary:")
    print(f"  train : {stats['train']:>6} samples")
    print(f"  val   : {stats['val']:>6} samples")
    print(f"  test  : {stats['test']:>6} samples")
    print(f"  total records read: {stats.get('total_records', 'n/a')}")


if __name__ == "__main__":
    main()
