"""External-app held-out benchmark: one name per VGA ``target_process``.

This loader is a shim over
``qontinui_train.evaluation.external_app_splits.loader.load_per_domain_splits``.
It registers one benchmark name per domain present in the VGA correction
log, of the form ``external_app_<target_process>``.

Unlike the other benchmark loaders in this package (ScreenSpot-v2 /
ScreenSpot-Pro / OSWorld-G), this one does **not** fetch anything from
HuggingFace. It reads the local VGA correction log and converts the
filtered, builder-confirmed subset into the VLM-SFT JSONL shape that
``grounding_eval.load_test_samples`` consumes.

Registration
------------
``BENCHMARK_LOADERS`` in the sibling ``__init__.py`` is static, but the
external-app benchmarks are dynamic (one per target process). We solve
that by exposing a class-level registry: ``external_app_benchmarks()``
returns a list of ``(name, loader)`` pairs that the
``__init__``-level dict merges at import time.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CORRECTIONS_DIR = Path("datasets/vga-corrections")
_ENV_OVERRIDE = "QONTINUI_VGA_CORRECTIONS_DIR"


def _resolve_corrections_dir() -> Path:
    env = os.environ.get(_ENV_OVERRIDE)
    return Path(env) if env else _DEFAULT_CORRECTIONS_DIR


def _known_target_processes(corrections_dir: Path) -> list[str]:
    """Return target_process values seen in the correction log.

    A single pass over the JSONL is cheap at today's scale. If the log
    grows past a few hundred MB we'll want a cached index alongside it.
    """
    jsonl = corrections_dir / "corrections.jsonl"
    if not jsonl.exists():
        return []

    seen: set[str] = set()
    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            tp = entry.get("target_process")
            if isinstance(tp, str):
                seen.add(tp)
    return sorted(seen)


def _make_loader(target_process: str) -> Callable[[Path], Path]:
    """Build a one-shot loader that emits a shadow jsonl for one domain.

    The returned callable matches the signature required by
    ``BENCHMARK_LOADERS`` entries: ``(cache_dir) -> Path``.
    """

    def loader(cache_dir: Path) -> Path:
        # Lazy-import to keep the heavy Pillow/PIL path off the module
        # import graph (only needed when this particular benchmark is
        # selected).
        try:
            from qontinui_train.evaluation.external_app_splits.loader import (
                load_per_domain_splits,
            )
        except ImportError:
            # When qontinui_train isn't importable (e.g. running
            # ``evaluation/grounding_eval.py`` as a top-level script
            # from a fresh checkout), fall back to the sibling
            # directory import.
            import sys

            pkg_root = Path(__file__).resolve().parents[3]
            sys.path.insert(0, str(pkg_root))
            from qontinui_train.evaluation.external_app_splits.loader import (  # type: ignore[no-redef]
                load_per_domain_splits,
            )

        corrections_dir = _resolve_corrections_dir()
        per_domain = load_per_domain_splits(corrections_dir)
        samples = per_domain.get(target_process, [])

        if not samples:
            logger.warning(
                "external_app_%s: no samples available at %s",
                target_process,
                corrections_dir,
            )

        # Write a shadow jsonl under the standard cache layout so the
        # rest of grounding_eval doesn't need special-casing.
        safe_name = _safe_benchmark_filename(target_process)
        out_path = cache_dir / safe_name / "test.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_vlm_dict(), separators=(",", ":")))
                f.write("\n")
        return out_path

    return loader


def _safe_benchmark_filename(target_process: str) -> str:
    """Filesystem-safe dirname for a Win32 process name like ``obs64.exe``.

    Just prefixes the target_process with ``external_app_`` — the
    characters allowed in a process name (alnum, dot, dash, underscore)
    are all filesystem-safe. But we replace path separators defensively.
    """
    sanitized = target_process.replace("/", "_").replace("\\", "_")
    return f"external_app_{sanitized}"


def external_app_benchmarks() -> dict[str, Callable[[Path], Path]]:
    """Return a ``{benchmark_name: loader}`` mapping, one entry per domain.

    Called at import time by the benchmarks package ``__init__`` to
    populate ``BENCHMARK_LOADERS``. If the correction log is empty or
    missing, this returns an empty dict — no benchmarks are registered.
    """
    corrections_dir = _resolve_corrections_dir()
    mapping: dict[str, Callable[[Path], Path]] = {}
    for tp in _known_target_processes(corrections_dir):
        name = _safe_benchmark_filename(tp)
        mapping[name] = _make_loader(tp)
    return mapping
