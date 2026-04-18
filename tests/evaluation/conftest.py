"""Shared pytest fixtures for evaluation/ tests.

Adds the `qontinui-train/evaluation` directory to `sys.path` so we can
import `benchmarks`, `report`, and `reproduction_check` without depending
on this project being installed as a package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parents[2] / "evaluation"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
