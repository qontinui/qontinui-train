"""Published-number reproduction check for grounding benchmarks.

When we evaluate ``UI-TARS-1.5-7B`` against a public benchmark, our measured
Acc@center should land near the number the paper / model card reports. If it
doesn't, that's evidence our eval harness or our model serving is
mis-configured — not a real score we should trust.

This module hardcodes a small table of published Acc@center numbers for the
benchmarks in ``benchmarks/__init__.py``. It exposes a single function
``check_published_reproduction`` that the grounding eval CLI calls after
computing metrics.

Source of published numbers
---------------------------
Numbers below are taken from the ClawGUI-Eval README's "reproduction rate"
section (Anthropic internal note: the ClawGUI-Eval project explicitly calls
this a "reproduction rate", not a "faithfulness" score). We cite that doc
rather than individual papers because ClawGUI-Eval re-runs the same harness
across papers and reports a single aligned number per (model, benchmark).

When a number here is stale — i.e. ClawGUI-Eval publishes a newer, more
accurate reproduction rate — update the table and bump ``_SOURCE_REV``
below so diff readers know the change is intentional.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Threshold for "reproduces" vs "diverges" (absolute Acc@center delta).
_REPRODUCE_TOLERANCE = 0.02

# Incremented whenever the published-number table is updated from a new
# ClawGUI-Eval README snapshot.
_SOURCE_REV = 1

# Published Acc@center for UI-TARS-1.5-7B per ClawGUI-Eval README
# reproduction-rate section.
# Keys: (benchmark_name, model_id) -> published Acc@center in [0, 1].
_PUBLISHED: dict[tuple[str, str], float] = {
    ("screenspot_v2", "ByteDance-Seed/UI-TARS-1.5-7B"): 0.943,
    ("screenspot_pro", "ByteDance-Seed/UI-TARS-1.5-7B"): 0.497,
    ("osworld_g", "ByteDance-Seed/UI-TARS-1.5-7B"): 0.419,
}


def check_published_reproduction(
    benchmark: str,
    model_id: str,
    actual_acc_center: float,
) -> tuple[bool, float]:
    """Check measured Acc@center against the published reference.

    Parameters
    ----------
    benchmark:
        Benchmark name as registered in ``BENCHMARK_LOADERS`` (e.g.
        ``"screenspot_v2"``).
    model_id:
        Model identifier matching the key used in ``_PUBLISHED`` (e.g.
        ``"ByteDance-Seed/UI-TARS-1.5-7B"``).
    actual_acc_center:
        The Acc@center we just measured, in ``[0, 1]``.

    Returns
    -------
    tuple[bool, float]
        ``(reproduces, delta)`` — ``reproduces`` is ``True`` when
        ``abs(delta) <= 0.02``; ``delta = actual - published`` (signed).
        When the (benchmark, model) pair is not in the table, returns
        ``(True, 0.0)`` — i.e. no evidence against reproduction.
    """
    published = _PUBLISHED.get((benchmark, model_id))
    if published is None:
        return (True, 0.0)
    delta = actual_acc_center - published
    reproduces = abs(delta) <= _REPRODUCE_TOLERANCE
    return (reproduces, delta)


def log_reproduction_check(
    benchmark: str,
    model_id: str,
    actual_acc_center: float,
) -> None:
    """Convenience: run the check and log at WARNING when it fails."""
    reproduces, delta = check_published_reproduction(
        benchmark, model_id, actual_acc_center
    )
    if (benchmark, model_id) not in _PUBLISHED:
        logger.debug(
            "No published Acc@center for (%s, %s); skipping reproduction check.",
            benchmark,
            model_id,
        )
        return
    if reproduces:
        logger.info(
            "Reproduction OK for %s on %s: actual=%.3f published=%.3f (delta=%+.3f)",
            model_id,
            benchmark,
            actual_acc_center,
            _PUBLISHED[(benchmark, model_id)],
            delta,
        )
    else:
        logger.warning(
            "Reproduction FAILED for %s on %s: actual=%.3f published=%.3f "
            "(delta=%+.3f, tolerance=%.3f). Investigate harness / serving.",
            model_id,
            benchmark,
            actual_acc_center,
            _PUBLISHED[(benchmark, model_id)],
            delta,
            _REPRODUCE_TOLERANCE,
        )
