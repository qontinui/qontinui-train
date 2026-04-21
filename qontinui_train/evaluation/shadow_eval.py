"""Shadow evaluation scaffold for the VGA v6 ship gate.

Compares a *candidate* grounding model (e.g. ``qontinui-grounding-v6``)
against the shipped *baseline* (``qontinui-grounding-v5``) on samples
that the runtime logged while serving with the baseline. The output is
a per-domain delta report + a pass/fail gate flag.

This is a **scaffold** — the DB query, image hydration, and model-call
plumbing are real but defensive (import-lazy, error-tolerant) so the
module can be imported and unit-tested without a live PG or a running
llama-swap.

Schema assumptions
------------------
``runner.vga_shadow_samples`` is introduced by the VGA PG schema
addendum (plan §13). Each row carries:

- ``image_sha: TEXT`` — content-addressable key for the screenshot
- ``image_path: TEXT`` — absolute path on the runner host
- ``prompt: TEXT`` — natural-language element description
- ``model_used: TEXT`` — model name at prediction time
- ``baseline_bbox: JSONB`` — ``{"x":..,"y":..,"w":..,"h":..}``
- ``target_process: TEXT``
- ``created_at: TIMESTAMPTZ``

If a correction exists for the same ``image_sha + prompt`` pair, that
correction is treated as ground truth. Otherwise the baseline's own
bbox is taken as "ground truth" for delta measurement — the shadow eval
then measures agreement between the models rather than absolute
accuracy. The scaffold's current implementation mirrors the agreement
mode; absolute-accuracy mode lands once the correction join is wired.

Gate rule (plan §13)
--------------------
`overall_gate_pass` is True iff every domain delta is ≥ +5pp Acc@center
AND no domain regresses below its baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Ship-gate threshold in percentage points. 5pp matches plan §13.
_SHIP_GATE_DELTA_PP = 5.0


@dataclass
class DomainResult:
    """Per-``target_process`` evaluation outcome."""

    baseline_acc: float
    candidate_acc: float
    delta_pp: float
    regression: bool
    samples: int = 0


@dataclass
class ShadowEvalReport:
    """Top-level result of a shadow-eval run."""

    candidate_model: str
    baseline_model: str
    per_domain_results: dict[str, DomainResult]
    overall_gate_pass: bool
    # Raw counts for the dashboard widget (plan §13 recommendation E).
    total_samples: int = 0
    started_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    ended_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_model": self.candidate_model,
            "baseline_model": self.baseline_model,
            "overall_gate_pass": self.overall_gate_pass,
            "total_samples": self.total_samples,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "per_domain_results": {
                domain: {
                    "baseline_acc": r.baseline_acc,
                    "candidate_acc": r.candidate_acc,
                    "delta_pp": r.delta_pp,
                    "regression": r.regression,
                    "samples": r.samples,
                }
                for domain, r in self.per_domain_results.items()
            },
        }


# ---------------------------------------------------------------------------
# DB access — lazy + pluggable
# ---------------------------------------------------------------------------


def _query_shadow_samples(
    pg_url: str,
    baseline_model: str,
    target_process: str | None,
    since: datetime,
    limit: int,
) -> list[dict[str, Any]]:
    """Return shadow rows. Lazy-imports psycopg for testability.

    Swallow-and-log on connection failure so the scaffold can be imported
    in test environments that have no PG.
    """
    try:
        import psycopg  # type: ignore[import-not-found]
    except ImportError:
        logger.error(
            "psycopg not installed; shadow_eval cannot query PG. "
            "Install with: pip install psycopg[binary]"
        )
        return []

    sql = """
    SELECT image_sha, image_path, prompt, baseline_bbox, target_process, created_at
    FROM runner.vga_shadow_samples
    WHERE model_used = %(baseline)s
      AND created_at >= %(since)s
    """
    params: dict[str, Any] = {"baseline": baseline_model, "since": since}
    if target_process is not None:
        sql += " AND target_process = %(target_process)s"
        params["target_process"] = target_process
    sql += " ORDER BY created_at DESC LIMIT %(limit)s"
    params["limit"] = limit

    try:
        with psycopg.connect(pg_url) as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [c.name for c in cur.description or []]
                return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]
    except Exception as exc:
        logger.error("shadow_eval: PG query failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Model inference — thin wrapper around grounding_eval's helper
# ---------------------------------------------------------------------------


def _predict(
    sample: dict[str, Any],
    model_name: str,
    api_base: str,
) -> tuple[float, float] | None:
    """Call the OpenAI-compatible endpoint for a single shadow sample.

    Returns the parsed ``(x, y)`` point in normalized [0, 1000] space, or
    ``None`` on any failure.
    """
    try:
        # Lazy import — grounding_eval sits at the sibling legacy path.
        import importlib
        import sys
        from pathlib import Path as _Path

        # Add the legacy evaluation/ dir to sys.path so
        # `grounding_eval` is importable as a top-level module.
        eval_legacy = _Path(__file__).resolve().parents[3] / "evaluation"
        if str(eval_legacy) not in sys.path:
            sys.path.insert(0, str(eval_legacy))
        grounding_eval = importlib.import_module("grounding_eval")
    except Exception as exc:  # noqa: BLE001 — scaffold import tolerance
        logger.error("shadow_eval: cannot import grounding_eval: %s", exc)
        return None

    # Build a VLM SFT-shaped sample on the fly so run_model_inference
    # works unmodified.
    uri = f"file:///{str(sample['image_path']).lstrip('/')}"
    vlm_sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": uri},
                    {
                        "type": "text",
                        "text": (
                            "Locate the following element in the screenshot "
                            "and output its position as <point>x y</point> "
                            "where x and y are integers between 0 and 1000 "
                            "(normalized coordinates).\n\n"
                            f"Element: {sample['prompt']}"
                        ),
                    },
                ],
            }
        ]
    }

    raw = grounding_eval.run_model_inference(vlm_sample, model_name, api_base)
    if raw is None:
        return None
    return grounding_eval.parse_point(raw)


def _bbox_center_normalized(
    bbox: dict[str, Any], image_w: int, image_h: int
) -> tuple[float, float]:
    cx_px = bbox["x"] + bbox["w"] / 2.0
    cy_px = bbox["y"] + bbox["h"] / 2.0
    return (cx_px / image_w * 1000.0, cy_px / image_h * 1000.0)


def _image_dims(image_path: str) -> tuple[int, int] | None:
    try:
        from pathlib import Path as _Path

        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with Image.open(_Path(image_path)) as im:
            return im.size
    except (OSError, ValueError):
        return None


def _within_tolerance(
    pred: tuple[float, float] | None,
    target: tuple[float, float],
    tolerance_normalized: float = 50.0,
) -> bool:
    """Hit within `tolerance_normalized` units of the 0–1000 normalized grid.

    50 units ≈ 5% of the frame. Matches the default tolerance in
    ``grounding_eval`` once re-scaled to 0–1000.
    """
    if pred is None:
        return False
    dx = pred[0] - target[0]
    dy = pred[1] - target[1]
    return (dx * dx + dy * dy) ** 0.5 <= tolerance_normalized


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_shadow_eval(
    pg_url: str,
    candidate_model: str,
    baseline_model: str = "qontinui-grounding-v5",
    api_base: str = "http://localhost:5800/v1",
    target_process: str | None = None,
    limit: int = 500,
    since: datetime | None = None,
) -> ShadowEvalReport:
    """Run the shadow eval and return a pass/fail report.

    Args:
        pg_url: postgres connection string (``runner`` schema).
        candidate_model: model name being considered for production.
        baseline_model: model name currently shipping.
        api_base: OpenAI-compatible base URL (llama-swap @ 5800).
        target_process: optional filter — only eval one domain.
        limit: max number of shadow rows to fetch.
        since: only consider rows created after this time; default is
            30 days back.

    Returns:
        :class:`ShadowEvalReport` with per-domain deltas and the
        overall gate flag.
    """
    started_at = datetime.now(UTC)
    if since is None:
        since = started_at - timedelta(days=30)

    rows = _query_shadow_samples(
        pg_url=pg_url,
        baseline_model=baseline_model,
        target_process=target_process,
        since=since,
        limit=limit,
    )

    # Bucket per-domain accumulators.
    accum: dict[str, dict[str, int]] = {}

    for row in rows:
        tp = row.get("target_process", "unknown")
        if tp not in accum:
            accum[tp] = {
                "baseline_correct": 0,
                "candidate_correct": 0,
                "total": 0,
            }

        dims = _image_dims(row["image_path"])
        if dims is None:
            continue
        image_w, image_h = dims

        baseline_bbox = row.get("baseline_bbox")
        if not isinstance(baseline_bbox, dict):
            continue
        baseline_point = _bbox_center_normalized(baseline_bbox, image_w, image_h)

        # TODO(milestone c.3): join against corrections on (image_sha,
        # prompt) and use the confirmed bbox as ground truth when present.
        # Until then, ground truth = the baseline's own prediction, so
        # the shadow eval measures candidate agreement with baseline —
        # useful as a regression smoke test but not a true accuracy
        # number.
        ground_truth = baseline_point

        baseline_pred = baseline_point  # baseline's logged prediction
        candidate_pred = _predict(row, candidate_model, api_base)

        if _within_tolerance(baseline_pred, ground_truth):
            accum[tp]["baseline_correct"] += 1
        if _within_tolerance(candidate_pred, ground_truth):
            accum[tp]["candidate_correct"] += 1
        accum[tp]["total"] += 1

    # Produce per-domain DomainResults.
    per_domain_results: dict[str, DomainResult] = {}
    all_passed = True
    for domain, counts in accum.items():
        total = max(1, counts["total"])
        baseline_acc = counts["baseline_correct"] / total
        candidate_acc = counts["candidate_correct"] / total
        delta_pp = (candidate_acc - baseline_acc) * 100.0
        regression = delta_pp < 0
        per_domain_results[domain] = DomainResult(
            baseline_acc=baseline_acc,
            candidate_acc=candidate_acc,
            delta_pp=delta_pp,
            regression=regression,
            samples=counts["total"],
        )
        # Ship gate: every domain must lift by ≥ 5pp and none may regress.
        if delta_pp < _SHIP_GATE_DELTA_PP or regression:
            all_passed = False

    # Empty result set → gate is implicitly False (nothing proven).
    if not per_domain_results:
        all_passed = False

    report = ShadowEvalReport(
        candidate_model=candidate_model,
        baseline_model=baseline_model,
        per_domain_results=per_domain_results,
        overall_gate_pass=all_passed,
        total_samples=sum(v.samples for v in per_domain_results.values()),
        started_at=started_at,
        ended_at=datetime.now(UTC),
    )

    logger.info(
        "shadow_eval: candidate=%s baseline=%s domains=%d gate_pass=%s",
        candidate_model,
        baseline_model,
        len(per_domain_results),
        all_passed,
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser():  # type: ignore[no-untyped-def]
    import argparse

    parser = argparse.ArgumentParser(
        description="Shadow-eval scaffold for the VGA v6 ship gate.",
    )
    parser.add_argument("--pg-url", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument(
        "--baseline-model", default="qontinui-grounding-v5"
    )
    parser.add_argument("--api-base", default="http://localhost:5800/v1")
    parser.add_argument("--target-process", default=None)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s - %(message)s",
    )

    report = run_shadow_eval(
        pg_url=args.pg_url,
        candidate_model=args.candidate_model,
        baseline_model=args.baseline_model,
        api_base=args.api_base,
        target_process=args.target_process,
        limit=args.limit,
    )

    import json

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.overall_gate_pass else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
