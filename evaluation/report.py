"""Cross-benchmark report generator.

Reads one or more ``grounding_eval.py`` JSON outputs and emits a single
markdown table with benchmarks as rows and models as columns. Cells show
``Acc@center / MDE``. Cells where the reproduction check failed (see
``reproduction_check.py``) are flagged with a bangs marker so reviewers
notice them.

Usage::

    python -m qontinui_train.evaluation.report \\
        --eval-output eval_results/screenspot_v2.json \\
        --eval-output eval_results/screenspot_pro.json \\
        --eval-output eval_results/osworld_g.json \\
        --output-md reports/grounding_v1.md

Each ``--eval-output`` file must be a JSON blob of the shape produced by
``grounding_eval.py``'s ``main()`` — specifically it must have:

- top-level ``test_jsonl`` (used to derive the benchmark name from the path);
- top-level ``results`` list, each element having ``model``, ``acc_center``,
  ``mean_distance_error``.

We derive the benchmark name from the eval JSON's ``benchmark`` field when
present, otherwise from the filename stem.

TODO (CI wiring)
----------------
Once a dedicated nightly eval job exists under
``qontinui-train/.github/workflows/``, add a step that:

1. Runs ``grounding_eval.py --benchmark screenspot_v2`` (and pro / osworld_g).
2. Collects the output JSONs.
3. Runs this report generator.
4. Uploads the markdown as a job artifact.

There is currently no ``qontinui-train/.github/workflows/`` directory; we
deliberately do NOT create one here because the repo has no existing
pattern for qontinui-train CI to extend.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

try:
    from .reproduction_check import check_published_reproduction
except ImportError:
    # Loaded as a top-level module (e.g. `python evaluation/report.py` or in
    # tests that put `evaluation/` on sys.path).
    from reproduction_check import (  # type: ignore[import-not-found,no-redef]
        check_published_reproduction,
    )

logger = logging.getLogger(__name__)


def _derive_benchmark(eval_blob: dict[str, Any], source_path: Path) -> str:
    """Return a benchmark name for *eval_blob*.

    Prefers an explicit ``benchmark`` key on the blob, falls back to the
    file stem if the test_jsonl path includes a known benchmark name, then
    to the source filename.
    """
    if "benchmark" in eval_blob and eval_blob["benchmark"]:
        return str(eval_blob["benchmark"])
    test_jsonl = eval_blob.get("test_jsonl") or ""
    for candidate in (
        "screenspot_v2",
        "screenspot_pro",
        "osworld_g",
        "internal",
    ):
        if candidate in test_jsonl:
            return candidate
    return source_path.stem


def _fmt_cell(
    acc: float | None,
    mde: float | None,
    *,
    flagged: bool,
) -> str:
    if acc is None or mde is None:
        return "-"
    marker = " !!" if flagged else ""
    return f"{acc:.3f} / {mde:.3f}{marker}"


def build_markdown(
    eval_blobs: list[tuple[Path, dict[str, Any]]],
) -> str:
    """Build the markdown table from parsed eval outputs.

    Parameters
    ----------
    eval_blobs:
        List of ``(source_path, parsed_json)`` pairs.
    """
    # benchmarks[benchmark][model] = (acc, mde, flagged)
    benchmarks: dict[str, dict[str, tuple[float, float, bool]]] = {}
    model_order: list[str] = []

    for source, blob in eval_blobs:
        benchmark = _derive_benchmark(blob, source)
        row = benchmarks.setdefault(benchmark, {})
        for result in blob.get("results", []):
            model = result.get("model")
            if not model:
                continue
            if model not in model_order:
                model_order.append(model)
            acc = float(result.get("acc_center", 0.0))
            mde = float(result.get("mean_distance_error", 0.0))
            reproduces, _delta = check_published_reproduction(
                benchmark=benchmark, model_id=model, actual_acc_center=acc
            )
            row[model] = (acc, mde, not reproduces)

    # Header
    lines: list[str] = []
    header = "| Benchmark | " + " | ".join(model_order) + " |"
    sep = "|" + "---|" * (len(model_order) + 1)
    lines.append(header)
    lines.append(sep)

    for benchmark in sorted(benchmarks.keys()):
        row_cells = [benchmark]
        for model in model_order:
            cell = benchmarks[benchmark].get(model)
            if cell is None:
                row_cells.append(_fmt_cell(None, None, flagged=False))
            else:
                acc, mde, flagged = cell
                row_cells.append(_fmt_cell(acc, mde, flagged=flagged))
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.append("")
    lines.append(
        "Legend: cell = `Acc@center / MDE`. `!!` marks a cell that failed the "
        "published-number reproduction check — see `reproduction_check.py`."
    )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine one or more grounding_eval.py JSON outputs into a "
            "markdown comparison table."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-output",
        action="append",
        required=True,
        type=Path,
        metavar="FILE",
        help="Path to an eval_results.json. Repeat for multiple benchmarks.",
    )
    parser.add_argument(
        "--output-md",
        required=True,
        type=Path,
        metavar="FILE",
        help="Where to write the combined markdown table.",
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
        format="%(levelname)s %(name)s - %(message)s",
    )

    blobs: list[tuple[Path, dict[str, Any]]] = []
    for eval_path in args.eval_output:
        if not eval_path.exists():
            logger.error("Eval output does not exist: %s", eval_path)
            raise SystemExit(1)
        with open(eval_path, encoding="utf-8") as fh:
            blobs.append((eval_path, json.load(fh)))

    markdown = build_markdown(blobs)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    logger.info("Wrote report to %s", args.output_md)
    print(f"Wrote report to {args.output_md}")


if __name__ == "__main__":
    main()
