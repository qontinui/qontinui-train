"""Grounding model evaluation script.

Phase 5 of the component-render synthetic grounding data pipeline.

Evaluates any OpenAI-format grounding model against a held-out test set
(``vlm_test.jsonl``) and reports:

- Acc@center  – predicted point falls within tolerance of ground-truth point
- Mean Distance Error (MDE) – normalised Euclidean distance
- Per-component breakdown
- Before/after comparison when both baseline and fine-tuned models are given

Usage::

    python -m qontinui_train.evaluation.grounding_eval \\
      --test-jsonl dataset/vlm_sft/vlm_test.jsonl \\
      --baseline-model ByteDance-Seed/UI-TARS-1.5-7B \\
      --finetuned-model qontinui-grounding-v1 \\
      --api-base http://localhost:5800/v1 \\
      --output-dir eval_results/ \\
      --max-samples 500 \\
      --tolerance 0.05
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for parsing model output
# ---------------------------------------------------------------------------

_POINT_RE = re.compile(r"<point>\s*([\d.]+)\s+([\d.]+)\s*</point>")
_BARE_RE = re.compile(r"^\s*([\d.]+)\s+([\d.]+)\s*$")

# Diagonal of the normalised unit square used for MDE normalisation
_UNIT_DIAGONAL = math.sqrt(2)

# ---------------------------------------------------------------------------
# Canned benchmarks
# ---------------------------------------------------------------------------
# Maps --benchmark=<name> to a test-jsonl path, resolved relative to this file
# so CI doesn't need to pass absolute paths. Keep this dict tight; only add a
# new entry when there's a real benchmark definition + README describing its
# origin and expected format.
#
# External benchmarks (ScreenSpot-v2, ScreenSpot-Pro, OSWorld-G) are NOT in
# this dict — they're resolved via ``benchmarks.BENCHMARK_LOADERS`` at
# runtime, which downloads from HF and caches a local shadow jsonl under
# ``--benchmark-cache-dir``.

_BENCHMARK_PATHS: dict[str, Path] = {
    "internal": Path(__file__).parent / "benchmarks" / "internal" / "test.jsonl",
}


def _available_benchmarks() -> list[str]:
    """Return the union of canned + external benchmark names."""
    # Import lazily so `grounding_eval` remains importable even if HF
    # datasets isn't installed (external loaders aren't needed for
    # --test-jsonl / --benchmark=internal paths).
    names = set(_BENCHMARK_PATHS.keys())
    try:
        from .benchmarks import BENCHMARK_LOADERS

        names.update(BENCHMARK_LOADERS.keys())
    except ImportError:
        # Running as a top-level script (``python grounding_eval.py``) —
        # the package-relative import fails. Fall back to an absolute
        # import using the file layout.
        try:
            import sys as _sys

            _sys.path.insert(0, str(Path(__file__).parent))
            from benchmarks import (  # type: ignore[import-not-found,no-redef]
                BENCHMARK_LOADERS,
            )

            names.update(BENCHMARK_LOADERS.keys())
        except Exception:
            logger.debug("External benchmark loaders unavailable", exc_info=True)
    return sorted(names)


def _resolve_benchmark(name: str, cache_dir: Path) -> Path:
    """Resolve a --benchmark name to a local test-jsonl path.

    Order of resolution:
    1. External loaders from ``benchmarks.BENCHMARK_LOADERS`` — downloads
       the HF dataset (if not cached) and returns the shadow jsonl path.
    2. Fallback: ``_BENCHMARK_PATHS`` lookup for canned local benchmarks
       (currently only ``"internal"``).

    Raises ``KeyError`` when *name* is not recognized by either layer.
    """
    # Try external loaders first — they emit local shadow files that the
    # existing load_test_samples() can consume.
    try:
        from .benchmarks import BENCHMARK_LOADERS
    except ImportError:
        try:
            import sys as _sys

            _sys.path.insert(0, str(Path(__file__).parent))
            from benchmarks import (  # type: ignore[import-not-found,no-redef]
                BENCHMARK_LOADERS,
            )
        except Exception:
            BENCHMARK_LOADERS = {}  # type: ignore[assignment]

    if name in BENCHMARK_LOADERS:
        loader = BENCHMARK_LOADERS[name]
        return loader(cache_dir)

    if name in _BENCHMARK_PATHS:
        return _BENCHMARK_PATHS[name]

    raise KeyError(name)


# ---------------------------------------------------------------------------
# Output parsing helpers
# ---------------------------------------------------------------------------


def parse_point(text: str) -> tuple[float, float] | None:
    """Extract (x, y) from ``<point>x y</point>`` or bare ``x y`` output.

    Returns ``None`` when no valid coordinate pair is found.
    """
    if text is None:
        return None
    m = _POINT_RE.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = _BARE_RE.match(text.strip())
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


# ---------------------------------------------------------------------------
# Ground-truth parsing from the JSONL record
# ---------------------------------------------------------------------------


def parse_gt_point(sample: dict[str, Any]) -> tuple[float, float] | None:
    """Extract the ground-truth (x, y) from a VLM SFT sample dict."""
    messages = sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            return parse_point(msg.get("content", ""))
    return None


def parse_prompt_text(sample: dict[str, Any]) -> str:
    """Extract the text prompt from the user message."""
    messages = sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        text = part.get("text", "")
                        return str(text) if text is not None else ""
            elif isinstance(content, str):
                return content
    return ""


def parse_image_path(sample: dict[str, Any]) -> Path | None:
    """Extract the image file path from the user message (``file:///…`` URI)."""
    messages = sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image":
                        uri: str = part.get("image", "")
                        if uri.startswith("file:///"):
                            # Strip the scheme
                            path_str = uri[len("file:///") :]
                            # On Windows paths come as C:/... — keep as-is
                            return Path(path_str)
    return None


# ---------------------------------------------------------------------------
# Component type extraction
# ---------------------------------------------------------------------------

# Capitalised component names that appear in prompts generated by
# grounding_to_vlm.py, e.g. "a disabled destructive Button labeled 'Delete'"
_COMPONENT_RE = re.compile(
    r"\b(Button|Badge|Input|Select|Checkbox|Radio|Toggle|Switch|"
    r"Slider|TextArea|Label|Link|Icon|Avatar|Card|Dialog|Modal|"
    r"Tooltip|Dropdown|Combobox|Spinner|Progress|Alert|Toast|Tab|"
    r"Accordion|Breadcrumb|Pagination|Table|Tag|Chip|Menu|MenuBar|"
    r"ContextMenu|Popover|DatePicker|TimePicker|ColorPicker|FileUpload)\b",
    re.IGNORECASE,
)


def extract_component_type(prompt_text: str) -> str:
    """Best-effort extraction of component type from the prompt string."""
    m = _COMPONENT_RE.search(prompt_text)
    if m:
        return m.group(1).capitalize()
    return "Unknown"


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_image_b64(image_path: Path) -> str | None:
    """Read *image_path* and return a base64-encoded string, or ``None`` on error."""
    try:
        data = image_path.read_bytes()
        return base64.b64encode(data).decode("ascii")
    except OSError as exc:
        logger.warning("Cannot read image %s: %s", image_path, exc)
        return None


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


def run_model_inference(
    sample: dict[str, Any],
    model_name: str,
    api_base: str,
    api_key: str = "not-needed",
) -> str | None:
    """Call the OpenAI-compatible API for a single sample.

    Returns the raw string output from the model, or ``None`` on any error.
    """
    try:
        import openai  # local import so the module is usable without openai installed
    except ImportError:
        logger.error(
            "openai package is required for inference. "
            "Install it with: pip install openai"
        )
        return None

    image_path = parse_image_path(sample)
    if image_path is None:
        logger.debug("No image path in sample; skipping")
        return None

    b64_image = load_image_b64(image_path)
    if b64_image is None:
        return None

    prompt_text = parse_prompt_text(sample)

    client = openai.OpenAI(base_url=api_base, api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            max_tokens=50,
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as exc:
        logger.warning("API error for model %s: %s", model_name, exc)
        return None


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def euclidean_distance(
    pred: tuple[float, float],
    gt: tuple[float, float],
) -> float:
    """Normalised Euclidean distance between two points in [0,1]² space."""
    dx = pred[0] - gt[0]
    dy = pred[1] - gt[1]
    return math.sqrt(dx * dx + dy * dy) / _UNIT_DIAGONAL


def is_within_tolerance(
    pred: tuple[float, float],
    gt: tuple[float, float],
    tolerance: float,
) -> bool:
    """Return True when the raw (un-normalised) distance is ≤ *tolerance*."""
    dx = pred[0] - gt[0]
    dy = pred[1] - gt[1]
    return math.sqrt(dx * dx + dy * dy) <= tolerance


# ---------------------------------------------------------------------------
# Single-model evaluation loop
# ---------------------------------------------------------------------------


def evaluate_model(
    samples: list[dict[str, Any]],
    model_name: str,
    api_base: str,
    tolerance: float,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run inference on *samples* and compute metrics.

    Returns a results dict with keys:
        ``acc_center``, ``mean_distance_error``, ``per_component``,
        ``n_total``, ``n_parsed``, ``n_failed``
    """
    try:
        from tqdm import tqdm  # type: ignore[import-untyped]
    except ImportError:
        # Graceful fallback when tqdm is not installed
        def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
            return iterable

    if max_samples is not None:
        samples = samples[:max_samples]

    n_total = len(samples)
    n_parsed = 0
    n_failed = 0
    correct = 0
    distance_sum = 0.0

    # Per-component accumulators: {component: {"correct": int, "total": int, "dist_sum": float}}
    per_component: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "dist_sum": 0.0}
    )

    logger.info("Evaluating model '%s' on %d samples …", model_name, n_total)

    for sample in tqdm(samples, desc=f"[{model_name}]", unit="sample"):
        gt_point = parse_gt_point(sample)
        if gt_point is None:
            logger.debug("No ground-truth point in sample; skipping")
            n_failed += 1
            continue

        prompt_text = parse_prompt_text(sample)
        component = extract_component_type(prompt_text)

        raw_output = run_model_inference(sample, model_name, api_base)
        if raw_output is None:
            n_failed += 1
            per_component[component]["total"] += 1
            continue

        pred_point = parse_point(raw_output)
        if pred_point is None:
            logger.debug("Could not parse point from output: %r", raw_output[:80])
            n_failed += 1
            per_component[component]["total"] += 1
            continue

        n_parsed += 1
        hit = is_within_tolerance(pred_point, gt_point, tolerance)
        dist = euclidean_distance(pred_point, gt_point)

        if hit:
            correct += 1
        distance_sum += dist

        per_component[component]["total"] += 1
        if hit:
            per_component[component]["correct"] += 1
        per_component[component]["dist_sum"] += dist

    # Aggregate
    acc_center = correct / n_total if n_total > 0 else 0.0
    mde = distance_sum / n_total if n_total > 0 else 0.0

    comp_results: dict[str, dict[str, Any]] = {}
    for comp, data in sorted(per_component.items()):
        total = data["total"]
        comp_results[comp] = {
            "samples": total,
            "acc_center": data["correct"] / total if total > 0 else 0.0,
            "mde": data["dist_sum"] / total if total > 0 else 0.0,
        }

    return {
        "model": model_name,
        "n_total": n_total,
        "n_parsed": n_parsed,
        "n_failed": n_failed,
        "acc_center": acc_center,
        "mean_distance_error": mde,
        "per_component": comp_results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def print_results(
    results: list[dict[str, Any]],
    *,
    show_per_component_for: str | None = None,
) -> None:
    """Print a formatted evaluation report to stdout."""
    print("\n=== Grounding Evaluation Results ===\n")

    for res in results:
        label = res["model"]
        print(f"Model: {label}")
        print(f"  Acc@center:          {_fmt(res['acc_center'])}")
        print(f"  Mean Distance Error: {_fmt(res['mean_distance_error'])}")
        print(
            f"  Samples: {res['n_total']}  parsed: {res['n_parsed']}  "
            f"failed: {res['n_failed']}"
        )
        print()

    # Per-component breakdown for the last (fine-tuned) model, or explicitly named one
    target_res: dict[str, Any] | None = None
    if show_per_component_for is not None:
        for res in results:
            if res["model"] == show_per_component_for:
                target_res = res
                break
    if target_res is None and results:
        target_res = results[-1]

    if target_res and target_res.get("per_component"):
        label = target_res["model"]
        print(f"--- Per-Component Breakdown ({label}) ---")
        header = f"{'Component':<20}  {'Samples':>7}  {'Acc@center':>10}  {'MDE':>8}"
        print(header)
        print("-" * len(header))
        for comp, data in sorted(
            target_res["per_component"].items(),
            key=lambda kv: kv[1]["samples"],
            reverse=True,
        ):
            print(
                f"{comp:<20}  {data['samples']:>7}  "
                f"{data['acc_center']:>10.3f}  {data['mde']:>8.3f}"
            )
        print()

    # Before/after comparison when two models are present
    if len(results) == 2:
        baseline, finetuned = results[0], results[1]
        acc_delta = finetuned["acc_center"] - baseline["acc_center"]
        mde_delta = finetuned["mean_distance_error"] - baseline["mean_distance_error"]

        def _pct(delta: float, base: float) -> str:
            if base == 0.0:
                return "n/a"
            return f"{delta / base * 100:+.1f}%"

        print("--- Improvement ---")
        print(
            f"Acc@center:   {acc_delta:+.3f} "
            f"({_pct(acc_delta, baseline['acc_center'])})"
        )
        print(
            f"MDE:          {mde_delta:+.3f} "
            f"({_pct(mde_delta, baseline['mean_distance_error'])})"
        )
        print()


# ---------------------------------------------------------------------------
# Test-set loading
# ---------------------------------------------------------------------------


def load_test_samples(
    test_jsonl: Path, max_samples: int | None = None
) -> list[dict[str, Any]]:
    """Load VLM SFT samples from *test_jsonl*.

    Each line is a JSON object with a ``"messages"`` key.
    """
    samples: list[dict[str, Any]] = []
    with open(test_jsonl, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                samples.append(obj)
            except json.JSONDecodeError as exc:
                logger.warning("%s:%d – invalid JSON: %s", test_jsonl.name, lineno, exc)
            if max_samples is not None and len(samples) >= max_samples:
                break

    logger.info("Loaded %d test samples from %s", len(samples), test_jsonl)
    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate grounding models on the VLM test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test-jsonl",
        required=False,
        default=None,
        type=Path,
        metavar="FILE",
        help=(
            "Path to vlm_test.jsonl. Required unless --benchmark is given; "
            "mutually exclusive with --benchmark."
        ),
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        choices=_available_benchmarks(),
        help=(
            "Use a canned or external benchmark test set. 'internal' resolves "
            "to qontinui-train/evaluation/benchmarks/internal/test.jsonl; "
            "'screenspot_v2' / 'screenspot_pro' / 'osworld_g' download from "
            "HuggingFace and cache a normalized shadow jsonl under "
            "--benchmark-cache-dir."
        ),
    )
    parser.add_argument(
        "--benchmark-cache-dir",
        default=Path(__file__).parent.parent / ".benchmark-cache",
        type=Path,
        metavar="DIR",
        help=(
            "Cache root for external benchmark loaders. Each loader writes "
            "<dir>/<benchmark>/test.jsonl (and an images/ subdir) on first "
            "run, then reuses it on subsequent runs."
        ),
    )
    parser.add_argument(
        "--baseline-model",
        default=None,
        metavar="MODEL",
        help="Baseline model name (OpenAI-compatible model identifier)",
    )
    parser.add_argument(
        "--finetuned-model",
        default=None,
        metavar="MODEL",
        help="Fine-tuned model name",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:5800/v1",
        metavar="URL",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        type=Path,
        metavar="DIR",
        help="Directory where eval_results.json is saved",
    )
    parser.add_argument(
        "--max-samples",
        default=None,
        type=int,
        metavar="N",
        help="Limit evaluation to the first N samples",
    )
    parser.add_argument(
        "--tolerance",
        default=0.05,
        type=float,
        metavar="T",
        help="Normalised-distance tolerance for Acc@center",
    )
    parser.add_argument(
        "--grounding-dir",
        default=None,
        type=Path,
        metavar="DIR",
        help=(
            "Optional: directory containing grounding.jsonl for bbox lookup. "
            "Not yet used in the current tolerance-based Acc@center mode."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m qontinui_train.evaluation.grounding_eval``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s – %(message)s",
    )

    # Validate: at least one model must be provided
    if args.baseline_model is None and args.finetuned_model is None:
        parser.error(
            "At least one of --baseline-model or --finetuned-model must be provided."
        )

    # Resolve --benchmark / --test-jsonl (mutually exclusive; exactly one required)
    if args.benchmark is not None and args.test_jsonl is not None:
        parser.error("--benchmark and --test-jsonl are mutually exclusive.")
    if args.benchmark is None and args.test_jsonl is None:
        parser.error("Provide either --test-jsonl or --benchmark.")
    if args.benchmark is not None:
        try:
            args.test_jsonl = _resolve_benchmark(
                args.benchmark, Path(args.benchmark_cache_dir).resolve()
            )
        except KeyError:
            parser.error(f"Unknown benchmark: {args.benchmark!r}")

    # Load test data
    test_jsonl = args.test_jsonl.resolve()
    if not test_jsonl.exists():
        if args.benchmark is not None:
            logger.error(
                "Benchmark '%s' points to %s which does not exist. "
                "Run `make benchmark-%s` or drop test.jsonl at that path. "
                "See %s/README.md for the expected JSONL format.",
                args.benchmark,
                test_jsonl,
                args.benchmark,
                test_jsonl.parent,
            )
        else:
            logger.error("Test JSONL not found: %s", test_jsonl)
        sys.exit(1)

    samples = load_test_samples(test_jsonl, max_samples=args.max_samples)
    if not samples:
        logger.error("No samples loaded from %s", test_jsonl)
        sys.exit(1)

    # Run evaluation for each requested model
    all_results: list[dict[str, Any]] = []

    models_to_eval: list[tuple[str, str]] = []
    if args.baseline_model:
        models_to_eval.append((args.baseline_model, "baseline"))
    if args.finetuned_model:
        models_to_eval.append((args.finetuned_model, "fine-tuned"))

    for model_name, _role in models_to_eval:
        result = evaluate_model(
            samples=samples,
            model_name=model_name,
            api_base=args.api_base,
            tolerance=args.tolerance,
            max_samples=None,  # already sliced by load_test_samples
        )
        all_results.append(result)

        # Reproduction check: warn when running a known benchmark with a
        # model that has a published Acc@center reference.
        if args.benchmark is not None:
            try:
                from .reproduction_check import log_reproduction_check
            except ImportError:
                import sys as _sys

                _sys.path.insert(0, str(Path(__file__).parent))
                from reproduction_check import (  # type: ignore[import-not-found,no-redef]
                    log_reproduction_check,
                )
            log_reproduction_check(
                benchmark=args.benchmark,
                model_id=model_name,
                actual_acc_center=result["acc_center"],
            )

    # Print report
    finetuned_name = args.finetuned_model if args.finetuned_model else None
    print_results(all_results, show_per_component_for=finetuned_name)

    # Persist results
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.json"

    payload: dict[str, Any] = {
        "test_jsonl": str(test_jsonl),
        "benchmark": args.benchmark,
        "tolerance": args.tolerance,
        "max_samples": args.max_samples,
        "results": all_results,
    }
    if len(all_results) == 2:
        baseline_acc = all_results[0]["acc_center"]
        finetuned_acc = all_results[1]["acc_center"]
        baseline_mde = all_results[0]["mean_distance_error"]
        finetuned_mde = all_results[1]["mean_distance_error"]
        payload["improvement"] = {
            "acc_center_delta": finetuned_acc - baseline_acc,
            "mde_delta": finetuned_mde - baseline_mde,
        }

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Results saved to %s", results_path)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
