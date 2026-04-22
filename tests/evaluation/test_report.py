"""Tests for the cross-benchmark markdown report generator."""

from __future__ import annotations

import json
from pathlib import Path

from report import build_markdown, main

MODEL_UITARS = "ByteDance-Seed/UI-TARS-1.5-7B"
MODEL_QONTINUI = "qontinui-grounding-v1"


def _write_eval_blob(
    path: Path, benchmark: str, entries: list[tuple[str, float, float]]
) -> None:
    blob = {
        "test_jsonl": f"/tmp/{benchmark}.jsonl",
        "benchmark": benchmark,
        "tolerance": 0.05,
        "max_samples": None,
        "results": [
            {
                "model": model,
                "n_total": 100,
                "n_parsed": 100,
                "n_failed": 0,
                "acc_center": acc,
                "mean_distance_error": mde,
                "per_component": {},
            }
            for (model, acc, mde) in entries
        ],
    }
    path.write_text(json.dumps(blob), encoding="utf-8")


def test_build_markdown_two_benchmarks_two_models(tmp_path: Path):
    p1 = tmp_path / "ss_v2.json"
    p2 = tmp_path / "osworld.json"
    _write_eval_blob(
        p1,
        "screenspot_v2",
        [
            (MODEL_UITARS, 0.943, 0.021),
            (MODEL_QONTINUI, 0.960, 0.018),
        ],
    )
    _write_eval_blob(
        p2,
        "osworld_g",
        [
            (MODEL_UITARS, 0.419, 0.092),
            (MODEL_QONTINUI, 0.500, 0.077),
        ],
    )

    with open(p1) as fh1, open(p2) as fh2:
        md = build_markdown([(p1, json.load(fh1)), (p2, json.load(fh2))])

    # Header has both model columns (order preserved from first seen)
    header_line = md.splitlines()[0]
    assert MODEL_UITARS in header_line
    assert MODEL_QONTINUI in header_line

    # Rows are sorted alphabetically by benchmark name.
    body = md.splitlines()[2:4]
    assert body[0].startswith("| osworld_g |")
    assert body[1].startswith("| screenspot_v2 |")

    # Cells include Acc / MDE
    assert "0.943 / 0.021" in md
    assert "0.500 / 0.077" in md


def test_build_markdown_flags_reproduction_failures(tmp_path: Path):
    # UI-TARS reported wildly wrong on screenspot_v2 -> should be flagged
    p = tmp_path / "bad.json"
    _write_eval_blob(
        p,
        "screenspot_v2",
        [(MODEL_UITARS, 0.70, 0.05)],  # published=0.943, delta=-0.243
    )
    with open(p) as fh:
        md = build_markdown([(p, json.load(fh))])
    assert "!!" in md, "expected flag marker on failed reproduction"


def test_build_markdown_omits_flag_when_within_tolerance(tmp_path: Path):
    p = tmp_path / "ok.json"
    _write_eval_blob(
        p,
        "screenspot_v2",
        [(MODEL_UITARS, 0.94, 0.02)],  # delta ~ -0.003
    )
    with open(p) as fh:
        md = build_markdown([(p, json.load(fh))])
    # UI-TARS row should not be flagged
    for line in md.splitlines():
        if line.startswith("| screenspot_v2"):
            assert "!!" not in line


def test_main_writes_output_md(tmp_path: Path):
    eval_json = tmp_path / "e.json"
    _write_eval_blob(
        eval_json,
        "screenspot_v2",
        [(MODEL_UITARS, 0.94, 0.02)],
    )
    out_md = tmp_path / "out" / "report.md"

    main(
        [
            "--eval-output",
            str(eval_json),
            "--output-md",
            str(out_md),
            "--log-level",
            "WARNING",
        ]
    )

    assert out_md.exists()
    body = out_md.read_text(encoding="utf-8")
    assert MODEL_UITARS in body
    assert "screenspot_v2" in body
