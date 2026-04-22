"""Tests for the published-number reproduction check."""

from __future__ import annotations

import logging

import pytest
from reproduction_check import (
    _PUBLISHED,
    _REPRODUCE_TOLERANCE,
    check_published_reproduction,
    log_reproduction_check,
)

MODEL_UITARS = "ByteDance-Seed/UI-TARS-1.5-7B"


def test_within_threshold_returns_true_and_correct_delta():
    # published screenspot_v2 = 0.943
    actual = 0.935  # delta = -0.008 (within 0.02)
    reproduces, delta = check_published_reproduction(
        "screenspot_v2", MODEL_UITARS, actual
    )
    assert reproduces is True
    assert delta == pytest.approx(actual - _PUBLISHED[("screenspot_v2", MODEL_UITARS)])
    assert abs(delta) <= _REPRODUCE_TOLERANCE


def test_outside_threshold_returns_false_with_signed_delta():
    # published screenspot_pro = 0.497
    actual = 0.60  # delta = +0.103
    reproduces, delta = check_published_reproduction(
        "screenspot_pro", MODEL_UITARS, actual
    )
    assert reproduces is False
    assert delta == pytest.approx(actual - _PUBLISHED[("screenspot_pro", MODEL_UITARS)])
    assert delta > _REPRODUCE_TOLERANCE


def test_outside_threshold_negative_delta():
    actual = 0.20  # published osworld_g = 0.419 -> delta ≈ -0.219
    reproduces, delta = check_published_reproduction("osworld_g", MODEL_UITARS, actual)
    assert reproduces is False
    assert delta < -_REPRODUCE_TOLERANCE


def test_unknown_benchmark_is_no_op():
    reproduces, delta = check_published_reproduction(
        "does_not_exist", MODEL_UITARS, 0.5
    )
    assert reproduces is True
    assert delta == 0.0


def test_unknown_model_is_no_op():
    reproduces, delta = check_published_reproduction(
        "screenspot_v2", "some-other-model", 0.5
    )
    assert reproduces is True
    assert delta == 0.0


def test_log_reproduction_check_warns_on_failure(caplog):
    with caplog.at_level(logging.WARNING, logger="reproduction_check"):
        log_reproduction_check("screenspot_v2", MODEL_UITARS, 0.70)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a WARNING log when reproduction fails"
    assert "Reproduction FAILED" in warnings[0].getMessage()


def test_log_reproduction_check_info_on_success(caplog):
    with caplog.at_level(logging.INFO, logger="reproduction_check"):
        log_reproduction_check(
            "screenspot_v2",
            MODEL_UITARS,
            _PUBLISHED[("screenspot_v2", MODEL_UITARS)],
        )
    msgs = [r.getMessage() for r in caplog.records]
    assert any("Reproduction OK" in m for m in msgs)
