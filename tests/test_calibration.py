"""
Tests for the calibration harness.

A probability source is only worth money if it is calibrated: when it says 60%,
the event should happen ~60% of the time. These tests pin Brier score, log loss,
and the reliability curve so we can measure any source (swarm, pricer, ML) before
trusting it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signals.calibration import (
    brier_score,
    log_loss,
    reliability_curve,
    calibration_report,
    Calibrator,
)


# ── Brier score ──────────────────────────────────────────────────────────────

def test_perfect_prediction_beats_a_hedge():
    outcomes = [1, 1, 0, 0]
    perfect = [1.0, 1.0, 0.0, 0.0]
    hedge = [0.5, 0.5, 0.5, 0.5]
    assert brier_score(perfect, outcomes) < brier_score(hedge, outcomes)
    assert brier_score(perfect, outcomes) == pytest.approx(0.0)


def test_always_base_rate_brier_equals_base_rate_variance():
    """3 ones, 7 zeros -> base rate 0.3. A source that always predicts 0.3 has
    Brier == 0.3 * 0.7 = 0.21."""
    outcomes = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    probs = [0.3] * 10
    assert brier_score(probs, outcomes) == pytest.approx(0.21)


# ── Log loss ─────────────────────────────────────────────────────────────────

def test_log_loss_is_finite_at_zero_and_one():
    """Without an eps clamp this would be infinite; the clamp keeps it finite."""
    assert math.isfinite(log_loss([1.0], [1]))
    assert math.isfinite(log_loss([0.0], [1]))


def test_confident_correct_beats_hedged_correct():
    assert log_loss([0.99], [1]) < log_loss([0.6], [1])


# ── Reliability curve ────────────────────────────────────────────────────────

def test_reliability_bins_sum_to_input_count_and_omit_empty():
    probs = [0.05, 0.15, 0.15, 0.95, 0.95, 0.95]
    outcomes = [0, 0, 1, 1, 1, 1]
    curve = reliability_curve(probs, outcomes, n_bins=10)
    assert sum(b["count"] for b in curve) == len(probs)
    # only 3 of 10 bins are populated; empties are omitted, not NaN
    assert len(curve) == 3
    for b in curve:
        assert 0.0 <= b["predicted"] <= 1.0
        assert 0.0 <= b["observed"] <= 1.0


# ── Report + validation ──────────────────────────────────────────────────────

def test_calibration_report_carries_core_metrics():
    outcomes = [1, 0, 1, 0, 1, 0]
    probs = [0.6, 0.4, 0.7, 0.3, 0.55, 0.45]
    rep = calibration_report(probs, outcomes)
    assert rep["n"] == 6
    assert rep["base_rate"] == pytest.approx(0.5)
    assert "brier" in rep and "log_loss" in rep
    assert isinstance(rep["reliability"], list)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        brier_score([0.5, 0.5], [1])


# ── Fitted calibration layer (Platt scaling) — swarm refinement ──────────────

def test_calibrator_corrects_an_overconfident_source():
    """An overconfident source (says 0.95 but only ~60% happen) gets pulled
    toward reality; fitted calibration lowers its Brier."""
    raw = [0.95, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05]
    outcomes = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]   # far less separable than 0.95/0.05 implies
    cal = Calibrator().fit(raw, outcomes)
    calibrated = cal.transform(raw)
    assert brier_score(calibrated, outcomes) <= brier_score(raw, outcomes)
    assert all(0.0 <= p <= 1.0 for p in calibrated)


def test_calibrator_transform_is_monotonic_in_the_raw_probability():
    """Calibration may rescale, but a higher raw probability must never map to a
    lower calibrated probability — it must preserve ranking."""
    raw = [0.2, 0.4, 0.6, 0.8]
    outcomes = [0, 0, 1, 1]
    cal = Calibrator().fit(raw, outcomes)
    out = cal.transform([0.1, 0.3, 0.5, 0.7, 0.9])
    assert out == sorted(out)
