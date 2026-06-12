"""
Tests for the ensemble / source registry.

The brain has several tools (swarm, fair-value pricer, ML). The ensemble blends
their probabilities (weighted by calibration) and only fires when they agree and
there is edge versus the market. Output must plug into the walk-forward harness
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signals.ensemble import combine_sources, make_ensemble_brain


# ── Weighted blend ───────────────────────────────────────────────────────────

def test_equal_weights_average_the_sources():
    assert combine_sources([0.4, 0.6]) == pytest.approx(0.5)


def test_unequal_weights_bias_toward_the_heavier_source():
    blended = combine_sources([0.4, 0.6], weights=[3.0, 1.0])
    assert blended == pytest.approx(0.45)        # (3*0.4 + 1*0.6)/4
    assert blended < 0.5                          # pulled toward the 0.4 source


def test_weight_length_mismatch_raises():
    with pytest.raises(ValueError):
        combine_sources([0.4, 0.6], weights=[1.0])


# ── Ensemble brain (harness-compatible) ──────────────────────────────────────

def test_ensemble_buys_when_agreeing_sources_beat_the_market():
    brain = make_ensemble_brain(sources=[lambda e: 0.70, lambda e: 0.72])
    sig = brain({"market_prob": 0.50}, {"edge_min": 0.05})
    assert sig["side"] == "buy"


def test_ensemble_stands_down_when_sources_disagree():
    """One source says 0.20, the other 0.80. No consensus -> no trade, even
    though the blend (0.50) might look like edge against some market."""
    brain = make_ensemble_brain(sources=[lambda e: 0.20, lambda e: 0.80],
                                max_disagreement=0.15)
    sig = brain({"market_prob": 0.30}, {"edge_min": 0.05})
    assert sig["side"] is None


def test_ensemble_stands_down_when_edge_is_too_small():
    brain = make_ensemble_brain(sources=[lambda e: 0.52, lambda e: 0.52])
    sig = brain({"market_prob": 0.50}, {"edge_min": 0.05})
    assert sig["side"] is None


def test_ensemble_brain_runs_in_the_walk_forward_harness():
    """The ensemble brain must be a drop-in signal_fn for the OOS engine."""
    from src.signals.walkforward import WalkForwardBacktester
    from datetime import datetime, timedelta, timezone

    def ev(i):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
        return {"id": i, "timestamp": ts.isoformat(), "market_prob": 0.50,
                "bid": 0.49, "ask": 0.51, "actual_outcome": float(i % 2)}

    events = [ev(i) for i in range(12)]
    brain = make_ensemble_brain(sources=[lambda e: 0.70, lambda e: 0.71])
    wf = WalkForwardBacktester(train_size=4, test_size=2)
    result = wf.run(events, signal_fn=brain, param_grid=[{"edge_min": 0.05}])
    assert isinstance(result.oos_trades, list)
