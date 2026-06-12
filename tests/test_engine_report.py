"""
Tests for the unified engine report — the instrument that makes refinement
calibration-driven.

`build_engine_report` runs a probability source through the walk-forward harness,
measures its calibration on resolved markets, and shows the portfolio allocation
its out-of-sample trades would produce. Swarm refinement is accepted only when
this report's Brier/log-loss improves out-of-sample.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signals.engine_report import build_engine_report


def _events(n: int) -> list[dict]:
    out = []
    for i in range(n):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
        out.append({"id": i, "timestamp": ts.isoformat(), "market_prob": 0.50,
                    "bid": 0.49, "ask": 0.51, "actual_outcome": float(i % 2)})
    return out


def test_report_bundles_calibration_oos_and_allocations():
    events = _events(12)
    report = build_engine_report(
        events,
        prob_fn=lambda e: 0.70,
        train_size=4, test_size=2, edge_min=0.05,
    )
    # calibration measured over every event
    assert report["calibration"]["n"] == 12
    assert "brier" in report["calibration"]
    assert "log_loss" in report["calibration"]
    # out-of-sample trade stats present
    assert "oos" in report and "n_trades" in report["oos"]
    # allocation per out-of-sample trade, capped book
    assert isinstance(report["allocations"], list)
    assert sum(a["size"] for a in report["allocations"]) <= 0.20 + 1e-9


def test_report_calibration_distinguishes_a_good_source_from_a_bad_one():
    """A source that nails the outcomes has a lower Brier than a constant 0.5
    hedge — the report must surface that difference."""
    events = _events(12)
    good = build_engine_report(events, prob_fn=lambda e: e["actual_outcome"],
                               train_size=4, test_size=2, edge_min=0.05)
    hedge = build_engine_report(events, prob_fn=lambda e: 0.5,
                                train_size=4, test_size=2, edge_min=0.05)
    assert good["calibration"]["brier"] < hedge["calibration"]["brier"]
