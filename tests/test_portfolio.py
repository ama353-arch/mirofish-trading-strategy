"""
Tests for the portfolio / capital-allocation layer.

Real trading fires many positions at once. This layer enforces the bankroll
rules across concurrent positions: fractional Kelly per position, a per-position
cap, a total-deployed cap, and correlation handling so we don't bet the same
risk many times over.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signals.portfolio import kelly_fraction, allocate


# ── Fractional Kelly ─────────────────────────────────────────────────────────

def test_kelly_is_zero_without_positive_edge():
    assert kelly_fraction(edge=0.0, odds=1.0) == 0.0
    assert kelly_fraction(edge=-0.1, odds=1.0) == 0.0


def test_kelly_grows_with_edge_and_respects_the_fraction():
    small = kelly_fraction(edge=0.1, odds=1.0)
    big = kelly_fraction(edge=0.2, odds=1.0)
    assert 0.0 < small < big
    # quarter-Kelly is exactly 1/4 of full-Kelly
    assert kelly_fraction(0.2, 1.0, fraction=0.25) == pytest.approx(
        0.25 * kelly_fraction(0.2, 1.0, fraction=1.0))


# ── Allocation ───────────────────────────────────────────────────────────────

def test_no_position_exceeds_the_per_position_cap():
    signals = [
        {"id": "a", "edge": 0.5, "odds": 1.0},
        {"id": "b", "edge": 0.6, "odds": 1.0},
    ]
    allocs = allocate(signals, max_position=0.05, kelly_fraction=1.0)
    assert all(a["size"] <= 0.05 + 1e-9 for a in allocs)


def test_total_deployed_is_capped_and_scaled_proportionally():
    # raw sizes 0.10 and 0.20 (odds=1, full Kelly); sum 0.30 > 0.20 cap.
    signals = [
        {"id": "a", "edge": 0.10, "odds": 1.0},
        {"id": "b", "edge": 0.20, "odds": 1.0},
    ]
    allocs = allocate(signals, max_deployed=0.20, max_position=1.0,
                      kelly_fraction=1.0)
    sizes = {a["id"]: a["size"] for a in allocs}
    assert sum(sizes.values()) == pytest.approx(0.20)
    # 1:2 ratio preserved after proportional scaling
    assert sizes["b"] == pytest.approx(2 * sizes["a"])


def test_correlated_signals_share_one_budget():
    """Two signals of equal edge in the same correlation group together get no
    more than a single uncorrelated signal of that edge would."""
    corr = [
        {"id": "a", "edge": 0.10, "odds": 1.0, "group": "g1"},
        {"id": "b", "edge": 0.10, "odds": 1.0, "group": "g1"},
    ]
    solo = [{"id": "c", "edge": 0.10, "odds": 1.0}]
    corr_alloc = allocate(corr, max_deployed=1.0, max_position=1.0, kelly_fraction=1.0)
    solo_alloc = allocate(solo, max_deployed=1.0, max_position=1.0, kelly_fraction=1.0)
    corr_total = sum(a["size"] for a in corr_alloc)
    solo_total = sum(a["size"] for a in solo_alloc)
    assert corr_total == pytest.approx(solo_total)
