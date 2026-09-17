"""
Tests for the arbitrage detector — the market-neutral, no-edge-model strategy.

Structural arb: a set of mutually-exclusive outcomes whose ask prices sum to less
than 1 (buy them all, one pays out 1, locked profit). Cross-venue arb: the same
event priced so you can buy YES on one venue and NO on the other for under 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signals.arbitrage import find_structural_arb, find_cross_venue_arb


# ── Structural arbitrage ─────────────────────────────────────────────────────

def test_underpriced_exclusive_set_is_an_arb():
    arb = find_structural_arb([0.30, 0.30, 0.30])
    assert arb is not None
    assert arb["guaranteed_profit"] == pytest.approx(0.10)


def test_fees_eat_into_the_guaranteed_profit():
    arb = find_structural_arb([0.30, 0.30, 0.30], fee=0.05)
    assert arb["guaranteed_profit"] == pytest.approx(0.05)


def test_fully_priced_set_is_not_an_arb():
    assert find_structural_arb([0.34, 0.33, 0.33]) is None      # sums to 1.0
    assert find_structural_arb([0.40, 0.40, 0.40]) is None      # over 1.0


def test_fee_that_erases_the_edge_returns_none():
    assert find_structural_arb([0.30, 0.30, 0.30], fee=0.15) is None


def test_single_or_empty_leg_is_not_an_arb():
    assert find_structural_arb([0.30]) is None
    assert find_structural_arb([]) is None


# ── Cross-venue arbitrage ────────────────────────────────────────────────────

def test_cross_venue_mispricing_is_an_arb():
    """Buy YES on A at 0.40 and NO on B at 0.55 -> 0.95 total, locks 0.05."""
    a = {"yes_ask": 0.40, "no_ask": 0.65}
    b = {"yes_ask": 0.50, "no_ask": 0.55}
    arb = find_cross_venue_arb(a, b)
    assert arb is not None
    assert arb["guaranteed_profit"] == pytest.approx(0.05)


def test_cross_venue_no_mispricing_returns_none():
    a = {"yes_ask": 0.40, "no_ask": 0.62}
    b = {"yes_ask": 0.50, "no_ask": 0.62}   # both directions sum to >= 1
    assert find_cross_venue_arb(a, b) is None
