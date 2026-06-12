"""
Tests for the unified historical-event schema and source adapters.

Every data source (Kalshi, Polymarket, a CSV export) must normalize into ONE
resolved-event shape so the same brain and the same walk-forward harness can
backtest any pathway identically. These tests pin the normalization so a
schema drift in one source can't silently corrupt the backtest.

Required unified fields:
    id, timestamp, market_prob, bid, ask, actual_outcome, source
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.historical import (
    from_kalshi_market,
    from_polymarket_market,
    REQUIRED_FIELDS,
)


# ── Kalshi adapter ────────────────────────────────────────────────────────────

def test_kalshi_resolved_market_maps_to_unified_schema():
    """Kalshi quotes YES in cents and settles to 'yes'/'no'. The adapter must
    convert cents->probability and result->binary outcome."""
    raw = {
        "ticker": "NBA-LAL-WIN-24",
        "close_time": "2026-04-01T03:00:00Z",
        "yes_bid": 58,     # cents
        "yes_ask": 62,
        "last_price": 60,
        "result": "yes",
    }
    ev = from_kalshi_market(raw)
    assert set(REQUIRED_FIELDS) <= set(ev)
    assert ev["id"] == "NBA-LAL-WIN-24"
    assert ev["bid"] == pytest.approx(0.58)
    assert ev["ask"] == pytest.approx(0.62)
    assert ev["market_prob"] == pytest.approx(0.60)
    assert ev["actual_outcome"] == 1.0
    assert ev["source"] == "kalshi"


def test_kalshi_no_result_means_outcome_no():
    raw = {"ticker": "X", "close_time": "2026-04-01T03:00:00Z",
           "yes_bid": 40, "yes_ask": 44, "last_price": 42, "result": "no"}
    assert from_kalshi_market(raw)["actual_outcome"] == 0.0


def test_kalshi_unsettled_market_is_rejected():
    """An unsettled market (no/blank result) cannot be backtested — the
    adapter returns None so it is filtered out, never silently scored as 0."""
    raw = {"ticker": "X", "close_time": "2026-04-01T03:00:00Z",
           "yes_bid": 40, "yes_ask": 44, "last_price": 42, "result": ""}
    assert from_kalshi_market(raw) is None


# ── Polymarket adapter ───────────────────────────────────────────────────────

def test_polymarket_resolved_market_maps_to_unified_schema():
    """Polymarket prices are already 0-1 probabilities. A resolved binary
    market carries a winning outcome string."""
    raw = {
        "id": "0xabc",
        "question": "Will it rain?",
        "end_date_iso": "2026-05-01T00:00:00Z",
        "best_bid": 0.33,
        "best_ask": 0.37,
        "last_trade_price": 0.35,
        "resolved": True,
        "winning_outcome": "Yes",
    }
    ev = from_polymarket_market(raw)
    assert set(REQUIRED_FIELDS) <= set(ev)
    assert ev["id"] == "0xabc"
    assert ev["bid"] == pytest.approx(0.33)
    assert ev["ask"] == pytest.approx(0.37)
    assert ev["market_prob"] == pytest.approx(0.35)
    assert ev["actual_outcome"] == 1.0
    assert ev["source"] == "polymarket"


def test_polymarket_unresolved_market_is_rejected():
    raw = {
        "id": "0xdef", "end_date_iso": "2026-05-01T00:00:00Z",
        "best_bid": 0.5, "best_ask": 0.54, "last_trade_price": 0.52,
        "resolved": False, "winning_outcome": None,
    }
    assert from_polymarket_market(raw) is None


def test_both_adapters_produce_schemas_the_harness_can_consume():
    """The whole point: a Kalshi event and a Polymarket event are
    interchangeable inputs to the walk-forward harness."""
    from src.signals.walkforward import _simulate_trade

    k = from_kalshi_market({
        "ticker": "K", "close_time": "2026-04-01T03:00:00Z",
        "yes_bid": 58, "yes_ask": 62, "last_price": 60, "result": "yes"})
    p = from_polymarket_market({
        "id": "P", "end_date_iso": "2026-05-01T00:00:00Z",
        "best_bid": 0.33, "best_ask": 0.37, "last_trade_price": 0.35,
        "resolved": True, "winning_outcome": "No"})

    # Both must be tradeable by the same simulator without adaptation.
    assert _simulate_trade(k, "buy", 0.02)["pnl_per_dollar"] > 0   # bought YES, won
    assert _simulate_trade(p, "buy", 0.02)["pnl_per_dollar"] == -1.0  # bought YES, lost
