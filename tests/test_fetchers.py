"""
Tests for the READ-ONLY historical data fetchers.

These parse real Kalshi / Polymarket API response shapes into the unified
resolved-event schema the backtest consumes. The parsing is pinned here against
fixtures so we never need a live network to know the mapping is correct — and so
a field rename on either venue fails loudly instead of silently corrupting data.

Scope guarantee: these fetchers are read-only. They never place orders.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetchers import parse_kalshi_markets, parse_polymarket_markets


# ── Kalshi: /markets response -> unified events ──────────────────────────────

def test_parse_kalshi_keeps_settled_drops_unsettled():
    response = {
        "markets": [
            {"ticker": "K1", "close_time": "2026-04-01T03:00:00Z",
             "yes_bid": 58, "yes_ask": 62, "last_price": 60, "result": "yes",
             "status": "settled"},
            {"ticker": "K2", "close_time": "2026-04-02T03:00:00Z",
             "yes_bid": 40, "yes_ask": 44, "last_price": 42, "result": "no",
             "status": "settled"},
            {"ticker": "K3", "close_time": "2026-04-03T03:00:00Z",
             "yes_bid": 50, "yes_ask": 54, "last_price": 52, "result": "",
             "status": "active"},
        ]
    }
    events = parse_kalshi_markets(response)
    assert len(events) == 2                      # unsettled K3 dropped
    by_id = {e["id"]: e for e in events}
    assert by_id["K1"]["actual_outcome"] == 1.0
    assert by_id["K2"]["actual_outcome"] == 0.0
    assert by_id["K1"]["source"] == "kalshi"


# ── Polymarket: Gamma market objects -> unified events ───────────────────────

def test_parse_polymarket_maps_gamma_fields_and_resolves_winner():
    """Gamma encodes the winner in outcomePrices (the '1' price) aligned with
    outcomes. A resolved YES market has outcomePrices ['1','0'] for
    outcomes ['Yes','No']."""
    gamma = [
        {"id": "0xPM1", "question": "Will it rain?",
         "endDate": "2026-05-01T00:00:00Z",
         "bestBid": 0.33, "bestAsk": 0.37, "lastTradePrice": 0.35,
         "closed": True, "outcomes": '["Yes","No"]',
         "outcomePrices": '["1","0"]'},
        {"id": "0xPM2", "question": "Still open?",
         "endDate": "2026-05-02T00:00:00Z",
         "bestBid": 0.5, "bestAsk": 0.54, "lastTradePrice": 0.52,
         "closed": False, "outcomes": '["Yes","No"]',
         "outcomePrices": '["0.52","0.48"]'},
    ]
    events = parse_polymarket_markets(gamma)
    assert len(events) == 1                      # open PM2 dropped
    ev = events[0]
    assert ev["id"] == "0xPM1"
    assert ev["actual_outcome"] == 1.0           # Yes won
    assert ev["bid"] == pytest.approx(0.33)
    assert ev["ask"] == pytest.approx(0.37)
    assert ev["source"] == "polymarket"


def test_parse_polymarket_resolves_a_no_winner():
    gamma = [
        {"id": "0xPM3", "question": "No-win case",
         "endDate": "2026-05-03T00:00:00Z",
         "bestBid": 0.6, "bestAsk": 0.64, "lastTradePrice": 0.62,
         "closed": True, "outcomes": '["Yes","No"]',
         "outcomePrices": '["0","1"]'},
    ]
    events = parse_polymarket_markets(gamma)
    assert events[0]["actual_outcome"] == 0.0


def test_parsed_events_feed_the_harness_directly():
    """A parsed event must be tradeable by the harness simulator unchanged."""
    from src.signals.walkforward import _simulate_trade
    events = parse_kalshi_markets({"markets": [
        {"ticker": "K1", "close_time": "2026-04-01T03:00:00Z",
         "yes_bid": 58, "yes_ask": 62, "last_price": 60, "result": "yes"}]})
    trade = _simulate_trade(events[0], "buy", 0.02)
    assert trade["pnl_per_dollar"] > 0           # bought YES, it won
