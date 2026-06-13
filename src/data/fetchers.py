"""
fetchers.py — READ-ONLY historical data fetchers for Kalshi and Polymarket.

These pull RESOLVED markets + prices and normalize them into the unified schema
(see historical.py) so the walk-forward harness and calibration can run on real
data. They are deliberately read-only: there is no order-placement code here, and
they read only read-only / market-data credentials from the environment.

The PARSE functions (pure, no network) are unit-tested against fixtures so the
field mapping is verified without a live API. The FETCH functions are the thin
network layer on top; they page through the read-only endpoints and call parse.
"""

from __future__ import annotations

import json
import os

from src.data.historical import from_kalshi_market, from_polymarket_market


# ── Parsing (pure, fixture-tested) ───────────────────────────────────────────

def parse_kalshi_markets(response: dict) -> list[dict]:
    """Normalize a Kalshi `/markets` response into unified resolved events.

    Unsettled markets (blank/non yes-no `result`) are dropped by the adapter.
    """
    out = []
    for raw in response.get("markets", []):
        ev = from_kalshi_market(raw)
        if ev is not None:
            out.append(ev)
    return out


def parse_polymarket_markets(markets: list[dict]) -> list[dict]:
    """Normalize Polymarket Gamma market objects into unified resolved events.

    Gamma encodes the winner in `outcomePrices` (a JSON-string array aligned with
    `outcomes`): the outcome whose price resolved to '1' won. We translate Gamma's
    field names into the shape `from_polymarket_market` expects, then normalize.
    """
    out = []
    for m in markets:
        resolved = bool(m.get("closed") or m.get("umaResolutionStatus") == "resolved")
        winning_outcome = None
        if resolved:
            winning_outcome = _gamma_winner(m)
        adapted = {
            "id": m.get("id") or m.get("conditionId"),
            "end_date_iso": m.get("endDate") or m.get("end_date_iso"),
            "best_bid": m.get("bestBid", m.get("best_bid")),
            "best_ask": m.get("bestAsk", m.get("best_ask")),
            "last_trade_price": m.get("lastTradePrice", m.get("last_trade_price")),
            "resolved": resolved,
            "winning_outcome": winning_outcome,
        }
        ev = from_polymarket_market(adapted)
        if ev is not None:
            out.append(ev)
    return out


def _gamma_winner(market: dict) -> str | None:
    """The outcome label whose resolved price is 1 (the winner), or None."""
    outcomes = _maybe_json(market.get("outcomes"))
    prices = _maybe_json(market.get("outcomePrices"))
    if not outcomes or not prices or len(outcomes) != len(prices):
        return None
    for label, price in zip(outcomes, prices):
        if abs(float(price) - 1.0) < 1e-9:
            return label
    return None


def _maybe_json(value):
    """Gamma returns these as JSON strings; accept already-parsed lists too."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


# ── Network layer (thin; read-only endpoints) ────────────────────────────────

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com/markets"


def fetch_kalshi_resolved(limit: int = 200, session=None) -> list[dict]:
    """Fetch settled Kalshi markets (read-only). Returns unified events.

    Read-only: hits only the public `/markets` endpoint with `status=settled`.
    A read-only API key (from env) is used only to relax rate limits; the call
    works unauthenticated for recent data.
    """
    import requests
    session = session or requests.Session()
    params = {"limit": limit, "status": "settled"}
    headers = {}
    key_id = os.environ.get("KALSHI_API_KEY_ID")
    if key_id:
        headers["KALSHI-ACCESS-KEY"] = key_id  # read-only id; no signing of orders
    resp = session.get(f"{KALSHI_BASE}/markets", params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return parse_kalshi_markets(resp.json())


def fetch_polymarket_resolved(limit: int = 200, session=None) -> list[dict]:
    """Fetch resolved Polymarket markets via the public Gamma API (no key).

    Read-only by construction: Gamma is a public read endpoint.
    """
    import requests
    session = session or requests.Session()
    params = {"limit": limit, "closed": "true"}
    resp = session.get(POLYMARKET_GAMMA, params=params, timeout=30)
    resp.raise_for_status()
    return parse_polymarket_markets(resp.json())
