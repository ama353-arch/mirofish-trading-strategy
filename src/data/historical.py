"""
historical.py — unified resolved-event schema + per-source adapters.

Every backtest pathway (Kalshi, Polymarket, CSV export) normalizes into ONE
shape so the brain and the walk-forward harness treat them interchangeably:

    {
        "id":             str,    # stable market identifier
        "timestamp":      str,    # ISO 8601, the resolution/close time
        "market_prob":    float,  # market-implied P(YES) in [0, 1]
        "bid":            float,  # best YES bid in [0, 1]
        "ask":            float,  # best YES ask in [0, 1]
        "actual_outcome": float,  # 1.0 if YES resolved true, else 0.0
        "source":         str,    # "kalshi" | "polymarket" | ...
    }

Adapters return None for unresolved/unsettled markets so they are filtered
out rather than silently scored as a loss. Live fetchers (network) map raw
API JSON into the dicts these adapters accept; the normalization logic lives
here so it is testable without a network.
"""

from __future__ import annotations

REQUIRED_FIELDS = ("id", "timestamp", "market_prob", "bid", "ask",
                   "actual_outcome", "source")

_YES_RESULTS = {"yes", "y", "true", "1"}
_NO_RESULTS = {"no", "n", "false", "0"}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _kalshi_price(raw: dict, dollars_key: str, cents_key: str):
    """Read a Kalshi price, preferring the current `*_dollars` field (already
    in 0-1) and falling back to the older cents field (0-100)."""
    if raw.get(dollars_key) is not None:
        return float(raw[dollars_key])
    if raw.get(cents_key) is not None:
        return raw[cents_key] / 100.0
    return None


def from_kalshi_market(raw: dict) -> dict | None:
    """Normalize a resolved Kalshi market. The live API quotes YES as
    `yes_bid_dollars` / `yes_ask_dollars` (0-1); older payloads used cents.
    `result` settles to 'yes'/'no'. Returns None if unsettled."""
    result = str(raw.get("result", "")).strip().lower()
    if result in _YES_RESULTS:
        outcome = 1.0
    elif result in _NO_RESULTS:
        outcome = 0.0
    else:
        return None  # unsettled — cannot backtest

    bid = _clamp01(_kalshi_price(raw, "yes_bid_dollars", "yes_bid") or 0.0)
    ask = _clamp01(_kalshi_price(raw, "yes_ask_dollars", "yes_ask") or 0.0)
    last = _kalshi_price(raw, "last_price_dollars", "last_price")
    market_prob = _clamp01(last) if last is not None else (bid + ask) / 2.0

    return {
        "id": raw["ticker"],
        "timestamp": raw.get("close_time") or raw.get("settlement_ts"),
        "market_prob": market_prob,
        "bid": bid,
        "ask": ask,
        "actual_outcome": outcome,
        "source": "kalshi",
    }


def from_polymarket_market(raw: dict) -> dict | None:
    """Normalize a resolved Polymarket binary market. Prices are already 0-1
    probabilities. Returns None if not yet resolved."""
    if not raw.get("resolved"):
        return None
    winner = raw.get("winning_outcome")
    if winner is None:
        return None
    w = str(winner).strip().lower()
    if w in _YES_RESULTS:
        outcome = 1.0
    elif w in _NO_RESULTS:
        outcome = 0.0
    else:
        return None  # ambiguous / multi-outcome — out of scope for binary backtest

    bid = _clamp01(raw["best_bid"])
    ask = _clamp01(raw["best_ask"])
    last = raw.get("last_trade_price")
    market_prob = _clamp01(last) if last is not None else (bid + ask) / 2.0

    return {
        "id": raw["id"],
        "timestamp": raw.get("end_date_iso") or raw.get("endDate"),
        "market_prob": market_prob,
        "bid": bid,
        "ask": ask,
        "actual_outcome": outcome,
        "source": "polymarket",
    }
