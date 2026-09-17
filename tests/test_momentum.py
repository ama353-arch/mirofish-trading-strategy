"""
Tests for the match-momentum detector and divergence signal.

A momentum event is a fast swing in model win-probability (a run). We only trade
AROUND momentum events — that's the whole thesis. The two competing theses:

  - FADE: the crowd overshoots the run, so trade toward model fair value.
  - FOLLOW: the run signals a real state change both model and market lag, so
    trade with the run.

Backtesting decides which (if either) actually makes money.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sports.momentum import momentum_score, momentum_trade


# ── Momentum detection ───────────────────────────────────────────────────────

def test_momentum_score_is_the_recent_swing_in_win_probability():
    assert momentum_score([0.5, 0.55, 0.6, 0.7], window=3) == pytest.approx(0.20)
    assert momentum_score([0.7, 0.6, 0.5], window=2) == pytest.approx(-0.20)


def test_momentum_score_is_zero_without_enough_history():
    assert momentum_score([0.5, 0.6], window=3) == 0.0


# ── Fade thesis: trade toward model fair value ───────────────────────────────

def test_fade_sells_when_the_market_overshot_above_the_model():
    """Home surged (momentum +), crowd pushed price to 0.70 but the model says
    0.55 — overpriced, so fade = sell."""
    sig = momentum_trade(model_wp=0.55, market_price=0.70, momentum=0.20,
                         thesis="fade", edge_min=0.05)
    assert sig["side"] == "sell"


def test_fade_buys_when_the_market_undershot_below_the_model():
    sig = momentum_trade(model_wp=0.75, market_price=0.60, momentum=0.20,
                         thesis="fade", edge_min=0.05)
    assert sig["side"] == "buy"


def test_fade_stands_down_when_edge_is_too_small():
    sig = momentum_trade(model_wp=0.52, market_price=0.50, momentum=0.20,
                         thesis="fade", edge_min=0.05)
    assert sig["side"] is None


# ── Follow thesis: trade with the run ────────────────────────────────────────

def test_follow_buys_into_positive_momentum_regardless_of_static_edge():
    sig = momentum_trade(model_wp=0.55, market_price=0.70, momentum=0.20,
                         thesis="follow", edge_min=0.05)
    assert sig["side"] == "buy"


def test_follow_sells_into_negative_momentum():
    sig = momentum_trade(model_wp=0.45, market_price=0.30, momentum=-0.20,
                         thesis="follow", edge_min=0.05)
    assert sig["side"] == "sell"


# ── Both theses only trade around a momentum event ───────────────────────────

def test_no_trade_without_a_momentum_event():
    for thesis in ("fade", "follow"):
        sig = momentum_trade(model_wp=0.60, market_price=0.50, momentum=0.02,
                             thesis=thesis, edge_min=0.05, momentum_threshold=0.10)
        assert sig["side"] is None

# ── Game-level backtest: walk the snapshots, settle the trades ───────────────

def test_backtest_game_settles_each_momentum_trade():
    """On a game the home team eventually WINS (outcome 1), a momentum run with
    the market above the model means FADE (sell home) loses while FOLLOW (buy
    home) wins. The backtest must reflect that."""
    from src.sports.momentum import backtest_game
    # rising win-prob (home running), market sitting above the model fair value
    snapshots = [
        {"model_wp": 0.50, "market_price": 0.55},
        {"model_wp": 0.55, "market_price": 0.62},
        {"model_wp": 0.60, "market_price": 0.70},
        {"model_wp": 0.65, "market_price": 0.80},   # model swung +0.15 (a run);
        # market overshot to 0.80, well above the 0.65 model fair value
    ]
    fade = backtest_game(snapshots, final_outcome=1.0, thesis="fade",
                         window=3, edge_min=0.05, momentum_threshold=0.10)
    follow = backtest_game(snapshots, final_outcome=1.0, thesis="follow",
                           window=3, edge_min=0.05, momentum_threshold=0.10)
    assert len(fade) >= 1 and len(follow) >= 1
    assert all(t["pnl_per_dollar"] < 0 for t in fade)     # sold the eventual winner
    assert all(t["pnl_per_dollar"] > 0 for t in follow)   # rode the run, it held


def test_backtest_game_makes_no_trades_without_momentum():
    from src.sports.momentum import backtest_game
    flat = [{"model_wp": 0.5, "market_price": 0.5} for _ in range(6)]
    trades = backtest_game(flat, final_outcome=1.0, thesis="fade",
                           momentum_threshold=0.10)
    assert trades == []


# ── Price-momentum backtest (market price only, no game-state model) ──────────

def test_price_momentum_follow_vs_fade_on_a_rising_winner():
    """Win-price rises through the game and the home team WINS (outcome 1).
    FOLLOW (ride the rise) wins; FADE (bet reversion) loses."""
    from src.sports.momentum import backtest_price_momentum
    prices = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    follow = backtest_price_momentum(prices, final_outcome=1.0, thesis="follow",
                                     window=3, momentum_threshold=0.05)
    fade = backtest_price_momentum(prices, final_outcome=1.0, thesis="fade",
                                   window=3, momentum_threshold=0.05)
    assert follow and all(t["pnl_per_dollar"] > 0 for t in follow)
    assert fade and all(t["pnl_per_dollar"] < 0 for t in fade)


def test_price_momentum_ignores_small_moves():
    from src.sports.momentum import backtest_price_momentum
    flat = [0.50, 0.51, 0.50, 0.51, 0.50, 0.51]
    assert backtest_price_momentum(flat, 1.0, thesis="fade",
                                   window=3, momentum_threshold=0.05) == []


def test_price_momentum_skips_prices_outside_the_contested_band():
    """Trades only fire while the game is contested (price in [min,max]); the
    blowout convergence toward 0/1 is excluded so it can't leak the result."""
    from src.sports.momentum import backtest_price_momentum
    prices = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    trades = backtest_price_momentum(prices, 1.0, thesis="follow", window=3,
                                     momentum_threshold=0.05,
                                     min_price=0.15, max_price=0.85)
    assert trades
    assert all(t["entry_price"] <= 0.85 for t in trades)


# ── True-edge test: does a move continue or revert over the NEXT few minutes? ──

def test_forward_returns_positive_when_momentum_continues():
    """A steadily rising price means momentum keeps going: the signed forward
    return (aligned to momentum direction) is positive."""
    from src.sports.momentum import momentum_forward_returns
    prices = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    fr = momentum_forward_returns(prices, window=2, horizon=2, momentum_threshold=0.05)
    assert fr and sum(fr) / len(fr) > 0


def test_forward_returns_negative_when_momentum_reverts():
    """A price that runs up then mean-reverts gives a negative signed forward
    return at the peak — reversion, the fade edge."""
    from src.sports.momentum import momentum_forward_returns
    # a spike that snaps back (true reversion), not a sustained reversal
    prices = [0.50, 0.50, 0.60, 0.50, 0.50, 0.50]
    fr = momentum_forward_returns(prices, window=2, horizon=2, momentum_threshold=0.05)
    assert fr and sum(fr) / len(fr) < 0
