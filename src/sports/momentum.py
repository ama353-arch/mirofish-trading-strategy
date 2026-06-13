"""
momentum.py — match-momentum detection and the divergence trade.

The thesis: we only trade AROUND momentum events (fast swings in model win
probability), because that is when the live market mis-prices the game. Two
competing theses, both supported here so the backtest can decide:

  - FADE   : the crowd overshoots the run; trade toward model fair value.
  - FOLLOW : the run signals a real state change the model/market lag; trade
             with the run.
"""

from __future__ import annotations


def momentum_score(wp_history, window: int = 3) -> float:
    """Recent swing in model win-probability: wp[-1] - wp[-1-window].

    Positive means the home team gained probability (a run in its favor).
    Returns 0.0 when there isn't enough history to span the window.
    """
    if len(wp_history) < window + 1:
        return 0.0
    return wp_history[-1] - wp_history[-1 - window]


def momentum_trade(
    model_wp: float,
    market_price: float,
    momentum: float,
    thesis: str = "fade",
    edge_min: float = 0.05,
    momentum_threshold: float = 0.10,
) -> dict:
    """Decide a trade given the current model fair value, live market price, and
    the momentum score. Returns {"side": "buy"|"sell"|None}.

    We stand down unless a momentum event is in play (|momentum| >= threshold).
    """
    if abs(momentum) < momentum_threshold:
        return {"side": None}

    if thesis == "fade":
        edge = model_wp - market_price
        if abs(edge) < edge_min:
            return {"side": None}
        return {"side": "buy" if edge > 0 else "sell"}

    if thesis == "follow":
        return {"side": "buy" if momentum > 0 else "sell"}

    raise ValueError(f"thesis must be 'fade' or 'follow', got {thesis!r}")


def backtest_game(
    snapshots,
    final_outcome: float,
    thesis: str = "fade",
    window: int = 3,
    edge_min: float = 0.05,
    momentum_threshold: float = 0.10,
) -> list[dict]:
    """Walk a single game's in-game snapshots and settle each momentum trade.

    `snapshots` is a time-ordered list of {"model_wp", "market_price"}. At each
    step we compute momentum from the model-WP history, ask `momentum_trade` for
    a side, and (if it trades) settle at the game's `final_outcome` (1.0 = home
    won). Returns the list of trade records — feed straight into the portfolio /
    stats layers.
    """
    from src.signals.walkforward import _simulate_trade

    wp_history: list[float] = []
    trades: list[dict] = []
    for i, snap in enumerate(snapshots):
        wp_history.append(snap["model_wp"])
        momentum = momentum_score(wp_history, window=window)
        sig = momentum_trade(snap["model_wp"], snap["market_price"], momentum,
                             thesis=thesis, edge_min=edge_min,
                             momentum_threshold=momentum_threshold)
        if sig["side"] is None:
            continue
        price = snap["market_price"]
        event = {"id": i, "timestamp": i, "market_prob": price,
                 "bid": price, "ask": price, "actual_outcome": final_outcome}
        trade = _simulate_trade(event, sig["side"], size_pct=0.02)
        if trade is not None:
            trades.append(trade)
    return trades


def momentum_forward_returns(
    prices,
    window: int = 10,
    horizon: int = 5,
    momentum_threshold: float = 0.05,
    min_price: float = 0.0,
    max_price: float = 1.0,
) -> list[float]:
    """The true-edge test: after a momentum event, what does the price do over
    the NEXT `horizon` ticks?

    Returns, per momentum event, the forward price change *aligned to the
    momentum direction*: positive means the move continued (a FOLLOW edge),
    negative means it reverted (a FADE edge). Uses no game outcome and no
    settlement, so the mechanical "price tracks the game" effect cannot leak in
    — this isolates genuine market over/under-reaction.
    """
    out: list[float] = []
    for i in range(window, len(prices) - horizon):
        p = prices[i]
        if not (min_price <= p <= max_price):
            continue
        mom = prices[i] - prices[i - window]
        if abs(mom) < momentum_threshold:
            continue
        fwd = prices[i + horizon] - prices[i]
        out.append(fwd if mom > 0 else -fwd)   # >0 = continued in momentum direction
    return out


def backtest_price_momentum(
    prices,
    final_outcome: float,
    thesis: str = "fade",
    window: int = 5,
    momentum_threshold: float = 0.05,
    min_price: float = 0.0,
    max_price: float = 1.0,
) -> list[dict]:
    """Price-only momentum backtest: momentum is the recent swing in the live
    market price itself (no game-state model). FOLLOW rides the move, FADE bets
    reversion. Each trade settles at the game's `final_outcome`.

    Use this when only the market price trajectory is available (Kalshi
    candlesticks). The model-anchored version (`backtest_game`) needs play-by-
    play for the win-probability model.
    """
    from src.signals.walkforward import _simulate_trade

    trades: list[dict] = []
    for i in range(len(prices)):
        if i < window:
            continue
        price = prices[i]
        if not (min_price <= price <= max_price):
            continue  # skip the resolution convergence; trade only contested states
        momentum = prices[i] - prices[i - window]
        if abs(momentum) < momentum_threshold:
            continue
        if thesis == "follow":
            side = "buy" if momentum > 0 else "sell"
        elif thesis == "fade":
            side = "sell" if momentum > 0 else "buy"
        else:
            raise ValueError(f"thesis must be 'fade' or 'follow', got {thesis!r}")
        event = {"id": i, "timestamp": i, "market_prob": price,
                 "bid": price, "ask": price, "actual_outcome": final_outcome}
        trade = _simulate_trade(event, side, size_pct=0.02)
        if trade is not None:
            trades.append(trade)
    return trades
