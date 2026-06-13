"""
walkforward.py — out-of-sample, walk-forward backtest harness.

This module exists to make backtest results trustworthy. The original
BacktestEngine evaluated every signal on a single pass with no separation
between the data used to choose parameters and the data used to score them,
and its demo fed it synthetic events where the swarm was defined to be more
accurate than the market. Both make profitable equity curves meaningless.

The walk-forward harness fixes the structural half of that:

    [ train ][ test ][ train ][ test ] ...
       tune    score    tune    score

Parameters (edge cutoff, R-hat gate, Kelly fraction, ...) are chosen on each
train fold and applied, unchanged, to the immediately following test fold.
Reported performance is the concatenation of test folds only — data the
parameter search never saw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

_DAYS_PER_YEAR = 365.25
_EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (stdlib only)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal inverse CDF (Acklam's rational approximation).

    Accurate to ~1e-9 on (0, 1); clamps the open-interval endpoints.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


def expected_max_sharpe_under_null(n_trials: int, sharpe_std: float = 1.0) -> float:
    """Expected maximum Sharpe across `n_trials` strategies that all truly have
    zero edge (Bailey & López de Prado).

    This is the bar a selected strategy must clear just to be distinguishable
    from luck. With one trial there is no selection, so the bar is 0.
    """
    if n_trials <= 1:
        return 0.0
    a = _norm_ppf(1.0 - 1.0 / n_trials)
    b = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sharpe_std * ((1.0 - _EULER_MASCHERONI) * a + _EULER_MASCHERONI * b)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_std: float = 1.0,
) -> float:
    """Probability the strategy's TRUE Sharpe is positive, after correcting for
    (a) how many strategies were tried and (b) non-normal returns.

    Returns a value in [0, 1]; below ~0.95 means "not convincingly real."
    """
    if n_obs <= 1:
        return 0.0
    sr0 = expected_max_sharpe_under_null(n_trials, sharpe_std)
    denom = math.sqrt(
        1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2
    )
    if denom <= 0:
        return 0.0
    z = (observed_sharpe - sr0) * math.sqrt(n_obs - 1) / denom
    return float(_norm_cdf(z))


def annualized_return(total_return: float, start: datetime, end: datetime) -> float:
    """Compound `total_return` over the actual elapsed calendar time.

    A +21% return over half a year is far better than +21% over five years;
    echoing the total return (as the original engine did) erases that. Returns
    0.0 when no time has elapsed, since annualization is then undefined.
    """
    elapsed_days = (end - start).total_seconds() / 86_400
    if elapsed_days <= 0:
        return 0.0
    years = elapsed_days / _DAYS_PER_YEAR
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def crossed_fill_price(
    side: str,
    bid: float,
    ask: float,
    slippage: float = 0.0,
) -> float:
    """Effective fill price for a marketable order on a binary market.

    A buy crosses up to the ask; a sell crosses down to the bid. `slippage`
    (a fraction of the touch price) models size that walks past top-of-book.
    The result is clamped to [0, 1] because a binary-outcome price cannot
    leave that range.

    This replaces the flat-percentage cost that hid how lethal wide spreads
    are on thin prediction markets.
    """
    if side == "buy":
        price = ask * (1.0 + slippage)
    elif side == "sell":
        price = bid * (1.0 - slippage)
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    return max(0.0, min(1.0, price))


def block_bootstrap_equity_paths(
    returns,
    block_size: int,
    n_paths: int = 1000,
    seed: int | None = None,
) -> "np.ndarray":
    """Circular block-bootstrap of per-trade returns into equity paths.

    Resampling whole contiguous BLOCKS (rather than individual trades, as the
    original IID bootstrap did) preserves the autocorrelation and streak
    structure of the strategy — winning and losing runs stay intact — so the
    resulting confidence bands are honest instead of artificially tight.

    Returns an array of shape (n_paths, len(returns) + 1); every path starts
    at bankroll 1.0.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n == 0:
        return np.ones((n_paths, 1))
    block_size = max(1, min(block_size, n))
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_size)

    paths = np.ones((n_paths, n + 1))
    for p in range(n_paths):
        starts = rng.integers(0, n, size=n_blocks)
        seq = np.concatenate([
            r[(s + np.arange(block_size)) % n] for s in starts
        ])[:n]
        paths[p, 1:] = np.cumprod(1.0 + seq)
    return paths


@dataclass
class Fold:
    """One walk-forward split: tune on `train`, score on `test`."""
    index: int
    train: list[dict] = field(default_factory=list)
    test: list[dict] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Out-of-sample result: every reported trade came from a test fold."""
    oos_trades: list[dict] = field(default_factory=list)
    chosen_params: list[dict] = field(default_factory=list)

    def stats(self) -> dict:
        """Honest summary computed from out-of-sample trades only.

        Returns sequentially-compounded total return (bankroll fraction per
        trade), win rate, and trade count. Empty result -> all zeros.
        """
        trades = self.oos_trades
        n = len(trades)
        if n == 0:
            return {"n_trades": 0, "win_rate": 0.0, "total_return": 0.0,
                    "avg_return": 0.0}

        wins = sum(1 for t in trades if t["pnl_per_dollar"] > 0)
        equity = 1.0
        for t in trades:
            equity *= 1.0 + t["pnl_per_dollar"] * t["size_pct"]
        return {
            "n_trades": n,
            "win_rate": wins / n,
            "total_return": equity - 1.0,
            "avg_return": sum(
                t["pnl_per_dollar"] * t["size_pct"] for t in trades
            ) / n,
        }


def _simulate_trade(event: dict, side: str, size_pct: float) -> dict | None:
    """Simulate one binary-market trade against a resolved event.

    Entry crosses the spread (realistic fill); payoff is the binary outcome.
    Returns a trade record, or None if the brain declined to trade.
    """
    if side not in ("buy", "sell"):
        return None

    bid = event["bid"]
    ask = event["ask"]
    outcome = event["actual_outcome"]
    entry = crossed_fill_price(side, bid=bid, ask=ask)  # YES fill: buy->ask, sell->bid

    # A buy pays the YES ask and is paid `outcome`. A sell of YES is a BUY of NO:
    # you pay (1 - yes_bid) and are paid (1 - outcome). Settling a sell at the
    # YES price (the old bug) overstated profit when the price was near 0.
    if side == "buy":
        cost = entry
        payoff = outcome
    else:  # sell == buy NO
        cost = 1.0 - entry
        payoff = 1.0 - outcome

    if cost <= 0.0 or cost >= 1.0:
        return None  # no room to make money

    pnl_per_dollar = payoff / cost - 1.0

    return {
        "event_id": event["id"],
        "timestamp": event["timestamp"],
        "side": side,
        "entry_price": entry,
        "cost": cost,
        "outcome": outcome,
        "size_pct": size_pct,
        "pnl_per_dollar": pnl_per_dollar,
    }


class WalkForwardBacktester:
    """Generate rolling, chronologically-ordered train/test folds."""

    def __init__(self, train_size: int, test_size: int):
        if train_size < 1 or test_size < 1:
            raise ValueError("train_size and test_size must both be >= 1")
        self.train_size = train_size
        self.test_size = test_size

    def make_folds(self, events: list[dict]) -> list[Fold]:
        """Split chronologically-sorted events into rolling train/test folds.

        Events are sorted by timestamp first so the no-look-ahead property
        holds even if the caller passes them out of order.
        """
        ordered = sorted(events, key=lambda e: e["timestamp"])

        folds: list[Fold] = []
        start = 0
        idx = 0
        while start + self.train_size + self.test_size <= len(ordered):
            train = ordered[start : start + self.train_size]
            test = ordered[start + self.train_size : start + self.train_size + self.test_size]
            folds.append(Fold(index=idx, train=train, test=test))
            start += self.test_size
            idx += 1

        return folds

    def _score_params(self, signal_fn, events: list[dict], params: dict) -> float:
        """Total per-dollar PnL of a parameter set over a block of events."""
        total = 0.0
        for event in events:
            sig = signal_fn(event, params) or {}
            trade = _simulate_trade(event, sig.get("side"), sig.get("size_pct", 0.0))
            if trade is not None:
                total += trade["pnl_per_dollar"] * trade["size_pct"]
        return total

    def run(self, events, signal_fn, param_grid) -> WalkForwardResult:
        """Walk-forward, out-of-sample backtest.

        For each fold: score every param set on the TRAIN block, pick the best,
        then apply that param set to the untouched TEST block. Reported trades
        are the concatenation of test-fold trades only. Parameters are never
        chosen using data they are later scored on.
        """
        if not param_grid:
            raise ValueError("param_grid must contain at least one parameter set")

        result = WalkForwardResult()
        for fold in self.make_folds(events):
            best_params = max(
                param_grid,
                key=lambda p: self._score_params(signal_fn, fold.train, p),
            )
            result.chosen_params.append(best_params)

            for event in fold.test:
                sig = signal_fn(event, best_params) or {}
                trade = _simulate_trade(event, sig.get("side"), sig.get("size_pct", 0.0))
                if trade is not None:
                    result.oos_trades.append(trade)

        return result
