"""
Tests for the walk-forward, out-of-sample backtesting harness.

The whole point of this module is to make backtest results *trustworthy* in
ways the original BacktestEngine was not:

  - Train/test separation (parameters tuned on train, scored on test only)
  - No look-ahead across the fold boundary
  - Honest, date-based annualization
  - Realistic per-market cost model
  - Multiple-testing / deflated-Sharpe awareness

These tests encode the CORRECT behavior up front so the harness cannot
silently lie the way a self-fulfilling synthetic backtest does.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signals.walkforward import (
    WalkForwardBacktester,
    annualized_return,
    crossed_fill_price,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _event(i: int, *, days: int) -> dict:
    """A minimal time-stamped event placeholder."""
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=days)
    return {"id": i, "timestamp": ts.isoformat()}


def _linear_events(n: int) -> list[dict]:
    """n events, one per day, already in chronological order."""
    return [_event(i, days=i) for i in range(n)]


# ── No look-ahead across the fold boundary ───────────────────────────────────

def test_test_fold_never_contains_events_before_its_train_fold():
    """Every test event must post-date every train event in the same fold.

    This is the property the original engine lacked entirely: there was no
    separation, so any threshold was tuned and scored on the same data.
    """
    events = _linear_events(20)
    wf = WalkForwardBacktester(train_size=5, test_size=3)

    folds = wf.make_folds(events)

    assert folds, "expected at least one fold"
    for fold in folds:
        latest_train = max(e["timestamp"] for e in fold.train)
        earliest_test = min(e["timestamp"] for e in fold.test)
        assert latest_train <= earliest_test, (
            "look-ahead: a test event pre-dates a train event in the same fold"
        )


# ── Honest, date-based annualization (replaces the fake sqrt(50)) ─────────────

def test_annualized_return_compounds_over_actual_elapsed_time():
    """A +21% total return earned over exactly half a year annualizes to
    (1.21)^2 - 1 = +46.4%, NOT just echoing the total return.

    The original engine set annualized_return = total_return, which is wrong
    for any horizon other than exactly one year.
    """
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=182.625)  # half a year

    ann = annualized_return(total_return=0.21, start=start, end=end)

    assert ann == pytest.approx(0.21 * 2 + 0.21**2, rel=1e-3)  # (1.21)^2 - 1


def test_annualized_return_is_zero_for_zero_elapsed_time():
    """No elapsed time means annualization is undefined; return 0.0 rather
    than dividing by zero or returning infinity."""
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert annualized_return(total_return=0.5, start=t, end=t) == 0.0


# ── Realistic cost model: you cross the spread, you don't fill at mid ─────────

def test_buy_fills_at_ask_and_sell_fills_at_bid():
    """A marketable order crosses the spread. A flat 2% cost hides this: on a
    market quoted 0.40 / 0.46 the real round-trip cost is the 6c spread
    (~14% of mid), not 2%."""
    assert crossed_fill_price("buy", bid=0.40, ask=0.46) == 0.46
    assert crossed_fill_price("sell", bid=0.40, ask=0.46) == 0.40


def test_slippage_worsens_the_fill_beyond_the_touch():
    """Size beyond top-of-book walks the book: a buy fills above the ask, a
    sell fills below the bid, by the slippage fraction of the price."""
    # 10% slippage on a 0.50 ask -> fill at 0.55
    assert crossed_fill_price("buy", bid=0.44, ask=0.50, slippage=0.10) == pytest.approx(0.55)
    # 10% slippage on a 0.44 bid -> fill at 0.396
    assert crossed_fill_price("sell", bid=0.44, ask=0.50, slippage=0.10) == pytest.approx(0.396)


def test_fill_price_is_clamped_to_valid_probability_range():
    """A binary-market price can never exceed 1.0 or fall below 0.0, no matter
    how wide the slippage assumption."""
    assert crossed_fill_price("buy", bid=0.95, ask=0.98, slippage=0.50) == 1.0
    # slippage > 100% would drive a sell below zero; it is floored at 0.0
    assert crossed_fill_price("sell", bid=0.03, ask=0.06, slippage=1.50) == 0.0


# ── The OOS run() loop: tune on train, report only test-fold trades ───────────

def _pm_event(i: int, *, days: int, mkt: float, outcome: float,
              bid: float | None = None, ask: float | None = None) -> dict:
    """A resolved prediction-market event in the unified schema."""
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=days)
    spread = 0.02
    return {
        "id": i,
        "timestamp": ts.isoformat(),
        "market_prob": mkt,
        "bid": bid if bid is not None else max(0.0, mkt - spread),
        "ask": ask if ask is not None else min(1.0, mkt + spread),
        "actual_outcome": outcome,
    }


def test_run_reports_only_test_fold_trades_never_train():
    """The reported equity curve must be built from out-of-sample test folds
    only. A train event must never appear as a reported trade — that was the
    original engine's fatal flaw."""
    # 20 events; brain always says "buy YES"; single trivial param.
    events = [_pm_event(i, days=i, mkt=0.50, outcome=float(i % 2)) for i in range(20)]
    wf = WalkForwardBacktester(train_size=5, test_size=3)

    def signal_fn(event, params):
        return {"side": "buy", "size_pct": 0.02}

    result = wf.run(events, signal_fn=signal_fn, param_grid=[{"edge_min": 0.0}])

    traded_ids = {t["event_id"] for t in result.oos_trades}
    expected_test_ids = {e["id"] for fold in wf.make_folds(events) for e in fold.test}
    assert traded_ids, "expected some out-of-sample trades"
    assert traded_ids <= expected_test_ids, "a non-test (train) event was reported as a trade"
    # the very first train block (ids 0-4) can never be a reported trade
    assert traded_ids.isdisjoint({0, 1, 2, 3, 4})


def test_run_selects_the_param_that_maximizes_train_score():
    """Parameter selection happens on train folds. Given a grid where only one
    param value trades profitably on the (visible) train data, run() must pick
    it — and we can see it in result.chosen_params."""
    # YES always resolves true; buying YES under market is +EV, selling is -EV.
    events = [_pm_event(i, days=i, mkt=0.50, outcome=1.0) for i in range(20)]
    wf = WalkForwardBacktester(train_size=5, test_size=3)

    def signal_fn(event, params):
        return {"side": params["side"], "size_pct": 0.02}

    result = wf.run(
        events,
        signal_fn=signal_fn,
        param_grid=[{"side": "buy"}, {"side": "sell"}],
    )

    # On train, buying YES (which always wins) dominates; run must choose it.
    assert all(p["side"] == "buy" for p in result.chosen_params)


# ── Deflated Sharpe: correct for how many strategies you tried ────────────────

def test_expected_max_sharpe_grows_with_number_of_trials():
    """The more strategies/params you try, the higher the best Sharpe you'd
    expect from pure luck. The selection bar must rise with trial count."""
    from src.signals.walkforward import expected_max_sharpe_under_null
    assert expected_max_sharpe_under_null(1) < expected_max_sharpe_under_null(10)
    assert expected_max_sharpe_under_null(10) < expected_max_sharpe_under_null(100)


def test_deflated_sharpe_falls_as_more_trials_are_run():
    """Holding the observed Sharpe fixed, confidence it is real must DROP as
    the number of trials rises — that is the whole point of deflation."""
    from src.signals.walkforward import deflated_sharpe_ratio
    few = deflated_sharpe_ratio(observed_sharpe=2.0, n_trials=1, n_obs=200)
    many = deflated_sharpe_ratio(observed_sharpe=2.0, n_trials=200, n_obs=200)
    assert 0.0 <= many <= few <= 1.0
    assert many < few


def test_deflated_sharpe_rises_with_a_stronger_observed_sharpe():
    """A genuinely higher Sharpe should survive deflation better."""
    from src.signals.walkforward import deflated_sharpe_ratio
    weak = deflated_sharpe_ratio(observed_sharpe=0.5, n_trials=50, n_obs=200)
    strong = deflated_sharpe_ratio(observed_sharpe=3.0, n_trials=50, n_obs=200)
    assert strong > weak


# ── OOS result metrics: honest stats from test-fold trades only ───────────────

def test_result_stats_compound_returns_and_count_wins():
    from src.signals.walkforward import WalkForwardResult
    r = WalkForwardResult(oos_trades=[
        {"pnl_per_dollar": 0.5, "size_pct": 0.1},    # +5% of bankroll
        {"pnl_per_dollar": -1.0, "size_pct": 0.1},   # -10%
        {"pnl_per_dollar": 0.5, "size_pct": 0.1},    # +5%
    ])
    s = r.stats()
    assert s["n_trades"] == 3
    assert s["win_rate"] == pytest.approx(2 / 3)
    assert s["total_return"] == pytest.approx(1.05 * 0.90 * 1.05 - 1.0)


def test_result_stats_are_zero_for_no_trades():
    from src.signals.walkforward import WalkForwardResult
    s = WalkForwardResult().stats()
    assert s["n_trades"] == 0
    assert s["win_rate"] == 0.0
    assert s["total_return"] == 0.0


# ── Block-bootstrap MC: preserve autocorrelation, don't shuffle it away ───────

def test_block_bootstrap_shape_starts_at_one():
    from src.signals.walkforward import block_bootstrap_equity_paths
    paths = block_bootstrap_equity_paths([0.1, -0.05, 0.2, -0.1],
                                         block_size=2, n_paths=64, seed=7)
    assert paths.shape == (64, 5)          # n_paths x (len+1)
    assert (paths[:, 0] == 1.0).all()      # every path starts at bankroll 1.0


def test_full_length_block_preserves_total_compounded_return():
    """A circular block bootstrap with block_size == len draws one wrapped
    block, so each path is a rotation of the return series. Rotations preserve
    the product (1+r), hence the final equity — proof the block structure is
    kept intact rather than IID-shuffled."""
    import numpy as np
    returns = [0.1, -0.05, 0.2, -0.1]
    from src.signals.walkforward import block_bootstrap_equity_paths
    paths = block_bootstrap_equity_paths(returns, block_size=4, n_paths=32, seed=1)
    expected_final = float(np.prod([1.0 + r for r in returns]))
    assert paths[:, -1] == pytest.approx(expected_final)
