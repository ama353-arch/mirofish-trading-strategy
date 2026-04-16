"""
BacktestEngine — historical backtesting framework for the swarm trading strategy.

Supports both prediction market and equity market backtesting with
realistic transaction costs, slippage, and bankroll management.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade."""
    trade_id: str
    timestamp: str
    asset: str
    direction: str          # "long_yes", "long_no", "long", "short"
    entry_price: float
    exit_price: float = 0.0
    position_size: float = 0.0   # In dollars
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = "open"    # "open", "closed", "stopped_out", "time_stopped"
    exit_timestamp: str = ""
    swarm_probability: float = 0.0
    market_probability: float = 0.0
    alpha: float = 0.0
    confidence: str = ""


@dataclass
class BacktestResult:
    """Complete backtest results."""
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_holding_period: float  # In trades/events for PM, days for equity
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return f"""
=== Backtest Summary: {self.strategy_name} ===
Period: {self.start_date} to {self.end_date}
Initial Capital: ${self.initial_capital:,.0f}
Final Capital:   ${self.final_capital:,.0f}

Total Return:      {self.total_return:+.2%}
Annualized Return: {self.annualized_return:+.2%}
Sharpe Ratio:      {self.sharpe_ratio:.2f}
Max Drawdown:      {self.max_drawdown:.2%}

Win Rate:          {self.win_rate:.1%}
Profit Factor:     {self.profit_factor:.2f}
Total Trades:      {self.total_trades}
Winning/Losing:    {self.winning_trades}/{self.losing_trades}
Avg Win:           {self.avg_win:+.2%}
Avg Loss:          {self.avg_loss:+.2%}
"""


class BacktestEngine:
    """
    Backtesting framework for swarm-generated trading signals.

    Can backtest against:
    1. Historical prediction market outcomes (mock or real)
    2. Historical equity events (earnings, Fed meetings, etc.)
    """

    def __init__(self, initial_capital: float = 10_000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = [initial_capital]

    def reset(self):
        """Reset backtest state."""
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]

    # ── Prediction Market Backtest ─────────────────────────────────────────

    def backtest_prediction_markets(
        self,
        events: list[dict],
        transaction_cost: float = 0.02,
    ) -> BacktestResult:
        """
        Backtest prediction market signals against historical events.

        Each event dict should have:
            - event: str (description)
            - swarm_probability: float (simulated estimate)
            - market_probability: float (market price at time of signal)
            - actual_outcome: float (1.0 for YES, 0.0 for NO)
            - signal: dict (from SignalGenerator)
            - timestamp: str
        """
        self.reset()

        for event in events:
            signal = event.get("signal", {})
            direction = signal.get("direction", "no_trade")

            if direction == "no_trade":
                continue

            position_pct = signal.get("position_size_pct", 0.02)
            position_size = self.capital * position_pct
            entry_price = event["market_probability"] if direction == "long_yes" else (1 - event["market_probability"])

            # Resolve: did the outcome match our bet?
            actual = event["actual_outcome"]
            if direction == "long_yes":
                exit_price = actual  # 1.0 if YES, 0.0 if NO
            else:
                exit_price = 1 - actual  # 1.0 if NO, 0.0 if YES

            # P&L (simplified: binary payout)
            gross_pnl = position_size * (exit_price / entry_price - 1) if entry_price > 0 else 0
            net_pnl = gross_pnl - (position_size * transaction_cost)

            self.capital += net_pnl

            trade = Trade(
                trade_id=f"pm_{len(self.trades):04d}",
                timestamp=event.get("timestamp", ""),
                asset=event.get("event", ""),
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position_size,
                pnl=net_pnl,
                pnl_pct=net_pnl / position_size if position_size > 0 else 0,
                status="closed",
                swarm_probability=event.get("swarm_probability", 0),
                market_probability=event.get("market_probability", 0),
                alpha=signal.get("alpha", 0),
                confidence=signal.get("confidence", ""),
            )
            self.trades.append(trade)
            self.equity_curve.append(self.capital)

        return self._compute_results("Prediction Market Swarm Strategy")

    # ── Equity Market Backtest ─────────────────────────────────────────────

    def backtest_equity_events(
        self,
        events: list[dict],
        transaction_cost_bps: float = 10,  # basis points per trade
    ) -> BacktestResult:
        """
        Backtest equity signals against historical price data.

        Each event dict should have:
            - ticker: str
            - event: str (description)
            - signal: dict (from SignalGenerator)
            - entry_price: float
            - exit_price: float (price after time_stop_days)
            - timestamp: str
        """
        self.reset()

        for event in events:
            signal = event.get("signal", {})
            direction = signal.get("direction", "no_trade")

            if direction == "no_trade":
                continue

            position_pct = signal.get("position_size_pct", 0.02)
            position_size = self.capital * position_pct
            entry = event["entry_price"]
            exit_ = event["exit_price"]

            # P&L
            if direction == "long":
                gross_return = (exit_ - entry) / entry
            else:  # short
                gross_return = (entry - exit_) / entry

            cost = transaction_cost_bps / 10_000 * 2  # Round trip
            net_return = gross_return - cost
            net_pnl = position_size * net_return

            self.capital += net_pnl

            trade = Trade(
                trade_id=f"eq_{len(self.trades):04d}",
                timestamp=event.get("timestamp", ""),
                asset=event.get("ticker", ""),
                direction=direction,
                entry_price=entry,
                exit_price=exit_,
                position_size=position_size,
                pnl=net_pnl,
                pnl_pct=net_return,
                status="closed",
                confidence=signal.get("confidence", ""),
            )
            self.trades.append(trade)
            self.equity_curve.append(self.capital)

        return self._compute_results("Equity Swarm Strategy")

    # ── Monte Carlo Backtest Simulation ────────────────────────────────────

    def monte_carlo_equity_curve(
        self,
        n_simulations: int = 1000,
    ) -> dict:
        """
        Run Monte Carlo simulation on the trade sequence to estimate
        confidence intervals for the equity curve.

        Returns percentile-based confidence bands.
        """
        if not self.trades:
            return {}

        # Use dollar P&L relative to capital at time of trade, not pct returns
        # This avoids compounding issues with binary payoff trades
        pnl_ratios = [t.pnl / self.initial_capital for t in self.trades]
        n_trades = len(pnl_ratios)
        curves = np.zeros((n_simulations, n_trades + 1))
        curves[:, 0] = self.initial_capital

        for sim in range(n_simulations):
            shuffled = np.random.choice(pnl_ratios, size=n_trades, replace=True)
            for i, ratio in enumerate(shuffled):
                curves[sim, i + 1] = curves[sim, i] + self.initial_capital * ratio

        return {
            "median": np.median(curves, axis=0).tolist(),
            "p5": np.percentile(curves, 5, axis=0).tolist(),
            "p25": np.percentile(curves, 25, axis=0).tolist(),
            "p75": np.percentile(curves, 75, axis=0).tolist(),
            "p95": np.percentile(curves, 95, axis=0).tolist(),
        }

    # ── Results Computation ────────────────────────────────────────────────

    def _compute_results(self, strategy_name: str) -> BacktestResult:
        """Compute performance metrics from completed trades."""
        if not self.trades:
            return BacktestResult(
                strategy_name=strategy_name,
                start_date="", end_date="",
                initial_capital=self.initial_capital,
                final_capital=self.capital,
                total_return=0, annualized_return=0,
                sharpe_ratio=0, max_drawdown=0,
                win_rate=0, profit_factor=0,
                total_trades=0, winning_trades=0, losing_trades=0,
                avg_win=0, avg_loss=0, avg_holding_period=0,
            )

        pnls = [t.pnl for t in self.trades]
        pnl_pcts = [t.pnl_pct for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Equity curve drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = float(abs(drawdown.min()))

        # Sharpe ratio (annualized, assuming ~50 events/year)
        if len(pnl_pcts) > 1 and np.std(pnl_pcts) > 0:
            sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts)) * np.sqrt(50)
        else:
            sharpe = 0.0

        total_return = (self.capital - self.initial_capital) / self.initial_capital

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=self.trades[0].timestamp if self.trades else "",
            end_date=self.trades[-1].timestamp if self.trades else "",
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            total_return=total_return,
            annualized_return=total_return,  # Simplified; would need date range for true annualization
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=len(wins) / len(self.trades) if self.trades else 0,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            avg_win=float(np.mean([t.pnl_pct for t in self.trades if t.pnl > 0])) if wins else 0,
            avg_loss=float(np.mean([t.pnl_pct for t in self.trades if t.pnl <= 0])) if losses else 0,
            avg_holding_period=1.0,  # Each trade = 1 event
            trades=self.trades,
            equity_curve=self.equity_curve,
        )
