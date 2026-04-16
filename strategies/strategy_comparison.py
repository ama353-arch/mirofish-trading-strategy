"""
Strategy Comparison Framework
──────────────────────────────
Runs all 5 MiroFish strategies through the backtest engine on the same
historical event data, then computes and displays comparative metrics.

Metrics compared:
  - Total return
  - Sharpe ratio
  - Max drawdown
  - Calmar ratio (return / max drawdown)
  - Win rate
  - Profit factor
  - Trade count
  - Average alpha per trade
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseStrategy, StrategySignal
from . import ALL_STRATEGIES

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.signals.backtest import BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Computed performance metrics for a single strategy."""
    name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_alpha: float
    avg_position_size: float
    high_conf_win_rate: float
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


@dataclass
class ComparisonResult:
    """Full comparison across all strategies."""
    event_count: int
    strategies: list[StrategyMetrics]
    ranking_by_sharpe: list[str]
    ranking_by_return: list[str]
    ranking_by_calmar: list[str]
    best_overall: str

    def to_dict(self) -> dict:
        return {
            "event_count": self.event_count,
            "strategies": [s.to_dict() for s in self.strategies],
            "ranking_by_sharpe": self.ranking_by_sharpe,
            "ranking_by_return": self.ranking_by_return,
            "ranking_by_calmar": self.ranking_by_calmar,
            "best_overall": self.best_overall,
        }


class StrategyComparison:
    """
    Compare multiple strategies on the same event dataset.

    Usage:
        comp = StrategyComparison(events=synthetic_events)
        result = comp.run()
        comp.print_dashboard(result)
    """

    def __init__(
        self,
        events: list[dict],
        initial_capital: float = 10_000,
        transaction_cost: float = 0.02,
        strategy_kwargs: dict | None = None,
    ):
        self.events = events
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.strategy_kwargs = strategy_kwargs or {
            "n_agents": 50,
            "n_rounds": 20,
            "n_sims": 5,
            "seed": 42,
        }

    def run(
        self,
        strategy_classes: list[type] | None = None,
    ) -> ComparisonResult:
        """
        Run all strategies through signal generation and backtesting.

        Args:
            strategy_classes: List of strategy classes to compare.
                Defaults to ALL_STRATEGIES.
        """
        classes = strategy_classes or ALL_STRATEGIES
        all_metrics = []

        for cls in classes:
            logger.info(f"Running strategy: {cls.name}...")
            start = time.time()

            try:
                strategy = cls(**self.strategy_kwargs)
                signals = strategy.generate_signals(self.events)
                bt_result = strategy.backtest(
                    events=self.events,
                    signals=signals,
                    initial_capital=self.initial_capital,
                    transaction_cost=self.transaction_cost,
                )
                elapsed = time.time() - start

                # Compute additional metrics
                trade_signals = [s for s in signals if s.direction != "no_trade"]
                avg_alpha = float(np.mean([s.alpha for s in trade_signals])) if trade_signals else 0
                avg_pos = float(np.mean([s.position_size_pct for s in trade_signals])) if trade_signals else 0
                high_conf = [s for s in trade_signals if s.confidence == "high"]

                # High-confidence win rate (needs to match against actual outcomes)
                high_conf_ids = {s.event_id for s in high_conf}
                high_conf_trades = [t for t in bt_result.trades
                                    if any(s.event_id == i and s.event_name == t.asset
                                           for i, s in [(s.event_id, s) for s in high_conf])]
                high_conf_wins = sum(1 for t in high_conf_trades if t.pnl > 0)
                high_conf_wr = high_conf_wins / max(1, len(high_conf_trades))

                calmar = (bt_result.total_return / bt_result.max_drawdown
                         if bt_result.max_drawdown > 0 else 0)

                metrics = StrategyMetrics(
                    name=strategy.name,
                    total_return=bt_result.total_return,
                    annualized_return=bt_result.annualized_return,
                    sharpe_ratio=bt_result.sharpe_ratio,
                    max_drawdown=bt_result.max_drawdown,
                    calmar_ratio=calmar,
                    win_rate=bt_result.win_rate,
                    profit_factor=bt_result.profit_factor,
                    total_trades=bt_result.total_trades,
                    winning_trades=bt_result.winning_trades,
                    losing_trades=bt_result.losing_trades,
                    avg_alpha=avg_alpha,
                    avg_position_size=avg_pos,
                    high_conf_win_rate=high_conf_wr,
                    elapsed_seconds=elapsed,
                )
                all_metrics.append(metrics)

            except Exception as e:
                logger.error(f"Strategy {cls.name} failed: {e}")
                all_metrics.append(StrategyMetrics(
                    name=cls.name,
                    total_return=0, annualized_return=0, sharpe_ratio=0,
                    max_drawdown=0, calmar_ratio=0, win_rate=0, profit_factor=0,
                    total_trades=0, winning_trades=0, losing_trades=0,
                    avg_alpha=0, avg_position_size=0, high_conf_win_rate=0,
                    elapsed_seconds=0,
                ))

        # Rankings
        by_sharpe = sorted(all_metrics, key=lambda m: m.sharpe_ratio, reverse=True)
        by_return = sorted(all_metrics, key=lambda m: m.total_return, reverse=True)
        by_calmar = sorted(all_metrics, key=lambda m: m.calmar_ratio, reverse=True)

        # Best overall: composite rank (lower is better)
        ranks = {}
        for i, m in enumerate(by_sharpe):
            ranks[m.name] = ranks.get(m.name, 0) + i
        for i, m in enumerate(by_return):
            ranks[m.name] = ranks.get(m.name, 0) + i
        for i, m in enumerate(by_calmar):
            ranks[m.name] = ranks.get(m.name, 0) + i
        best = min(ranks, key=ranks.get) if ranks else ""

        return ComparisonResult(
            event_count=len(self.events),
            strategies=all_metrics,
            ranking_by_sharpe=[m.name for m in by_sharpe],
            ranking_by_return=[m.name for m in by_return],
            ranking_by_calmar=[m.name for m in by_calmar],
            best_overall=best,
        )

    @staticmethod
    def print_dashboard(result: ComparisonResult):
        """Print a formatted comparison dashboard to stdout."""
        print()
        print("=" * 100)
        print("  STRATEGY COMPARISON DASHBOARD")
        print(f"  Events: {result.event_count}")
        print("=" * 100)

        # Header
        col_w = 22
        header = f"{'Strategy':<{col_w}} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8} {'WinRate':>8} {'PF':>8} {'Trades':>7} {'AvgAlpha':>9}"
        print()
        print(header)
        print("-" * len(header))

        for m in result.strategies:
            row = (
                f"{m.name:<{col_w}} "
                f"{m.total_return:>+7.1%} "
                f"{m.sharpe_ratio:>8.2f} "
                f"{m.max_drawdown:>7.1%} "
                f"{m.calmar_ratio:>8.2f} "
                f"{m.win_rate:>7.1%} "
                f"{m.profit_factor:>8.2f} "
                f"{m.total_trades:>7d} "
                f"{m.avg_alpha:>8.3f}"
            )
            print(row)

        print()
        print("-" * len(header))

        # Rankings
        print(f"\n  Rankings:")
        print(f"    By Sharpe Ratio:  {' > '.join(result.ranking_by_sharpe)}")
        print(f"    By Total Return:  {' > '.join(result.ranking_by_return)}")
        print(f"    By Calmar Ratio:  {' > '.join(result.ranking_by_calmar)}")
        print(f"\n    Best Overall: {result.best_overall}")

        # Per-strategy detail
        print(f"\n{'='*100}")
        print("  STRATEGY DETAILS")
        print(f"{'='*100}")
        for m in result.strategies:
            print(f"\n  {m.name}")
            print(f"  {'─' * len(m.name)}")
            print(f"    Return: {m.total_return:+.2%} | Sharpe: {m.sharpe_ratio:.2f} | MaxDD: {m.max_drawdown:.2%}")
            print(f"    Trades: {m.total_trades} (W:{m.winning_trades}/L:{m.losing_trades}) | Win Rate: {m.win_rate:.1%}")
            print(f"    Profit Factor: {m.profit_factor:.2f} | Calmar: {m.calmar_ratio:.2f}")
            print(f"    Avg Alpha: {m.avg_alpha:.4f} | Avg Position: {m.avg_position_size:.2%}")
            print(f"    High-Conf Win Rate: {m.high_conf_win_rate:.1%}")
            print(f"    Elapsed: {m.elapsed_seconds:.1f}s")

        print(f"\n{'='*100}\n")

    @staticmethod
    def save_results(result: ComparisonResult, path: str | Path):
        """Save comparison results to JSON."""
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info(f"Results saved to {path}")
