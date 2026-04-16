"""
BaseStrategy — abstract base class for all MiroFish-inspired trading strategies.

Each strategy implements:
  1. generate_signals() — run its unique simulation + signal logic on a batch of events
  2. backtest() — evaluate signals against historical outcomes
  3. describe() — return a human-readable summary of the strategy logic

All strategies share the same event data format so they can be compared apples-to-apples.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simulation.engine import SimulationEngine, SwarmResult
from src.agents.persona import PersonaGenerator, PREDICTION_MARKET_ARCHETYPES, EQUITY_MARKET_ARCHETYPES
from src.signals.backtest import BacktestEngine, BacktestResult, Trade


@dataclass
class StrategySignal:
    """Universal signal format shared across all strategies."""
    event_id: int
    event_name: str
    direction: str                # "long_yes", "long_no", "long", "short", "no_trade"
    position_size_pct: float      # Fraction of capital to risk
    confidence: str               # "high", "medium", "low"
    alpha: float                  # Estimated edge
    strategy_metadata: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""

    # Subclasses must set these
    name: str = "BaseStrategy"
    short_description: str = ""

    def __init__(
        self,
        llm_client=None,
        n_agents: int = 50,
        n_rounds: int = 20,
        n_sims: int = 5,
        seed: int = 42,
    ):
        self.llm = llm_client
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.n_sims = n_sims
        self.seed = seed
        self.engine = SimulationEngine(
            llm_client=llm_client,
            n_agents=n_agents,
            n_rounds=n_rounds,
            n_simulations=n_sims,
        )

    @abstractmethod
    def generate_signals(self, events: list[dict]) -> list[StrategySignal]:
        """
        Given a list of event dicts, run the strategy logic and return signals.

        Each event dict has at minimum:
            - event_id: int
            - event_name: str
            - market_probability: float (0-1)
            - context: str (optional background)
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """Return a one-paragraph description of how this strategy works."""
        ...

    def backtest(
        self,
        events: list[dict],
        signals: list[StrategySignal],
        initial_capital: float = 10_000,
        transaction_cost: float = 0.02,
    ) -> BacktestResult:
        """
        Backtest signals against actual outcomes.

        Each event dict must also have:
            - actual_outcome: float (1.0 for YES, 0.0 for NO)
        """
        bt = BacktestEngine(initial_capital=initial_capital)
        event_map = {e["event_id"]: e for e in events}
        bt_events = []

        for sig in signals:
            ev = event_map.get(sig.event_id)
            if ev is None or sig.direction == "no_trade":
                continue
            bt_events.append({
                "event": sig.event_name,
                "swarm_probability": sig.alpha + ev["market_probability"]
                    if sig.direction == "long_yes" else ev["market_probability"] - sig.alpha,
                "market_probability": ev["market_probability"],
                "actual_outcome": ev["actual_outcome"],
                "signal": {
                    "direction": sig.direction,
                    "position_size_pct": sig.position_size_pct,
                    "alpha": sig.alpha,
                    "confidence": sig.confidence,
                },
                "timestamp": ev.get("timestamp", ""),
            })

        result = bt.backtest_prediction_markets(bt_events, transaction_cost=transaction_cost)
        result.strategy_name = self.name
        return result

    def _run_sim(self, event_name: str, context: str = "", market_type: str = "prediction") -> SwarmResult:
        """Convenience: run a full swarm simulation for one event."""
        return self.engine.run(
            event_description=event_name,
            context=context,
            market_type=market_type,
            seed=self.seed,
        )
