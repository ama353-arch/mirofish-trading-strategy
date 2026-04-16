"""
SignalGenerator — converts swarm simulation results into actionable trade signals.

Implements the divergence-based edge estimation and Kelly criterion sizing
described in the strategy document.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..simulation.engine import SwarmResult

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    PM_SIGNAL_THRESHOLD, PM_MAX_SWARM_STD, PM_KELLY_FRACTION,
    PM_MAX_POSITION_PCT, PM_MAX_DEPLOYED_PCT,
    PM_STOP_LOSS_PCT, PM_TAKE_PROFIT_PCT,
    EQ_BULL_THRESHOLD, EQ_BEAR_THRESHOLD, EQ_MIN_CONVICTION,
    EQ_BASE_POSITION_PCT, EQ_MAX_POSITION_PCT,
    EQ_STOP_LOSS_ATR_MULT, EQ_TIME_STOP_DAYS,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionMarketSignal:
    """Trade signal for prediction markets."""
    event: str
    direction: str              # "long_yes", "long_no", "no_trade"
    swarm_probability: float
    market_probability: float
    alpha: float                # swarm - market
    swarm_std: float
    confidence: str             # "high", "medium", "low"
    converged: bool
    kelly_fraction: float       # Optimal Kelly bet fraction
    position_size_pct: float    # As % of bankroll
    stop_loss_price: float
    take_profit_price: float
    expected_edge: float
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "direction": self.direction,
            "swarm_prob": round(self.swarm_probability, 4),
            "market_prob": round(self.market_probability, 4),
            "alpha": round(self.alpha, 4),
            "swarm_std": round(self.swarm_std, 4),
            "confidence": self.confidence,
            "converged": self.converged,
            "kelly_fraction": round(self.kelly_fraction, 4),
            "position_size_pct": round(self.position_size_pct, 4),
            "stop_loss": round(self.stop_loss_price, 4),
            "take_profit": round(self.take_profit_price, 4),
            "expected_edge": round(self.expected_edge, 4),
            "reasoning": self.reasoning,
        }


@dataclass
class EquitySignal:
    """Trade signal for equity markets."""
    ticker: str
    event: str
    direction: str              # "long", "short", "no_trade"
    bullish_fraction: float
    mean_conviction: float
    sentiment_momentum: float   # Rate of change of sentiment
    swarm_probability: float
    position_size_pct: float
    stop_loss_atr: float
    time_stop_days: int
    confidence: str
    archetype_breakdown: dict = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "event": self.event,
            "direction": self.direction,
            "bullish_fraction": round(self.bullish_fraction, 4),
            "mean_conviction": round(self.mean_conviction, 2),
            "sentiment_momentum": round(self.sentiment_momentum, 4),
            "position_size_pct": round(self.position_size_pct, 4),
            "stop_loss_atr_mult": self.stop_loss_atr,
            "time_stop_days": self.time_stop_days,
            "confidence": self.confidence,
            "archetype_breakdown": {k: round(v, 3) for k, v in self.archetype_breakdown.items()},
        }


class SignalGenerator:
    """Converts SwarmResult into actionable trade signals."""

    # ── Prediction Market Signals ──────────────────────────────────────────

    @staticmethod
    def generate_prediction_market_signal(
        swarm_result: SwarmResult,
        market_probability: float,
        bankroll: float = 10_000,
    ) -> PredictionMarketSignal:
        """
        Generate a prediction market trade signal.

        Args:
            swarm_result: Output of SimulationEngine.run()
            market_probability: Current market-implied probability (0-1)
            bankroll: Current bankroll in dollars
        """
        p_swarm = swarm_result.swarm_probability
        p_market = market_probability
        alpha = p_swarm - p_market
        converged = swarm_result.converged

        # Kelly criterion
        if alpha > 0:
            # Bet YES: odds = 1/p_market - 1
            b = (1 / p_market) - 1 if p_market > 0 else 0
            kelly_raw = (p_swarm * b - (1 - p_swarm)) / b if b > 0 else 0
        else:
            # Bet NO: odds = 1/(1-p_market) - 1
            p_no_swarm = 1 - p_swarm
            p_no_market = 1 - p_market
            b = (1 / p_no_market) - 1 if p_no_market > 0 else 0
            kelly_raw = (p_no_swarm * b - (1 - p_no_swarm)) / b if b > 0 else 0

        kelly_frac = max(0, kelly_raw * PM_KELLY_FRACTION)

        # Position sizing
        position_pct = min(kelly_frac, PM_MAX_POSITION_PCT)

        # Expected edge
        if alpha > 0:
            expected_edge = (p_swarm - p_market) / p_market
        else:
            expected_edge = ((1 - p_swarm) - (1 - p_market)) / (1 - p_market) if p_market < 1 else 0

        # Determine direction
        if abs(alpha) < PM_SIGNAL_THRESHOLD:
            direction = "no_trade"
            reasoning = f"Alpha ({alpha:.3f}) below threshold ({PM_SIGNAL_THRESHOLD})"
        elif swarm_result.swarm_std > PM_MAX_SWARM_STD:
            direction = "no_trade"
            reasoning = f"Swarm std ({swarm_result.swarm_std:.3f}) too high (>{PM_MAX_SWARM_STD})"
        elif not converged:
            direction = "no_trade"
            reasoning = f"Simulation not converged (R-hat={swarm_result.r_hat:.3f})"
        elif alpha > 0:
            direction = "long_yes"
            reasoning = f"Swarm ({p_swarm:.1%}) > Market ({p_market:.1%}). Alpha: {alpha:+.1%}"
        else:
            direction = "long_no"
            reasoning = f"Swarm ({p_swarm:.1%}) < Market ({p_market:.1%}). Alpha: {alpha:+.1%}"

        # Confidence level
        if abs(alpha) > 0.15 and swarm_result.swarm_std < 0.08 and converged:
            confidence = "high"
        elif abs(alpha) > 0.08 and swarm_result.swarm_std < 0.12:
            confidence = "medium"
        else:
            confidence = "low"

        # Stop loss / take profit prices
        if direction == "long_yes":
            entry_price = p_market
            stop_loss = entry_price * (1 - PM_STOP_LOSS_PCT)
            take_profit = min(0.98, entry_price + (p_swarm - entry_price) * 0.8)
        elif direction == "long_no":
            entry_price = 1 - p_market
            stop_loss = entry_price * (1 - PM_STOP_LOSS_PCT)
            take_profit = min(0.98, entry_price + ((1 - p_swarm) - entry_price) * 0.8)
        else:
            entry_price = p_market
            stop_loss = 0
            take_profit = 0

        return PredictionMarketSignal(
            event=swarm_result.event_description,
            direction=direction,
            swarm_probability=p_swarm,
            market_probability=p_market,
            alpha=alpha,
            swarm_std=swarm_result.swarm_std,
            confidence=confidence,
            converged=converged,
            kelly_fraction=kelly_frac,
            position_size_pct=position_pct,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            expected_edge=expected_edge,
            reasoning=reasoning,
        )

    # ── Equity Market Signals ──────────────────────────────────────────────

    @staticmethod
    def generate_equity_signal(
        swarm_result: SwarmResult,
        ticker: str,
        current_atr: float = 0.0,
        portfolio_nav: float = 100_000,
    ) -> EquitySignal:
        """
        Generate an equity trade signal.

        Args:
            swarm_result: Output of SimulationEngine.run()
            ticker: Stock ticker
            current_atr: Current Average True Range (for stop loss)
            portfolio_nav: Portfolio net asset value
        """
        bull_frac = swarm_result.bullish_fraction
        conviction = swarm_result.mean_conviction * 10  # Scale to 1-10

        # Sentiment momentum: slope of sentiment trajectory
        traj = []
        for sim in swarm_result.simulation_results:
            traj.extend(sim.sentiment_trajectory)
        if len(traj) >= 5:
            # Simple linear regression slope
            x = np.arange(len(traj))
            slope = np.polyfit(x, traj, 1)[0]
            momentum = float(slope)
        else:
            momentum = 0.0

        # Position sizing
        conv_factor = conviction / 10.0
        signal_strength = min(1.0, abs(bull_frac - 0.5) / 0.2)
        position_pct = EQ_BASE_POSITION_PCT * conv_factor * signal_strength
        position_pct = min(position_pct, EQ_MAX_POSITION_PCT)

        # Direction
        if bull_frac > EQ_BULL_THRESHOLD and conviction >= EQ_MIN_CONVICTION and momentum > 0:
            direction = "long"
            reasoning = (f"Bullish: {bull_frac:.0%} of agents bullish, "
                        f"conviction {conviction:.1f}/10, momentum +{momentum:.4f}")
        elif bull_frac < EQ_BEAR_THRESHOLD and conviction >= EQ_MIN_CONVICTION and momentum < 0:
            direction = "short"
            reasoning = (f"Bearish: only {bull_frac:.0%} of agents bullish, "
                        f"conviction {conviction:.1f}/10, momentum {momentum:.4f}")
        else:
            direction = "no_trade"
            reasons = []
            if EQ_BEAR_THRESHOLD <= bull_frac <= EQ_BULL_THRESHOLD:
                reasons.append(f"bull fraction {bull_frac:.0%} in neutral zone")
            if conviction < EQ_MIN_CONVICTION:
                reasons.append(f"conviction {conviction:.1f} < {EQ_MIN_CONVICTION}")
            if abs(momentum) < 0.001:
                reasons.append("flat sentiment momentum")
            reasoning = "No trade: " + ", ".join(reasons)

        # Confidence
        if abs(bull_frac - 0.5) > 0.25 and conviction > 8 and swarm_result.converged:
            confidence = "high"
        elif abs(bull_frac - 0.5) > 0.15 and conviction > 6:
            confidence = "medium"
        else:
            confidence = "low"

        return EquitySignal(
            ticker=ticker,
            event=swarm_result.event_description,
            direction=direction,
            bullish_fraction=bull_frac,
            mean_conviction=conviction,
            sentiment_momentum=momentum,
            swarm_probability=swarm_result.swarm_probability,
            position_size_pct=position_pct,
            stop_loss_atr=EQ_STOP_LOSS_ATR_MULT,
            time_stop_days=EQ_TIME_STOP_DAYS,
            confidence=confidence,
            archetype_breakdown=swarm_result.archetype_probabilities,
            reasoning=reasoning,
        )
