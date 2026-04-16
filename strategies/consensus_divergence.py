"""
Consensus Divergence Strategy
─────────────────────────────
Core idea: Run a standard agent swarm simulation and trade when the swarm's
consensus probability diverges significantly from the current market-implied
probability. Buy YES when the swarm says underpriced, buy NO when overpriced.

This is the "vanilla" MiroFish strategy — the most direct application of
swarm intelligence to finding mispricings.

Signal logic:
  alpha = p_swarm - p_market
  Trade when |alpha| > threshold AND swarm has converged (R-hat < 1.1)
  Size via fractional Kelly criterion
"""

import logging
import numpy as np

from .base import BaseStrategy, StrategySignal

logger = logging.getLogger(__name__)


class ConsensusDivergenceStrategy(BaseStrategy):
    name = "Consensus Divergence"
    short_description = "Trade swarm vs. market probability divergence"

    def __init__(
        self,
        alpha_threshold: float = 0.05,
        max_swarm_std: float = 0.15,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha_threshold = alpha_threshold
        self.max_swarm_std = max_swarm_std
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct

    def describe(self) -> str:
        return (
            "Consensus Divergence runs a full agent swarm simulation for each event "
            "and compares the swarm's mean probability estimate to the current market "
            "price. When the divergence exceeds a threshold and the swarm has converged "
            "(low cross-simulation variance), we trade the direction of the gap using "
            "fractional Kelly sizing. This is the simplest and most direct application "
            "of the MiroFish methodology — pure 'wisdom of the simulated crowd' vs. "
            "the real market."
        )

    def generate_signals(self, events: list[dict]) -> list[StrategySignal]:
        signals = []
        for ev in events:
            eid = ev["event_id"]
            name = ev["event_name"]
            mp = ev["market_probability"]
            ctx = ev.get("context", "")

            result = self._run_sim(name, context=ctx)
            p_swarm = result.swarm_probability
            alpha = p_swarm - mp

            # Kelly
            if alpha > 0:
                b = (1 / mp) - 1 if mp > 0.01 else 100
                kelly_raw = (p_swarm * b - (1 - p_swarm)) / b if b > 0 else 0
            else:
                p_no = 1 - p_swarm
                mp_no = 1 - mp
                b = (1 / mp_no) - 1 if mp_no > 0.01 else 100
                kelly_raw = (p_no * b - (1 - p_no)) / b if b > 0 else 0

            kelly_frac = max(0, kelly_raw * self.kelly_fraction)
            pos_size = min(kelly_frac, self.max_position_pct)

            # Direction
            if abs(alpha) < self.alpha_threshold:
                direction = "no_trade"
                confidence = "low"
            elif result.swarm_std > self.max_swarm_std:
                direction = "no_trade"
                confidence = "low"
            elif not result.converged:
                direction = "no_trade"
                confidence = "low"
            elif alpha > 0:
                direction = "long_yes"
                confidence = "high" if alpha > 0.12 and result.swarm_std < 0.08 else "medium"
            else:
                direction = "long_no"
                confidence = "high" if abs(alpha) > 0.12 and result.swarm_std < 0.08 else "medium"

            signals.append(StrategySignal(
                event_id=eid,
                event_name=name,
                direction=direction,
                position_size_pct=pos_size,
                confidence=confidence,
                alpha=abs(alpha),
                strategy_metadata={
                    "swarm_prob": round(p_swarm, 4),
                    "market_prob": round(mp, 4),
                    "swarm_std": round(result.swarm_std, 4),
                    "r_hat": round(result.r_hat, 4),
                    "converged": result.converged,
                    "bullish_frac": round(result.bullish_fraction, 4),
                },
            ))

        n_trades = sum(1 for s in signals if s.direction != "no_trade")
        logger.info(f"[ConsensusDivergence] Generated {n_trades}/{len(signals)} trade signals")
        return signals
