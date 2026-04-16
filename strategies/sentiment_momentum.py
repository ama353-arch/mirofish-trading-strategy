"""
Sentiment Momentum Strategy
────────────────────────────
Core idea: Run the agent swarm simulation and track how agent sentiment
*evolves* over simulation rounds. Trade directional shifts — go long when
sentiment is accelerating positive, short when accelerating negative.

Unlike Consensus Divergence (which cares about the final level), this
strategy cares about the *trajectory*. A swarm that starts bearish and
shifts bullish over 30 rounds is a stronger signal than one that starts
and ends bullish.

Signal logic:
  1. Run simulation, extract per-round aggregate sentiment trajectory
  2. Fit a linear trend to the trajectory → slope = sentiment momentum
  3. Compute acceleration = change in slope over first-half vs. second-half
  4. Trade when momentum and acceleration are aligned and strong
"""

import logging
import numpy as np

from .base import BaseStrategy, StrategySignal

logger = logging.getLogger(__name__)


class SentimentMomentumStrategy(BaseStrategy):
    name = "Sentiment Momentum"
    short_description = "Trade directional shifts in agent sentiment trajectory"

    def __init__(
        self,
        momentum_threshold: float = 0.003,
        acceleration_threshold: float = 0.001,
        base_position_pct: float = 0.03,
        max_position_pct: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.momentum_threshold = momentum_threshold
        self.acceleration_threshold = acceleration_threshold
        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct

    def describe(self) -> str:
        return (
            "Sentiment Momentum tracks how agent opinion evolves over the course "
            "of a simulation rather than just the final consensus. It fits a linear "
            "trend to the aggregate sentiment trajectory across simulation rounds and "
            "computes both momentum (slope) and acceleration (change in slope between "
            "the first and second halves). Trades are taken when momentum and acceleration "
            "are aligned — a swarm that's shifting increasingly bullish is a stronger buy "
            "signal than one with stable bullish consensus. This captures the 'narrative "
            "is turning' dynamic that often precedes real market moves."
        )

    def generate_signals(self, events: list[dict]) -> list[StrategySignal]:
        signals = []
        for ev in events:
            eid = ev["event_id"]
            name = ev["event_name"]
            mp = ev["market_probability"]
            ctx = ev.get("context", "")

            result = self._run_sim(name, context=ctx)

            # Aggregate sentiment trajectories across all simulation runs
            all_trajectories = []
            for sim in result.simulation_results:
                traj = sim.sentiment_trajectory
                if traj:
                    all_trajectories.append(traj)

            if not all_trajectories:
                signals.append(self._no_trade_signal(eid, name, "no trajectory data"))
                continue

            # Average trajectory across simulations
            min_len = min(len(t) for t in all_trajectories)
            avg_traj = np.mean([t[:min_len] for t in all_trajectories], axis=0)

            if len(avg_traj) < 6:
                signals.append(self._no_trade_signal(eid, name, "trajectory too short"))
                continue

            # Overall momentum: slope of linear fit
            x = np.arange(len(avg_traj))
            momentum = float(np.polyfit(x, avg_traj, 1)[0])

            # Acceleration: compare first-half slope to second-half slope
            mid = len(avg_traj) // 2
            first_half = avg_traj[:mid]
            second_half = avg_traj[mid:]
            slope_1 = float(np.polyfit(np.arange(len(first_half)), first_half, 1)[0])
            slope_2 = float(np.polyfit(np.arange(len(second_half)), second_half, 1)[0])
            acceleration = slope_2 - slope_1

            # Sentiment level at end vs. start
            sentiment_shift = float(avg_traj[-1] - avg_traj[0])

            # Signal logic: momentum-driven, acceleration as confidence modifier
            strong_mom = abs(momentum) > self.momentum_threshold
            strong_accel = abs(acceleration) > self.acceleration_threshold
            aligned = (momentum > 0 and acceleration > 0) or (momentum < 0 and acceleration < 0)

            if strong_mom:
                if momentum > 0:
                    direction = "long_yes"
                else:
                    direction = "long_no"

                # Scale position by momentum strength
                mom_strength = min(1.0, abs(momentum) / (self.momentum_threshold * 3))
                # Reduce size if acceleration opposes momentum (decelerating)
                accel_mod = 1.0 if aligned else 0.6
                pos_size = min(
                    self.base_position_pct * (1 + mom_strength) * accel_mod,
                    self.max_position_pct,
                )
                alpha = abs(sentiment_shift) * 0.5  # Rough edge estimate
                if aligned and strong_accel and abs(momentum) > self.momentum_threshold * 2:
                    confidence = "high"
                elif aligned:
                    confidence = "medium"
                else:
                    confidence = "low"
            else:
                direction = "no_trade"
                pos_size = 0
                alpha = 0
                confidence = "low"

            signals.append(StrategySignal(
                event_id=eid,
                event_name=name,
                direction=direction,
                position_size_pct=pos_size,
                confidence=confidence,
                alpha=alpha,
                strategy_metadata={
                    "momentum": round(momentum, 6),
                    "acceleration": round(acceleration, 6),
                    "sentiment_shift": round(sentiment_shift, 4),
                    "final_sentiment": round(float(avg_traj[-1]), 4),
                    "initial_sentiment": round(float(avg_traj[0]), 4),
                    "trajectory_length": len(avg_traj),
                },
            ))

        n_trades = sum(1 for s in signals if s.direction != "no_trade")
        logger.info(f"[SentimentMomentum] Generated {n_trades}/{len(signals)} trade signals")
        return signals

    def _no_trade_signal(self, eid: int, name: str, reason: str) -> StrategySignal:
        return StrategySignal(
            event_id=eid, event_name=name, direction="no_trade",
            position_size_pct=0, confidence="low", alpha=0,
            strategy_metadata={"skip_reason": reason},
        )
