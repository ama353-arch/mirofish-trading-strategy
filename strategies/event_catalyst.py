"""
Event Catalyst Strategy
───────────────────────
Core idea: Run TWO simulations for each event — one with pre-event
information only, and one with post-event information injected. Trade
the *delta* between pre and post-event simulated probabilities.

This captures how the market SHOULD react to new information, according
to the agent swarm. If the swarm's post-event probability shifts sharply
but the actual market hasn't moved yet (or moves the wrong way), that's
the edge.

Signal logic:
  1. Run pre-event simulation (agents see only background context)
  2. Run post-event simulation (agents see the event outcome/data)
  3. Compute delta = post_prob - pre_prob
  4. Trade when |delta| is large and the market price hasn't fully adjusted
  5. Size proportional to delta magnitude and conviction
"""

import logging
import numpy as np

from .base import BaseStrategy, StrategySignal

logger = logging.getLogger(__name__)


class EventCatalystStrategy(BaseStrategy):
    name = "Event Catalyst"
    short_description = "Trade pre/post event probability deltas"

    def __init__(
        self,
        delta_threshold: float = 0.03,
        market_adjustment_tolerance: float = 0.02,
        base_position_pct: float = 0.03,
        max_position_pct: float = 0.06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.delta_threshold = delta_threshold
        self.market_adjustment_tolerance = market_adjustment_tolerance
        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct

    def describe(self) -> str:
        return (
            "Event Catalyst runs two parallel simulations for each event: a pre-event "
            "simulation where agents only see background context, and a post-event "
            "simulation where agents receive the new information (earnings data, Fed "
            "decision, election result, etc.). The strategy trades the delta between "
            "these two estimates when the actual market price hasn't fully adjusted. "
            "This captures 'information processing lag' — the time it takes for diverse "
            "market participants to digest and act on new information, modeled by our "
            "agent population."
        )

    def generate_signals(self, events: list[dict]) -> list[StrategySignal]:
        signals = []
        for ev in events:
            eid = ev["event_id"]
            name = ev["event_name"]
            mp = ev["market_probability"]
            pre_ctx = ev.get("context", "")
            post_info = ev.get("post_event_info", "")

            # If no post-event info provided, synthesize a simulated event outcome
            if not post_info:
                post_info = self._synthesize_event_info(name, mp)

            # Simulation 1: Pre-event (background context only)
            pre_result = self.engine.run(
                event_description=name,
                context=pre_ctx,
                market_type="prediction",
                seed=self.seed,
            )
            pre_prob = pre_result.swarm_probability

            # Simulation 2: Post-event (with new information)
            post_result = self.engine.run(
                event_description=name,
                context=pre_ctx,
                initial_info=post_info,
                market_type="prediction",
                seed=self.seed + 1000,  # Different seed for independence
            )
            post_prob = post_result.swarm_probability

            # Delta: how much the swarm's view shifted after the event
            delta = post_prob - pre_prob

            # How much has the market already adjusted?
            # (In real-time, mp would be the CURRENT market price post-event)
            implied_market_adjustment = mp - pre_prob
            unadjusted_delta = delta - implied_market_adjustment

            # Trade when the swarm sees a bigger shift than the market has priced in
            strong_delta = abs(delta) > self.delta_threshold
            market_lag = abs(unadjusted_delta) > self.market_adjustment_tolerance

            if strong_delta and market_lag:
                if unadjusted_delta > 0:
                    direction = "long_yes"
                else:
                    direction = "long_no"

                # Size proportional to delta magnitude
                delta_strength = min(1.0, abs(delta) / 0.20)
                pos_size = min(
                    self.base_position_pct * (1 + delta_strength),
                    self.max_position_pct,
                )

                # Confidence based on convergence and delta alignment
                both_converged = pre_result.converged and post_result.converged
                confidence = "high" if both_converged and abs(delta) > 0.12 else "medium"
            else:
                direction = "no_trade"
                pos_size = 0
                confidence = "low"
                unadjusted_delta = 0

            signals.append(StrategySignal(
                event_id=eid,
                event_name=name,
                direction=direction,
                position_size_pct=pos_size,
                confidence=confidence,
                alpha=abs(unadjusted_delta),
                strategy_metadata={
                    "pre_prob": round(pre_prob, 4),
                    "post_prob": round(post_prob, 4),
                    "delta": round(delta, 4),
                    "market_prob": round(mp, 4),
                    "unadjusted_delta": round(unadjusted_delta, 4),
                    "pre_converged": pre_result.converged,
                    "post_converged": post_result.converged,
                },
            ))

        n_trades = sum(1 for s in signals if s.direction != "no_trade")
        logger.info(f"[EventCatalyst] Generated {n_trades}/{len(signals)} trade signals")
        return signals

    @staticmethod
    def _synthesize_event_info(event_name: str, market_prob: float) -> str:
        """
        If no post-event info is provided, generate a generic event stimulus.
        This creates a slight informational nudge to see how the swarm responds.
        """
        # Simulate a mild surprise in either direction
        if market_prob > 0.5:
            return (
                f"New information regarding '{event_name}': Recent developments "
                f"suggest the outcome may be more uncertain than previously thought. "
                f"Some analysts are revising their estimates downward."
            )
        else:
            return (
                f"New information regarding '{event_name}': Recent developments "
                f"suggest the outcome may be more likely than previously thought. "
                f"Several indicators have shifted in the affirmative direction."
            )
