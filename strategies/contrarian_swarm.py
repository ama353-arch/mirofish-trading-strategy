"""
Contrarian Swarm Strategy
─────────────────────────
Core idea: Overweight contrarian agent personas in the simulation and use
their high-conviction signals to fade the crowd. When contrarian agents
strongly disagree with the market consensus and have high conviction,
bet with the contrarians.

This exploits the well-documented tendency of prediction markets and equity
markets to overshoot on narrative-driven moves. Contrarian agents are
calibrated to be skeptical of consensus, anchored to base rates, and
attuned to mean-reversion dynamics.

Signal logic:
  1. Run simulation with a custom archetype distribution (40% contrarians)
  2. Extract the contrarian sub-population's mean probability estimate
  3. Compare contrarian estimate to market price
  4. Trade when contrarians strongly disagree with market AND have high conviction
  5. Only take trades where contrarian estimate differs from full-swarm estimate
     (i.e., the contrarians are seeing something the crowd doesn't)
"""

import logging
import random

import numpy as np

from .base import BaseStrategy, StrategySignal

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.simulation.engine import SimulationEngine
from src.agents.persona import PersonaGenerator

logger = logging.getLogger(__name__)

# Custom archetype distribution: heavy on contrarians
CONTRARIAN_HEAVY_ARCHETYPES = {
    "retail_bettor":       0.10,
    "political_analyst":   0.10,
    "contrarian_trader":   0.25,  # 2.5x overweight
    "institutional_trader":0.10,
    "social_influencer":   0.05,
    "data_scientist":      0.10,
    "domain_expert":       0.10,
    "noise_trader":        0.05,
    # Add an extra contrarian variant
    "contrarian_trader":   0.25,
}
# Note: the dict deduplication means contrarian_trader = 0.25
# We compensate by shrinking others. Net distribution:
CONTRARIAN_ARCHETYPES_FINAL = {
    "retail_bettor":       0.10,
    "political_analyst":   0.10,
    "contrarian_trader":   0.30,
    "institutional_trader":0.08,
    "social_influencer":   0.05,
    "data_scientist":      0.10,
    "domain_expert":       0.10,
    "noise_trader":        0.02,
}
# Equity variant
CONTRARIAN_EQUITY_ARCHETYPES = {
    "quant_trader":        0.10,
    "fundamental_analyst": 0.10,
    "retail_investor":     0.10,
    "market_maker":        0.05,
    "macro_strategist":    0.05,
    "event_driven":        0.05,
    "contrarian":          0.40,  # 4x overweight
    "algo_trader":         0.15,
}


class ContrarianSwarmStrategy(BaseStrategy):
    name = "Contrarian Swarm"
    short_description = "Overweight contrarians, fade the crowd when they have conviction"

    def __init__(
        self,
        contrarian_alpha_threshold: float = 0.06,
        contrarian_conviction_min: float = 0.55,
        divergence_from_crowd_min: float = 0.03,
        base_position_pct: float = 0.03,
        max_position_pct: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contrarian_alpha_threshold = contrarian_alpha_threshold
        self.contrarian_conviction_min = contrarian_conviction_min
        self.divergence_from_crowd_min = divergence_from_crowd_min
        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct

        # Monkey-patch the archetype distributions for contrarian-heavy simulation
        import config.settings as cfg
        self._orig_pm = cfg.PREDICTION_MARKET_ARCHETYPES.copy()
        self._orig_eq = cfg.EQUITY_MARKET_ARCHETYPES.copy()

    def describe(self) -> str:
        return (
            "Contrarian Swarm runs simulations with a population deliberately "
            "overweighted toward contrarian agent personas (30-40% vs. the usual 10%). "
            "It then isolates the contrarian sub-population's probability estimate and "
            "compares it to both the market price and the full-swarm consensus. Trades "
            "are taken when contrarians strongly disagree with the market AND their view "
            "diverges from the crowd — meaning the contrarians are seeing something "
            "the consensus misses. This exploits narrative-driven overshooting and "
            "crowd herding dynamics."
        )

    def generate_signals(self, events: list[dict]) -> list[StrategySignal]:
        signals = []

        # Temporarily override archetypes
        import config.settings as cfg
        cfg.PREDICTION_MARKET_ARCHETYPES.clear()
        cfg.PREDICTION_MARKET_ARCHETYPES.update(CONTRARIAN_ARCHETYPES_FINAL)

        try:
            for ev in events:
                eid = ev["event_id"]
                name = ev["event_name"]
                mp = ev["market_probability"]
                ctx = ev.get("context", "")

                result = self._run_sim(name, context=ctx)

                # Extract contrarian sub-population's estimates
                contrarian_archetypes = {"contrarian_trader", "contrarian"}
                contrarian_probs = []
                contrarian_convictions = []
                crowd_probs = []

                for sim in result.simulation_results:
                    for est in sim.agent_estimates:
                        if est["archetype"] in contrarian_archetypes:
                            contrarian_probs.append(est["final_probability"])
                            contrarian_convictions.append(est["final_conviction"])
                        else:
                            crowd_probs.append(est["final_probability"])

                if not contrarian_probs:
                    signals.append(StrategySignal(
                        event_id=eid, event_name=name, direction="no_trade",
                        position_size_pct=0, confidence="low", alpha=0,
                        strategy_metadata={"skip_reason": "no contrarian agents"},
                    ))
                    continue

                contrarian_mean = float(np.mean(contrarian_probs))
                contrarian_conviction = float(np.mean(contrarian_convictions))
                crowd_mean = float(np.mean(crowd_probs)) if crowd_probs else result.swarm_probability

                # Alpha: contrarian view vs. market
                alpha_vs_market = contrarian_mean - mp
                # Divergence: contrarian view vs. crowd
                divergence_vs_crowd = contrarian_mean - crowd_mean

                # Trade when:
                # 1. Contrarians strongly disagree with market
                # 2. Contrarians disagree with the crowd (unique information)
                # 3. Contrarian conviction is above threshold
                strong_alpha = abs(alpha_vs_market) > self.contrarian_alpha_threshold
                unique_view = abs(divergence_vs_crowd) > self.divergence_from_crowd_min
                high_conviction = contrarian_conviction > self.contrarian_conviction_min

                if strong_alpha and unique_view and high_conviction:
                    if alpha_vs_market > 0:
                        direction = "long_yes"
                    else:
                        direction = "long_no"

                    strength = min(1.0, abs(alpha_vs_market) / 0.15)
                    pos_size = min(self.base_position_pct * (1 + strength * 0.5), self.max_position_pct)
                    confidence = "high" if abs(alpha_vs_market) > 0.10 and contrarian_conviction > 0.65 else "medium"
                else:
                    direction = "no_trade"
                    pos_size = 0
                    confidence = "low"

                signals.append(StrategySignal(
                    event_id=eid,
                    event_name=name,
                    direction=direction,
                    position_size_pct=pos_size,
                    confidence=confidence,
                    alpha=abs(alpha_vs_market),
                    strategy_metadata={
                        "contrarian_prob": round(contrarian_mean, 4),
                        "crowd_prob": round(crowd_mean, 4),
                        "market_prob": round(mp, 4),
                        "contrarian_conviction": round(contrarian_conviction, 4),
                        "alpha_vs_market": round(alpha_vs_market, 4),
                        "divergence_vs_crowd": round(divergence_vs_crowd, 4),
                        "n_contrarian_agents": len(contrarian_probs),
                    },
                ))

        finally:
            # Restore original archetypes
            cfg.PREDICTION_MARKET_ARCHETYPES.clear()
            cfg.PREDICTION_MARKET_ARCHETYPES.update(self._orig_pm)

        n_trades = sum(1 for s in signals if s.direction != "no_trade")
        logger.info(f"[ContrarianSwarm] Generated {n_trades}/{len(signals)} trade signals")
        return signals
