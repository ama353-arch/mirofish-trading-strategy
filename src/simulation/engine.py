"""
SimulationEngine — orchestrates the full agent swarm simulation.

Manages the lifecycle: persona generation → agent initialization →
simulation loop → result aggregation → convergence diagnostics.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..agents.persona import PersonaGenerator, AgentPersona
from ..agents.agent import SwarmAgent
from .environment import SimulationEnvironment

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from config.settings import (
    DEFAULT_AGENT_COUNT,
    DEFAULT_SIMULATION_ROUNDS,
    DEFAULT_NUM_SIMULATIONS,
    CONVERGENCE_THRESHOLD,
    MIN_EFFECTIVE_SAMPLE_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result of a single simulation run."""
    simulation_id: int
    agent_estimates: list[dict]         # Per-agent final estimates
    mean_probability: float
    std_probability: float
    median_probability: float
    bullish_fraction: float
    mean_conviction: float
    sentiment_trajectory: list[float]
    n_agents: int
    n_rounds: int
    elapsed_seconds: float
    environment_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "simulation_id": self.simulation_id,
            "mean_probability": self.mean_probability,
            "std_probability": self.std_probability,
            "median_probability": self.median_probability,
            "bullish_fraction": self.bullish_fraction,
            "mean_conviction": self.mean_conviction,
            "sentiment_trajectory": self.sentiment_trajectory,
            "n_agents": self.n_agents,
            "n_rounds": self.n_rounds,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class SwarmResult:
    """Aggregated result across multiple simulation runs."""
    event_description: str
    simulation_results: list[SimulationResult]
    swarm_probability: float            # Mean across simulations
    swarm_std: float                    # Std across simulations
    swarm_median: float
    r_hat: float                        # Gelman-Rubin convergence diagnostic
    effective_sample_size: float
    converged: bool
    bullish_fraction: float
    mean_conviction: float

    # Per-archetype breakdown
    archetype_probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event": self.event_description,
            "swarm_probability": round(self.swarm_probability, 4),
            "swarm_std": round(self.swarm_std, 4),
            "swarm_median": round(self.swarm_median, 4),
            "r_hat": round(self.r_hat, 4),
            "effective_sample_size": round(self.effective_sample_size, 1),
            "converged": self.converged,
            "bullish_fraction": round(self.bullish_fraction, 4),
            "mean_conviction": round(self.mean_conviction, 4),
            "n_simulations": len(self.simulation_results),
            "archetype_probabilities": {
                k: round(v, 4) for k, v in self.archetype_probabilities.items()
            },
        }


class SimulationEngine:
    """
    Orchestrates multiple simulation runs and aggregates results.

    Usage:
        engine = SimulationEngine(llm_client=llm)
        result = engine.run(
            event_description="Will the Fed cut rates at the March 2026 meeting?",
            context="CPI came in at 2.3%, labor market cooling...",
            market_type="prediction",
        )
        print(result.swarm_probability)
    """

    def __init__(
        self,
        llm_client=None,
        n_agents: int | None = None,
        n_rounds: int | None = None,
        n_simulations: int | None = None,
    ):
        self.llm = llm_client
        self.n_agents = n_agents or DEFAULT_AGENT_COUNT
        self.n_rounds = n_rounds or DEFAULT_SIMULATION_ROUNDS
        self.n_simulations = n_simulations or DEFAULT_NUM_SIMULATIONS
        self.persona_generator = PersonaGenerator(llm_client=llm_client)

    def run(
        self,
        event_description: str,
        context: str = "",
        market_type: str = "prediction",
        initial_info: str = "",
        info_injections: list[tuple[int, str]] | None = None,
        use_llm_personas: bool = False,
        use_llm_agents: bool = False,
        seed: int | None = None,
    ) -> SwarmResult:
        """
        Run the full swarm simulation pipeline.

        Args:
            event_description: What event/question to simulate
            context: Background knowledge graph context
            market_type: "prediction" or "equity"
            initial_info: Information available at simulation start
            info_injections: List of (round, info_text) to inject during simulation
            use_llm_personas: Use LLM for richer persona generation
            use_llm_agents: Use LLM for agent reasoning (expensive!)
            seed: Random seed for reproducibility
        """
        logger.info(f"Starting swarm simulation: {event_description[:80]}")
        logger.info(f"Config: {self.n_agents} agents, {self.n_rounds} rounds, {self.n_simulations} simulations")

        sim_results = []
        for sim_id in range(self.n_simulations):
            if seed is not None:
                np.random.seed(seed + sim_id)
                import random
                random.seed(seed + sim_id)

            result = self._run_single_simulation(
                sim_id=sim_id,
                event_description=event_description,
                context=context,
                market_type=market_type,
                initial_info=initial_info,
                info_injections=info_injections or [],
                use_llm_personas=use_llm_personas,
                use_llm_agents=use_llm_agents,
            )
            sim_results.append(result)
            logger.info(f"Simulation {sim_id + 1}/{self.n_simulations}: "
                       f"mean_prob={result.mean_probability:.3f}, "
                       f"std={result.std_probability:.3f}")

        # Aggregate across simulations
        return self._aggregate_results(event_description, sim_results)

    def _run_single_simulation(
        self,
        sim_id: int,
        event_description: str,
        context: str,
        market_type: str,
        initial_info: str,
        info_injections: list[tuple[int, str]],
        use_llm_personas: bool,
        use_llm_agents: bool,
    ) -> SimulationResult:
        """Run a single simulation instance."""
        start_time = time.time()

        # 1. Generate personas
        personas = self.persona_generator.generate_population(
            market_type=market_type,
            n_agents=self.n_agents,
            event_context=event_description,
            use_llm=use_llm_personas,
        )

        # 2. Create agents
        agent_llm = self.llm if use_llm_agents else None
        agents = [SwarmAgent(persona=p, llm_client=agent_llm) for p in personas]

        # 3. Create environment
        env = SimulationEnvironment(
            event_description=event_description,
            initial_context=context,
        )
        for round_num, info in info_injections:
            env.inject_information(round_num, info)

        # 4. Initialize agents
        for agent in agents:
            agent.initialize(event_description, context, initial_info)

        # 5. Simulation loop
        for round_num in range(self.n_rounds):
            # Agents observe the timeline
            timeline = env.get_timeline(round_num)
            for agent in agents:
                agent.observe(timeline)

            # Agents act
            for agent in agents:
                post = agent.act(round_num, event_description)
                if post:
                    env.add_post(post)

            # Process round
            env.process_round(round_num)

        # 6. Collect results
        estimates = [agent.get_final_estimate() for agent in agents]
        probs = [e["final_probability"] for e in estimates]
        convictions = [e["final_conviction"] for e in estimates]
        stances = [e["final_stance"] for e in estimates]

        elapsed = time.time() - start_time

        return SimulationResult(
            simulation_id=sim_id,
            agent_estimates=estimates,
            mean_probability=float(np.mean(probs)),
            std_probability=float(np.std(probs)),
            median_probability=float(np.median(probs)),
            bullish_fraction=sum(1 for s in stances if s == "bullish") / len(stances),
            mean_conviction=float(np.mean(convictions)),
            sentiment_trajectory=env.get_sentiment_trajectory(),
            n_agents=len(agents),
            n_rounds=self.n_rounds,
            elapsed_seconds=round(elapsed, 2),
            environment_stats=env.get_activity_stats(),
        )

    def _aggregate_results(
        self,
        event_description: str,
        sim_results: list[SimulationResult],
    ) -> SwarmResult:
        """Aggregate results across multiple simulation runs."""
        sim_means = [r.mean_probability for r in sim_results]
        sim_stds = [r.std_probability for r in sim_results]

        # Gelman-Rubin R-hat (simplified)
        r_hat = self._compute_r_hat(sim_results)

        # Effective sample size
        n_eff = self._compute_effective_sample_size(sim_results, r_hat)

        # Per-archetype breakdown
        archetype_probs = self._compute_archetype_breakdown(sim_results)

        overall_mean = float(np.mean(sim_means))
        overall_std = float(np.std(sim_means))
        overall_median = float(np.median(sim_means))

        return SwarmResult(
            event_description=event_description,
            simulation_results=sim_results,
            swarm_probability=overall_mean,
            swarm_std=overall_std,
            swarm_median=overall_median,
            r_hat=r_hat,
            effective_sample_size=n_eff,
            converged=r_hat < CONVERGENCE_THRESHOLD and n_eff > MIN_EFFECTIVE_SAMPLE_SIZE,
            bullish_fraction=float(np.mean([r.bullish_fraction for r in sim_results])),
            mean_conviction=float(np.mean([r.mean_conviction for r in sim_results])),
            archetype_probabilities=archetype_probs,
        )

    @staticmethod
    def _compute_r_hat(sim_results: list[SimulationResult]) -> float:
        """
        Simplified Gelman-Rubin R-hat convergence diagnostic.

        Compares between-simulation variance to within-simulation variance.
        R-hat < 1.1 indicates convergence.
        """
        if len(sim_results) < 2:
            return 1.0

        # Between-chain variance
        chain_means = [r.mean_probability for r in sim_results]
        B = np.var(chain_means, ddof=1)

        # Within-chain variance (average of per-simulation agent variance)
        W = np.mean([r.std_probability ** 2 for r in sim_results])

        if W == 0:
            return 1.0

        # Pooled variance estimate
        n = sim_results[0].n_agents
        m = len(sim_results)
        var_hat = (1 - 1/n) * W + (1/n) * B

        r_hat = np.sqrt(var_hat / W) if W > 0 else 1.0
        return float(r_hat)

    @staticmethod
    def _compute_effective_sample_size(
        sim_results: list[SimulationResult],
        r_hat: float,
    ) -> float:
        """Estimate effective sample size from R-hat."""
        total_agents = sum(r.n_agents for r in sim_results)
        if r_hat <= 1.0:
            return float(total_agents)
        return float(total_agents / (r_hat ** 2))

    @staticmethod
    def _compute_archetype_breakdown(sim_results: list[SimulationResult]) -> dict[str, float]:
        """Compute mean probability estimate by archetype across simulations."""
        archetype_probs: dict[str, list[float]] = {}
        for sim in sim_results:
            for est in sim.agent_estimates:
                arch = est["archetype"]
                if arch not in archetype_probs:
                    archetype_probs[arch] = []
                archetype_probs[arch].append(est["final_probability"])

        return {
            arch: float(np.mean(probs))
            for arch, probs in archetype_probs.items()
        }
