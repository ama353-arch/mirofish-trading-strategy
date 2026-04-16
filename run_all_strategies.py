#!/usr/bin/env python3
"""
run_all_strategies.py — Run all 5 MiroFish trading strategies on synthetic
event data and produce a comparison dashboard.

No API keys needed. Runs entirely on rule-based agent simulation.

Usage:
    python run_all_strategies.py
    python run_all_strategies.py --events 100 --agents 80 --rounds 25 --sims 8
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from strategies import ALL_STRATEGIES
from strategies.strategy_comparison import StrategyComparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mirofish.compare")


def generate_synthetic_events(
    n_events: int = 60,
    seed: int = 12345,
) -> list[dict]:
    """
    Generate a diverse set of synthetic prediction market events for backtesting.

    Each event has:
      - A ground-truth probability (drawn from Beta distribution)
      - A noisy market probability (true_prob + noise)
      - An actual outcome (Bernoulli draw from true probability)
      - Optional enrichment data for cross-market arb

    The events span different "difficulty" regimes:
      - Easy events (true prob near 0 or 1, market is close)
      - Medium events (true prob 30-70%, market has moderate noise)
      - Hard events (true prob 40-60%, market is noisy)
    """
    rng = np.random.RandomState(seed)
    events = []

    # Event categories for realistic naming
    categories = [
        ("Fed cuts rates at {month} FOMC", ["March", "June", "September", "December"]),
        ("US GDP growth exceeds {pct}% in Q{q}", ["2", "2.5", "3", "3.5"]),
        ("{company} beats earnings expectations", ["Apple", "Google", "Microsoft", "Amazon", "NVIDIA", "Tesla", "Meta"]),
        ("CPI comes in below {pct}%", ["2.5", "3.0", "3.5", "4.0"]),
        ("{party} wins {state} in 2026 midterms", ["Democrats", "Republicans"]),
        ("Oil prices above ${price}/barrel by year-end", ["70", "80", "90", "100"]),
        ("Bitcoin exceeds ${price}K by Q{q} 2026", ["80", "100", "120", "150"]),
        ("{country} central bank raises rates", ["ECB", "BoE", "BoJ", "RBI"]),
        ("S&P 500 closes above {level} by end of Q{q}", ["5500", "5800", "6000", "6200"]),
        ("US unemployment rate exceeds {pct}%", ["4.0", "4.5", "5.0"]),
    ]

    for i in range(n_events):
        # Pick category and fill in template
        cat_idx = i % len(categories)
        template, options = categories[cat_idx]
        fill_key = template.split("{")[1].split("}")[0] if "{" in template else ""
        option = options[i % len(options)]
        extra = {"q": str((i % 4) + 1), "month": ["March", "June", "Sep", "Dec"][i % 4],
                 "state": ["Pennsylvania", "Georgia", "Arizona", "Michigan", "Wisconsin"][i % 5]}
        event_name = template
        for k, v in {fill_key: option, **extra}.items():
            event_name = event_name.replace(f"{{{k}}}", v)

        # Difficulty: cycle through easy/medium/hard
        difficulty = i % 3
        if difficulty == 0:  # Easy
            true_prob = rng.beta(0.5, 0.5)  # Bimodal (near 0 or 1)
            market_noise = rng.normal(0, 0.05)
        elif difficulty == 1:  # Medium
            true_prob = rng.beta(2, 2)  # Centered around 0.5
            market_noise = rng.normal(0, 0.08)
        else:  # Hard
            true_prob = rng.beta(5, 5)  # Tight around 0.5
            market_noise = rng.normal(0, 0.12)

        market_prob = float(np.clip(true_prob + market_noise, 0.05, 0.95))
        actual_outcome = 1.0 if rng.random() < true_prob else 0.0

        event = {
            "event_id": i,
            "event_name": event_name,
            "market_probability": market_prob,
            "actual_outcome": actual_outcome,
            "true_probability": float(true_prob),
            "difficulty": ["easy", "medium", "hard"][difficulty],
            "timestamp": f"2026-{(i // 5) + 1:02d}-{(i % 28) + 1:02d}",
            "context": f"Analyzing: {event_name}",
        }

        # Add options data for some events (for cross-market arb)
        if cat_idx in (2, 8):  # Earnings and S&P events
            event["implied_vol"] = float(rng.uniform(0.15, 0.60))
            event["implied_move_pct"] = float(rng.uniform(0.03, 0.12))
            event["options_tail_prob"] = float(rng.uniform(0.05, 0.30))

        events.append(event)

    return events


def print_event_summary(events: list[dict]):
    """Print a summary of the synthetic event dataset."""
    n = len(events)
    outcomes = [e["actual_outcome"] for e in events]
    probs = [e["market_probability"] for e in events]
    diffs = [e["difficulty"] for e in events]

    print(f"\n{'='*80}")
    print(f"  SYNTHETIC EVENT DATASET")
    print(f"{'='*80}")
    print(f"  Total events:        {n}")
    print(f"  YES outcomes:        {sum(outcomes):.0f} ({sum(outcomes)/n:.0%})")
    print(f"  NO outcomes:         {n - sum(outcomes):.0f} ({(n - sum(outcomes))/n:.0%})")
    print(f"  Avg market prob:     {np.mean(probs):.2%}")
    print(f"  Difficulty mix:      Easy={diffs.count('easy')}, Medium={diffs.count('medium')}, Hard={diffs.count('hard')}")
    print(f"  Events w/ options:   {sum(1 for e in events if 'implied_vol' in e)}")
    print()

    # Sample events
    print(f"  Sample events:")
    for e in events[:5]:
        marker = "YES" if e["actual_outcome"] == 1.0 else " NO"
        print(f"    [{marker}] (mkt: {e['market_probability']:.0%}) {e['event_name']}")
    print(f"    ... and {n-5} more")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Run and compare all MiroFish strategies")
    parser.add_argument("--events", type=int, default=60, help="Number of synthetic events")
    parser.add_argument("--agents", type=int, default=50, help="Agents per simulation")
    parser.add_argument("--rounds", type=int, default=20, help="Simulation rounds")
    parser.add_argument("--sims", type=int, default=5, help="Parallel simulations per event")
    parser.add_argument("--capital", type=float, default=10_000, help="Starting capital")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for events")
    parser.add_argument("--save", type=str, default="", help="Path to save results JSON")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  MIROFISH TRADING STRATEGY — MULTI-STRATEGY COMPARISON")
    print("  Agent Swarm Simulation for Market Prediction")
    print("  (Rule-based mode — no API keys needed)")
    print("=" * 80)

    # 1. Generate events
    print(f"\n[1/3] Generating {args.events} synthetic prediction market events...")
    events = generate_synthetic_events(n_events=args.events, seed=args.seed)
    print_event_summary(events)

    # 2. Run comparison
    print(f"[2/3] Running all 5 strategies...")
    print(f"       Config: {args.agents} agents, {args.rounds} rounds, {args.sims} sims per event")
    print(f"       This will run ~{len(ALL_STRATEGIES) * args.events * args.sims} total simulations")
    print()

    start = time.time()
    comp = StrategyComparison(
        events=events,
        initial_capital=args.capital,
        strategy_kwargs={
            "n_agents": args.agents,
            "n_rounds": args.rounds,
            "n_sims": args.sims,
            "seed": 42,
        },
    )
    result = comp.run()
    total_elapsed = time.time() - start

    # 3. Display results
    print(f"\n[3/3] Results (total time: {total_elapsed:.1f}s)")
    comp.print_dashboard(result)

    # Strategy descriptions
    print("=" * 80)
    print("  STRATEGY DESCRIPTIONS")
    print("=" * 80)
    for cls in ALL_STRATEGIES:
        strat = cls(n_agents=10, n_rounds=5, n_sims=1)
        print(f"\n  {strat.name}")
        print(f"  {'─' * len(strat.name)}")
        # Word-wrap description
        desc = strat.describe()
        words = desc.split()
        lines = []
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > 78:
                lines.append(line)
                line = "    " + w
            else:
                line += " " + w if line.strip() else "    " + w
        lines.append(line)
        print("\n".join(lines))
    print()

    # Save results
    save_path = args.save or str(
        Path(__file__).parent / "data" / "processed" / "strategy_comparison.json"
    )
    comp.save_results(result, save_path)
    print(f"Results saved to {save_path}\n")


if __name__ == "__main__":
    main()
