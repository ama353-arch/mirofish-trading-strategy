#!/usr/bin/env python3
"""
run_walkforward.py — honest, out-of-sample backtest of the MiroFish swarm.

This is the replacement for the old synthetic demo. Instead of feeding the
backtest synthetic events where the swarm was DEFINED to beat the market, it
runs the *real* swarm engine as the brain and scores it walk-forward, on test
folds only, with realistic spread-crossing costs and a deflated-Sharpe sanity
check.

    python3 run_walkforward.py                 # uses bundled sample markets
    python3 run_walkforward.py path/to/markets.json

Input JSON is a list of resolved markets in the unified schema
(see src/data/historical.py). Each may also carry a "question" string used to
drive the swarm.

IMPORTANT — what this does and does NOT prove:
  • It proves the PLUMBING: the swarm -> edge -> sizing -> OOS scoring path is
    correct and free of look-ahead.
  • It does NOT yet prove EDGE. With rule-based agents and no real-time
    information feeding the swarm, the brain is largely uninformed, so a
    break-even-minus-costs result is the honest, expected outcome. Edge comes
    only once (a) real resolved-market data and (b) an informed swarm
    (news/knowledge-graph context, LLM agents) are wired in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.signals.walkforward import (
    WalkForwardBacktester,
    block_bootstrap_equity_paths,
    deflated_sharpe_ratio,
)
from src.signals.engine_report import build_engine_report
from src.simulation.engine import SimulationEngine

SAMPLE_PATH = Path(__file__).parent / "data" / "processed" / "sample_resolved_markets.json"


def make_swarm_prob_fn(n_agents=30, n_rounds=10, n_sims=4):
    """A cached swarm probability source: `prob_fn(event) -> probability`.

    The swarm probability depends only on the event, so we simulate each event
    once and reuse it across folds and the parameter grid.
    """
    engine = SimulationEngine(n_agents=n_agents, n_rounds=n_rounds, n_simulations=n_sims)
    cache: dict = {}

    def prob_fn(event: dict) -> float:
        eid = event["id"]
        if eid not in cache:
            seed = abs(hash(eid)) % (2**31)
            res = engine.run(
                event_description=event.get("question", eid),
                market_type="prediction",
                seed=seed,
            )
            cache[eid] = res.swarm_probability
        return cache[eid]

    return prob_fn


def load_events(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_PATH
    if not path.exists():
        print(f"No event file at {path}. Provide a unified-schema JSON list.")
        sys.exit(1)

    events = load_events(path)
    print(f"Loaded {len(events)} resolved markets from {path.name}\n")

    prob_fn = make_swarm_prob_fn()
    report = build_engine_report(
        events, prob_fn,
        train_size=max(3, len(events) // 4),
        test_size=max(2, len(events) // 6),
        edge_min=0.05,
    )
    cal = report["calibration"]
    oos = report["oos"]

    print("=== CALIBRATION (swarm vs resolved outcomes, all events) ===")
    print(f"  Samples:         {cal['n']}")
    print(f"  Base rate:       {cal['base_rate']:.1%}")
    print(f"  Brier score:     {cal['brier']:.4f}   (lower is better; "
          f"{cal['base_rate'] * (1 - cal['base_rate']):.4f} = always-base-rate)")
    print(f"  Log loss:        {cal['log_loss']:.4f}")

    print("\n=== OUT-OF-SAMPLE RESULT (test folds only) ===")
    print(f"  OOS trades:      {oos['n_trades']}")
    print(f"  Win rate:        {oos['win_rate']:.1%}")
    print(f"  Total return:    {oos['total_return']:+.2%}")

    print("\n=== PORTFOLIO ALLOCATION (concurrent book, bankroll fraction) ===")
    deployed = sum(a["size"] for a in report["allocations"])
    print(f"  Positions:       {len(report['allocations'])}")
    print(f"  Total deployed:  {deployed:.1%}  (cap 20%)")

    print("\n  NOTE: plumbing check only. A poor Brier here means the swarm is")
    print("  not yet calibrated on this sample — exactly what E5 refinement and")
    print("  real data are for. The harness is honest; the edge is unproven.")


if __name__ == "__main__":
    main()
