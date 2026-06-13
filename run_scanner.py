#!/usr/bin/env python3
"""
run_scanner.py — CLI entry point for the Kalshi market scanner.

Examples:
    # Default scan (top 10), uses live Kalshi API → pmxt → mock fallback chain.
    python run_scanner.py

    # Tight filters, faster sims, JSON output.
    python run_scanner.py --top 5 --min-liquidity 5000 --n-agents 40 --json

    # Force mock universe (deterministic, offline-friendly).
    python run_scanner.py --mock --top 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.scanner import KalshiScanner, ScanConfig
from src.scanner.kalshi_scanner import KalshiDiscovery


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_config(args: argparse.Namespace) -> ScanConfig:
    return ScanConfig(
        min_liquidity=args.min_liquidity,
        min_volume_24h=args.min_volume,
        min_days_to_close=args.min_days,
        max_days_to_close=args.max_days,
        max_candidates=args.max_candidates,
        top_n=args.top,
        n_agents=args.n_agents,
        n_rounds=args.n_rounds,
        n_sims=args.n_sims,
        seed=args.seed,
    )


def _print_table(result) -> None:
    print()
    print("=" * 100)
    print(f"  KALSHI MARKET SCANNER  —  source: {result.source}  "
          f"|  scored {result.n_scored}/{result.n_after_filter} of {result.n_discovered}  "
          f"|  {result.elapsed_seconds:.1f}s")
    print("=" * 100)
    if not result.opportunities:
        print("\n  No opportunities passed filters. Try lowering --min-liquidity or --min-volume.\n")
        return

    header = f"{'#':<3} {'SCORE':>6}  {'DIR':<9} {'EDGE':>7} {'P_SWARM':>8} {'P_MKT':>7} {'DAYS':>6} {'LIQ':>10}  QUESTION"
    print(header)
    print("-" * 100)
    for i, opp in enumerate(result.opportunities, 1):
        q = opp.question if len(opp.question) <= 38 else opp.question[:35] + "..."
        print(
            f"{i:<3} {opp.score:>6.3f}  "
            f"{opp.direction:<9} "
            f"{opp.edge:>+7.3f} "
            f"{opp.swarm_probability:>8.3f} "
            f"{opp.yes_price:>7.3f} "
            f"{opp.days_to_close:>6.1f} "
            f"{opp.liquidity:>10,.0f}  "
            f"{q}"
        )
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="MiroFish Kalshi market scanner")

    ap.add_argument("--top", type=int, default=10, help="Number of top opportunities to return")
    ap.add_argument("--max-candidates", type=int, default=40, help="Cap on markets passed to simulation")
    ap.add_argument("--min-liquidity", type=float, default=1_000.0)
    ap.add_argument("--min-volume", type=float, default=500.0)
    ap.add_argument("--min-days", type=float, default=0.5, help="Minimum days to close")
    ap.add_argument("--max-days", type=float, default=60.0, help="Maximum days to close")

    ap.add_argument("--n-agents", type=int, default=60)
    ap.add_argument("--n-rounds", type=int, default=15)
    ap.add_argument("--n-sims", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--mock", action="store_true", help="Force the offline mock market universe")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    ap.add_argument("-v", "--verbose", action="store_true")

    args = ap.parse_args()
    _setup_logging(args.verbose)
    cfg = _build_config(args)

    if args.mock:
        discovery = lambda _n: (KalshiDiscovery._mock_markets(), "mock")  # noqa: E731
        scanner = KalshiScanner(config=cfg, discovery=discovery)
    else:
        scanner = KalshiScanner(config=cfg)

    result = scanner.scan()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_table(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
