#!/usr/bin/env python3
"""
run_live.py — Live trading pipeline for MiroFish agent swarm strategy.

This is the production entry point that:
  1. Connects to prediction market exchanges via pmxt (unified API)
  2. Discovers and ranks markets using semantic search (RAG)
  3. Runs agent swarm simulations on the best candidates
  4. Generates trade signals with safety rails
  5. Executes trades (dry-run by default)

Designed to be demoed to Polymarket, Kalshi, and other trading desks.

Usage:
    # Dry run (no real trades, no API keys needed)
    python run_live.py --dry-run

    # Live discovery with Polymarket data
    python run_live.py --exchange polymarket --query "2026 elections"

    # Full pipeline with LLM agents + live trading
    python run_live.py --exchange polymarket --query "Fed rate" --use-llm --live

    # Multi-exchange scan
    python run_live.py --exchange polymarket --exchange kalshi --query "economy"
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import *
from src.utils.pmxt_client import UnifiedPredictionClient, UnifiedMarket, UnifiedOrder, get_live_markets
from src.utils.market_rag import MarketRAG
from src.data.ingest import DataIngestionPipeline
from src.data.knowledge_graph import KnowledgeGraph
from src.simulation.engine import SimulationEngine
from src.signals.generator import SignalGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mirofish.live")


def print_header():
    print("\n" + "=" * 78)
    print("  MIROFISH LIVE TRADING PIPELINE")
    print("  Agent Swarm Simulation × Unified Prediction Market API")
    print("=" * 78)


def run_live_pipeline(
    exchanges: list[str],
    query: str,
    n_agents: int = 80,
    n_rounds: int = 25,
    n_sims: int = 8,
    max_markets: int = 5,
    use_llm: bool = False,
    live: bool = False,
    bankroll: float = 10_000,
    fetch_news: bool = True,
):
    """Full live trading pipeline."""
    print_header()

    print(f"\n  Config:")
    print(f"    Exchanges:    {', '.join(exchanges)}")
    print(f"    Query:        '{query}'")
    print(f"    Agents:       {n_agents}")
    print(f"    Sims:         {n_sims}")
    print(f"    LLM agents:   {'Yes' if use_llm else 'No (rule-based)'}")
    print(f"    Mode:         {'LIVE' if live else 'DRY RUN'}")
    print(f"    Bankroll:     ${bankroll:,.0f}")
    print()

    # ── Step 1: Connect to exchanges ──────────────────────────────────────
    print("[1/6] Connecting to exchanges...")
    client = UnifiedPredictionClient(bankroll=bankroll)
    for ex in exchanges:
        connected = client.connect(ex)
        status = "connected" if connected else "FAILED"
        print(f"  {ex}: {status}")

    # ── Step 2: Discover markets ──────────────────────────────────────────
    print(f"\n[2/6] Searching markets for '{query}'...")
    markets = client.search_markets(query, min_volume=0, max_results=50)

    if not markets:
        print("  No markets found via live API. Using mock markets for demo...")
        markets = _generate_mock_markets(query, n=10)

    print(f"  Found {len(markets)} markets")

    # ── Step 3: Rank via semantic search (RAG) ────────────────────────────
    print(f"\n[3/6] Ranking markets by relevance (RAG)...")
    rag = MarketRAG()
    market_dicts = []
    for m in markets:
        market_dicts.append({
            "id": m.market_id,
            "question": m.market_question,
            "exchange": m.exchange,
            "outcome_prices": m.outcome_prices,
            "volume_24h": m.volume_24h,
            "outcomes": m.outcomes,
        })
    rag.index_markets(market_dicts)
    ranked = rag.search(query, top_k=max_markets)

    print(f"  Top {len(ranked)} markets by relevance:")
    for i, r in enumerate(ranked):
        print(f"    {i+1}. [{r.exchange}] {r.question}")
        print(f"       YES @ {r.yes_price:.0%} | Volume: ${r.volume:,.0f} | Relevance: {r.similarity_score:.2f}")

    # ── Step 4: Run swarm simulations ─────────────────────────────────────
    print(f"\n[4/6] Running agent swarm simulations...")
    llm_client = None
    if use_llm:
        from src.utils.llm import LLMClient
        llm_client = LLMClient()

    engine = SimulationEngine(
        llm_client=llm_client,
        n_agents=n_agents,
        n_rounds=n_rounds,
        n_simulations=n_sims,
    )

    signals = []
    for i, r in enumerate(ranked):
        print(f"\n  Simulating: {r.question[:70]}...")

        # Optionally fetch live news context
        context = ""
        if fetch_news:
            pipeline = DataIngestionPipeline()
            docs = pipeline.ingest_for_event(r.question, subreddits=["politics", "economics", "wallstreetbets"])
            context = "\n".join([d["text"][:500] for d in docs[:5]])

        start = time.time()
        result = engine.run(
            event_description=r.question,
            context=context,
            market_type="prediction",
            use_llm_agents=use_llm,
            seed=42 + i,
        )
        elapsed = time.time() - start

        # Generate signal
        signal = SignalGenerator.generate_prediction_market_signal(
            swarm_result=result,
            market_probability=r.yes_price,
            bankroll=bankroll,
        )

        signals.append({
            "market": r,
            "result": result,
            "signal": signal,
            "elapsed": elapsed,
        })

        # Display result
        arrow = "▲" if signal.alpha > 0 else "▼" if signal.alpha < 0 else "─"
        print(f"    Swarm: {result.swarm_probability:.1%} | Market: {r.yes_price:.1%} | "
              f"Alpha: {signal.alpha:+.1%} {arrow}")
        print(f"    Direction: {signal.direction} | Confidence: {signal.confidence} | "
              f"Kelly: {signal.kelly_fraction:.2%} | Time: {elapsed:.1f}s")
        if result.archetype_probabilities:
            top_arch = sorted(result.archetype_probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"    Top archetypes: {', '.join(f'{a}={p:.0%}' for a,p in top_arch)}")

    # ── Step 5: Generate trade recommendations ────────────────────────────
    trade_signals = [s for s in signals if s["signal"].direction != "no_trade"]

    print(f"\n{'='*78}")
    print(f"[5/6] TRADE RECOMMENDATIONS")
    print(f"{'='*78}")

    if not trade_signals:
        print("\n  No actionable trades found. All markets are fairly priced according to the swarm.")
    else:
        for s in trade_signals:
            sig = s["signal"]
            mkt = s["market"]
            print(f"\n  {'BUY YES' if sig.direction == 'long_yes' else 'BUY NO'}: {mkt.question[:65]}")
            print(f"  ├─ Exchange:     {mkt.exchange}")
            print(f"  ├─ Swarm prob:   {sig.swarm_probability:.1%}")
            print(f"  ├─ Market price: {sig.market_probability:.1%}")
            print(f"  ├─ Alpha:        {sig.alpha:+.1%}")
            print(f"  ├─ Edge:         {sig.expected_edge:+.1%}")
            print(f"  ├─ Confidence:   {sig.confidence}")
            print(f"  ├─ Position:     {sig.position_size_pct:.2%} of bankroll (${bankroll * sig.position_size_pct:,.0f})")
            print(f"  ├─ Stop loss:    {sig.stop_loss_price:.2f}")
            print(f"  └─ Take profit:  {sig.take_profit_price:.2f}")

    # ── Step 6: Execute (dry run or live) ─────────────────────────────────
    print(f"\n{'='*78}")
    print(f"[6/6] EXECUTION ({'LIVE' if live else 'DRY RUN'})")
    print(f"{'='*78}")

    for s in trade_signals:
        sig = s["signal"]
        mkt = s["market"]

        outcome = "Yes" if sig.direction == "long_yes" else "No"
        amount = bankroll * sig.position_size_pct
        price = sig.market_probability if sig.direction == "long_yes" else (1 - sig.market_probability)

        order = UnifiedOrder(
            exchange=mkt.exchange,
            market_id=mkt.market_id,
            outcome=outcome,
            side="buy",
            price=price,
            amount=amount,
        )

        result = client.place_order(order, dry_run=not live)
        status = result.get("status", "unknown")
        print(f"\n  Order: BUY {outcome} on '{mkt.question[:50]}...'")
        print(f"  Price: {price:.2f} | Amount: ${amount:.2f} | Status: {status}")
        if status == "rejected":
            print(f"  Reason: {result.get('reason', '')}")

    # Exposure summary
    exposure = client.get_exposure()
    print(f"\n  Portfolio Exposure:")
    print(f"    Deployed:  ${exposure['total_deployed']:,.2f} ({exposure['exposure_pct']:.1%})")
    print(f"    Remaining: ${exposure['remaining_capacity']:,.2f}")
    print(f"    Positions: {exposure['n_positions']}")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "exchanges": exchanges, "query": query,
            "n_agents": n_agents, "n_sims": n_sims,
            "mode": "live" if live else "dry_run",
        },
        "markets_scanned": len(markets),
        "markets_simulated": len(ranked),
        "trades_recommended": len(trade_signals),
        "signals": [
            {
                "market": s["market"].question,
                "exchange": s["market"].exchange,
                "direction": s["signal"].direction,
                "swarm_prob": s["signal"].swarm_probability,
                "market_prob": s["signal"].market_probability,
                "alpha": s["signal"].alpha,
                "confidence": s["signal"].confidence,
                "position_pct": s["signal"].position_size_pct,
            }
            for s in signals
        ],
    }
    out_path = Path(__file__).parent / "data" / "processed" / "live_run.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to {out_path}")
    print(f"\n{'='*78}\n")

    return output


def _generate_mock_markets(query: str, n: int = 10) -> list[UnifiedMarket]:
    """Generate mock markets for demo purposes."""
    from src.utils.market_data import PolymarketClient
    templates = [
        f"Will {query} happen before end of 2026?",
        f"Will {query} lead to a market correction?",
        f"Will {query} be resolved by June 2026?",
        f"Will {query} impact GDP growth positively?",
        f"Is {query} more likely than not?",
        f"Will {query} exceed analyst expectations?",
        f"Will {query} be the biggest story of Q3 2026?",
        f"Will {query} happen in the next 90 days?",
        f"Will markets react positively to {query}?",
        f"Is the consensus on {query} correct?",
    ]
    markets = []
    rng = np.random.RandomState(hash(query) % 2**31)
    for i in range(min(n, len(templates))):
        price = float(rng.beta(2, 2))
        vol = float(rng.uniform(10_000, 500_000))
        m = PolymarketClient.mock_market(templates[i], yes_price=round(price, 2), volume_24h=vol)
        markets.append(UnifiedMarket(
            exchange="polymarket_mock",
            market_id=m["condition_id"],
            event_title=templates[i],
            market_question=templates[i],
            outcomes=m["outcomes"],
            outcome_prices=m["outcome_prices"],
            volume_24h=vol,
            liquidity=vol * 0.3,
            end_date=m["end_date"],
            active=True,
        ))
    return markets


def main():
    parser = argparse.ArgumentParser(
        description="MiroFish Live Trading Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--exchange", action="append", default=[], help="Exchange to connect (can repeat)")
    parser.add_argument("--query", type=str, default="US economy 2026", help="Market search query")
    parser.add_argument("--agents", type=int, default=80, help="Agents per simulation")
    parser.add_argument("--rounds", type=int, default=25, help="Simulation rounds")
    parser.add_argument("--sims", type=int, default=8, help="Parallel simulations")
    parser.add_argument("--max-markets", type=int, default=5, help="Max markets to simulate")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM-powered agents")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: dry run)")
    parser.add_argument("--bankroll", type=float, default=10_000, help="Trading bankroll")
    parser.add_argument("--dry-run", action="store_true", help="Explicit dry run (default)")
    parser.add_argument("--no-news", action="store_true", help="Skip news fetching")

    args = parser.parse_args()
    exchanges = args.exchange or ["polymarket"]

    if args.live and not args.dry_run:
        print("\n  WARNING: --live mode enabled. Real orders will be placed.")
        print("  Press Ctrl+C within 5 seconds to cancel...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n  Cancelled.")
            sys.exit(0)

    run_live_pipeline(
        exchanges=exchanges,
        query=args.query,
        n_agents=args.agents,
        n_rounds=args.rounds,
        n_sims=args.sims,
        max_markets=args.max_markets,
        use_llm=args.use_llm,
        live=args.live and not args.dry_run,
        bankroll=args.bankroll,
        fetch_news=not args.no_news,
    )


if __name__ == "__main__":
    main()
