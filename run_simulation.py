#!/usr/bin/env python3
"""
run_simulation.py — Main entry point for running a MiroFish-inspired
agent swarm simulation and generating trade signals.

Usage:
    # Quick demo (rule-based, no API keys needed)
    python run_simulation.py --demo

    # Prediction market simulation
    python run_simulation.py --mode prediction \
        --event "Will the Fed cut rates at the June 2026 FOMC meeting?" \
        --market-prob 0.45

    # Equity catalyst simulation
    python run_simulation.py --mode equity \
        --event "NVDA Q1 2026 earnings beat expectations" \
        --ticker NVDA

    # Full pipeline with LLM-powered agents (requires OPENAI_API_KEY)
    python run_simulation.py --mode prediction \
        --event "Will there be a US government shutdown in 2026?" \
        --market-prob 0.30 \
        --use-llm
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import *
from src.data.ingest import DataIngestionPipeline
from src.data.knowledge_graph import KnowledgeGraph
from src.simulation.engine import SimulationEngine
from src.signals.generator import SignalGenerator
from src.signals.backtest import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mirofish")


def run_prediction_market(
    event: str,
    market_prob: float,
    n_agents: int = 100,
    n_rounds: int = 30,
    n_sims: int = 10,
    use_llm: bool = False,
    fetch_news: bool = False,
) -> dict:
    """Run a prediction market simulation and generate a signal."""
    print(f"\n{'='*70}")
    print(f"  MIROFISH SWARM SIMULATION — PREDICTION MARKET")
    print(f"{'='*70}")
    print(f"  Event: {event}")
    print(f"  Market probability: {market_prob:.1%}")
    print(f"  Agents: {n_agents} | Rounds: {n_rounds} | Simulations: {n_sims}")
    print(f"  LLM agents: {'Yes' if use_llm else 'No (rule-based)'}")
    print(f"{'='*70}\n")

    # 1. Data ingestion
    context = ""
    if fetch_news:
        print("[1/4] Ingesting data...")
        pipeline = DataIngestionPipeline()
        docs = pipeline.ingest_for_event(event)
        print(f"  Fetched {len(docs)} documents")

        # Build knowledge graph
        llm = None
        if use_llm:
            from src.utils.llm import LLMClient
            llm = LLMClient()

        kg = KnowledgeGraph(llm_client=llm)
        for doc in docs:
            kg.add_document(doc["text"], source_id=doc.get("source", ""))
        stats = kg.build_from_documents(use_llm=use_llm)
        print(f"  Knowledge graph: {stats}")
        context = kg.get_context_for_event(event)
    else:
        print("[1/4] Skipping data ingestion (use --fetch-news to enable)")
        context = f"Event under analysis: {event}"

    # 2. Run simulation
    print(f"\n[2/4] Running {n_sims} simulations with {n_agents} agents each...")
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

    start = time.time()
    result = engine.run(
        event_description=event,
        context=context,
        market_type="prediction",
        use_llm_agents=use_llm,
        seed=42,
    )
    elapsed = time.time() - start

    print(f"\n  Swarm probability:  {result.swarm_probability:.1%}")
    print(f"  Swarm std:          {result.swarm_std:.4f}")
    print(f"  R-hat:              {result.r_hat:.4f}")
    print(f"  Converged:          {'Yes' if result.converged else 'No'}")
    print(f"  Bullish fraction:   {result.bullish_fraction:.1%}")
    print(f"  Mean conviction:    {result.mean_conviction:.2f}")
    print(f"  Elapsed:            {elapsed:.1f}s")

    # Archetype breakdown
    print(f"\n  Archetype Breakdown:")
    for arch, prob in sorted(result.archetype_probabilities.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 40)
        print(f"    {arch:25s} {prob:.1%} {bar}")

    # 3. Generate signal
    print(f"\n[3/4] Generating trade signal...")
    signal = SignalGenerator.generate_prediction_market_signal(
        swarm_result=result,
        market_probability=market_prob,
    )

    print(f"\n  Direction:          {signal.direction}")
    print(f"  Alpha:              {signal.alpha:+.1%}")
    print(f"  Expected edge:      {signal.expected_edge:+.1%}")
    print(f"  Confidence:         {signal.confidence}")
    print(f"  Kelly fraction:     {signal.kelly_fraction:.2%}")
    print(f"  Position size:      {signal.position_size_pct:.2%} of bankroll")
    print(f"  Reasoning:          {signal.reasoning}")

    # 4. Summary
    print(f"\n[4/4] Done!")
    print(f"{'='*70}\n")

    return {
        "swarm_result": result.to_dict(),
        "signal": signal.to_dict(),
    }


def run_equity_simulation(
    event: str,
    ticker: str,
    n_agents: int = 100,
    n_rounds: int = 40,
    n_sims: int = 10,
    use_llm: bool = False,
) -> dict:
    """Run an equity catalyst simulation and generate a signal."""
    print(f"\n{'='*70}")
    print(f"  MIROFISH SWARM SIMULATION — EQUITY CATALYST")
    print(f"{'='*70}")
    print(f"  Event: {event}")
    print(f"  Ticker: {ticker}")
    print(f"  Agents: {n_agents} | Rounds: {n_rounds} | Simulations: {n_sims}")
    print(f"{'='*70}\n")

    # Get current price and ATR
    from src.utils.market_data import EquityDataClient
    print(f"[1/4] Fetching market data for {ticker}...")
    price = EquityDataClient.get_current_price(ticker)
    if price:
        print(f"  Current price: ${price:.2f}")
        df = EquityDataClient.get_price_history(ticker, period="3mo")
        atr = float(EquityDataClient.compute_atr(df).iloc[-1]) if not df.empty else price * 0.02
        print(f"  14-day ATR: ${atr:.2f}")
    else:
        print(f"  Could not fetch live data, using mock data")
        price = 100.0
        atr = 2.0

    # Run simulation
    print(f"\n[2/4] Running {n_sims} simulations...")
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

    result = engine.run(
        event_description=event,
        context=f"Analyzing {ticker} around catalyst event: {event}",
        market_type="equity",
        use_llm_agents=use_llm,
        seed=42,
    )

    print(f"\n  Swarm probability:  {result.swarm_probability:.1%}")
    print(f"  Bullish fraction:   {result.bullish_fraction:.1%}")
    print(f"  Mean conviction:    {result.mean_conviction:.2f}")
    print(f"  Converged:          {'Yes' if result.converged else 'No'}")

    # Generate signal
    print(f"\n[3/4] Generating trade signal...")
    signal = SignalGenerator.generate_equity_signal(
        swarm_result=result,
        ticker=ticker,
        current_atr=atr,
    )

    print(f"\n  Direction:          {signal.direction}")
    print(f"  Bullish fraction:   {signal.bullish_fraction:.1%}")
    print(f"  Conviction:         {signal.mean_conviction:.1f}/10")
    print(f"  Sentiment momentum: {signal.sentiment_momentum:+.4f}")
    print(f"  Position size:      {signal.position_size_pct:.2%} of NAV")
    print(f"  Stop loss:          {signal.stop_loss_atr}x ATR (${atr * signal.stop_loss_atr:.2f})")
    print(f"  Time stop:          {signal.time_stop_days} days")
    print(f"  Confidence:         {signal.confidence}")

    print(f"\n  Archetype Breakdown:")
    for arch, prob in sorted(signal.archetype_breakdown.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 40)
        print(f"    {arch:25s} {prob:.1%} {bar}")

    print(f"\n[4/4] Done!")
    print(f"{'='*70}\n")

    return {
        "swarm_result": result.to_dict(),
        "signal": signal.to_dict(),
    }


def run_demo():
    """Run a quick demo with mock data — no API keys needed."""
    print("\n" + "="*70)
    print("  MIROFISH TRADING STRATEGY — DEMO MODE")
    print("  (No API keys needed — using rule-based agents and mock data)")
    print("="*70)

    # Demo 1: Prediction market
    print("\n\n--- DEMO 1: Prediction Market ---\n")
    pm_result = run_prediction_market(
        event="Will the US enter a recession by Q4 2026?",
        market_prob=0.35,
        n_agents=50,
        n_rounds=20,
        n_sims=5,
        use_llm=False,
    )

    # Demo 2: Equity catalyst
    print("\n\n--- DEMO 2: Equity Catalyst ---\n")
    eq_result = run_equity_simulation(
        event="Apple Q2 2026 earnings report — focus on AI services revenue",
        ticker="AAPL",
        n_agents=50,
        n_rounds=25,
        n_sims=5,
        use_llm=False,
    )

    # Demo 3: Quick backtest with synthetic events
    print("\n\n--- DEMO 3: Synthetic Backtest ---\n")
    backtest = BacktestEngine(initial_capital=10_000)

    # Generate synthetic prediction market events
    np.random.seed(42)
    synthetic_events = []
    for i in range(50):
        true_prob = np.random.beta(2, 2)  # True probability
        market_prob = true_prob + np.random.normal(0, 0.08)  # Market with noise
        market_prob = np.clip(market_prob, 0.05, 0.95)
        swarm_prob = true_prob + np.random.normal(0, 0.05)   # Swarm (better estimate)
        swarm_prob = np.clip(swarm_prob, 0.05, 0.95)
        outcome = 1.0 if np.random.random() < true_prob else 0.0

        alpha = swarm_prob - market_prob
        if abs(alpha) > 0.05:
            direction = "long_yes" if alpha > 0 else "long_no"
        else:
            direction = "no_trade"

        synthetic_events.append({
            "event": f"Synthetic Event #{i+1}",
            "swarm_probability": swarm_prob,
            "market_probability": market_prob,
            "actual_outcome": outcome,
            "signal": {
                "direction": direction,
                "position_size_pct": 0.03,
                "alpha": alpha,
                "confidence": "medium",
            },
            "timestamp": f"2026-{(i//4)+1:02d}-{(i%28)+1:02d}",
        })

    result = backtest.backtest_prediction_markets(synthetic_events)
    print(result.summary())

    # Monte Carlo confidence intervals
    mc = backtest.monte_carlo_equity_curve(n_simulations=500)
    if mc:
        print(f"  Monte Carlo Final Capital (500 sims):")
        print(f"    5th percentile:  ${mc['p5'][-1]:,.0f}")
        print(f"    Median:          ${mc['median'][-1]:,.0f}")
        print(f"    95th percentile: ${mc['p95'][-1]:,.0f}")

    # Save results
    output = {
        "prediction_market": pm_result,
        "equity": eq_result,
        "backtest": {
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
        },
    }

    output_path = Path(__file__).parent / "data" / "processed" / "demo_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MiroFish-Inspired Agent Swarm Trading Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--demo", action="store_true", help="Run demo mode (no API keys needed)")
    parser.add_argument("--mode", choices=["prediction", "equity"], default="prediction")
    parser.add_argument("--event", type=str, help="Event description to simulate")
    parser.add_argument("--market-prob", type=float, default=0.5, help="Market-implied probability (prediction markets)")
    parser.add_argument("--ticker", type=str, default="SPY", help="Stock ticker (equity mode)")
    parser.add_argument("--agents", type=int, default=100, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=30, help="Simulation rounds")
    parser.add_argument("--sims", type=int, default=10, help="Number of parallel simulations")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM-powered agents (requires API key)")
    parser.add_argument("--fetch-news", action="store_true", help="Fetch live news data")

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.event:
        if args.mode == "prediction":
            run_prediction_market(
                event=args.event,
                market_prob=args.market_prob,
                n_agents=args.agents,
                n_rounds=args.rounds,
                n_sims=args.sims,
                use_llm=args.use_llm,
                fetch_news=args.fetch_news,
            )
        else:
            run_equity_simulation(
                event=args.event,
                ticker=args.ticker,
                n_agents=args.agents,
                n_rounds=args.rounds,
                n_sims=args.sims,
                use_llm=args.use_llm,
            )
    else:
        print("Usage: python run_simulation.py --demo")
        print("   or: python run_simulation.py --event 'Your event' --market-prob 0.5")
        parser.print_help()


if __name__ == "__main__":
    main()
