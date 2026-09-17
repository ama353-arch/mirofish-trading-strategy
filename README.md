![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Strategies](https://img.shields.io/badge/strategies-5-orange)
![Demo](https://img.shields.io/badge/demo-zero%20API%20keys-brightgreen)

# MiroFish Trading Strategy

> **Agent-swarm Monte Carlo simulation for market prediction.**

MiroFish replaces the parametric stochastic processes that drive traditional quant strategies (GBM, jump-diffusion, regime-switching) with a population of heterogeneous AI agents. Each agent has its own persona, knowledge state, and behavioral policy; they interact on a simulated social platform; the swarm's collective probability estimate is the trade signal. Five strategies, full backtesting, prediction markets and equities — runs in demo mode with zero API keys in ~2 seconds.

---

## Features

- **16 AI agent archetypes** — retail bettors, institutional traders, political analysts, contrarians, domain experts, noise traders. Each has MBTI-derived cognitive profile (risk tolerance, contrarian bias, decision speed, information diet).
- **5 trading strategies** — Consensus Divergence, Sentiment Momentum, Contrarian Swarm, Event Catalyst Delta, Cross-Market Arbitrage. Each extracts a different signal from the same swarm engine.
- **Full backtesting framework** — Sharpe, Sortino, max drawdown, Calmar ratio, profit factor, win rate, full trade log, and Monte Carlo confidence intervals on the equity curve.
- **Dual market support** — prediction markets (Polymarket / Kalshi) and equities (yfinance, with options-implied probability comparisons).
- **Gelman-Rubin convergence diagnostics** — borrows from MCMC: only trade when R-hat < 1.1 across parallel simulations.
- **Kelly criterion sizing** — fractional Kelly with bankroll constraints (≤5% per position, ≤20% deployed).
- **Strategy comparison dashboard** — run all five against the same event stream; compare risk-adjusted returns side-by-side.
- **Zero-API demo mode** — heuristic agents, no keys, runs in ~2 seconds. Plug in GPT-4o-mini for richer reasoning.

---

## Quick Start

```bash
git clone https://github.com/aananthandey/mirofish-trading-strategy.git && cd mirofish-trading-strategy
pip install -r requirements.txt
python run_simulation.py --demo
python run_all_strategies.py
```

The first run is the demo (prediction market + equity catalyst + synthetic backtest, no API keys). The second compares all five strategies across the same event stream.

---

## Usage

```bash
# Prediction market
python run_simulation.py --mode prediction \
  --event "Will the Fed cut rates at the June 2026 FOMC meeting?" \
  --market-prob 0.45 --agents 100 --rounds 30 --sims 10

# Equity catalyst
python run_simulation.py --mode equity \
  --event "NVDA Q1 2026 earnings beat expectations" \
  --ticker NVDA --agents 100 --rounds 40 --sims 10

# LLM-powered agents (requires OPENAI_API_KEY)
python run_simulation.py --mode prediction --use-llm --fetch-news \
  --event "Supreme Court rules against the SEC in 2026" --market-prob 0.40
```

Sample output:

```
  Swarm probability:  52.3%
  R-hat:              1.0087       ← converged
  Bullish fraction:   61.0%
  Direction:          long_yes
  Alpha:              +7.3%
  Kelly fraction:     1.57%
```

> ⚠️ The `+7.3% Alpha` above is illustrative, not a performance claim. See the
> backtesting note below.

## Backtesting — read this before trusting any number

There are **two** backtest paths in this repo, and they are not equal:

- `src.signals.backtest.BacktestEngine` — the original single-pass engine. Its
  demo (`run_simulation.py --demo`) feeds it **synthetic events where the swarm
  is defined in code to be a more accurate estimator than the market**. Such a
  backtest cannot lose and proves nothing about real edge. Treat any number it
  produces — including older README figures — as **not evidence**.

- `src.signals.walkforward` — the **hardened, out-of-sample harness** (use
  this). Walk-forward train/test folds with provably no look-ahead, realistic
  spread-crossing costs, honest date-based annualization, **deflated Sharpe**
  (corrects for how many strategies/params were tried), and block-bootstrap
  Monte Carlo. Reported performance is **test-fold-only**.

Run the honest end-to-end backtest (real swarm, out-of-sample):

```bash
python3 run_walkforward.py                      # bundled sample markets
python3 run_walkforward.py path/to/markets.json # your resolved markets
```

```python
from src.signals.walkforward import WalkForwardBacktester, deflated_sharpe_ratio
wf = WalkForwardBacktester(train_size=8, test_size=5)
result = wf.run(events, signal_fn=brain, param_grid=[{"edge_min": 0.05}])
print(result.stats())   # OOS win rate, total return — test folds only
```

**The gate:** nothing touches real money until it is positive, statistically
significant, and out-of-sample (deflated Sharpe > ~0.95). See
[`docs/SCOPE.md`](docs/SCOPE.md).

---

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  DATA INGEST │───▶ │  KNOWLEDGE GRAPH │───▶ │  PERSONA GENERATOR│
│  News / API  │     │   (GraphRAG)     │     │  (LLM or rules)   │
└──────────────┘     └──────────────────┘     └─────────┬─────────┘
                                                        │ N agents × M sims
                                                        ▼
┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  TRADE EXEC  │ ◀───│ SIGNAL GENERATOR │ ◀───│  SIMULATION ENGINE│
│  (Broker API)│     │ (Kelly, sizing)  │     │  (Agent Swarm)    │
└──────────────┘     └──────────────────┘     └───────────────────┘
```

**The math.** Classical Monte Carlo samples a parametric distribution P(X). MiroFish replaces P(X) with a generative process:

```
P_swarm(X) = (1/N) · Σ δ(X − outcome(agent_i))
```

Each agent has a persona θᵢ (risk tolerance, biases, expertise), a knowledge state Kᵢ(t) drawn from a shared knowledge graph plus private information, and a behavioral policy πᵢ(a | s, θᵢ, Kᵢ). This is formally equivalent to **adaptive importance sampling** — the swarm's proposal distribution Q approximates the true distribution P better than market-implied prices M when D_KL(P ‖ Q) < D_KL(P ‖ M).

Run M parallel simulations, compute Gelman-Rubin R̂ = √(Var_between / Var_within), and trade only when R̂ < 1.1 and effective sample size N_eff > 100.

```
mirofish-trading-strategy/
├── run_simulation.py           # Single-event entry point
├── run_all_strategies.py       # Compare all 5 strategies
├── run_live.py                 # Live signal generation
├── config/settings.py          # All parameters centralized
├── src/
│   ├── agents/                 # 16 archetypes, MBTI profiles, SwarmAgent
│   ├── simulation/             # Social platform + multi-sim engine + R-hat
│   ├── data/                   # News ingest + GraphRAG knowledge graph
│   ├── signals/                # Signal generator + backtest + Monte Carlo CI
│   └── utils/                  # LLM client, market data, RAG
├── strategies/
│   ├── consensus_divergence.py # Strategy 1 — alpha = p_swarm − p_market
│   ├── sentiment_momentum.py   # Strategy 2 — d(S_bull)/d(round)
│   ├── contrarian_swarm.py     # Strategy 3 — overweight contrarian personas
│   ├── event_catalyst.py       # Strategy 4 — pre/post-event delta
│   ├── cross_market_arb.py     # Strategy 5 — swarm tail vs. options-implied
│   └── strategy_comparison.py  # Multi-strategy comparison framework
└── tests/
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Core simulation | Python 3.10+, NumPy, pandas |
| Agents | 16 MBTI-derived archetypes, rule-based or LLM |
| Knowledge graph | NetworkX (GraphRAG-inspired), scikit-learn TF-IDF |
| LLM | OpenAI-compatible (`gpt-4o-mini` default) |
| Market data | yfinance, Polymarket CLOB, Kalshi |
| News | NewsAPI, newspaper3k, PRAW |
| Backtesting | pandas, scipy, custom Monte Carlo CI |
| Visualization | matplotlib, seaborn |
| Config | pydantic, python-dotenv |

---

## Two Modes

**Rule-based** (no API keys) — heuristic reasoning from persona traits, social influence as weighted belief averaging with contrarian inversion. 100 agents × 30 rounds × 10 sims runs in seconds. Right for backtesting and parameter sweeps.

**LLM-powered** (`OPENAI_API_KEY`) — each agent reasons through structured prompts. ~$2-5 per event with GPT-4o-mini. Right for live signals and research.

---

## Disclaimer

Research and educational use only. Nothing here is financial advice. Trading carries substantial risk of loss. Simulated past performance does not guarantee future results.

---

## Built by

**[Aananth Andey](https://github.com/aananthandey)** — Cornell '28, founder of [Journi AI](https://journi.ai). Inspired by [MiroFish](https://github.com/666ghj/MiroFish) and [OASIS](https://github.com/camel-ai/oasis).

---

## License

MIT — see [LICENSE](LICENSE).
