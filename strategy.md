# MiroFish-Inspired Trading Strategy: Agent Swarm Simulation for Market Prediction

**Author:** Aananth Andey
**Affiliation:** Cornell Dyson School — Finance & Entrepreneurship
**Date:** March 2026
**Version:** 1.0

---

## 1. Executive Summary

This document describes a trading strategy that replaces traditional Monte Carlo simulation with **AI agent swarm simulation** — inspired by [MiroFish](https://github.com/666ghj/MiroFish). Instead of sampling from parameterized probability distributions, we construct a population of heterogeneous AI agents (each with distinct personas, beliefs, information sets, and behavioral heuristics) and let them interact on a simulated social platform. The **emergent collective behavior** of the swarm produces probability distributions over market outcomes that are richer, more narrative-aware, and more sensitive to regime changes than anything a standard stochastic model can deliver.

The strategy targets two asset classes:

1. **Prediction markets** (Polymarket, Kalshi) — binary and multi-outcome event contracts.
2. **Equity markets** — directional trades around catalysts (earnings, FOMC, macro surprises).

---

## 2. Theoretical Framework

### 2.1 From Monte Carlo to Agent Monte Carlo

Classical Monte Carlo estimation of an expected value $E[f(X)]$ draws $N$ i.i.d. samples $x_1, \dots, x_N$ from some distribution $P(X)$ and computes:

$$\hat{\mu}_N = \frac{1}{N} \sum_{i=1}^{N} f(x_i)$$

The convergence rate is $O(1/\sqrt{N})$ regardless of dimension — a strength, but the quality of the estimate is bounded by the quality of $P(X)$. In markets, $P(X)$ is unknown, nonstationary, and shaped by reflexive feedback loops. Parameterizing it (e.g., GBM, jump-diffusion, copulas) embeds structural assumptions that break during the exact regimes where accurate estimation matters most.

**Agent Monte Carlo** replaces the parametric distribution with a generative process:

$$P(X) \approx P_{\text{swarm}}(X) = \frac{1}{N} \sum_{i=1}^{N} \delta\bigl(X - \text{outcome}(\text{agent}_i)\bigr)$$

where each $\text{agent}_i$ is an LLM-driven entity with:

- A **persona** $\theta_i$ (demographics, risk tolerance, expertise, cognitive biases)
- A **knowledge state** $K_i(t)$ drawn from a shared knowledge graph plus private information
- A **behavioral policy** $\pi_i(a \mid s, \theta_i, K_i)$ governing actions (post, trade, share, ignore)

The agents interact on a simulated social platform, and the **emergent distribution of opinions, positions, and predicted outcomes** constitutes $P_{\text{swarm}}(X)$.

### 2.2 Why This Works

The key insight is that markets are **socially constructed information aggregation mechanisms**. Price is not determined by a mathematical process but by the collective actions of heterogeneous participants processing information through diverse cognitive lenses. Agent swarm simulation models this process directly:

1. **Narrative sensitivity.** Agents read and react to narratives (news, tweets, rumors), not just numerical features. This captures how a single headline can move markets more than a basis point of data.

2. **Heterogeneous priors.** A population of agents with different expertise, biases, and risk preferences naturally produces fat-tailed, skewed, and multimodal distributions — without needing to assume them parametrically.

3. **Reflexivity.** Agents observe each other's behavior and update accordingly. This captures herding, panic, FOMO, and contrarian dynamics that are endogenous to markets.

4. **Regime awareness.** Because agent behavior is conditioned on semantic content (not just historical statistics), the swarm adapts to novel regimes in ways that backward-looking models cannot.

### 2.3 Formal Connection to Importance Sampling

Agent Monte Carlo can be understood as a form of **adaptive importance sampling**. Let $Q$ be the agent-generated distribution and $P$ be the true (unknown) market distribution. The swarm estimate:

$$\hat{\mu}_{\text{swarm}} = \frac{1}{N} \sum_{i=1}^{N} f(x_i), \quad x_i \sim Q$$

is an unbiased estimator of $E_Q[f(X)]$. The trading edge comes from the hypothesis that $Q$ is a better approximation of $P$ than the market-implied distribution $M$:

$$D_{KL}(P \| Q) < D_{KL}(P \| M)$$

When this holds, the swarm's probability estimates are more accurate than market prices, and we can extract edge by trading the difference.

### 2.4 Variance Reduction Through Persona Diversification

In classical Monte Carlo, variance reduction techniques (antithetic variates, control variates, stratified sampling) improve estimator efficiency. In Agent Monte Carlo, the analogous technique is **persona diversification**:

- **Stratified personas:** Ensure the agent population covers all relevant market participant archetypes (retail, institutional, quant, fundamental, contrarian, momentum, etc.).
- **Antithetic personas:** For every bullish agent, include a structurally bearish counterpart with symmetric but opposite priors.
- **Control personas:** Include "baseline" agents with well-calibrated priors (e.g., base-rate forecasters) to anchor the swarm.

This reduces the variance of the swarm estimate without increasing agent count.

---

## 3. System Architecture

### 3.1 Pipeline Overview

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  DATA INGEST │────▸│  KNOWLEDGE GRAPH │────▸│  PERSONA GENERATOR│
│  (News, API, │     │  (GraphRAG)      │     │  (LLM-driven)     │
│   Social)    │     │                  │     │                   │
└──────────────┘     └──────────────────┘     └─────────┬─────────┘
                                                        │
                                                        ▼
┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  TRADE EXEC  │◂────│ SIGNAL GENERATOR │◂────│  SIMULATION ENGINE│
│  (Broker API)│     │ (Swarm→Signals)  │     │  (Agent Swarm)    │
└──────────────┘     └──────────────────┘     └───────────────────┘
```

### 3.2 Component Details

**Data Ingestion Layer**
- News: NewsAPI, GDELT, RSS feeds
- Social media: Reddit API (PRAW), Twitter/X API (academic access)
- Market data: yfinance (equities), Polymarket API (prediction markets), Kalshi API
- Economic data: FRED API (macro indicators)
- Polling data: FiveThirtyEight, RealClearPolitics scrapers

**Knowledge Graph (GraphRAG)**
- Entities: People, organizations, events, policies, financial instruments, macro indicators
- Relationships: causal links, temporal sequences, correlations, influences
- Implementation: NetworkX (local) or Neo4j (production), with LLM-based entity extraction
- Update cadence: Continuous streaming for breaking events, batch for background context

**Persona Generator**
- Uses knowledge graph entities to seed realistic agent personas
- Each persona includes: demographics, expertise domain, risk appetite, cognitive biases, information sources, typical trading behavior
- LLM generates rich backstories and behavioral tendencies
- Population calibrated to approximate real market participant mix

**Simulation Engine**
- Agents interact on a simulated social/trading platform
- Each simulation round: agents observe environment → form opinions → take actions → update beliefs
- Actions: post opinion, share information, express trading intent, react to others
- Configurable rounds (default: 20-50 per event)
- Multiple parallel simulations for convergence estimation

**Signal Generator**
- Aggregates agent opinions into probability distributions
- Compares swarm probabilities to market-implied probabilities
- Generates signals when divergence exceeds threshold
- Kelly criterion for position sizing

---

## 4. Strategy A: Prediction Markets

### 4.1 Target Markets

- **Polymarket:** Binary and multi-outcome contracts (politics, macro events, crypto, culture)
- **Kalshi:** Regulated U.S. event contracts (economic data, elections, weather, Fed decisions)

### 4.2 Simulation Design for Event Prediction

**Step 1: Event Selection**
Select events where:
- Contract liquidity > $50K (sufficient depth for entry/exit)
- Resolution date > 3 days out (time for information to propagate)
- Market-implied probability between 15-85% (avoid extreme odds where edge is small in absolute terms)
- Rich information environment (news, polling, social discussion available)

**Step 2: Knowledge Graph Construction**
For each target event:
1. Ingest all relevant news articles (last 30 days)
2. Ingest relevant social media discussions (Reddit threads, Twitter discourse)
3. Ingest any available polling or survey data
4. Ingest historical analogues (similar past events and their outcomes)
5. Extract entities and relationships via LLM
6. Build event-specific knowledge graph

**Step 3: Agent Population Design**

| Agent Archetype | Count | Description |
|----------------|-------|-------------|
| Retail Bettor | 30% | Recency-biased, narrative-driven, moderate risk tolerance |
| Political Analyst | 15% | Deep domain knowledge, base-rate aware, methodical |
| Contrarian Trader | 10% | Systematically bets against consensus, looks for overreaction |
| Institutional Trader | 15% | Risk-averse, focuses on expected value, large positions |
| Social Media Influencer | 10% | Amplifies narratives, generates herding behavior |
| Data Scientist | 10% | Quantitative, model-driven, skeptical of narratives |
| Insider/Expert | 5% | Domain-specific deep knowledge, low noise |
| Noise Trader | 5% | Random-ish behavior, provides liquidity and noise |

Each archetype is further diversified with random personality traits (MBTI, age, geography, information diet) to avoid artificial consensus.

**Step 4: Simulation Execution**
1. Initialize agents with personas and knowledge states
2. Inject event context into the simulated environment
3. Run 30 rounds of interaction:
   - Agents read shared timeline
   - Agents form/update opinions
   - Agents post predictions and reasoning
   - Agents react to each other's posts (agree, disagree, amplify)
4. After simulation, extract each agent's final predicted probability
5. Repeat 10x with different random seeds for robustness

**Step 5: Signal Extraction**

Let $p_{\text{swarm}}$ be the mean predicted probability across all agents and simulations, and $p_{\text{market}}$ be the current market-implied probability.

**Signal strength:**
$$\alpha = p_{\text{swarm}} - p_{\text{market}}$$

**Confidence:**
$$\sigma_{\text{swarm}} = \text{std}(\text{agent predictions across simulations})$$

**Trade signal:**
- **Long (buy YES):** $\alpha > 0.05$ and $\sigma_{\text{swarm}} < 0.15$
- **Short (buy NO):** $\alpha < -0.05$ and $\sigma_{\text{swarm}} < 0.15$
- **No trade:** Otherwise

### 4.3 Position Sizing (Kelly Criterion)

$$f^* = \frac{p_{\text{swarm}} \cdot b - (1 - p_{\text{swarm}})}{b}$$

where $b$ is the decimal odds implied by $p_{\text{market}}$.

In practice, use **fractional Kelly** ($f = 0.25 \cdot f^*$) to account for estimation error.

**Bankroll rules:**
- Maximum 5% of bankroll on any single contract
- Maximum 20% of bankroll deployed at any time
- Stop loss: Exit if position loses 50% of initial stake
- Take profit: Exit if position gains 100% or market moves to within 2% of swarm estimate

### 4.4 Edge Cases and Risk Management

**Model disagreement:** If the standard deviation across simulation runs exceeds 0.20, the swarm has not converged. Do not trade.

**Information shock:** If a major new development occurs after simulation, re-run with updated knowledge graph before acting on stale signals.

**Liquidity risk:** Only enter positions where 24h volume > 5x position size.

**Correlation risk:** If multiple active positions share a common factor (e.g., all political events in the same election), treat them as a single bet for bankroll allocation purposes.

---

## 5. Strategy B: Equity Markets

### 5.1 Target Events

- Earnings announcements (mega-cap tech, high-volatility names)
- FOMC meetings and Fed chair press conferences
- CPI/PPI/NFP releases
- Geopolitical events (tariffs, sanctions, conflict escalation)
- Sector-specific catalysts (FDA approvals, antitrust rulings, tech regulation)

### 5.2 Simulation Design for Equity Sentiment

**Step 1: Event Selection**
- Focus on events with high implied volatility (options market pricing significant moves)
- Prioritize events where narrative/sentiment matters more than pure quantitative surprise (e.g., Fed tone vs. actual rate decision)

**Step 2: Knowledge Graph Construction**
For each target event:
1. Ingest company/sector-specific news (last 60 days)
2. Ingest analyst reports and earnings estimates (consensus data)
3. Ingest social media discussion (r/wallstreetbets, FinTwit, StockTwits)
4. Ingest macroeconomic context (recent data releases, policy environment)
5. Ingest options market data (implied volatility, skew, put/call ratio)
6. Build event-specific knowledge graph with entity relationships

**Step 3: Agent Population Design**

| Agent Archetype | Count | Description |
|----------------|-------|-------------|
| Quant Trader | 15% | Stat-arb, mean-reversion, systematic factor exposure |
| Fundamental Analyst | 20% | DCF-driven, focus on earnings quality and guidance |
| Retail Investor | 20% | Momentum-chasing, FOMO-driven, social media influenced |
| Market Maker | 10% | Delta-neutral, profits from spread, provides liquidity |
| Macro Strategist | 10% | Top-down, rates/FX/commodities, cross-asset |
| Activist/Event-Driven | 5% | Catalysts, special situations, corporate actions |
| Contrarian | 10% | Fades consensus, buys fear, sells greed |
| Algorithmic Trader | 10% | Momentum signals, volume triggers, technical patterns |

**Step 4: Simulation Execution**
1. Initialize agents with pre-event context
2. Inject the event stimulus (earnings beat/miss, Fed statement, data release)
3. Run 40 rounds of post-event simulation:
   - Agents process the event through their cognitive lens
   - Agents form directional views and conviction levels
   - Agents discuss, debate, and influence each other
   - Agents express trading intentions
4. Track sentiment trajectory (how opinion evolves over rounds)
5. Run 10 parallel simulations with varied initial conditions

**Step 5: Signal Extraction**

**Directional signal:** Fraction of agents that are net bullish in final round.

$$S_{\text{bull}} = \frac{|\{i : \text{agent}_i \text{ is bullish}\}|}{N}$$

**Conviction signal:** Average conviction level (1-10 scale) across agents.

**Sentiment momentum:** Rate of change of $S_{\text{bull}}$ over simulation rounds — is opinion crystallizing or fragmenting?

**Trade signal generation:**
- **Long:** $S_{\text{bull}} > 0.65$ and conviction > 7 and sentiment momentum positive
- **Short:** $S_{\text{bull}} < 0.35$ and conviction > 7 and sentiment momentum negative
- **No trade:** Otherwise

### 5.3 Position Sizing and Risk Management

**Position sizing:** Based on conviction and signal strength.

$$\text{Position} = \text{BaseSize} \times \frac{\text{Conviction}}{10} \times \min\left(1, \frac{|S_{\text{bull}} - 0.5|}{0.2}\right)$$

where BaseSize is 2% of portfolio NAV.

**Risk rules:**
- Maximum single position: 5% of portfolio
- Maximum sector exposure: 15% of portfolio
- Stop loss: 2x ATR below entry (long) or above entry (short)
- Time stop: Close position 5 trading days after entry if thesis hasn't played out
- Maximum drawdown trigger: Halt all new trades if portfolio drawdown exceeds 10%

### 5.4 Portfolio Construction

Maintain a **long/short equity book** with:
- Gross exposure: 50-100% of NAV
- Net exposure: -30% to +30% (approximately market-neutral)
- Maximum positions: 10 at any time
- Rebalance: After each catalyst event simulation

---

## 6. Mathematical Formulation

### 6.1 Swarm Probability Estimator

Let $\Omega = \{o_1, \dots, o_K\}$ be the set of possible outcomes for an event. Each agent $i$ produces a probability vector:

$$\mathbf{p}_i = (p_i^{(1)}, \dots, p_i^{(K)}), \quad \sum_{k=1}^{K} p_i^{(k)} = 1$$

The swarm probability estimate is the **weighted average** across agents:

$$\hat{\mathbf{p}}_{\text{swarm}} = \sum_{i=1}^{N} w_i \cdot \mathbf{p}_i$$

where weights $w_i$ reflect agent calibration quality (tracked over time):

$$w_i = \frac{\exp(-\lambda \cdot \text{BrierScore}_i)}{\sum_j \exp(-\lambda \cdot \text{BrierScore}_j)}$$

Newly created agents start with uniform weights. As the system accumulates track records, well-calibrated agent archetypes receive higher weight.

### 6.2 Divergence-Based Edge Estimation

The **expected edge** of a trade is:

$$E[\text{edge}] = \sum_{k=1}^{K} \hat{p}_{\text{swarm}}^{(k)} \cdot \left(\frac{1}{p_{\text{market}}^{(k)}} - 1\right) \cdot \hat{p}_{\text{swarm}}^{(k)} - \sum_{k \neq k^*} \hat{p}_{\text{swarm}}^{(k)}$$

Simplified for a binary market (YES at price $m$, swarm estimate $p$):

$$E[\text{edge}] = p \cdot \frac{1}{m} - 1 = \frac{p - m}{m}$$

We trade when $|E[\text{edge}]| > \text{threshold}$ and the swarm has converged (low cross-simulation variance).

### 6.3 Convergence Diagnostics

Borrow from MCMC convergence diagnostics:

**Cross-simulation variance (Gelman-Rubin-like):**

$$\hat{R} = \sqrt{\frac{\text{Var}_{\text{between simulations}}}{\text{Var}_{\text{within simulations}}}}$$

Trade only when $\hat{R} < 1.1$ (simulations have converged to similar distributions).

**Effective sample size:**

$$N_{\text{eff}} = \frac{N \cdot M}{\hat{R}^2}$$

where $N$ is agent count and $M$ is simulation count. Require $N_{\text{eff}} > 100$ for trade entry.

---

## 7. Implementation Roadmap

### Phase 1: Proof of Concept (Weeks 1-3)
- Simplified agent simulation with 5 archetypes, 50 agents
- Single knowledge source (news headlines only)
- Binary prediction market focus (Polymarket)
- Manual backtesting against historical prediction market data

### Phase 2: Full Pipeline (Weeks 4-8)
- Full knowledge graph with multiple data sources
- LLM-driven persona generation (10+ archetypes, 200+ agents)
- Automated simulation and signal generation
- Backtesting framework with historical equity catalysts
- Paper trading on live prediction markets

### Phase 3: Production (Weeks 9-12)
- Real-time data ingestion pipeline
- Automated trade execution (Polymarket API)
- Performance tracking and agent weight calibration
- Risk management automation
- Dashboard for monitoring

---

## 8. Expected Performance and Limitations

### 8.1 Expected Edge
- Prediction markets: 3-8% edge on well-researched events (before fees)
- Equity catalysts: Sharpe ratio target of 1.0-1.5 on the catalyst trading book

### 8.2 Key Risks
1. **LLM hallucination:** Agents may form views based on fabricated information. Mitigated by grounding all agent knowledge in the verified knowledge graph.
2. **Overfitting to narratives:** If the dominant narrative is wrong, the swarm may converge on the wrong answer. Mitigated by persona diversification and contrarian agents.
3. **Cost:** Running 200+ LLM-powered agents across 10 simulations per event is expensive. Estimated $2-5 per event with GPT-4o-mini level models.
4. **Latency:** Full simulation takes 5-15 minutes per event. Not suitable for latency-sensitive trading.
5. **Market impact:** Prediction markets are thin. Large positions move prices.
6. **Regulatory risk:** Prediction market legality varies by jurisdiction.

### 8.3 When This Strategy Fails
- Pure quantitative surprises (CPI comes in 50bps above consensus — no narrative will save you)
- Flash crashes driven by technical/structural factors (margin calls, ETF rebalancing)
- Events with zero prior information (true black swans)
- When market pricing already reflects diverse, well-informed opinion (efficient markets)

---

## 9. Comparison to Existing Approaches

| Approach | Strengths | Weaknesses | MiroFish Advantage |
|----------|-----------|------------|-------------------|
| Traditional Monte Carlo | Mathematically rigorous, fast | Requires parameterization, misses narratives | Agents generate distributions from semantics |
| Sentiment Analysis (NLP) | Scalable, real-time | Shallow — counts positive/negative words | Agents reason about *why* sentiment exists |
| Prediction Market Prices | Aggregates real beliefs | Only reflects current state, not dynamics | Simulates how beliefs will evolve |
| Expert Forecasting | High quality per forecast | Expensive, slow, small sample | Scales to 1000s of simulated experts |
| LLM Direct Prompting | Cheap, fast | Single viewpoint, no interaction effects | Multi-agent interaction captures emergence |

---

## 10. References

1. MiroFish GitHub Repository: https://github.com/666ghj/MiroFish
2. OASIS: Open Agent Social Interaction Simulations (CAMEL-AI)
3. Surowiecki, J. (2005). *The Wisdom of Crowds.* Anchor Books.
4. Tetlock, P. (2015). *Superforecasting.* Crown Publishers.
5. Kelly, J.L. (1956). A New Interpretation of Information Rate. *Bell System Technical Journal.*
6. Gelman, A. & Rubin, D. (1992). Inference from Iterative Simulation Using Multiple Sequences. *Statistical Science.*
7. Soros, G. (1987). *The Alchemy of Finance.* John Wiley & Sons.
