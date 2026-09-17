# MiroFish — Scope (source of truth)

> This file is the contract we build against. If a decision contradicts it,
> update this file first. Last updated during the prediction-market hardening
> loop (2026-06-11).

## The one-line thesis

**MiroFish is a brain. Prediction markets are the arena. We become provably,
best-in-class good at predicting binary-event markets before we touch anything
else.**

Equity markets are explicitly **out of near-term scope** — they are efficient,
HFT-saturated, and capital-hungry; a solo operator has little edge there. The
swarm is architecturally *native* to prediction markets (a crowd simulating a
narrative-driven binary outcome), so that is where we specialize.

## The brain (tools)

One normalized output per opportunity — `probability → edge vs market → size →
rationale + stats` — produced by an ensemble of:

1. **MiroFish swarm** — heterogeneous agent Monte Carlo (the core moat).
2. **ML Monte Carlo** — statistical/ML probability models.
3. **Fundamental / news** — knowledge-graph context feeding the swarm.

A trade only fires when sources **agree** *and* there is **edge vs the market
price**. (Kalman denoising applies to any live price/line series.)

## The pathways (arenas)

| Pathway | Modes in scope |
|---|---|
| **Kalshi** | Pure prediction (swarm+ML+MC edge), structural arbitrage, NBA in-game (later) |
| **Polymarket** | Same as Kalshi + cross-venue arb + **wallet-follow** (watch given wallets, score their fills vs our edge) |

## The execution boundary (firm)

The brain decides, displays, and alerts. **The live order / bet / fund-transfer
is human-triggered** (one tap from the dashboard). Everything up to that tap is
autonomous. Rationale: prop-firm compliance, anonymous-wallet risk, and the
fact that an unattended bug should never be able to drain an account. Arbitrage
is the most defensible autonomous candidate but still surfaces for a tap.

## The dashboard (the brain you follow)

Every candidate trade card shows: each tool's vote, the edge math, the Kelly
size, *and* the strategy's **out-of-sample** track record (win rate, Sharpe,
max DD, sample size). You never see a number that came from in-sample fitting.

## The validation gate (non-negotiable)

Nothing touches real money until it is **positive, statistically significant,
and out-of-sample**. Concretely: walk-forward test folds, realistic
spread-crossing costs, and a **deflated Sharpe > ~0.95** (corrects for how many
strategies/params we tried). Until a pathway clears this, the dashboard labels
it "paper / unproven."

## Status (this loop)

**DONE — hardened backtest harness** (`src/signals/walkforward.py`, 21 tests):
- Walk-forward train/test folds, provably no look-ahead.
- Realistic spread-crossing cost model.
- Honest date-based annualization.
- Deflated Sharpe (multiple-testing correction).
- Block-bootstrap Monte Carlo (preserves autocorrelation).
- OOS-only `run()` loop + result stats.

**DONE — unified event schema + adapters** (`src/data/historical.py`):
Kalshi & Polymarket resolved markets normalize into one shape the harness
consumes interchangeably.

**DONE — end-to-end integration** (`run_walkforward.py`): the real swarm scored
out-of-sample. Current sample result is intentionally break-even-minus-costs —
proof the harness is honest, not proof of edge.

## Next (future loops)

1. **Live data fetchers** — pull real resolved Kalshi + Polymarket history
   (needs read-only API access / keys) into the unified schema.
2. **Inform the swarm** — wire news/knowledge-graph context + optional LLM
   agents so the brain actually has information to find edge.
3. **Wallet-follow backtest** — pull given wallets' on-chain history, prove
   following them is +EV before any mirroring.
4. **Arbitrage detector** — structural + cross-venue.
5. **Dashboard** — the rationale + OOS-stats surface.

## Out of scope (do not silently add)

Equity strategies, live order execution, real-money credential handling,
auto-mirroring wallets without a proven backtest, live in-game tick feeds
(until the resolved-market backtest passes the gate).
