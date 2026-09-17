---
spec: quant-for-prediction-markets pipeline
status: built (E1-E5 done, TDD, 97 tests green) — pending live data + commit
branch: feat/quant-prediction-markets
created: 2026-06-12
depends_on: src/signals/walkforward.py, src/signals/pricing.py, src/data/historical.py
---

> BUILD LOG (2026-06-12): All five units landed test-first.
> - E1 calibration → `src/signals/calibration.py` (7 tests)
> - E2 arbitrage → `src/signals/arbitrage.py` (7 tests)
> - E3 portfolio → `src/signals/portfolio.py` (5 tests)
> - E4 ensemble → `src/signals/ensemble.py` (7 tests)
> - E5 engine report (calibration-driven instrument) → `src/signals/engine_report.py` (2 tests)
> - Wired into `run_walkforward.py`; end-to-end report runs the real swarm.
> Independent Codex review skipped this round (quota exceeded); self-review clean.
> Remaining: live Kalshi/Polymarket data fetchers (needs keys), then real validation.

# Spec — Quant for Prediction Markets

## Context

The walk-forward backtest harness is hardened (`src/signals/walkforward.py`, 15
tests) and a quantitative Fair-Value Pricer exists (`src/signals/pricing.py`, 8
tests). What's missing is the layer that turns "a brain that emits probabilities"
into "an autonomous signal/decision engine that covers everything Kalshi and
Polymarket offer." This spec builds that layer. Execution boundary is firm: the
engine **decides and surfaces; it never auto-fires live orders or moves money.**

## Build units (dependency-ordered)

```
E1 Calibration ──┬─> E4 Ensemble/Registry ──> (dashboard, later)
                 │
E2 Arbitrage ────┤   (independent; market-neutral, no edge model)
                 │
E3 Portfolio ────┘   (allocates whatever signals exist)

E5 MiroFish refinement — parallel, driven by E1 calibration output
```

Sequencing rationale: **E1 first** because every other unit needs a way to know
whether a probability source is trustworthy — calibration is the measuring stick.
E2 is independent and can land anytime (no model). E3 needs signals to allocate,
so it follows. E4 (ensemble) needs E1 to weight sources. E5 uses E1's numbers to
tell whether swarm changes help.

---

## E1 — Calibration harness  `src/signals/calibration.py`

**Why:** a probability source is worthless if it isn't calibrated — when it says
60%, the thing should happen ~60% of the time. We must measure this before
trusting any source (swarm, pricer, ML) with money.

**API:**
- `brier_score(probs, outcomes) -> float` — mean squared error of probabilities.
- `log_loss(probs, outcomes, eps=1e-15) -> float` — penalizes confident wrong.
- `reliability_curve(probs, outcomes, n_bins=10) -> list[Bin]` — predicted vs
  observed frequency per bin (the reliability diagram data).
- `calibration_report(probs, outcomes) -> dict` — Brier, log loss, base rate,
  count, plus the reliability curve.

**Acceptance criteria:**
1. A perfectly calibrated source (prob == empirical frequency) scores Brier near
   the irreducible variance and lower than any miscalibrated source.
2. `brier_score` of a source that always predicts the base rate equals the base
   rate variance `p(1-p)`.
3. `log_loss` returns a finite number even when a prob is exactly 0 or 1 (eps
   clamp); a confident-correct prediction beats a hedged one.
4. `reliability_curve` bins sum to the input count; empty bins are omitted, not
   NaN.
5. Mismatched-length inputs raise `ValueError`.

## E2 — Arbitrage detector  `src/signals/arbitrage.py`

**Why:** market-neutral, no edge model needed, the most defensible autonomous
strategy. Two kinds: structural (a set of mutually-exclusive outcomes whose ask
prices sum to < 1 => buy them all for guaranteed profit) and cross-venue (same
event priced differently on Kalshi vs Polymarket).

**API:**
- `find_structural_arb(legs, fee=0.0) -> Arb | None` — legs = mutually exclusive
  outcomes with ask prices; returns the arb if sum(asks)+fees < 1.
- `find_cross_venue_arb(market_a, market_b, fee=0.0) -> Arb | None` — same event,
  two venues; returns the arb if you can buy YES on one and NO on the other for
  < 1 total.

**Acceptance criteria:**
1. Three mutually-exclusive legs at asks 0.30/0.30/0.30 (sum 0.90) yield an arb
   with guaranteed_profit ~0.10 before fees; with fee 0.05 total it shrinks to
   ~0.05; at sum >= 1 returns None.
2. Cross-venue: YES@0.40 on A and NO@0.55 on B (0.95 total) is an arb; 0.40 +
   0.62 (1.02) is not.
3. Fees are subtracted from the guaranteed profit, never ignored.
4. A single leg, empty legs, or legs summing to exactly 1.0 return None.

## E3 — Portfolio / capital-allocation layer  `src/signals/portfolio.py`

**Why:** the backtest scores trades sequentially, but real trading fires many
positions at once "in conjunction." This layer enforces the bankroll rules across
*concurrent* positions: <= 20% total deployed, per-position fractional-Kelly cap,
and it down-weights correlated bets so we don't bet the same risk ten times.

**API:**
- `kelly_fraction(edge, odds, fraction=0.25) -> float` — fractional Kelly size.
- `allocate(signals, max_deployed=0.20, max_position=0.05, kelly_fraction=0.25,
  correlation_groups=None) -> list[Allocation]` — sizes each signal, clamps per
  position, scales the whole book down if total > max_deployed, splits a Kelly
  budget across members of a correlation group.

**Acceptance criteria:**
1. `kelly_fraction` is 0 when edge <= 0; positive and increasing in edge; never
   exceeds the full-Kelly value times `fraction`.
2. Each allocation <= `max_position` of bankroll.
3. Sum of allocations <= `max_deployed`; if raw sizes exceed it, all are scaled
   down proportionally (not truncated arbitrarily).
4. Two signals in the same correlation group share one group's Kelly budget (sum
   of the pair <= what one uncorrelated signal of that edge would get).

## E4 — Ensemble / source registry  `src/signals/ensemble.py`

**Why:** the brain has multiple tools (swarm, pricer, ML). The engine should
combine them and only fire when they agree and there's edge. Calibration (E1)
supplies the per-source weights.

**API:**
- `combine_sources(source_probs, weights=None) -> float` — calibration-weighted
  blend of probabilities.
- `make_ensemble_brain(sources, edge_min) -> signal_fn` — harness-compatible
  brain that blends sources and applies the agreement + edge gate.

**Acceptance criteria:**
1. Equal weights average the sources; unequal weights bias toward the
   higher-weighted source.
2. The ensemble brain stands down (side=None) when sources disagree beyond a
   threshold or edge < edge_min.
3. Output plugs into `WalkForwardBacktester.run()` unchanged.

## E5 — MiroFish refinement (parallel)

**Why:** the swarm is the unproven narrative-edge bet. Use E1 calibration to
measure whether changes (persona mix, info injection, convergence gate) actually
improve calibration on resolved markets. No change ships unless it improves the
calibration report out-of-sample.

**Acceptance criteria:**
1. A `calibration_report` is produced for the swarm on the sample markets.
2. Any swarm change is accepted only if it lowers Brier/log-loss OOS.

---

## Testing pyramid (per unit)

| Layer | What | Count (target) |
|---|---|---|
| Unit | each scoring/sizing/detection function | +4-6 each |
| Integration | each brain plugged into `WalkForwardBacktester.run()` | +1 each |
| E2E | `run_walkforward.py` produces a calibration + allocation report | +1 |

## Out of scope (do not add)

Live order execution, real-money credential handling, live data fetchers (separate
phase, needs keys), the dashboard UI, equity-market strategies.

## Rollback

All new modules are additive (new files). Revert = delete the new file + its test;
nothing existing is modified except `run_walkforward.py` reporting.
