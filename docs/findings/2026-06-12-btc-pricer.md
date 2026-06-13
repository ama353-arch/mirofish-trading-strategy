---
type: findings
topic: BTC hourly pricer — is the computable edge tradeable?
date: 2026-06-12
data: 100 near-money settled KXBTCD hourly markets + Coinbase spot (read-only)
status: real-but-faint edge; not tradeable as taker; maker unproven
---

# BTC Hourly Pricer: Findings

## Question
On Kalshi hourly "BTC above $X" markets (KXBTCD), does our lognormal N(d2) fair
value beat the market, and is the edge tradeable after the spread?

## Method
Near-money markets only (strike within 1.5% of spot). BTC spot from Coinbase;
volatility = realized from the prior hour (no look-ahead, no Deribit IV yet).
One trade per market (independent), walk-forward, deflated Sharpe.

## Results
Calibration (Brier vs outcome): pricer **0.1580** < market 0.1596. On the 288
points where they disagree >10c, the **pricer is more right** (0.180 vs 0.190).
So the model is genuinely competitive on accuracy.

OOS profit, one trade/market, walk-forward (36 trades):

| Scenario | Avg PnL/$ | Win | Deflated Sharpe |
|---|---|---|---|
| Taker (cross spread) | -0.0056 | 52.8% | 0.000 |
| Maker (fill at mid) | +0.1993 | 52.8% | 0.044 |

## Conclusion
- The pricer has a **real but faint** signal (52.8% win on near-money binaries).
- **The spread is the killer**: positive at mid, gone once you cross bid/ask.
- Even as a maker it is **not yet statistically convincing** (DSR 0.044, n=36) and
  ignores adverse selection.
- Unlike NBA (market strictly better), BTC is a live candidate — the edge exists,
  it's just under the spread.

## Next levers (to confirm or kill)
1. Deribit implied vol instead of realized (research's specific recommendation).
2. More markets (n=36 is small) for significance.
3. Realistic maker-fill model with adverse selection (the honest maker test).
