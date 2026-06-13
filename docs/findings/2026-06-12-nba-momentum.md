---
type: findings
topic: NBA in-game momentum — does it make money?
date: 2026-06-12
data: 60 real settled KXNBAGAME markets, 1-min candlestick price series (read-only)
status: validated — real but thin
---

# NBA In-Game Momentum: Findings

## Question
On Kalshi NBA live "win" markets, does a momentum move (fast swing in the win
price) tend to **continue** (FOLLOW edge) or **revert** (FADE edge), and is it
tradeable after costs?

## Method (disciplined, in order)
1. Fixed a real settlement bug first (`sell` = buy NO at `1-bid`; the old code
   overstated sell profit ~19x at low prices and produced a fake +347% result).
2. Cached 60 real games' in-game price trajectories (gitignored `data/raw/`).
3. **OOS walk-forward** (`oos_momentum.py`): tune thesis/window/threshold on
   train games, score held-out games. Result: +12.5%/trade, Sharpe 3.49,
   deflated Sharpe 1.000. **But a 3.49 Sharpe is a red flag** — settling at the
   final outcome conflates true edge with the mechanical "price tracks the game"
   effect and the favorite-longshot bias.
4. **True-edge test** (`momentum_forward_returns`): measure the price move over
   the NEXT few minutes, aligned to momentum direction. No outcome, no
   settlement, so the mechanical effect cannot leak in.

## Result (the honest one)
Across all window/horizon combos on 60 games:

| window | horizon | n | mean (signed fwd) | t-stat | tradeable vs ~1c spread |
|---|---|---|---|---|---|
| 5  | 5  | 2143 | +0.0086 | 3.54 | below spread |
| 5  | 10 | 2114 | +0.0125 | 3.74 | **above spread (barely)** |
| 10 | 10 | 2722 | +0.0101 | 3.62 | **above spread (barely)** |
| 15 | 5  | 2972 | +0.0076 | 4.07 | below spread |

- **Direction: CONTINUATION (FOLLOW).** Every combo positive, t > 3. Robust.
  My fade/overreaction prior was wrong.
- **Magnitude: marginal.** 0.5-1.25c per move. At/below the bid-ask spread for
  most settings; only the ~10-min horizon clears it, and barely.

## Conclusion
The momentum-continuation effect in NBA win prices is **real and statistically
significant but economically thin** — not harvestable as a *taker* (crossing the
spread eats it). This matches the research: the edge exists but is mostly
captured; surviving edge is maker-only / longer-horizon.

## Open directions (need a human call)
- **Maker execution** — post limit orders, never cross the spread; can the thin
  continuation be harvested without paying the spread?
- **Game-state model (play-by-play)** — join free NBA play-by-play to hunt the
  *rare, large* mispricings (a move that genuinely overshoots fair value), which
  are where real money is vs the thin average move.
- **Other sports** — soccer/NFL may show a larger or different effect.
