# Crypto 99c-favourite forward test — running tally

The measurement has moved between machines, so the evidence lives in more than one ledger. All of them
use identical methodology (frozen rule, same fee model, same fill logic), so their break-even budgets
add and their loss counts add. With zero losses throughout, combined p = exp(-total break-even).

| Sample | Trades | Losses | Break-even budget | Ledger |
|---|---|---|---|---|
| Local, to 2026-09-17 | 318 | 0 | 1.63 | `ledger_local_318.json` |
| Railway, to 2026-09-17 | 174 | 0 | 0.92 | lost — ran on ephemeral disk before the volume existed |
| Railway, from 2026-09-17 | accumulating | — | — | `/data/ledger.json` on the volume |

**Banked so far: 492 trades, 0 losses, break-even 2.55, exact p = 0.078.**

Significance needs a break-even budget around 4.6 (p < 0.01), so this is roughly 56% of the way.
At ~5 trades/hour with the machine always awake, that is about a week more.

A single loss resets this arithmetic — which is the entire point of running it.
