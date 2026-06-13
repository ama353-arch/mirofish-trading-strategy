"""
arbitrage.py — market-neutral arbitrage detection across binary markets.

No edge model required: arbitrage is pure price arithmetic, which makes it the
most defensible autonomous strategy. The engine still only *surfaces* the arb;
a human (or the user's own script) places the legs.

Two kinds:
  - Structural: mutually-exclusive outcomes whose ask prices sum below 1.
  - Cross-venue: the same event priced such that YES on one venue + NO on the
    other costs less than 1.
"""

from __future__ import annotations


def find_structural_arb(leg_asks, fee: float = 0.0) -> dict | None:
    """Detect a structural arb across mutually-exclusive outcomes.

    `leg_asks` are the ask prices to buy each outcome. If buying the full set
    costs less than 1 (after `fee`), exactly one resolves YES and pays 1, so the
    profit is locked. Needs at least two legs to form a partition.
    """
    if len(leg_asks) < 2:
        return None
    total_cost = sum(leg_asks) + fee
    profit = 1.0 - total_cost
    if profit <= 0:
        return None
    return {
        "kind": "structural",
        "legs": list(leg_asks),
        "total_cost": total_cost,
        "guaranteed_profit": profit,
    }


def find_cross_venue_arb(market_a: dict, market_b: dict, fee: float = 0.0) -> dict | None:
    """Detect a cross-venue arb on the same event across two venues.

    Each market exposes `yes_ask` and `no_ask`. You can lock profit by buying
    YES on one venue and NO on the other; we take the cheaper of the two
    directions. Profit exists when that combined cost (plus `fee`) is below 1.
    """
    dir1 = market_a["yes_ask"] + market_b["no_ask"]   # YES@A + NO@B
    dir2 = market_b["yes_ask"] + market_a["no_ask"]   # YES@B + NO@A
    if dir1 <= dir2:
        cost, direction = dir1, "yes_a+no_b"
    else:
        cost, direction = dir2, "yes_b+no_a"

    total_cost = cost + fee
    profit = 1.0 - total_cost
    if profit <= 0:
        return None
    return {
        "kind": "cross_venue",
        "direction": direction,
        "total_cost": total_cost,
        "guaranteed_profit": profit,
    }
