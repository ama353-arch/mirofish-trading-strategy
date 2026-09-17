"""
portfolio.py — concurrent capital allocation across simultaneous positions.

The backtest scores trades one at a time, but a live engine fires many at once.
This layer turns a set of signals into position sizes that respect:
  - fractional Kelly per position,
  - a per-position cap (no single bet too large),
  - a total-deployed cap (keep dry powder),
  - correlation groups (don't bet the same risk repeatedly).

Correlation is modeled as explicit groups for v1 (a full covariance matrix is a
future refinement): members of a group share a single group's budget.
"""

from __future__ import annotations


def _kelly(edge: float, odds: float, fraction: float = 0.25) -> float:
    """Fractional Kelly stake as a fraction of bankroll.

    `edge` is expected value per unit staked; `odds` are the net decimal odds.
    Returns 0 for non-positive edge. `fraction` < 1 is the standard safety
    haircut (quarter-Kelly by default).
    """
    if edge <= 0 or odds <= 0:
        return 0.0
    full_kelly = edge / odds
    return fraction * full_kelly


# Public alias; `allocate` takes a `kelly_fraction` parameter that would
# otherwise shadow this name inside the function body.
kelly_fraction = _kelly


def allocate(
    signals,
    max_deployed: float = 0.20,
    max_position: float = 0.05,
    kelly_fraction: float = 0.25,
    correlation_groups=None,
) -> list[dict]:
    """Size a set of concurrent signals into bankroll-fraction allocations.

    Each signal is a dict with `id`, `edge`, `odds`, and an optional `group`.
    Pipeline: raw fractional-Kelly size -> share within correlation groups ->
    clamp to `max_position` -> scale the whole book down if total exceeds
    `max_deployed`.
    """
    # 1. raw fractional-Kelly size per signal
    raw = {}
    for s in signals:
        raw[s["id"]] = _kelly(s["edge"], s["odds"], fraction=kelly_fraction)

    # 2. correlation groups share a single budget (the max member size),
    #    split equally among members
    group_members: dict[str, list[str]] = {}
    for s in signals:
        g = s.get("group")
        if g is not None:
            group_members.setdefault(g, []).append(s["id"])
    for members in group_members.values():
        budget = max(raw[m] for m in members)
        share = budget / len(members)
        for m in members:
            raw[m] = share

    # 3. per-position cap
    capped = {sid: min(size, max_position) for sid, size in raw.items()}

    # 4. total-deployed cap with proportional scaling
    total = sum(capped.values())
    if total > max_deployed and total > 0:
        scale = max_deployed / total
        capped = {sid: size * scale for sid, size in capped.items()}

    return [{"id": s["id"], "size": capped[s["id"]]} for s in signals]
