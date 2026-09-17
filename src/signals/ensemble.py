"""
ensemble.py — combine multiple brain tools into one decision.

The engine runs several probability sources (MiroFish swarm, the quantitative
fair-value pricer, ML models). The ensemble blends them — weighted by their
measured calibration — and only trades when they agree and there is edge against
the market. The resulting brain is a drop-in `signal_fn` for the walk-forward
harness.
"""

from __future__ import annotations

import statistics


def combine_sources(source_probs, weights=None) -> float:
    """Weighted average of source probabilities. Equal weights by default."""
    if not source_probs:
        return 0.0
    if weights is None:
        return sum(source_probs) / len(source_probs)
    if len(weights) != len(source_probs):
        raise ValueError(
            f"weights and source_probs must match: "
            f"{len(weights)} != {len(source_probs)}"
        )
    total_w = sum(weights)
    if total_w <= 0:
        return sum(source_probs) / len(source_probs)
    return sum(p * w for p, w in zip(source_probs, weights)) / total_w


def make_ensemble_brain(sources, weights=None, max_disagreement: float = 0.15):
    """Build a harness-compatible brain that blends `sources`.

    `sources` is a list of callables `source(event) -> prob`. The brain stands
    down (side=None) when the sources disagree more than `max_disagreement`
    (population stdev of their probabilities) or when the blended edge versus
    `event['market_prob']` is below `params['edge_min']`.
    """
    def brain(event: dict, params: dict) -> dict:
        probs = [s(event) for s in sources]

        # consensus gate: too much spread across sources => no trade
        if len(probs) > 1 and statistics.pstdev(probs) > max_disagreement:
            return {"side": None}

        blended = combine_sources(probs, weights)
        edge = blended - event["market_prob"]
        if abs(edge) < params["edge_min"]:
            return {"side": None}
        return {"side": "buy" if edge > 0 else "sell",
                "size_pct": params.get("size_pct", 0.02)}

    return brain
