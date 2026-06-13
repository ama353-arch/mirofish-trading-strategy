"""
pricing.py — quantitative fair-value pricer for markets with a computable
underlying (BTC price, an index level, anything with a liquid spot + vol).

This is a *distinct brain tool* from the MiroFish swarm. Narrative markets need
an information edge; markets like "Will BTC be above $K at the top of the hour"
have a fair value that falls straight out of a lognormal model. The trade edge
comes from the contract price lagging the underlying:

    edge = model_fair_value - market_implied_probability

Drift over minute/hour horizons is ~0 (optionally funding-implied), so the
dominant inputs are spot, strike, volatility, and time-to-expiry.
"""

from __future__ import annotations

import math

from src.signals.walkforward import _norm_cdf


def prob_finish_above(
    spot: float,
    strike: float,
    sigma_annual: float,
    t_years: float,
    drift: float = 0.0,
) -> float:
    """Lognormal probability that the underlying finishes strictly above the
    strike at expiry (geometric Brownian motion).

        P(S_T > K) = N(d2),
        d2 = [ln(S/K) + (drift - sigma^2/2) * T] / (sigma * sqrt(T))

    At t_years == 0 this is a hard settlement (1.0 above strike, 0.0 below).
    """
    if t_years <= 0.0 or sigma_annual <= 0.0:
        return 1.0 if spot > strike else 0.0
    if spot <= 0.0:
        return 0.0

    vol = sigma_annual * math.sqrt(t_years)
    d2 = (math.log(spot / strike) + (drift - 0.5 * sigma_annual**2) * t_years) / vol
    return _norm_cdf(d2)


def make_pricing_brain(drift: float = 0.0):
    """A brain backed by the quantitative pricer, compatible with the
    walk-forward harness `signal_fn(event, params)` contract.

    The event must carry `spot`, `strike`, `sigma_annual`, `t_years`, and the
    `market_prob` (market-implied P(YES)). Trades only when the model edge
    exceeds `params['edge_min']`.
    """
    def brain(event: dict, params: dict) -> dict:
        fair = prob_finish_above(
            spot=event["spot"],
            strike=event["strike"],
            sigma_annual=event["sigma_annual"],
            t_years=event["t_years"],
            drift=drift,
        )
        edge = fair - event["market_prob"]
        if abs(edge) < params["edge_min"]:
            return {"side": None}
        # fair > market => YES underpriced => buy YES; else sell (buy NO)
        return {"side": "buy" if edge > 0 else "sell",
                "size_pct": params.get("size_pct", 0.02)}

    return brain
