"""
Tests for the quantitative fair-value pricer.

For markets with a computable underlying (BTC price, an index level), the
probability of finishing above a strike is not a matter of opinion — it falls
out of a lognormal model. This is the "quant for prediction markets" core:
fair value from spot + volatility + time, then edge = fair - market price.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signals.pricing import prob_finish_above


# ── Lognormal probability of finishing above a strike ────────────────────────

def test_at_the_money_is_about_half():
    """Spot == strike, tiny horizon, no drift -> ~50/50 (a hair under 0.5 from
    the -sigma^2/2 term)."""
    p = prob_finish_above(spot=100.0, strike=100.0, sigma_annual=0.6,
                          t_years=1.0 / (365 * 24))  # one hour
    assert p == pytest.approx(0.5, abs=0.01)
    assert p < 0.5  # the variance drag pulls it just below


def test_deep_in_the_money_approaches_one_and_out_approaches_zero():
    """Spot far above strike -> almost certain YES; far below -> almost certain
    NO."""
    high = prob_finish_above(spot=120.0, strike=100.0, sigma_annual=0.6,
                             t_years=1.0 / (365 * 24))
    low = prob_finish_above(spot=80.0, strike=100.0, sigma_annual=0.6,
                            t_years=1.0 / (365 * 24))
    assert high > 0.99
    assert low < 0.01


def test_probability_is_monotonic_increasing_in_spot():
    """As spot rises toward and past the strike, P(finish above) must rise."""
    probs = [
        prob_finish_above(spot=s, strike=100.0, sigma_annual=0.6,
                          t_years=1.0 / (365 * 24 * 12))  # five minutes
        for s in (98, 99, 100, 101, 102)
    ]
    assert probs == sorted(probs)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_more_time_pulls_a_winning_position_back_toward_even():
    """A position that is currently ITM is more certain with LESS time left:
    more time = more variance = more chance of reversal, so P drops toward 0.5
    as the horizon lengthens."""
    near = prob_finish_above(spot=101.0, strike=100.0, sigma_annual=0.6,
                             t_years=1.0 / (365 * 24 * 12))   # 5 min
    far = prob_finish_above(spot=101.0, strike=100.0, sigma_annual=0.6,
                            t_years=1.0 / (365 * 24))         # 1 hour
    assert near > far > 0.5


def test_zero_time_left_is_a_hard_settlement():
    """At expiry the outcome is decided: above strike -> 1, below -> 0."""
    assert prob_finish_above(spot=101.0, strike=100.0, sigma_annual=0.6,
                             t_years=0.0) == 1.0
    assert prob_finish_above(spot=99.0, strike=100.0, sigma_annual=0.6,
                             t_years=0.0) == 0.0


# ── Fair-value edge -> tradeable signal (plugs into the walk-forward harness) ──

def test_pricing_brain_buys_when_yes_is_underpriced():
    """Spot well above strike => model fair value ~1.0. If the market prices
    YES at only 0.40, YES is underpriced and the brain buys."""
    from src.signals.pricing import make_pricing_brain
    brain = make_pricing_brain()
    event = {"spot": 105.0, "strike": 100.0, "sigma_annual": 0.6,
             "t_years": 1.0 / (365 * 24), "market_prob": 0.40}
    sig = brain(event, {"edge_min": 0.05})
    assert sig["side"] == "buy"


def test_pricing_brain_sells_when_yes_is_overpriced():
    """Spot well below strike => fair value ~0.0. If the market prices YES at
    0.55, YES is overpriced and the brain sells (buys NO)."""
    from src.signals.pricing import make_pricing_brain
    brain = make_pricing_brain()
    event = {"spot": 95.0, "strike": 100.0, "sigma_annual": 0.6,
             "t_years": 1.0 / (365 * 24), "market_prob": 0.55}
    sig = brain(event, {"edge_min": 0.05})
    assert sig["side"] == "sell"


def test_pricing_brain_stands_down_when_edge_is_below_threshold():
    """When model fair value and market price agree within edge_min, no trade."""
    from src.signals.pricing import make_pricing_brain
    brain = make_pricing_brain()
    event = {"spot": 100.0, "strike": 100.0, "sigma_annual": 0.6,
             "t_years": 1.0 / (365 * 24), "market_prob": 0.49}
    sig = brain(event, {"edge_min": 0.05})
    assert sig["side"] is None
