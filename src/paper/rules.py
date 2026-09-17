"""
Candidate rules, and the pricing they depend on.

The one thing to get right here is WHICH PRICE EACH SIDE COSTS. A Kalshi book
quotes YES; buying YES lifts the yes ask, and buying NO lifts the NO ask, which
is `1 - yes_bid`. Pricing the NO side at `1 - yes_ask` puts the buyer on the wrong
side of the spread and invents an edge exactly the size of that spread. That error
produced a fake +5c/contract result in the two widest-spread series on 2026-09-11.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Quote:
    ticker: str
    ts: float
    seconds_to_close: float
    yes_bid: float | None
    yes_ask: float | None

    @property
    def valid(self) -> bool:
        b, a = self.yes_bid, self.yes_ask
        return b is not None and a is not None and 0 < b <= a < 1

    @property
    def mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2


@dataclass
class PaperOrder:
    side: str
    price: float
    rule: str


def _sides(q: Quote) -> tuple[tuple[str, float], tuple[str, float]]:
    """(favourite, longshot), each as (side, what it costs at its OWN ask)."""
    yes = ("yes", q.yes_ask)
    no = ("no", 1 - q.yes_bid)
    return (yes, no) if q.mid >= 0.5 else (no, yes)


def favourite_at_ask(q: Quote) -> tuple[str, float] | None:
    return _sides(q)[0] if q.valid else None


def longshot_at_ask(q: Quote) -> tuple[str, float] | None:
    return _sides(q)[1] if q.valid else None


def final_seconds_favourite(
    q: Quote, min_price: float = 0.99, window: tuple[float, float] = (2, 15)
) -> PaperOrder | None:
    """Buy the favourite in the closing seconds when it is already priced near certain.

    This is the one lead history could not resolve: 1,612 historical trades produced
    zero losses against 3.9 break-even losses (pooled p = 0.02, neither half
    significant alone). Only forward samples can settle it.
    """
    if not q.valid or not (window[0] <= q.seconds_to_close <= window[1]):
        return None
    side, price = _sides(q)[0]
    if not (min_price <= price < 1):
        return None
    return PaperOrder(side=side, price=price, rule="final_seconds_favourite")


def model_disagreement(q: Quote, model_p: float, threshold: float) -> PaperOrder | None:
    """Trade when a model's probability beats the price it would actually pay, net of the fee.

    The threshold is applied AFTER the fee, because a gross gap is not an edge: at 50c the taker fee
    alone is 1.75c, so a 2c disagreement is a losing trade dressed up as a winning one.
    """
    from src.paper.ledger import taker_fee

    if not q.valid:
        return None
    ask = q.yes_ask
    if model_p - ask - taker_fee(ask) > threshold:
        return PaperOrder(side="yes", price=ask, rule="model_disagreement")
    no_ask = 1 - q.yes_bid
    if (1 - model_p) - no_ask - taker_fee(no_ask) > threshold:
        return PaperOrder(side="no", price=no_ask, rule="model_disagreement")
    return None
