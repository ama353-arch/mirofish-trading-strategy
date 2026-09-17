"""
Paper-trade ledger: fees, settlement, and the statistic that decides the question.

Every Kalshi 15-minute crypto series measured efficient, so this ledger's real job
is not to report profit — it is to report whether the observed LOSS COUNT is below
the number of losses the prices could absorb and still break even. On bets priced
near certainty a t-statistic is worthless (a run with no losses looks infinitely
significant), so `stats()` reports an exact Poisson lower-tail p-value instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

TAKER_RATE = 0.07
MAKER_RATE = 0.0175
CENTICENT = 10_000  # Kalshi rounds a fee up to $0.0001


def _round_up_centicent(raw: float) -> float:
    return math.ceil(round(raw * CENTICENT, 9)) / CENTICENT


def taker_fee(price: float, contracts: float = 1) -> float:
    """Kalshi taker fee: 0.07 * C * P * (1-P), rounded up to a centicent."""
    return _round_up_centicent(TAKER_RATE * contracts * price * (1 - price))


def maker_fee(price: float, contracts: float = 1) -> float:
    """Kalshi maker fee: a quarter of the taker rate, same shape."""
    return _round_up_centicent(MAKER_RATE * contracts * price * (1 - price))


def poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for X ~ Poisson(lam). Computed in logs so large lam stays finite."""
    if lam <= 0:
        return 1.0
    total = sum(math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1)) for i in range(k + 1))
    return min(1.0, total)


@dataclass
class PaperTrade:
    """One hypothetical fill, at a price that was really quoted, with the real fee."""

    ticker: str
    series: str
    decision_ts: float
    side: str  # "yes" or "no" — the side BOUGHT, at its own ask
    price: float  # what that side cost, per contract
    contracts: float
    rule: str
    meta: dict[str, Any] = field(default_factory=dict)
    result: int | None = None  # 1 = market resolved yes, 0 = no

    @property
    def fee(self) -> float:
        return taker_fee(self.price, self.contracts)

    def won(self, result: int) -> bool:
        return (self.side == "yes") == (result == 1)

    def gross(self, result: int) -> float:
        return self.contracts * (1 - self.price) if self.won(result) else -self.contracts * self.price

    def pnl(self, result: int) -> float:
        return self.gross(result) - self.fee

    @property
    def break_even_loss_rate(self) -> float:
        """The loss rate at which this trade's expected value is exactly zero."""
        return (1 - self.price) - self.fee / self.contracts


class PaperLedger:
    """Holds paper trades and settles them against the market's published result."""

    def __init__(self) -> None:
        self.trades: list[PaperTrade] = []

    def add(self, trade: PaperTrade) -> None:
        self.trades.append(trade)

    def settle(self, ticker: str, result: int) -> int:
        """Settle every open trade on `ticker`. Unknown tickers are a no-op."""
        n = 0
        for t in self.trades:
            if t.ticker == ticker and t.result is None:
                t.result = result
                n += 1
        return n

    @property
    def settled(self) -> list[PaperTrade]:
        return [t for t in self.trades if t.result is not None]

    def stats(self) -> dict[str, Any]:
        done = self.settled
        losses = sum(1 for t in done if not t.won(t.result))
        be = sum(t.break_even_loss_rate for t in done)
        gross = sum(t.gross(t.result) for t in done)
        fees = sum(t.fee for t in done)
        contracts = sum(t.contracts for t in done)
        return {
            "open": len(self.trades) - len(done),
            "settled": len(done),
            "wins": len(done) - losses,
            "losses": losses,
            "contracts": contracts,
            "gross": gross,
            "fees": fees,
            "net": gross - fees,
            "net_per_contract": (gross - fees) / contracts if contracts else float("nan"),
            "break_even_losses": be,
            "exact_p": poisson_cdf(losses, be) if done else float("nan"),
        }
