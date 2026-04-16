"""
pmxt_client.py — Unified prediction market client powered by pmxt.

pmxt (https://github.com/pmxt-dev/pmxt) provides a CCXT-style unified API
for Polymarket, Kalshi, Limitless, and 6+ other prediction market exchanges.

This module wraps pmxt to provide:
  1. Multi-exchange market discovery and price fetching
  2. Orderbook and trade data across all supported exchanges
  3. Order placement with safety rails (position limits, confirmation)
  4. Fallback to our existing mock/direct clients when pmxt isn't installed

Install: pip install pmxt
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Try to import pmxt; gracefully degrade if not installed
try:
    from pmxt import Exchange as PmxtExchange
    PMXT_AVAILABLE = True
    logger.info("pmxt library loaded — unified prediction market API available")
except ImportError:
    PMXT_AVAILABLE = False
    logger.info("pmxt not installed — using built-in clients. Run: pip install pmxt")


@dataclass
class UnifiedMarket:
    """Normalized market representation across all exchanges."""
    exchange: str               # "polymarket", "kalshi", "limitless", etc.
    market_id: str
    event_title: str
    market_question: str
    outcomes: list[str]         # e.g., ["Yes", "No"]
    outcome_prices: list[float] # e.g., [0.65, 0.35]
    volume_24h: float
    liquidity: float
    end_date: str
    active: bool
    url: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def yes_price(self) -> float:
        return self.outcome_prices[0] if self.outcome_prices else 0.5

    @property
    def implied_probability(self) -> float:
        return self.yes_price

    @property
    def spread(self) -> float:
        """Bid-ask spread estimate from outcome prices."""
        if len(self.outcome_prices) >= 2:
            return abs(1.0 - sum(self.outcome_prices))
        return 0.0


@dataclass
class UnifiedOrder:
    """Order to place on any exchange."""
    exchange: str
    market_id: str
    outcome: str         # "Yes" or "No"
    side: str            # "buy" or "sell"
    price: float         # Limit price (0-1)
    amount: float        # In dollars
    order_type: str = "limit"


class UnifiedPredictionClient:
    """
    Unified client for all prediction market exchanges.

    Uses pmxt when available, falls back to direct API calls for Polymarket.

    Supported exchanges (via pmxt):
      - polymarket, polymarket_us, kalshi, limitless, probable,
        myriad, opinion, metaculus, smarkets

    Usage:
        client = UnifiedPredictionClient()

        # Connect to exchanges (read-only for market data)
        client.connect("polymarket")
        client.connect("kalshi")

        # Search markets across all connected exchanges
        markets = client.search_markets("Fed rate cut")

        # Get orderbook
        book = client.get_orderbook("polymarket", market_id)

        # Place order (requires credentials)
        client.connect("polymarket", credentials={...})
        order = client.place_order(UnifiedOrder(...))
    """

    # Safety limits
    MAX_ORDER_SIZE = 500         # Max $500 per order
    MAX_POSITION_PCT = 0.05      # Max 5% of bankroll per position
    MAX_TOTAL_EXPOSURE = 0.20    # Max 20% total deployed

    def __init__(self, bankroll: float = 10_000):
        self.bankroll = bankroll
        self.exchanges: dict[str, Any] = {}
        self.positions: list[dict] = []
        self._total_deployed = 0.0

    def connect(self, exchange_name: str, credentials: dict | None = None) -> bool:
        """
        Connect to a prediction market exchange.

        Args:
            exchange_name: e.g., "polymarket", "kalshi", "limitless"
            credentials: Exchange-specific auth (see pmxt docs)
                Polymarket: {"privateKey": "0x...", "signatureType": "EOA"}
                Kalshi: {"apiKey": "...", "privateKey": "RSA key"}
        """
        if PMXT_AVAILABLE:
            try:
                config = credentials or {}
                ex = PmxtExchange(exchange_name, config)
                self.exchanges[exchange_name] = ex
                logger.info(f"Connected to {exchange_name} via pmxt")
                return True
            except Exception as e:
                logger.error(f"Failed to connect to {exchange_name}: {e}")
                return False
        else:
            # Stub connection for non-pmxt usage
            self.exchanges[exchange_name] = {"name": exchange_name, "stub": True}
            logger.info(f"Stub connection to {exchange_name} (pmxt not installed)")
            return True

    def search_markets(
        self,
        query: str,
        exchanges: list[str] | None = None,
        active_only: bool = True,
        min_volume: float = 0,
        max_results: int = 50,
    ) -> list[UnifiedMarket]:
        """
        Search for markets across all connected exchanges.

        Returns normalized UnifiedMarket objects sorted by volume.
        """
        target_exchanges = exchanges or list(self.exchanges.keys())
        all_markets = []

        for ex_name in target_exchanges:
            ex = self.exchanges.get(ex_name)
            if ex is None:
                continue

            if PMXT_AVAILABLE and not isinstance(ex, dict):
                try:
                    raw_markets = ex.fetchMarkets(query=query, limit=max_results)
                    for m in raw_markets:
                        market = self._normalize_pmxt_market(ex_name, m)
                        if market and (not active_only or market.active):
                            if market.volume_24h >= min_volume:
                                all_markets.append(market)
                except Exception as e:
                    logger.warning(f"Market search failed on {ex_name}: {e}")
            else:
                # Fallback: use our built-in Polymarket client
                all_markets.extend(self._fallback_search(ex_name, query, max_results))

        # Sort by volume descending
        all_markets.sort(key=lambda m: m.volume_24h, reverse=True)
        return all_markets[:max_results]

    def get_orderbook(self, exchange: str, market_id: str) -> dict | None:
        """Fetch the order book for a specific market."""
        ex = self.exchanges.get(exchange)
        if ex is None:
            return None

        if PMXT_AVAILABLE and not isinstance(ex, dict):
            try:
                return ex.fetchOrderBook(market_id)
            except Exception as e:
                logger.error(f"Orderbook fetch failed: {e}")
                return None
        return None

    def get_market_price(self, exchange: str, market_id: str) -> float | None:
        """Get the current YES price for a market."""
        ex = self.exchanges.get(exchange)
        if ex is None:
            return None

        if PMXT_AVAILABLE and not isinstance(ex, dict):
            try:
                book = ex.fetchOrderBook(market_id)
                if book and "bids" in book and book["bids"]:
                    return float(book["bids"][0][0])
                return None
            except Exception as e:
                logger.error(f"Price fetch failed: {e}")
                return None
        return None

    def place_order(self, order: UnifiedOrder, dry_run: bool = True) -> dict:
        """
        Place an order with safety rails.

        IMPORTANT: dry_run=True by default. Set to False for live trading.

        Safety checks:
          1. Order size <= MAX_ORDER_SIZE
          2. Position <= MAX_POSITION_PCT of bankroll
          3. Total exposure <= MAX_TOTAL_EXPOSURE of bankroll
        """
        # Safety check 1: absolute size limit
        if order.amount > self.MAX_ORDER_SIZE:
            return {"status": "rejected", "reason": f"Order ${order.amount} exceeds max ${self.MAX_ORDER_SIZE}"}

        # Safety check 2: position size
        max_pos = self.bankroll * self.MAX_POSITION_PCT
        if order.amount > max_pos:
            return {"status": "rejected", "reason": f"Order ${order.amount} exceeds position limit ${max_pos:.0f}"}

        # Safety check 3: total exposure
        max_total = self.bankroll * self.MAX_TOTAL_EXPOSURE
        if self._total_deployed + order.amount > max_total:
            return {"status": "rejected", "reason": f"Would exceed total exposure limit ${max_total:.0f}"}

        if dry_run:
            logger.info(f"DRY RUN: {order.side} {order.outcome} on {order.exchange} "
                       f"@ {order.price:.2f} for ${order.amount:.2f}")
            return {
                "status": "dry_run",
                "order": {
                    "exchange": order.exchange,
                    "market_id": order.market_id,
                    "outcome": order.outcome,
                    "side": order.side,
                    "price": order.price,
                    "amount": order.amount,
                },
            }

        # Live order execution
        ex = self.exchanges.get(order.exchange)
        if ex is None:
            return {"status": "error", "reason": f"Not connected to {order.exchange}"}

        if PMXT_AVAILABLE and not isinstance(ex, dict):
            try:
                result = ex.createOrder(
                    outcome=order.outcome,
                    side=order.side,
                    price=order.price,
                    amount=order.amount,
                )
                self._total_deployed += order.amount
                self.positions.append({
                    "exchange": order.exchange,
                    "market_id": order.market_id,
                    "outcome": order.outcome,
                    "price": order.price,
                    "amount": order.amount,
                    "timestamp": datetime.now().isoformat(),
                })
                return {"status": "filled", "result": result}
            except Exception as e:
                return {"status": "error", "reason": str(e)}
        else:
            return {"status": "error", "reason": "pmxt not installed — cannot place live orders"}

    def get_positions(self) -> list[dict]:
        """Return current open positions."""
        return self.positions

    def get_exposure(self) -> dict:
        """Return current exposure metrics."""
        return {
            "total_deployed": self._total_deployed,
            "bankroll": self.bankroll,
            "exposure_pct": self._total_deployed / self.bankroll if self.bankroll > 0 else 0,
            "remaining_capacity": max(0, self.bankroll * self.MAX_TOTAL_EXPOSURE - self._total_deployed),
            "n_positions": len(self.positions),
        }

    def list_exchanges(self) -> list[str]:
        """Return list of connected exchanges."""
        return list(self.exchanges.keys())

    # ── Internal Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _normalize_pmxt_market(exchange: str, raw: dict) -> UnifiedMarket | None:
        """Convert pmxt market object to our UnifiedMarket format."""
        try:
            outcomes = raw.get("outcomes", ["Yes", "No"])
            prices = raw.get("outcomePrices", raw.get("outcome_prices", [0.5, 0.5]))
            if isinstance(prices, str):
                import json
                prices = json.loads(prices)
            prices = [float(p) for p in prices]

            return UnifiedMarket(
                exchange=exchange,
                market_id=str(raw.get("id", raw.get("market_id", ""))),
                event_title=raw.get("event_title", raw.get("title", "")),
                market_question=raw.get("question", raw.get("title", "")),
                outcomes=outcomes,
                outcome_prices=prices,
                volume_24h=float(raw.get("volume_24h", raw.get("volume", 0))),
                liquidity=float(raw.get("liquidity", 0)),
                end_date=raw.get("end_date", raw.get("close_time", "")),
                active=raw.get("active", True),
                url=raw.get("url", ""),
                raw=raw,
            )
        except Exception as e:
            logger.warning(f"Failed to normalize market from {exchange}: {e}")
            return None

    def _fallback_search(self, exchange: str, query: str, limit: int) -> list[UnifiedMarket]:
        """Fallback market search using our built-in client."""
        from .market_data import PolymarketClient

        if exchange in ("polymarket", "polymarket_us"):
            client = PolymarketClient()
            raw_markets = client.get_markets(limit=limit)
            markets = []
            for m in raw_markets:
                question = m.get("question", m.get("title", ""))
                if query.lower() in question.lower() or not query:
                    prices = m.get("outcome_prices", m.get("outcomePrices", [0.5, 0.5]))
                    if isinstance(prices, str):
                        import json
                        prices = json.loads(prices)
                    markets.append(UnifiedMarket(
                        exchange=exchange,
                        market_id=str(m.get("condition_id", m.get("id", ""))),
                        event_title=question,
                        market_question=question,
                        outcomes=m.get("outcomes", ["Yes", "No"]),
                        outcome_prices=[float(p) for p in prices],
                        volume_24h=float(m.get("volume_24h", m.get("volume", 0))),
                        liquidity=float(m.get("liquidity", 0)),
                        end_date=m.get("end_date", ""),
                        active=m.get("active", True),
                        raw=m,
                    ))
            return markets
        return []


# ── Convenience Functions ──────────────────────────────────────────────────

def get_live_markets(
    query: str = "",
    exchanges: list[str] | None = None,
    min_volume: float = 10_000,
    limit: int = 20,
) -> list[UnifiedMarket]:
    """
    Quick function to search live markets across exchanges.

    Usage:
        markets = get_live_markets("Fed rate cut", exchanges=["polymarket", "kalshi"])
        for m in markets:
            print(f"{m.exchange}: {m.market_question} — YES @ {m.yes_price:.0%}")
    """
    client = UnifiedPredictionClient()
    for ex in (exchanges or ["polymarket"]):
        client.connect(ex)
    return client.search_markets(query, min_volume=min_volume, max_results=limit)
