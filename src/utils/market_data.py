"""
Market data clients for Polymarket (prediction markets) and equities (yfinance).
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import POLYMARKET_API_URL

logger = logging.getLogger(__name__)


# ── Polymarket Client ──────────────────────────────────────────────────────

class PolymarketClient:
    """Read-only client for Polymarket's CLOB API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or POLYMARKET_API_URL).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_markets(self, limit: int = 50, active_only: bool = True) -> list[dict]:
        """Fetch active prediction markets."""
        url = f"{self.base_url}/markets"
        params = {"limit": limit}
        if active_only:
            params["active"] = "true"
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Polymarket API error: {e}")
            return []

    def get_market(self, condition_id: str) -> dict | None:
        """Fetch a single market by condition ID."""
        url = f"{self.base_url}/markets/{condition_id}"
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Polymarket market fetch error: {e}")
            return None

    def get_orderbook(self, token_id: str) -> dict | None:
        """Fetch the order book for a token."""
        url = f"{self.base_url}/book"
        params = {"token_id": token_id}
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Polymarket orderbook error: {e}")
            return None

    @staticmethod
    def implied_probability(price: float) -> float:
        """Convert a CLOB price (0-1) to implied probability."""
        return max(0.0, min(1.0, price))

    @staticmethod
    def mock_market(
        question: str,
        yes_price: float = 0.55,
        volume_24h: float = 100_000,
    ) -> dict:
        """Create a mock market for testing without API access."""
        return {
            "condition_id": "mock_" + question[:20].replace(" ", "_").lower(),
            "question": question,
            "outcomes": ["Yes", "No"],
            "outcome_prices": [yes_price, 1 - yes_price],
            "volume_24h": volume_24h,
            "end_date": (datetime.now() + timedelta(days=30)).isoformat(),
            "active": True,
            "mock": True,
        }


# ── Equity Data Client ────────────────────────────────────────────────────

class EquityDataClient:
    """Equity market data via yfinance."""

    @staticmethod
    def get_price_history(
        ticker: str,
        period: str = "3mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV price history."""
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
            return df
        except Exception as e:
            logger.error(f"yfinance error for {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_current_price(ticker: str) -> float | None:
        """Get latest closing price."""
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1d")
            if hist.empty:
                return None
            return float(hist["Close"].iloc[-1])
        except Exception as e:
            logger.error(f"Price fetch error for {ticker}: {e}")
            return None

    @staticmethod
    def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    @staticmethod
    def get_implied_volatility(ticker: str) -> float | None:
        """Estimate implied volatility from options (nearest expiry ATM)."""
        try:
            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                return None
            chain = t.option_chain(expirations[0])
            calls = chain.calls
            # Find near-the-money
            current = EquityDataClient.get_current_price(ticker)
            if current is None:
                return None
            calls["moneyness"] = (calls["strike"] - current).abs()
            atm = calls.nsmallest(3, "moneyness")
            return float(atm["impliedVolatility"].mean())
        except Exception as e:
            logger.error(f"IV fetch error for {ticker}: {e}")
            return None

    @staticmethod
    def mock_price_data(ticker: str, days: int = 90) -> pd.DataFrame:
        """Generate mock price data for testing."""
        np.random.seed(hash(ticker) % 2**32)
        dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
        price = 100.0
        prices = []
        for _ in range(days):
            ret = np.random.normal(0.0005, 0.02)
            price *= (1 + ret)
            prices.append(price)
        df = pd.DataFrame({
            "Open": prices,
            "High": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            "Low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 10_000_000, size=days),
        }, index=dates)
        return df
