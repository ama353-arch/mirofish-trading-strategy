"""
Market scanner module — discovers, filters, and ranks trading opportunities.

Bridges market discovery (pmxt / direct Kalshi API) and the MiroFish
agent swarm simulation. Answers the question: "what should I trade?"
"""

from .kalshi_scanner import (
    KalshiScanner,
    MarketOpportunity,
    ScanConfig,
    ScanResult,
)

__all__ = [
    "KalshiScanner",
    "MarketOpportunity",
    "ScanConfig",
    "ScanResult",
]
