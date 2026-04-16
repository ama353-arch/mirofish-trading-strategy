from .base import BaseStrategy
from .consensus_divergence import ConsensusDivergenceStrategy
from .sentiment_momentum import SentimentMomentumStrategy
from .contrarian_swarm import ContrarianSwarmStrategy
from .event_catalyst import EventCatalystStrategy
from .cross_market_arb import CrossMarketArbStrategy

ALL_STRATEGIES = [
    ConsensusDivergenceStrategy,
    SentimentMomentumStrategy,
    ContrarianSwarmStrategy,
    EventCatalystStrategy,
    CrossMarketArbStrategy,
]

__all__ = [
    "BaseStrategy",
    "ConsensusDivergenceStrategy",
    "SentimentMomentumStrategy",
    "ContrarianSwarmStrategy",
    "EventCatalystStrategy",
    "CrossMarketArbStrategy",
    "ALL_STRATEGIES",
]
