"""
market_rag.py — Semantic search over prediction markets using RAG.

Inspired by Polymarket/agents' architecture: vectorize market descriptions
and use embedding-based search to find the most relevant markets for a
given event or query. This is far superior to keyword matching for
finding related markets.

Operates in two modes:
  1. Full mode (with sentence-transformers): proper embedding-based search
  2. Lite mode (no extra deps): TF-IDF based semantic search via scikit-learn

Both modes support:
  - Index a batch of markets
  - Semantic search by natural language query
  - Filter by exchange, volume, activity
  - Persist the index for reuse
"""

import json
import logging
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Try sentence-transformers for full embedding mode
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# TF-IDF fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False


@dataclass
class MarketSearchResult:
    """A market returned from semantic search."""
    market_id: str
    exchange: str
    question: str
    similarity_score: float
    yes_price: float
    volume: float
    metadata: dict


class MarketRAG:
    """
    Semantic search index over prediction markets.

    Usage:
        rag = MarketRAG()

        # Index markets (from UnifiedPredictionClient or raw dicts)
        rag.index_markets([
            {"id": "abc", "question": "Will the Fed cut rates?", "exchange": "polymarket", ...},
            ...
        ])

        # Search
        results = rag.search("monetary policy easing", top_k=5)
        for r in results:
            print(f"{r.similarity_score:.2f} | {r.question} ({r.exchange})")
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.markets: list[dict] = []
        self.texts: list[str] = []
        self.embeddings: np.ndarray | None = None

        # Initialize embedding model or TF-IDF fallback
        self._encoder = None
        self._tfidf = None
        self._tfidf_matrix = None

        if EMBEDDINGS_AVAILABLE:
            try:
                self._encoder = SentenceTransformer(model_name)
                logger.info(f"MarketRAG using sentence-transformers ({model_name})")
            except Exception as e:
                logger.warning(f"Failed to load sentence-transformers: {e}")

        if self._encoder is None and TFIDF_AVAILABLE:
            logger.info("MarketRAG using TF-IDF fallback (install sentence-transformers for better results)")
        elif self._encoder is None:
            logger.warning("MarketRAG: no embedding or TF-IDF available. Install scikit-learn or sentence-transformers.")

    def index_markets(self, markets: list[dict]):
        """
        Index a batch of markets for semantic search.

        Each market dict should have at minimum:
          - id or market_id: str
          - question or market_question: str
          - exchange: str

        Optional: yes_price, volume, active, outcomes, end_date
        """
        self.markets = markets
        self.texts = []

        for m in markets:
            # Build a rich text representation for embedding
            question = m.get("question", m.get("market_question", m.get("event_title", "")))
            outcomes = m.get("outcomes", [])
            exchange = m.get("exchange", "")
            text = f"{question}"
            if outcomes:
                text += f" Outcomes: {', '.join(str(o) for o in outcomes)}."
            if exchange:
                text += f" Exchange: {exchange}."
            self.texts.append(text)

        if not self.texts:
            logger.warning("No markets to index")
            return

        # Build embeddings
        if self._encoder is not None:
            self.embeddings = self._encoder.encode(self.texts, show_progress_bar=False)
            logger.info(f"Indexed {len(self.texts)} markets with sentence-transformers")
        elif TFIDF_AVAILABLE:
            self._tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
            self._tfidf_matrix = self._tfidf.fit_transform(self.texts)
            logger.info(f"Indexed {len(self.texts)} markets with TF-IDF")
        else:
            logger.warning("No search backend available")

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_volume: float = 0,
        exchanges: list[str] | None = None,
    ) -> list[MarketSearchResult]:
        """
        Semantic search for markets matching a query.

        Args:
            query: Natural language query (e.g., "US monetary policy")
            top_k: Number of results to return
            min_volume: Minimum 24h volume filter
            exchanges: Filter to specific exchanges
        """
        if not self.texts:
            return []

        # Compute similarities
        if self._encoder is not None and self.embeddings is not None:
            query_emb = self._encoder.encode([query], show_progress_bar=False)
            sims = cosine_similarity_np(query_emb, self.embeddings)[0]
        elif self._tfidf is not None and self._tfidf_matrix is not None:
            query_vec = self._tfidf.transform([query])
            sims = cosine_similarity(query_vec, self._tfidf_matrix)[0]
        else:
            # Last resort: keyword matching
            sims = self._keyword_similarity(query)

        # Build results with filtering
        results = []
        for idx in np.argsort(sims)[::-1]:
            m = self.markets[idx]
            score = float(sims[idx])

            # Filters
            vol = float(m.get("volume_24h", m.get("volume", 0)))
            if vol < min_volume:
                continue
            ex = m.get("exchange", "")
            if exchanges and ex not in exchanges:
                continue

            prices = m.get("outcome_prices", m.get("outcomePrices", [0.5]))
            if isinstance(prices, str):
                prices = json.loads(prices)

            results.append(MarketSearchResult(
                market_id=str(m.get("id", m.get("market_id", m.get("condition_id", "")))),
                exchange=ex,
                question=m.get("question", m.get("market_question", m.get("event_title", ""))),
                similarity_score=score,
                yes_price=float(prices[0]) if prices else 0.5,
                volume=vol,
                metadata=m,
            ))

            if len(results) >= top_k:
                break

        return results

    def find_arbitrage_opportunities(
        self,
        similarity_threshold: float = 0.85,
        price_divergence_threshold: float = 0.05,
    ) -> list[dict]:
        """
        Find potential arbitrage: same/similar markets on different exchanges
        with different prices.
        """
        if self.embeddings is None and self._tfidf_matrix is None:
            return []

        opportunities = []
        n = len(self.markets)

        # Compute pairwise similarities for cross-exchange pairs
        for i in range(n):
            for j in range(i + 1, n):
                ex_i = self.markets[i].get("exchange", "")
                ex_j = self.markets[j].get("exchange", "")
                if ex_i == ex_j:
                    continue  # Only cross-exchange

                # Similarity
                if self.embeddings is not None:
                    sim = float(np.dot(self.embeddings[i], self.embeddings[j]) /
                               (np.linalg.norm(self.embeddings[i]) * np.linalg.norm(self.embeddings[j]) + 1e-8))
                else:
                    continue

                if sim < similarity_threshold:
                    continue

                # Price divergence
                p_i = self.markets[i].get("outcome_prices", [0.5])
                p_j = self.markets[j].get("outcome_prices", [0.5])
                if isinstance(p_i, str): p_i = json.loads(p_i)
                if isinstance(p_j, str): p_j = json.loads(p_j)
                price_diff = abs(float(p_i[0]) - float(p_j[0]))

                if price_diff > price_divergence_threshold:
                    opportunities.append({
                        "market_a": {
                            "exchange": ex_i,
                            "question": self.markets[i].get("question", ""),
                            "price": float(p_i[0]),
                        },
                        "market_b": {
                            "exchange": ex_j,
                            "question": self.markets[j].get("question", ""),
                            "price": float(p_j[0]),
                        },
                        "similarity": round(sim, 3),
                        "price_divergence": round(price_diff, 4),
                    })

        opportunities.sort(key=lambda x: x["price_divergence"], reverse=True)
        return opportunities

    def _keyword_similarity(self, query: str) -> np.ndarray:
        """Fallback: simple keyword overlap similarity."""
        query_words = set(query.lower().split())
        sims = []
        for text in self.texts:
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            total = len(query_words | text_words)
            sims.append(overlap / total if total > 0 else 0)
        return np.array(sims)

    def save_index(self, path: str | Path):
        """Save the market index to disk."""
        data = {
            "markets": self.markets,
            "texts": self.texts,
        }
        if self.embeddings is not None:
            data["embeddings"] = self.embeddings.tolist()
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info(f"Saved market index to {path}")

    def load_index(self, path: str | Path):
        """Load a saved market index."""
        with open(path) as f:
            data = json.load(f)
        self.markets = data["markets"]
        self.texts = data["texts"]
        if "embeddings" in data:
            self.embeddings = np.array(data["embeddings"])
        elif TFIDF_AVAILABLE and self.texts:
            self._tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
            self._tfidf_matrix = self._tfidf.fit_transform(self.texts)
        logger.info(f"Loaded market index: {len(self.markets)} markets")


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query vector(s) and a matrix."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)
