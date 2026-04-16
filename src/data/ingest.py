"""
Data ingestion pipeline — fetches news, social media, market data, and
economic indicators to feed into the knowledge graph.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import requests
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import NEWS_API_KEY

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    Aggregates data from multiple sources into a unified format suitable
    for knowledge graph construction.
    """

    def __init__(self, news_api_key: str | None = None):
        self.news_api_key = news_api_key or NEWS_API_KEY
        self.session = requests.Session()

    # ── News Ingestion ─────────────────────────────────────────────────────

    def fetch_news(
        self,
        query: str,
        days_back: int = 7,
        max_articles: int = 50,
        language: str = "en",
    ) -> list[dict]:
        """
        Fetch news articles from NewsAPI.

        Returns list of dicts with: title, description, content, source, url, published_at
        """
        if not self.news_api_key:
            logger.warning("No NEWS_API_KEY set, returning mock data")
            return self._mock_news(query, max_articles)

        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "language": language,
            "sortBy": "relevancy",
            "pageSize": min(max_articles, 100),
            "apiKey": self.news_api_key,
        }

        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for a in data.get("articles", []):
                articles.append({
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "content": a.get("content", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "url": a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                })
            logger.info(f"Fetched {len(articles)} news articles for query: {query}")
            return articles
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return self._mock_news(query, max_articles)

    # ── Reddit Ingestion ───────────────────────────────────────────────────

    def fetch_reddit_posts(
        self,
        subreddit: str,
        query: str = "",
        limit: int = 50,
    ) -> list[dict]:
        """
        Fetch Reddit posts via the public JSON API (no auth needed for read-only).

        Returns list of dicts with: title, selftext, score, num_comments, url, created_utc
        """
        base_url = f"https://www.reddit.com/r/{subreddit}"
        if query:
            url = f"{base_url}/search.json?q={query}&restrict_sr=1&limit={limit}&sort=relevance"
        else:
            url = f"{base_url}/hot.json?limit={limit}"

        try:
            headers = {"User-Agent": "mirofish-trading/1.0"}
            resp = self.session.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            posts = []
            for child in data.get("data", {}).get("children", []):
                p = child.get("data", {})
                posts.append({
                    "title": p.get("title", ""),
                    "selftext": p.get("selftext", "")[:2000],
                    "score": p.get("score", 0),
                    "num_comments": p.get("num_comments", 0),
                    "url": f"https://reddit.com{p.get('permalink', '')}",
                    "created_utc": p.get("created_utc", 0),
                    "subreddit": subreddit,
                })
            logger.info(f"Fetched {len(posts)} Reddit posts from r/{subreddit}")
            return posts
        except Exception as e:
            logger.error(f"Reddit API error: {e}")
            return []

    # ── FRED Economic Data ─────────────────────────────────────────────────

    def fetch_fred_series(
        self,
        series_id: str,
        observation_start: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch economic data from FRED (public, no API key needed for basic access).

        Common series: GDP, UNRATE, CPIAUCSL, FEDFUNDS, T10Y2Y, VIXCLS
        """
        if not observation_start:
            observation_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={observation_start}"
        try:
            df = pd.read_csv(url, parse_dates=["DATE"])
            df.columns = ["date", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df.dropna(inplace=True)
            logger.info(f"Fetched {len(df)} observations for FRED series {series_id}")
            return df
        except Exception as e:
            logger.error(f"FRED fetch error for {series_id}: {e}")
            return pd.DataFrame(columns=["date", "value"])

    # ── Aggregation ────────────────────────────────────────────────────────

    def ingest_for_event(
        self,
        event_description: str,
        keywords: list[str] | None = None,
        subreddits: list[str] | None = None,
    ) -> list[dict]:
        """
        Run the full ingestion pipeline for a target event.

        Returns a list of document dicts ready for knowledge graph construction.
        """
        documents = []
        kw = keywords or event_description.split()[:5]
        query = " ".join(kw)

        # News
        articles = self.fetch_news(query, days_back=14, max_articles=30)
        for a in articles:
            text = f"{a['title']}. {a['description']} {a['content']}"
            if len(text.strip()) > 50:
                documents.append({
                    "text": text,
                    "source": f"news:{a['source']}",
                    "type": "news",
                    "url": a["url"],
                    "published": a["published_at"],
                })

        # Reddit
        subs = subreddits or ["politics", "economics", "wallstreetbets", "stocks"]
        for sub in subs:
            posts = self.fetch_reddit_posts(sub, query=query, limit=20)
            for p in posts:
                text = f"{p['title']}. {p['selftext']}"
                if len(text.strip()) > 50:
                    documents.append({
                        "text": text,
                        "source": f"reddit:r/{sub}",
                        "type": "reddit",
                        "url": p["url"],
                        "score": p["score"],
                    })

        logger.info(f"Ingested {len(documents)} documents for event: {event_description[:80]}")
        return documents

    # ── Mock Data ──────────────────────────────────────────────────────────

    @staticmethod
    def _mock_news(query: str, n: int = 10) -> list[dict]:
        """Generate mock news articles for testing without API keys."""
        templates = [
            "Analysts debate the implications of {q} for markets",
            "New developments in {q} raise questions about future direction",
            "Expert panel split on {q} outcome, citing competing factors",
            "Recent data suggests shifting dynamics around {q}",
            "Market participants reassess positions following {q} developments",
            "Institutional investors weigh in on {q} with mixed signals",
            "Social media buzz around {q} reaches new highs",
            "Historical patterns suggest caution regarding {q}",
            "Policy implications of {q} remain uncertain, sources say",
            "Breaking: Key stakeholders announce positions on {q}",
        ]
        articles = []
        for i in range(min(n, len(templates))):
            title = templates[i].format(q=query)
            articles.append({
                "title": title,
                "description": f"Detailed analysis of {query} and its potential market impact.",
                "content": f"In a comprehensive review, analysts have examined the various factors "
                          f"surrounding {query}. Key considerations include market sentiment, "
                          f"historical precedent, and emerging data points that could influence "
                          f"the outcome. Multiple stakeholders have expressed divergent views.",
                "source": f"MockNews_{i}",
                "url": f"https://example.com/news/{i}",
                "published_at": datetime.now().isoformat(),
            })
        return articles
