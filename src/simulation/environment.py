"""
SimulationEnvironment — the simulated social platform where agents interact.

Inspired by MiroFish's OASIS-powered Twitter/Reddit simulation, simplified
for a trading-focused use case.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from ..agents.agent import Post

logger = logging.getLogger(__name__)


@dataclass
class TimelineState:
    """State of the simulated social platform at a point in time."""
    round_num: int
    posts: list[Post] = field(default_factory=list)
    trending_topics: list[str] = field(default_factory=list)
    aggregate_sentiment: float = 0.5  # 0=bearish, 1=bullish


class SimulationEnvironment:
    """
    The shared social environment where agents observe and post.

    Models a simplified social media timeline with:
    - Chronological post feed
    - Trending topics (most discussed)
    - Aggregate sentiment tracking
    - Information injection (breaking news, data releases)
    """

    def __init__(self, event_description: str, initial_context: str = ""):
        self.event_description = event_description
        self.initial_context = initial_context
        self.timeline: list[Post] = []
        self.round_states: list[TimelineState] = []
        self.injected_info: list[tuple[int, str]] = []  # (round, info_text)

    def inject_information(self, round_num: int, info: str):
        """Schedule information to be injected at a specific round."""
        self.injected_info.append((round_num, info))
        logger.info(f"Scheduled info injection at round {round_num}: {info[:80]}")

    def get_timeline(self, round_num: int, max_posts: int = 50) -> list[Post]:
        """
        Get the current timeline visible to agents.

        Returns the most recent posts, weighted by engagement (likes).
        """
        # Include any injected information as "system" posts
        for inj_round, info in self.injected_info:
            if inj_round == round_num:
                system_post = Post(
                    author_id="BREAKING_NEWS",
                    round_num=round_num,
                    content=info,
                    stance="neutral",
                    confidence=1.0,
                )
                self.timeline.append(system_post)

        # Return recent posts, prioritizing high-engagement ones
        recent = self.timeline[-max_posts * 2:]
        # Sort by a mix of recency and engagement
        recent.sort(key=lambda p: p.round_num + p.likes * 0.1, reverse=True)
        return recent[:max_posts]

    def add_post(self, post: Post):
        """Add a new post to the timeline."""
        self.timeline.append(post)

    def process_round(self, round_num: int):
        """
        End-of-round processing: compute aggregate metrics and save state.
        """
        round_posts = [p for p in self.timeline if p.round_num == round_num]

        # Compute aggregate sentiment
        if round_posts:
            sentiments = []
            for p in round_posts:
                if p.stance == "bullish":
                    sentiments.append(1.0)
                elif p.stance == "bearish":
                    sentiments.append(0.0)
                else:
                    sentiments.append(0.5)
            avg_sentiment = sum(sentiments) / len(sentiments)
        else:
            avg_sentiment = self.round_states[-1].aggregate_sentiment if self.round_states else 0.5

        # Simulate engagement (likes) — popular posts get more likes
        for post in round_posts:
            if post.author_id != "BREAKING_NEWS":
                # Posts aligned with current sentiment get more engagement
                alignment = 1 - abs(
                    (1.0 if post.stance == "bullish" else 0.0 if post.stance == "bearish" else 0.5)
                    - avg_sentiment
                )
                post.likes = int(alignment * post.confidence * 10 + 1)

        state = TimelineState(
            round_num=round_num,
            posts=round_posts,
            aggregate_sentiment=avg_sentiment,
        )
        self.round_states.append(state)

    def get_sentiment_trajectory(self) -> list[float]:
        """Return the aggregate sentiment across all rounds."""
        return [s.aggregate_sentiment for s in self.round_states]

    def get_activity_stats(self) -> dict:
        """Return summary statistics about the simulation environment."""
        return {
            "total_posts": len(self.timeline),
            "total_rounds": len(self.round_states),
            "posts_per_round": len(self.timeline) / max(1, len(self.round_states)),
            "final_sentiment": self.round_states[-1].aggregate_sentiment if self.round_states else 0.5,
            "sentiment_trajectory": self.get_sentiment_trajectory(),
        }
