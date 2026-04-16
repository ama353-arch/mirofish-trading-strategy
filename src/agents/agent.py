"""
SwarmAgent — an LLM-driven agent that forms opinions, posts on the simulated
social platform, reacts to other agents, and produces a final probability
estimate for a target event.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from .persona import AgentPersona

logger = logging.getLogger(__name__)


@dataclass
class Post:
    """A single post on the simulated social platform."""
    author_id: str
    round_num: int
    content: str
    stance: str          # "bullish", "bearish", "neutral"
    confidence: float    # 0-1
    likes: int = 0
    replies: list["Post"] = field(default_factory=list)


class SwarmAgent:
    """
    A single agent in the swarm simulation.

    Each agent has a persona, maintains a belief state, observes the social
    environment, and takes actions (post, react, update beliefs) each round.
    """

    def __init__(self, persona: AgentPersona, llm_client=None):
        self.persona = persona
        self.llm = llm_client
        self.belief_history: list[float] = []       # probability estimates over rounds
        self.conviction_history: list[float] = []   # conviction (0-1) over rounds
        self.posts: list[Post] = []
        self.observed_posts: list[Post] = []
        self.current_probability: float = 0.5       # initial uninformed prior
        self.current_conviction: float = 0.5
        self.stance: str = "neutral"

    @property
    def agent_id(self) -> str:
        return self.persona.agent_id

    def initialize(self, event_description: str, context: str, initial_info: str = ""):
        """
        Set the agent's initial state based on event context.

        In LLM mode, the agent processes the event through its persona lens.
        In rule-based mode, uses heuristics based on persona traits.
        """
        if self.llm:
            self._llm_initialize(event_description, context, initial_info)
        else:
            self._rule_initialize(event_description)

    def _llm_initialize(self, event_desc: str, context: str, initial_info: str):
        """LLM-driven initial belief formation."""
        system = self.persona.to_system_prompt()
        user = f"""You are being asked to form an initial opinion about the following event:

EVENT: {event_desc}

AVAILABLE CONTEXT:
{context}

{f"ADDITIONAL INFORMATION: {initial_info}" if initial_info else ""}

Based on your background, expertise, and cognitive tendencies, provide:
1. Your initial probability estimate that the event outcome is YES (0.0 to 1.0)
2. Your confidence in this estimate (0.0 to 1.0)
3. Your stance: "bullish" (likely YES), "bearish" (likely NO), or "neutral"
4. A brief reasoning (2-3 sentences, in character)

Respond in JSON: {{"probability": float, "confidence": float, "stance": string, "reasoning": string}}"""

        try:
            result = self.llm.chat_json(system, user, temperature=0.8)
            self.current_probability = max(0.01, min(0.99, float(result.get("probability", 0.5))))
            self.current_conviction = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
            self.stance = result.get("stance", "neutral")
        except Exception as e:
            logger.warning(f"Agent {self.agent_id} LLM init failed: {e}")
            self._rule_initialize(event_desc)

        self.belief_history.append(self.current_probability)
        self.conviction_history.append(self.current_conviction)

    def _rule_initialize(self, event_desc: str):
        """Rule-based initial belief formation using persona traits."""
        # Start with a base prior around 0.5, shift based on archetype
        base = 0.5
        noise = random.gauss(0, 0.15)

        # Contrarians start skeptical of consensus
        if self.persona.contrarian_bias > 0.5:
            noise -= 0.05 * self.persona.contrarian_bias

        # Noise traders are nearly random
        if self.persona.archetype == "noise_trader":
            noise = random.gauss(0, 0.3)

        self.current_probability = max(0.01, min(0.99, base + noise))
        self.current_conviction = max(0.1, min(0.9,
            0.3 + self.persona.risk_tolerance * 0.3 + random.gauss(0, 0.1)
        ))
        if self.current_probability > 0.6:
            self.stance = "bullish"
        elif self.current_probability < 0.4:
            self.stance = "bearish"
        else:
            self.stance = "neutral"

        self.belief_history.append(self.current_probability)
        self.conviction_history.append(self.current_conviction)

    def observe(self, timeline: list[Post]):
        """Observe the latest posts from the simulated social platform."""
        self.observed_posts = timeline

    def act(self, round_num: int, event_description: str) -> Post | None:
        """
        Take an action this round: post an opinion, react, or stay silent.
        Returns a Post if the agent decides to speak, None otherwise.
        """
        # Decide whether to post (based on persona extroversion and decision speed)
        post_probability = 0.3
        if self.persona.decision_speed == "fast":
            post_probability += 0.2
        if self.persona.mbti[0] == "E":  # Extroverted
            post_probability += 0.15

        if random.random() > post_probability:
            return None  # Stay silent this round

        if self.llm:
            return self._llm_act(round_num, event_description)
        else:
            return self._rule_act(round_num)

    def _llm_act(self, round_num: int, event_desc: str) -> Post:
        """LLM-driven action: generate a post."""
        recent = self.observed_posts[-10:] if self.observed_posts else []
        timeline_summary = "\n".join([
            f"[{p.author_id}] ({p.stance}, conf={p.confidence:.1f}): {p.content[:200]}"
            for p in recent
        ]) or "No posts yet."

        system = self.persona.to_system_prompt()
        user = f"""Round {round_num} of the simulation. You are participating in a discussion about:

EVENT: {event_desc}

RECENT DISCUSSION:
{timeline_summary}

Your current belief: {self.current_probability:.2f} probability of YES, confidence {self.current_conviction:.2f}

Write a brief post (1-3 sentences) sharing your current view IN CHARACTER.
Also update your probability estimate based on what you've seen.

Respond in JSON: {{"post": string, "updated_probability": float, "confidence": float, "stance": string}}"""

        try:
            result = self.llm.chat_json(system, user, temperature=0.8)
            self.current_probability = max(0.01, min(0.99, float(result.get("updated_probability", self.current_probability))))
            self.current_conviction = max(0.0, min(1.0, float(result.get("confidence", self.current_conviction))))
            self.stance = result.get("stance", self.stance)
            content = result.get("post", f"I estimate {self.current_probability:.0%} probability.")
        except Exception as e:
            logger.warning(f"Agent {self.agent_id} LLM act failed: {e}")
            content = f"My current estimate: {self.current_probability:.0%} chance of YES."

        self.belief_history.append(self.current_probability)
        self.conviction_history.append(self.current_conviction)

        post = Post(
            author_id=self.agent_id,
            round_num=round_num,
            content=content,
            stance=self.stance,
            confidence=self.current_conviction,
        )
        self.posts.append(post)
        return post

    def _rule_act(self, round_num: int) -> Post:
        """Rule-based action: update beliefs from observed posts and generate a post."""
        if self.observed_posts:
            # Social influence: shift toward observed consensus
            observed_probs = [p.confidence * (1.0 if p.stance == "bullish" else 0.0 if p.stance == "bearish" else 0.5)
                             for p in self.observed_posts[-20:]]
            if observed_probs:
                social_signal = sum(observed_probs) / len(observed_probs)

                # Influence strength depends inversely on contrarian bias
                influence = (1 - self.persona.contrarian_bias) * 0.15
                if self.persona.contrarian_bias > 0.6:
                    # Contrarians move opposite to social signal
                    self.current_probability += influence * (0.5 - social_signal)
                else:
                    # Conformists move toward social signal
                    self.current_probability += influence * (social_signal - self.current_probability)

                self.current_probability = max(0.01, min(0.99, self.current_probability))

        # Add small random walk
        self.current_probability += random.gauss(0, 0.03)
        self.current_probability = max(0.01, min(0.99, self.current_probability))

        # Update conviction (increases over time as agents become more certain)
        self.current_conviction = min(0.95, self.current_conviction + random.gauss(0.01, 0.02))

        # Update stance
        if self.current_probability > 0.6:
            self.stance = "bullish"
        elif self.current_probability < 0.4:
            self.stance = "bearish"
        else:
            self.stance = "neutral"

        self.belief_history.append(self.current_probability)
        self.conviction_history.append(self.current_conviction)

        content = f"Round {round_num}: I see a {self.current_probability:.0%} chance. Feeling {self.stance}."
        post = Post(
            author_id=self.agent_id,
            round_num=round_num,
            content=content,
            stance=self.stance,
            confidence=self.current_conviction,
        )
        self.posts.append(post)
        return post

    def get_final_estimate(self) -> dict:
        """Return the agent's final probability estimate and metadata."""
        return {
            "agent_id": self.agent_id,
            "archetype": self.persona.archetype,
            "final_probability": self.current_probability,
            "final_conviction": self.current_conviction,
            "final_stance": self.stance,
            "belief_trajectory": self.belief_history,
            "conviction_trajectory": self.conviction_history,
            "n_posts": len(self.posts),
        }
