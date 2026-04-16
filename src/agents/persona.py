"""
Agent persona generation — the MiroFish-inspired approach to creating diverse,
realistic market participant personas from a knowledge graph.
"""

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    PREDICTION_MARKET_ARCHETYPES,
    EQUITY_MARKET_ARCHETYPES,
    DEFAULT_AGENT_COUNT,
)

logger = logging.getLogger(__name__)

# ── MBTI types and their trading implications ──────────────────────────────
MBTI_TRADING_PROFILES = {
    "INTJ": {"risk_tolerance": 0.7, "contrarian_bias": 0.6, "info_processing": "systematic", "decision_speed": "slow"},
    "ENTJ": {"risk_tolerance": 0.8, "contrarian_bias": 0.4, "info_processing": "systematic", "decision_speed": "fast"},
    "INTP": {"risk_tolerance": 0.5, "contrarian_bias": 0.7, "info_processing": "analytical", "decision_speed": "slow"},
    "ENTP": {"risk_tolerance": 0.7, "contrarian_bias": 0.8, "info_processing": "exploratory", "decision_speed": "fast"},
    "INFJ": {"risk_tolerance": 0.3, "contrarian_bias": 0.5, "info_processing": "intuitive", "decision_speed": "slow"},
    "ENFJ": {"risk_tolerance": 0.4, "contrarian_bias": 0.3, "info_processing": "intuitive", "decision_speed": "fast"},
    "INFP": {"risk_tolerance": 0.3, "contrarian_bias": 0.4, "info_processing": "intuitive", "decision_speed": "slow"},
    "ENFP": {"risk_tolerance": 0.6, "contrarian_bias": 0.5, "info_processing": "exploratory", "decision_speed": "fast"},
    "ISTJ": {"risk_tolerance": 0.2, "contrarian_bias": 0.2, "info_processing": "systematic", "decision_speed": "slow"},
    "ESTJ": {"risk_tolerance": 0.4, "contrarian_bias": 0.2, "info_processing": "systematic", "decision_speed": "fast"},
    "ISFJ": {"risk_tolerance": 0.2, "contrarian_bias": 0.1, "info_processing": "conservative", "decision_speed": "slow"},
    "ESFJ": {"risk_tolerance": 0.3, "contrarian_bias": 0.1, "info_processing": "consensus", "decision_speed": "fast"},
    "ISTP": {"risk_tolerance": 0.6, "contrarian_bias": 0.5, "info_processing": "analytical", "decision_speed": "fast"},
    "ESTP": {"risk_tolerance": 0.9, "contrarian_bias": 0.4, "info_processing": "action-oriented", "decision_speed": "fast"},
    "ISFP": {"risk_tolerance": 0.4, "contrarian_bias": 0.3, "info_processing": "intuitive", "decision_speed": "slow"},
    "ESFP": {"risk_tolerance": 0.7, "contrarian_bias": 0.2, "info_processing": "experiential", "decision_speed": "fast"},
}

ARCHETYPE_TEMPLATES = {
    # ── Prediction Market Archetypes ──
    "retail_bettor": {
        "description": "Recreational prediction market bettor. Follows news and social media. Prone to recency bias and narrative-driven betting. Moderate position sizes.",
        "expertise": ["current events", "social media trends", "sports"],
        "cognitive_biases": ["recency_bias", "anchoring", "availability_heuristic", "bandwagon"],
        "information_sources": ["twitter", "reddit", "news_headlines"],
        "typical_mbti": ["ESFP", "ENFP", "ESTP", "ENTP"],
    },
    "political_analyst": {
        "description": "Professional or amateur political analyst. Deep domain knowledge in elections, policy, and geopolitics. Uses base rates and historical analogues. Methodical and calibrated.",
        "expertise": ["elections", "policy analysis", "geopolitics", "polling methodology"],
        "cognitive_biases": ["overconfidence_in_models", "anchoring_to_base_rates"],
        "information_sources": ["polling_data", "academic_research", "policy_documents", "expert_networks"],
        "typical_mbti": ["INTJ", "INTP", "ISTJ", "ENTJ"],
    },
    "contrarian_trader": {
        "description": "Systematically bets against consensus. Looks for markets where the crowd has overreacted. Profits from mean reversion in sentiment.",
        "expertise": ["market microstructure", "behavioral finance", "contrarian strategies"],
        "cognitive_biases": ["contrarian_bias", "overconfidence"],
        "information_sources": ["market_data", "sentiment_indicators", "positioning_data"],
        "typical_mbti": ["ENTP", "INTP", "INTJ", "ISTP"],
    },
    "institutional_trader": {
        "description": "Risk-averse professional trader at a fund or prop shop. Focuses on expected value calculations, proper position sizing, and portfolio risk. Large positions.",
        "expertise": ["risk management", "portfolio theory", "derivatives", "market microstructure"],
        "cognitive_biases": ["loss_aversion", "status_quo_bias"],
        "information_sources": ["bloomberg", "research_reports", "internal_models", "options_market"],
        "typical_mbti": ["ISTJ", "INTJ", "ESTJ", "ENTJ"],
    },
    "social_influencer": {
        "description": "High-follower social media personality who amplifies narratives. May or may not have deep expertise. Drives herding behavior in their followers.",
        "expertise": ["content creation", "narrative construction", "audience engagement"],
        "cognitive_biases": ["narrative_bias", "spotlight_effect", "overconfidence"],
        "information_sources": ["twitter", "youtube", "podcasts", "viral_content"],
        "typical_mbti": ["ENFJ", "ENFP", "ENTJ", "ESFP"],
    },
    "data_scientist": {
        "description": "Quantitative analyst who builds statistical models. Relies on data over narratives. Skeptical of anecdotal evidence. May miss qualitative signals.",
        "expertise": ["statistics", "machine_learning", "data_analysis", "causal_inference"],
        "cognitive_biases": ["model_overfit", "quantitative_bias", "neglect_of_qualitative"],
        "information_sources": ["structured_data", "academic_papers", "model_outputs"],
        "typical_mbti": ["INTP", "INTJ", "ISTP", "ISTJ"],
    },
    "domain_expert": {
        "description": "Deep specialist in the specific domain of the event (e.g., epidemiologist for pandemic markets, climate scientist for weather markets). High signal-to-noise.",
        "expertise": ["domain_specific"],
        "cognitive_biases": ["curse_of_knowledge", "anchoring_to_expertise"],
        "information_sources": ["academic_literature", "professional_networks", "primary_data"],
        "typical_mbti": ["INTJ", "ISTJ", "INTP", "INFJ"],
    },
    "noise_trader": {
        "description": "Trades on noise rather than signal. Random-ish behavior that adds liquidity and prevents the swarm from being artificially clean.",
        "expertise": [],
        "cognitive_biases": ["all_biases"],
        "information_sources": ["random"],
        "typical_mbti": list(MBTI_TRADING_PROFILES.keys()),
    },
    # ── Equity Market Archetypes ──
    "quant_trader": {
        "description": "Systematic trader using statistical arbitrage, factor models, and mean-reversion signals. Data-driven, low conviction per trade, high throughput.",
        "expertise": ["factor_models", "statistical_arbitrage", "time_series", "options_pricing"],
        "cognitive_biases": ["model_overfit", "overfitting_to_backtest"],
        "information_sources": ["price_data", "factor_databases", "options_flow"],
        "typical_mbti": ["INTP", "INTJ", "ISTP", "ISTJ"],
    },
    "fundamental_analyst": {
        "description": "Bottoms-up equity analyst. DCF models, earnings quality assessment, management evaluation. Long time horizons.",
        "expertise": ["financial_modeling", "accounting", "valuation", "industry_analysis"],
        "cognitive_biases": ["anchoring_to_models", "endowment_effect", "confirmation_bias"],
        "information_sources": ["10K_filings", "earnings_calls", "sell_side_research", "industry_data"],
        "typical_mbti": ["ISTJ", "INTJ", "ESTJ", "INTP"],
    },
    "retail_investor": {
        "description": "Individual investor. Mix of FOMO, social media influence, and basic fundamental awareness. Concentrated positions, emotional decision-making.",
        "expertise": ["basic_finance", "social_media_trends"],
        "cognitive_biases": ["recency_bias", "loss_aversion", "herding", "FOMO", "disposition_effect"],
        "information_sources": ["reddit_wsb", "twitter_fintwit", "cnbc", "tiktok_finance"],
        "typical_mbti": ["ESFP", "ENFP", "ESTP", "ENTP"],
    },
    "market_maker": {
        "description": "Delta-neutral liquidity provider. Profits from bid-ask spread. Views are about volatility and flow, not direction.",
        "expertise": ["market_microstructure", "options_greeks", "inventory_management", "flow_analysis"],
        "cognitive_biases": ["vol_anchoring"],
        "information_sources": ["order_flow", "options_surface", "cross_asset_correlations"],
        "typical_mbti": ["ISTP", "INTP", "ISTJ", "INTJ"],
    },
    "macro_strategist": {
        "description": "Top-down global macro thinker. Trades rates, FX, commodities, and equities based on macro regime views.",
        "expertise": ["monetary_policy", "fiscal_policy", "global_macro", "cross_asset"],
        "cognitive_biases": ["narrative_bias", "overconfidence_in_framework"],
        "information_sources": ["fed_minutes", "economic_data", "global_flows", "central_bank_speeches"],
        "typical_mbti": ["ENTJ", "INTJ", "ENTP", "INTP"],
    },
    "event_driven": {
        "description": "Catalyst-focused trader. M&A, spin-offs, regulatory decisions, earnings surprises. Short time horizon, high conviction.",
        "expertise": ["corporate_actions", "regulatory_process", "activism", "special_situations"],
        "cognitive_biases": ["overconfidence", "anchoring_to_thesis"],
        "information_sources": ["SEC_filings", "court_documents", "regulatory_calendars", "expert_networks"],
        "typical_mbti": ["ENTJ", "INTJ", "ESTP", "ENTP"],
    },
    "contrarian": {
        "description": "Fades consensus. Buys when others panic, sells when others are greedy. Deep value orientation.",
        "expertise": ["behavioral_finance", "value_investing", "sentiment_analysis"],
        "cognitive_biases": ["contrarian_bias", "patience_bias"],
        "information_sources": ["sentiment_surveys", "positioning_data", "valuation_metrics"],
        "typical_mbti": ["INTP", "INTJ", "ISTP", "ENTP"],
    },
    "algo_trader": {
        "description": "Momentum and technical signal trader. Follows price trends, volume breakouts, and technical patterns.",
        "expertise": ["technical_analysis", "momentum", "trend_following", "volume_analysis"],
        "cognitive_biases": ["pattern_seeking", "confirmation_bias"],
        "information_sources": ["price_charts", "volume_data", "technical_indicators"],
        "typical_mbti": ["ISTP", "ESTP", "INTP", "INTJ"],
    },
}


@dataclass
class AgentPersona:
    """Complete persona for a swarm agent."""
    agent_id: str
    archetype: str
    name: str
    age: int
    gender: str
    country: str
    mbti: str
    profession: str
    bio: str
    persona_description: str  # Rich backstory + behavioral tendencies
    risk_tolerance: float     # 0-1
    contrarian_bias: float    # 0-1 (how much they fade consensus)
    info_processing: str      # systematic, analytical, intuitive, etc.
    decision_speed: str       # fast or slow
    expertise: list[str] = field(default_factory=list)
    cognitive_biases: list[str] = field(default_factory=list)
    information_sources: list[str] = field(default_factory=list)
    interested_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_system_prompt(self) -> str:
        """Generate the system prompt that will drive this agent's LLM behavior."""
        return f"""You are {self.name}, a {self.age}-year-old {self.gender} {self.profession} from {self.country}.

PERSONALITY: {self.mbti}
RISK TOLERANCE: {self.risk_tolerance:.1f}/1.0 ({'aggressive' if self.risk_tolerance > 0.6 else 'moderate' if self.risk_tolerance > 0.3 else 'conservative'})
CONTRARIAN TENDENCY: {self.contrarian_bias:.1f}/1.0 ({'strong contrarian' if self.contrarian_bias > 0.6 else 'moderate' if self.contrarian_bias > 0.3 else 'consensus-following'})
DECISION STYLE: {self.info_processing}, {self.decision_speed} decisions

BACKGROUND: {self.persona_description}

EXPERTISE: {', '.join(self.expertise)}
COGNITIVE BIASES YOU TEND TO EXHIBIT: {', '.join(self.cognitive_biases)}
INFORMATION SOURCES YOU TRUST: {', '.join(self.information_sources)}

You must respond IN CHARACTER. Your opinions should reflect your background, biases, and expertise.
When asked to evaluate events or markets, reason from YOUR perspective — not as a neutral AI."""


class PersonaGenerator:
    """
    Generates diverse agent personas for swarm simulation.

    Can operate in two modes:
    1. Rule-based (no LLM) — fast, deterministic, good for backtesting
    2. LLM-enhanced — richer personas, better for live simulation
    """

    # Name pools for rule-based generation
    FIRST_NAMES = [
        "Alex", "Jordan", "Morgan", "Casey", "Taylor", "Riley", "Quinn",
        "Avery", "Cameron", "Dakota", "Reese", "Skyler", "Finley", "Sage",
        "James", "Sarah", "Wei", "Priya", "Carlos", "Yuki", "Omar",
        "Elena", "Dmitri", "Aisha", "Raj", "Mei", "Fatima", "Hans",
        "Sofia", "Kenji", "Amara", "Lucas", "Zara", "Nikolai", "Ingrid",
    ]
    COUNTRIES = [
        "United States", "United Kingdom", "Germany", "Japan", "India",
        "Brazil", "Canada", "Australia", "Singapore", "South Korea",
        "France", "Netherlands", "Switzerland", "China", "Israel",
    ]
    PROFESSIONS_BY_ARCHETYPE = {
        "retail_bettor": ["college student", "software engineer", "teacher", "sales rep", "freelancer"],
        "political_analyst": ["political consultant", "think tank researcher", "journalist", "professor of political science"],
        "contrarian_trader": ["hedge fund analyst", "independent trader", "prop desk trader"],
        "institutional_trader": ["portfolio manager", "risk analyst", "prop trader", "CIO"],
        "social_influencer": ["content creator", "podcast host", "newsletter writer", "YouTuber"],
        "data_scientist": ["ML engineer", "quantitative researcher", "data analyst", "PhD candidate"],
        "domain_expert": ["industry specialist", "professor", "former regulator", "subject-matter consultant"],
        "noise_trader": ["retiree", "part-time day trader", "hobbyist investor", "student"],
        "quant_trader": ["quantitative analyst", "algo developer", "strat at a prop firm"],
        "fundamental_analyst": ["equity research analyst", "buy-side analyst", "CFA charterholder"],
        "retail_investor": ["Robinhood trader", "WSB member", "self-taught investor", "gig worker"],
        "market_maker": ["options market maker", "electronic trader", "liquidity provider"],
        "macro_strategist": ["global macro PM", "rates strategist", "chief economist"],
        "event_driven": ["special situations analyst", "merger arb trader", "activist fund analyst"],
        "contrarian": ["deep value investor", "distressed debt analyst", "contrarian fund PM"],
        "algo_trader": ["systematic trader", "CTA analyst", "technical analyst"],
    }

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def generate_population(
        self,
        market_type: str = "prediction",
        n_agents: int | None = None,
        event_context: str = "",
        use_llm: bool = False,
    ) -> list[AgentPersona]:
        """
        Generate a diverse population of agent personas.

        Args:
            market_type: "prediction" or "equity"
            n_agents: Total number of agents (defaults to config)
            event_context: Description of the event being simulated
            use_llm: Whether to use LLM for richer persona generation
        """
        n = n_agents or DEFAULT_AGENT_COUNT
        archetypes = PREDICTION_MARKET_ARCHETYPES if market_type == "prediction" else EQUITY_MARKET_ARCHETYPES

        personas = []
        for archetype, fraction in archetypes.items():
            count = max(1, round(n * fraction))
            template = ARCHETYPE_TEMPLATES[archetype]
            for j in range(count):
                if use_llm and self.llm:
                    persona = self._generate_llm_persona(archetype, template, event_context, j)
                else:
                    persona = self._generate_rule_persona(archetype, template, j)
                personas.append(persona)

        # Trim or pad to exact count
        if len(personas) > n:
            personas = personas[:n]
        elif len(personas) < n:
            # Pad with random archetypes
            while len(personas) < n:
                arch = random.choice(list(archetypes.keys()))
                tmpl = ARCHETYPE_TEMPLATES[arch]
                personas.append(self._generate_rule_persona(arch, tmpl, len(personas)))

        random.shuffle(personas)
        logger.info(f"Generated {len(personas)} agent personas for {market_type} market")
        return personas

    def _generate_rule_persona(self, archetype: str, template: dict, idx: int) -> AgentPersona:
        """Fast, deterministic persona generation."""
        mbti = random.choice(template["typical_mbti"])
        mbti_profile = MBTI_TRADING_PROFILES[mbti]
        name = random.choice(self.FIRST_NAMES)
        country = random.choice(self.COUNTRIES)
        profs = self.PROFESSIONS_BY_ARCHETYPE.get(archetype, ["analyst"])
        profession = random.choice(profs)
        age = random.randint(22, 65)
        gender = random.choice(["male", "female", "non-binary"])

        # Add noise to MBTI-derived traits
        risk_tol = np.clip(mbti_profile["risk_tolerance"] + random.gauss(0, 0.1), 0, 1)
        contrarian = np.clip(mbti_profile["contrarian_bias"] + random.gauss(0, 0.1), 0, 1)

        return AgentPersona(
            agent_id=f"{archetype}_{idx:03d}",
            archetype=archetype,
            name=name,
            age=age,
            gender=gender,
            country=country,
            mbti=mbti,
            profession=profession,
            bio=f"{name} is a {age}-year-old {profession} from {country}.",
            persona_description=template["description"],
            risk_tolerance=round(risk_tol, 2),
            contrarian_bias=round(contrarian, 2),
            info_processing=mbti_profile["info_processing"],
            decision_speed=mbti_profile["decision_speed"],
            expertise=template["expertise"],
            cognitive_biases=template["cognitive_biases"],
            information_sources=template["information_sources"],
        )

    def _generate_llm_persona(self, archetype: str, template: dict, context: str, idx: int) -> AgentPersona:
        """LLM-enhanced persona generation with rich backstory."""
        if not self.llm:
            return self._generate_rule_persona(archetype, template, idx)

        system_prompt = """You are a persona generator for a market simulation.
Generate a realistic, detailed persona for a market participant.
Respond in JSON with these fields:
- name (string)
- age (int, 22-65)
- gender (string)
- country (string)
- mbti (string, valid MBTI type)
- profession (string)
- bio (string, ~100 chars)
- persona_description (string, ~500 chars, detailed backstory and behavioral tendencies)
- interested_topics (list of strings)"""

        user_prompt = f"""Generate a persona for this archetype:
ARCHETYPE: {archetype}
DESCRIPTION: {template['description']}
EXPERTISE AREAS: {', '.join(template['expertise'])}
EVENT CONTEXT: {context or 'General market prediction'}

Make the persona unique, specific, and realistic. Include concrete details about their
background, how they make decisions, what biases they might exhibit, and how they'd
approach this specific event."""

        try:
            data = self.llm.chat_json(system_prompt, user_prompt, temperature=0.9)
            mbti = data.get("mbti", random.choice(template["typical_mbti"]))
            if mbti not in MBTI_TRADING_PROFILES:
                mbti = random.choice(template["typical_mbti"])
            mbti_profile = MBTI_TRADING_PROFILES[mbti]

            return AgentPersona(
                agent_id=f"{archetype}_{idx:03d}",
                archetype=archetype,
                name=data.get("name", f"Agent_{idx}"),
                age=data.get("age", 35),
                gender=data.get("gender", "non-binary"),
                country=data.get("country", "United States"),
                mbti=mbti,
                profession=data.get("profession", "analyst"),
                bio=data.get("bio", template["description"][:100]),
                persona_description=data.get("persona_description", template["description"]),
                risk_tolerance=round(np.clip(mbti_profile["risk_tolerance"] + random.gauss(0, 0.1), 0, 1), 2),
                contrarian_bias=round(np.clip(mbti_profile["contrarian_bias"] + random.gauss(0, 0.1), 0, 1), 2),
                info_processing=mbti_profile["info_processing"],
                decision_speed=mbti_profile["decision_speed"],
                expertise=template["expertise"],
                cognitive_biases=template["cognitive_biases"],
                information_sources=template["information_sources"],
                interested_topics=data.get("interested_topics", []),
            )
        except Exception as e:
            logger.warning(f"LLM persona generation failed for {archetype}_{idx}: {e}, falling back to rule-based")
            return self._generate_rule_persona(archetype, template, idx)
