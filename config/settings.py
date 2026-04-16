"""
Global configuration for the MiroFish Trading Strategy.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ── LLM Configuration ─────────────────────────────────────────────────────
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── Market Data APIs ──────────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
POLYMARKET_API_URL = "https://clob.polymarket.com"
KALSHI_API_URL = "https://trading-api.kalshi.com/trade-api/v2"

# ── Simulation Parameters ─────────────────────────────────────────────────
DEFAULT_AGENT_COUNT = 100
DEFAULT_SIMULATION_ROUNDS = 30
DEFAULT_NUM_SIMULATIONS = 10
CONVERGENCE_THRESHOLD = 1.1        # Gelman-Rubin R-hat
MIN_EFFECTIVE_SAMPLE_SIZE = 100

# ── Agent Archetype Distribution (Prediction Markets) ─────────────────────
PREDICTION_MARKET_ARCHETYPES = {
    "retail_bettor":       0.30,
    "political_analyst":   0.15,
    "contrarian_trader":   0.10,
    "institutional_trader":0.15,
    "social_influencer":   0.10,
    "data_scientist":      0.10,
    "domain_expert":       0.05,
    "noise_trader":        0.05,
}

# ── Agent Archetype Distribution (Equity Markets) ─────────────────────────
EQUITY_MARKET_ARCHETYPES = {
    "quant_trader":        0.15,
    "fundamental_analyst": 0.20,
    "retail_investor":     0.20,
    "market_maker":        0.10,
    "macro_strategist":    0.10,
    "event_driven":        0.05,
    "contrarian":          0.10,
    "algo_trader":         0.10,
}

# ── Trading Rules ─────────────────────────────────────────────────────────
# Prediction markets
PM_SIGNAL_THRESHOLD = 0.05         # Min |alpha| to trade
PM_MAX_SWARM_STD = 0.15            # Max swarm std for trade entry
PM_KELLY_FRACTION = 0.25           # Fractional Kelly multiplier
PM_MAX_POSITION_PCT = 0.05         # Max % of bankroll per contract
PM_MAX_DEPLOYED_PCT = 0.20         # Max total deployed capital
PM_STOP_LOSS_PCT = 0.50            # Stop loss as % of initial stake
PM_TAKE_PROFIT_PCT = 1.00          # Take profit as % of initial stake

# Equity markets
EQ_BULL_THRESHOLD = 0.65           # Min bullish fraction for long
EQ_BEAR_THRESHOLD = 0.35           # Max bullish fraction for short
EQ_MIN_CONVICTION = 7              # Min avg conviction (1-10)
EQ_BASE_POSITION_PCT = 0.02        # Base position as % of NAV
EQ_MAX_POSITION_PCT = 0.05         # Max position as % of NAV
EQ_MAX_SECTOR_PCT = 0.15           # Max sector exposure
EQ_STOP_LOSS_ATR_MULT = 2.0        # Stop loss in ATR multiples
EQ_TIME_STOP_DAYS = 5              # Days before time-based exit
EQ_MAX_DRAWDOWN_PCT = 0.10         # Max portfolio drawdown before halt
