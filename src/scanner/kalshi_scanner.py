"""
kalshi_scanner.py — discover and rank Kalshi trading opportunities.

This module answers the practical question every trader asks first:
"Out of thousands of active Kalshi markets, which ones should I trade right now?"

Pipeline:
    1. Discover  — pull active markets (Kalshi public API or pmxt fallback)
    2. Filter    — drop markets that fail liquidity / volume / time guards
    3. Score     — run agent swarm sim on each survivor, compute composite score
    4. Rank      — return top-N opportunities sorted by score

The score is a transparent linear combination of six factors. Each factor is
normalized into [0, 1] before weighting so weights are directly interpretable:

    edge        — |swarm_probability - market_probability|   (the alpha)
    convergence — 1 if R-hat < CONVERGENCE_THRESHOLD else 0  (trust gate)
    liquidity   — log1p(liquidity)  normalized by reference scale
    volume      — log1p(volume_24h) normalized by reference scale
    time_value  — sweet-spot bell curve over days-to-resolution
    confidence  — 1 - swarm_std    (lower disagreement → higher conf)

Designed to run offline: when no network is available the scanner still
produces a working ranking against mock markets. This makes it demo-safe.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import KALSHI_API_URL, CONVERGENCE_THRESHOLD


def _import_module_by_path(module_name: str, file_path: Path):
    """Load a module directly from a file path, bypassing its parent package __init__.

    The src.utils package imports openai at package init, which makes it impossible
    to use lighter-weight siblings (like pmxt_client) when that dep isn't installed.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_pmxt_module = _import_module_by_path(
    "_mirofish_pmxt_client",
    Path(__file__).resolve().parents[1] / "utils" / "pmxt_client.py",
)
UnifiedMarket = _pmxt_module.UnifiedMarket


def _load_simulation_engine():
    """Lazy import for the simulation engine, used only when a real scan runs."""
    from src.simulation.engine import SimulationEngine, SwarmResult  # noqa: F401
    return SimulationEngine

logger = logging.getLogger(__name__)


# ── Public dataclasses ─────────────────────────────────────────────────────


@dataclass
class ScanConfig:
    """Knobs controlling a Kalshi scan."""

    min_liquidity: float = 1_000.0
    min_volume_24h: float = 500.0
    min_days_to_close: float = 0.5
    max_days_to_close: float = 60.0
    min_yes_price: float = 0.05
    max_yes_price: float = 0.95
    max_candidates: int = 40           # cap before scoring (sim is expensive)
    top_n: int = 10                    # how many opportunities to return

    # Simulation settings (kept small — scanner runs many sims back-to-back)
    n_agents: int = 60
    n_rounds: int = 15
    n_sims: int = 4
    seed: int | None = 42

    # Score weights (must be non-negative). Defaults sum to 1.0 but
    # the final score is min-max scaled, so absolute weights only need
    # to encode RELATIVE importance.
    w_edge: float = 0.35
    w_convergence: float = 0.10
    w_liquidity: float = 0.15
    w_volume: float = 0.15
    w_time: float = 0.10
    w_confidence: float = 0.15

    # Reference scales for log normalization
    liquidity_ref: float = 50_000.0
    volume_ref: float = 100_000.0

    # Time-value bell-curve params (days)
    time_sweet_spot: float = 7.0
    time_bandwidth: float = 14.0


@dataclass
class MarketOpportunity:
    """A scored trading opportunity returned by the scanner."""

    market_id: str
    ticker: str
    question: str
    yes_price: float
    no_price: float
    liquidity: float
    volume_24h: float
    days_to_close: float
    swarm_probability: float
    swarm_std: float
    r_hat: float
    converged: bool
    edge: float                 # signed: swarm - market
    direction: str              # "long_yes", "long_no", "no_trade"
    score: float
    score_components: dict = field(default_factory=dict)

    @property
    def abs_edge(self) -> float:
        return abs(self.edge)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edge"] = round(self.edge, 4)
        d["score"] = round(self.score, 4)
        d["swarm_probability"] = round(self.swarm_probability, 4)
        d["swarm_std"] = round(self.swarm_std, 4)
        d["r_hat"] = round(self.r_hat, 4)
        return d


@dataclass
class ScanResult:
    """End-to-end scan output."""

    opportunities: list[MarketOpportunity]
    n_discovered: int
    n_after_filter: int
    n_scored: int
    elapsed_seconds: float
    config: ScanConfig
    source: str = "unknown"      # "kalshi_api" | "pmxt" | "mock"

    def to_dict(self) -> dict:
        return {
            "n_discovered": self.n_discovered,
            "n_after_filter": self.n_after_filter,
            "n_scored": self.n_scored,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "source": self.source,
            "opportunities": [o.to_dict() for o in self.opportunities],
        }


# ── Discovery ──────────────────────────────────────────────────────────────


class KalshiDiscovery:
    """
    Pulls active Kalshi markets. Tries:
        1. Direct Kalshi public API (no auth required for read-only endpoints).
        2. pmxt unified client (if installed).
        3. A built-in mock universe (last-resort, keeps the scanner runnable).
    """

    def __init__(self, base_url: str | None = None, http_timeout: float = 10.0):
        self.base_url = (base_url or KALSHI_API_URL).rstrip("/")
        self.http_timeout = http_timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def fetch(self, max_markets: int = 200) -> tuple[list[UnifiedMarket], str]:
        """Return (markets, source_label)."""
        # Try direct API first — fastest path, no extra deps.
        markets = self._fetch_via_api(max_markets)
        if markets:
            return markets, "kalshi_api"

        # Then pmxt.
        markets = self._fetch_via_pmxt(max_markets)
        if markets:
            return markets, "pmxt"

        # Last resort — mock universe so demos & tests still produce output.
        logger.warning("Falling back to mock Kalshi markets (no network / no pmxt)")
        return self._mock_markets(), "mock"

    def _fetch_via_api(self, max_markets: int) -> list[UnifiedMarket]:
        url = f"{self.base_url}/markets"
        try:
            resp = self.session.get(
                url,
                params={"limit": min(max_markets, 200), "status": "open"},
                timeout=self.http_timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:                                  # noqa: BLE001
            logger.info(f"Kalshi direct API unavailable: {e}")
            return []

        raw_markets = payload.get("markets") if isinstance(payload, dict) else payload
        if not raw_markets:
            return []

        return [m for m in (self._normalize_kalshi(r) for r in raw_markets) if m is not None]

    def _fetch_via_pmxt(self, max_markets: int) -> list[UnifiedMarket]:
        try:
            from src.utils.pmxt_client import UnifiedPredictionClient, PMXT_AVAILABLE
        except ImportError:
            return []
        if not PMXT_AVAILABLE:
            return []
        try:
            client = UnifiedPredictionClient()
            if not client.connect("kalshi"):
                return []
            return client.search_markets(query="", exchanges=["kalshi"], max_results=max_markets)
        except Exception as e:                                  # noqa: BLE001
            logger.info(f"pmxt Kalshi fetch failed: {e}")
            return []

    @staticmethod
    def _normalize_kalshi(raw: dict) -> UnifiedMarket | None:
        try:
            ticker = raw.get("ticker") or raw.get("market_ticker") or raw.get("id")
            if not ticker:
                return None
            yes_price = _safe_cents_to_dollars(raw.get("yes_ask") or raw.get("yes_price"))
            no_price = _safe_cents_to_dollars(raw.get("no_ask") or raw.get("no_price"))
            if yes_price is None and no_price is not None:
                yes_price = max(0.0, min(1.0, 1.0 - no_price))
            if yes_price is None:
                yes_price = 0.5
            outcome_prices = [yes_price, max(0.0, min(1.0, 1.0 - yes_price))]
            return UnifiedMarket(
                exchange="kalshi",
                market_id=str(ticker),
                event_title=str(raw.get("event_ticker") or raw.get("category") or ticker),
                market_question=str(raw.get("title") or raw.get("subtitle") or raw.get("yes_sub_title") or ticker),
                outcomes=["Yes", "No"],
                outcome_prices=outcome_prices,
                volume_24h=float(raw.get("volume_24h") or raw.get("volume") or 0.0),
                liquidity=float(raw.get("liquidity") or raw.get("open_interest") or 0.0),
                end_date=str(raw.get("close_time") or raw.get("expiration_time") or ""),
                active=bool(raw.get("status", "open") == "open" or raw.get("active", True)),
                url=f"https://kalshi.com/markets/{ticker}",
                raw=raw,
            )
        except Exception as e:                                  # noqa: BLE001
            logger.debug(f"Kalshi normalize skipped: {e}")
            return None

    @staticmethod
    def _mock_markets() -> list[UnifiedMarket]:
        """A small, deterministic mock universe used when offline."""
        seeds = [
            ("FED-RATE-CUT-MAR26", "Will the Fed cut rates at the March 2026 meeting?",       0.62, 250_000, 38_000, 21),
            ("ELECTION-TURNOUT-26", "Will US midterm voter turnout exceed 50% in 2026?",       0.48, 180_000, 22_000, 45),
            ("BTC-100K-EOY",       "Will BTC close above $100k on Dec 31?",                     0.71, 410_000, 95_000, 12),
            ("RECESSION-Q3",       "Will the NBER declare a recession by Q3 2026?",            0.27,  90_000, 11_000, 35),
            ("CPI-BELOW-3",        "Will headline CPI come in below 3% next print?",           0.55, 140_000, 17_000,  4),
            ("OPENAI-IPO-2026",    "Will OpenAI file an S-1 in 2026?",                         0.18,  65_000,  6_000, 55),
            ("SUPER-BOWL-FAV",     "Will the favored team win the Super Bowl?",                0.58, 320_000, 48_000,  9),
            ("WEATHER-NYC-SNOW",   "Will NYC see >6in of snow this week?",                     0.34,  42_000,  4_500,  3),
            ("ECB-CUT-MAY",        "Will the ECB cut rates at its May meeting?",               0.66, 110_000, 13_000, 28),
            ("US-CHINA-TARIFF",    "Will US announce new China tariffs in Q2?",                0.41,  88_000,  9_800, 18),
            ("AAPL-EARNINGS-BEAT", "Will Apple beat consensus EPS next earnings?",             0.63, 215_000, 31_000,  6),
            ("OIL-OVER-90",        "Will WTI close above $90 by month end?",                   0.22,  55_000,  5_500, 14),
        ]
        now = datetime.now(timezone.utc).timestamp()
        out: list[UnifiedMarket] = []
        for ticker, q, yes, vol, liq, days in seeds:
            end = datetime.fromtimestamp(now + days * 86400, tz=timezone.utc).isoformat()
            out.append(UnifiedMarket(
                exchange="kalshi",
                market_id=ticker,
                event_title=ticker,
                market_question=q,
                outcomes=["Yes", "No"],
                outcome_prices=[yes, max(0.0, min(1.0, 1.0 - yes))],
                volume_24h=float(vol),
                liquidity=float(liq),
                end_date=end,
                active=True,
                url=f"https://kalshi.com/markets/{ticker}",
                raw={"mock": True},
            ))
        return out


# ── Scoring ────────────────────────────────────────────────────────────────


def _safe_cents_to_dollars(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # Kalshi historically returned prices in cents (0-100). pmxt returns
    # dollars (0-1). Treat anything > 1.5 as cents to be safe.
    return f / 100.0 if f > 1.5 else f


def _days_until(end_date_str: str) -> float:
    """Best-effort parse of Kalshi close-time strings -> days from now."""
    if not end_date_str:
        return float("nan")
    raw = end_date_str.replace("Z", "+00:00")
    try:
        end = datetime.fromisoformat(raw)
    except ValueError:
        # Try common alternative formats
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                end = datetime.strptime(end_date_str, fmt)
                break
            except ValueError:
                continue
        else:
            return float("nan")
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    delta = end - datetime.now(timezone.utc)
    return delta.total_seconds() / 86400.0


def _normalize_log(v: float, ref: float) -> float:
    if v <= 0 or ref <= 0:
        return 0.0
    return min(1.0, math.log1p(v) / math.log1p(ref))


def _time_value(days: float, sweet_spot: float, bandwidth: float) -> float:
    """
    Bell-curve preference for days-to-resolution.

    Too soon (< 1 day) → barely tradable; too far (> 60d) → capital tied up.
    Peaks at `sweet_spot` and decays with std `bandwidth`.
    """
    if math.isnan(days) or days < 0:
        return 0.0
    z = (days - sweet_spot) / max(bandwidth, 1e-6)
    return math.exp(-0.5 * z * z)


# ── Scanner ────────────────────────────────────────────────────────────────


class KalshiScanner:
    """
    End-to-end Kalshi market scanner.

    Usage:
        scanner = KalshiScanner()
        result = scanner.scan()
        for opp in result.opportunities:
            print(f"{opp.score:.3f}  {opp.direction:9s}  {opp.question}")

    For testing / dependency injection, you can pass a `discovery` callable
    that returns (list[UnifiedMarket], source_label), and / or an
    `engine_factory` that returns a SimulationEngine-shaped object.
    """

    def __init__(
        self,
        config: ScanConfig | None = None,
        discovery: Callable[[int], tuple[list[UnifiedMarket], str]] | None = None,
        engine_factory: Callable[[], Any] | None = None,
    ):
        self.config = config or ScanConfig()
        self._discovery = discovery
        self._engine_factory = engine_factory

    # ── Public API ─────────────────────────────────────────────────────────

    def scan(self) -> ScanResult:
        cfg = self.config
        start = datetime.now(timezone.utc).timestamp()

        markets, source = self._discover(max_markets=max(cfg.max_candidates * 5, 200))
        n_discovered = len(markets)

        filtered = self.filter_markets(markets)
        n_after_filter = len(filtered)

        # Cap candidates before running simulations (sim is the slow step).
        candidates = filtered[: cfg.max_candidates]

        opportunities = [self._score_market(m) for m in candidates]
        opportunities = [o for o in opportunities if o is not None]
        opportunities.sort(key=lambda o: o.score, reverse=True)
        top = opportunities[: cfg.top_n]

        elapsed = datetime.now(timezone.utc).timestamp() - start
        return ScanResult(
            opportunities=top,
            n_discovered=n_discovered,
            n_after_filter=n_after_filter,
            n_scored=len(opportunities),
            elapsed_seconds=elapsed,
            config=cfg,
            source=source,
        )

    def filter_markets(self, markets: list[UnifiedMarket]) -> list[UnifiedMarket]:
        cfg = self.config
        kept: list[UnifiedMarket] = []
        for m in markets:
            if not m.active:
                continue
            if m.liquidity < cfg.min_liquidity:
                continue
            if m.volume_24h < cfg.min_volume_24h:
                continue
            if not (cfg.min_yes_price <= m.yes_price <= cfg.max_yes_price):
                continue
            days = _days_until(m.end_date)
            if math.isnan(days):
                # Unknown close date — keep, but penalise via time_value=0 later.
                kept.append(m)
                continue
            if days < cfg.min_days_to_close or days > cfg.max_days_to_close:
                continue
            kept.append(m)

        # Sort by a cheap pre-score: liquidity * volume (we'll re-score after sim).
        kept.sort(key=lambda m: (m.liquidity + 1) * (m.volume_24h + 1), reverse=True)
        return kept

    def score_opportunity(
        self,
        market: UnifiedMarket,
        swarm: Any,
    ) -> MarketOpportunity:
        """Pure-function scoring — exposed so tests can verify the math."""
        cfg = self.config
        p_swarm = float(swarm.swarm_probability)
        p_market = float(market.yes_price)
        edge = p_swarm - p_market
        abs_edge = abs(edge)
        days = _days_until(market.end_date)

        f_edge = min(1.0, abs_edge / 0.20)              # 20% edge = max signal
        f_conv = 1.0 if swarm.converged else 0.0
        f_liq = _normalize_log(market.liquidity, cfg.liquidity_ref)
        f_vol = _normalize_log(market.volume_24h, cfg.volume_ref)
        f_time = _time_value(days, cfg.time_sweet_spot, cfg.time_bandwidth)
        f_conf = max(0.0, 1.0 - float(swarm.swarm_std) / 0.25)  # std 0.25 → 0

        weighted = (
            cfg.w_edge * f_edge
            + cfg.w_convergence * f_conv
            + cfg.w_liquidity * f_liq
            + cfg.w_volume * f_vol
            + cfg.w_time * f_time
            + cfg.w_confidence * f_conf
        )
        total_w = (
            cfg.w_edge + cfg.w_convergence + cfg.w_liquidity
            + cfg.w_volume + cfg.w_time + cfg.w_confidence
        )
        score = weighted / total_w if total_w > 0 else 0.0

        direction = self._direction_for(edge, swarm)

        return MarketOpportunity(
            market_id=market.market_id,
            ticker=market.market_id,
            question=market.market_question,
            yes_price=p_market,
            no_price=max(0.0, min(1.0, 1.0 - p_market)),
            liquidity=float(market.liquidity),
            volume_24h=float(market.volume_24h),
            days_to_close=days if not math.isnan(days) else -1.0,
            swarm_probability=p_swarm,
            swarm_std=float(swarm.swarm_std),
            r_hat=float(swarm.r_hat),
            converged=bool(swarm.converged),
            edge=edge,
            direction=direction,
            score=score,
            score_components={
                "edge": round(f_edge, 4),
                "convergence": round(f_conv, 4),
                "liquidity": round(f_liq, 4),
                "volume": round(f_vol, 4),
                "time": round(f_time, 4),
                "confidence": round(f_conf, 4),
            },
        )

    # ── Internals ──────────────────────────────────────────────────────────

    def _discover(self, max_markets: int) -> tuple[list[UnifiedMarket], str]:
        if self._discovery is not None:
            return self._discovery(max_markets)
        return KalshiDiscovery().fetch(max_markets=max_markets)

    def _make_engine(self):
        if self._engine_factory is not None:
            return self._engine_factory()
        SimulationEngine = _load_simulation_engine()
        return SimulationEngine(
            n_agents=self.config.n_agents,
            n_rounds=self.config.n_rounds,
            n_simulations=self.config.n_sims,
        )

    def _score_market(self, market: UnifiedMarket) -> MarketOpportunity | None:
        try:
            engine = self._make_engine()
            swarm = engine.run(
                event_description=market.market_question,
                context="",
                market_type="prediction",
                seed=self.config.seed,
            )
        except Exception as e:                                  # noqa: BLE001
            logger.warning(f"Sim failed for {market.market_id}: {e}")
            return None
        return self.score_opportunity(market, swarm)

    @staticmethod
    def _direction_for(edge: float, swarm: SwarmResult) -> str:
        if abs(edge) < 0.02 or not swarm.converged:
            return "no_trade"
        return "long_yes" if edge > 0 else "long_no"
