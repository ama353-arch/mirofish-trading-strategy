"""
Tests for the Kalshi market scanner.

Covers:
  - Filtering rules (liquidity, volume, price range, time to close)
  - Scoring math (each factor, weight respect, monotonicity)
  - Time-value bell curve
  - Discovery fallback chain (api → pmxt → mock)
  - End-to-end scan with injected discovery + injected engine
  - Direction logic (long_yes / long_no / no_trade)
  - Edge cases (empty universe, unknown close dates, all filtered out)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scanner.kalshi_scanner import (
    KalshiDiscovery,
    KalshiScanner,
    MarketOpportunity,
    ScanConfig,
    ScanResult,
    UnifiedMarket,
    _days_until,
    _normalize_log,
    _safe_cents_to_dollars,
    _time_value,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _iso_days_from_now(days: float) -> str:
    when = datetime.now(timezone.utc) + timedelta(days=days)
    return when.isoformat()


def _make_market(
    ticker: str = "MKT-1",
    question: str = "Will X happen?",
    yes_price: float = 0.5,
    volume: float = 50_000.0,
    liquidity: float = 10_000.0,
    days: float = 7.0,
    active: bool = True,
) -> UnifiedMarket:
    return UnifiedMarket(
        exchange="kalshi",
        market_id=ticker,
        event_title=ticker,
        market_question=question,
        outcomes=["Yes", "No"],
        outcome_prices=[yes_price, max(0.0, min(1.0, 1.0 - yes_price))],
        volume_24h=volume,
        liquidity=liquidity,
        end_date=_iso_days_from_now(days),
        active=active,
        url=f"https://kalshi.com/{ticker}",
    )


@dataclass
class FakeSwarmResult:
    """Minimal SwarmResult stand-in — exposes only the fields the scanner reads."""
    swarm_probability: float = 0.5
    swarm_std: float = 0.05
    swarm_median: float = 0.5
    r_hat: float = 1.0
    effective_sample_size: float = 250.0
    converged: bool = True
    bullish_fraction: float = 0.5
    mean_conviction: float = 7.0
    event_description: str = ""
    simulation_results: list = field(default_factory=list)
    archetype_probabilities: dict = field(default_factory=dict)


class FakeEngine:
    """Stand-in for SimulationEngine that returns a configurable swarm result."""

    def __init__(self, result_fn):
        self._result_fn = result_fn
        self.calls: list[dict] = []

    def run(self, event_description, context="", market_type="prediction", seed=None, **_):
        self.calls.append({"event": event_description, "seed": seed})
        return self._result_fn(event_description)


# ── _safe_cents_to_dollars ────────────────────────────────────────────────


def test_safe_cents_handles_none():
    assert _safe_cents_to_dollars(None) is None


def test_safe_cents_treats_above_1_5_as_cents():
    assert _safe_cents_to_dollars(65) == pytest.approx(0.65)
    assert _safe_cents_to_dollars(100) == pytest.approx(1.0)


def test_safe_cents_treats_below_1_5_as_dollars():
    assert _safe_cents_to_dollars(0.55) == pytest.approx(0.55)
    assert _safe_cents_to_dollars(1.0) == pytest.approx(1.0)


def test_safe_cents_rejects_garbage():
    assert _safe_cents_to_dollars("not-a-number") is None


# ── _days_until ───────────────────────────────────────────────────────────


def test_days_until_future_iso():
    when = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
    assert _days_until(when) == pytest.approx(10.0, abs=0.05)


def test_days_until_past_returns_negative():
    when = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    assert _days_until(when) < 0


def test_days_until_empty_is_nan():
    assert math.isnan(_days_until(""))


def test_days_until_garbage_is_nan():
    assert math.isnan(_days_until("not-a-date"))


# ── _normalize_log ────────────────────────────────────────────────────────


def test_normalize_log_zero():
    assert _normalize_log(0, 1000) == 0.0


def test_normalize_log_monotonic():
    a = _normalize_log(1_000, 50_000)
    b = _normalize_log(10_000, 50_000)
    c = _normalize_log(100_000, 50_000)
    assert a < b < c


def test_normalize_log_caps_at_one():
    assert _normalize_log(10_000_000, 50_000) == 1.0


# ── _time_value (bell curve) ──────────────────────────────────────────────


def test_time_value_peaks_at_sweet_spot():
    cfg = ScanConfig()
    peak = _time_value(cfg.time_sweet_spot, cfg.time_sweet_spot, cfg.time_bandwidth)
    off = _time_value(cfg.time_sweet_spot + 30, cfg.time_sweet_spot, cfg.time_bandwidth)
    assert peak == pytest.approx(1.0)
    assert off < peak


def test_time_value_symmetric_around_peak():
    cfg = ScanConfig()
    left = _time_value(cfg.time_sweet_spot - 5, cfg.time_sweet_spot, cfg.time_bandwidth)
    right = _time_value(cfg.time_sweet_spot + 5, cfg.time_sweet_spot, cfg.time_bandwidth)
    assert left == pytest.approx(right)


def test_time_value_handles_nan():
    assert _time_value(float("nan"), 7, 14) == 0.0


# ── filter_markets ────────────────────────────────────────────────────────


def test_filter_drops_low_liquidity():
    scanner = KalshiScanner(config=ScanConfig(min_liquidity=5_000))
    universe = [
        _make_market("A", liquidity=100.0),     # too thin
        _make_market("B", liquidity=20_000.0),
    ]
    kept = scanner.filter_markets(universe)
    assert [m.market_id for m in kept] == ["B"]


def test_filter_drops_low_volume():
    scanner = KalshiScanner(config=ScanConfig(min_volume_24h=10_000))
    universe = [
        _make_market("A", volume=100.0),
        _make_market("B", volume=50_000.0),
    ]
    kept = scanner.filter_markets(universe)
    assert [m.market_id for m in kept] == ["B"]


def test_filter_respects_price_band():
    cfg = ScanConfig(min_yes_price=0.10, max_yes_price=0.90)
    scanner = KalshiScanner(config=cfg)
    universe = [
        _make_market("EDGE-LOW", yes_price=0.02),
        _make_market("EDGE-HIGH", yes_price=0.98),
        _make_market("MIDDLE", yes_price=0.55),
    ]
    kept = scanner.filter_markets(universe)
    assert [m.market_id for m in kept] == ["MIDDLE"]


def test_filter_respects_time_window():
    cfg = ScanConfig(min_days_to_close=1.0, max_days_to_close=30.0)
    scanner = KalshiScanner(config=cfg)
    universe = [
        _make_market("TOO-SOON", days=0.1),
        _make_market("OK", days=14.0),
        _make_market("TOO-FAR", days=120.0),
    ]
    kept = scanner.filter_markets(universe)
    assert [m.market_id for m in kept] == ["OK"]


def test_filter_drops_inactive_markets():
    scanner = KalshiScanner()
    kept = scanner.filter_markets([_make_market("DEAD", active=False)])
    assert kept == []


def test_filter_keeps_unknown_close_date():
    """Unknown close date shouldn't auto-drop — let scoring penalise it."""
    scanner = KalshiScanner()
    m = _make_market("NO-DATE")
    m.end_date = ""
    kept = scanner.filter_markets([m])
    assert len(kept) == 1


def test_filter_sorts_by_pre_score():
    scanner = KalshiScanner()
    universe = [
        _make_market("LOW", volume=5_000, liquidity=2_000),
        _make_market("HIGH", volume=200_000, liquidity=40_000),
        _make_market("MID", volume=50_000, liquidity=10_000),
    ]
    kept = scanner.filter_markets(universe)
    assert [m.market_id for m in kept] == ["HIGH", "MID", "LOW"]


# ── score_opportunity ────────────────────────────────────────────────────


def test_score_edge_drives_long_yes():
    scanner = KalshiScanner()
    market = _make_market(yes_price=0.40)
    swarm = FakeSwarmResult(swarm_probability=0.70, swarm_std=0.04, converged=True)
    opp = scanner.score_opportunity(market, swarm)
    assert opp.direction == "long_yes"
    assert opp.edge == pytest.approx(0.30)
    assert opp.score > 0


def test_score_edge_drives_long_no():
    scanner = KalshiScanner()
    market = _make_market(yes_price=0.80)
    swarm = FakeSwarmResult(swarm_probability=0.40, swarm_std=0.04, converged=True)
    opp = scanner.score_opportunity(market, swarm)
    assert opp.direction == "long_no"
    assert opp.edge < 0


def test_score_no_trade_when_edge_tiny():
    scanner = KalshiScanner()
    market = _make_market(yes_price=0.50)
    swarm = FakeSwarmResult(swarm_probability=0.505, converged=True)
    opp = scanner.score_opportunity(market, swarm)
    assert opp.direction == "no_trade"


def test_score_no_trade_when_unconverged():
    scanner = KalshiScanner()
    market = _make_market(yes_price=0.50)
    swarm = FakeSwarmResult(swarm_probability=0.80, converged=False, r_hat=1.5)
    opp = scanner.score_opportunity(market, swarm)
    assert opp.direction == "no_trade"


def test_score_higher_edge_higher_score():
    scanner = KalshiScanner()
    market = _make_market(yes_price=0.50)
    small = scanner.score_opportunity(
        market, FakeSwarmResult(swarm_probability=0.55, converged=True)
    )
    large = scanner.score_opportunity(
        market, FakeSwarmResult(swarm_probability=0.85, converged=True)
    )
    assert large.score > small.score


def test_score_components_in_unit_interval():
    scanner = KalshiScanner()
    market = _make_market()
    opp = scanner.score_opportunity(market, FakeSwarmResult(swarm_probability=0.60, converged=True))
    for k, v in opp.score_components.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"
    assert 0.0 <= opp.score <= 1.0


def test_score_unconverged_zeros_convergence_component():
    scanner = KalshiScanner()
    opp = scanner.score_opportunity(
        _make_market(),
        FakeSwarmResult(swarm_probability=0.60, converged=False),
    )
    assert opp.score_components["convergence"] == 0.0


# ── End-to-end scan with injected discovery + engine ──────────────────────


def _make_scanner_with(markets, swarm_fn=None, cfg=None):
    swarm_fn = swarm_fn or (lambda q: FakeSwarmResult(swarm_probability=0.70, converged=True))
    engine = FakeEngine(swarm_fn)
    return KalshiScanner(
        config=cfg or ScanConfig(top_n=3, n_agents=10, n_rounds=5, n_sims=2),
        discovery=lambda _n: (markets, "injected"),
        engine_factory=lambda: engine,
    ), engine


def test_scan_returns_top_n_sorted_descending():
    markets = [
        _make_market(f"M-{i}", yes_price=0.50, volume=50_000, liquidity=20_000)
        for i in range(5)
    ]
    # Wider edges → higher scores; we want the top 3 to be the ones we make most extreme.
    def swarm_fn(q):
        # M-4 strongest, M-0 weakest
        idx = int(q.split()[-1].rstrip("?")) if q.split()[-1].rstrip("?").isdigit() else 0
        prob = 0.50 + 0.05 * (idx + 1)
        return FakeSwarmResult(swarm_probability=prob, converged=True)

    scanner, _ = _make_scanner_with(markets, swarm_fn)
    result = scanner.scan()
    assert len(result.opportunities) == 3
    scores = [o.score for o in result.opportunities]
    assert scores == sorted(scores, reverse=True)


def test_scan_calls_engine_once_per_candidate():
    markets = [_make_market(f"M-{i}") for i in range(4)]
    scanner, engine = _make_scanner_with(markets)
    scanner.scan()
    assert len(engine.calls) == 4


def test_scan_propagates_source_label():
    scanner = KalshiScanner(
        config=ScanConfig(top_n=2, n_agents=10, n_rounds=5, n_sims=2),
        discovery=lambda _n: ([_make_market("X")], "mock"),
        engine_factory=lambda: FakeEngine(
            lambda q: FakeSwarmResult(swarm_probability=0.70, converged=True)
        ),
    )
    result = scanner.scan()
    assert result.source == "mock"


def test_scan_empty_universe():
    scanner, _ = _make_scanner_with([])
    result = scanner.scan()
    assert result.opportunities == []
    assert result.n_discovered == 0
    assert result.n_after_filter == 0


def test_scan_all_markets_filtered_out():
    # All sub-threshold liquidity → nothing survives filter
    markets = [_make_market(f"M-{i}", liquidity=10.0) for i in range(5)]
    scanner, engine = _make_scanner_with(
        markets, cfg=ScanConfig(min_liquidity=1_000, top_n=3, n_agents=10, n_rounds=5, n_sims=2)
    )
    result = scanner.scan()
    assert result.opportunities == []
    assert result.n_discovered == 5
    assert result.n_after_filter == 0
    assert engine.calls == []


def test_scan_handles_simulation_failure_gracefully():
    """If one market's simulation throws, the scan should keep going."""
    markets = [_make_market(f"M-{i}") for i in range(3)]

    failing_engine_factory_state = {"calls": 0}

    class FlakyEngine:
        def run(self, event_description, **_):
            failing_engine_factory_state["calls"] += 1
            if failing_engine_factory_state["calls"] == 2:
                raise RuntimeError("boom")
            return FakeSwarmResult(swarm_probability=0.70, converged=True)

    scanner = KalshiScanner(
        config=ScanConfig(top_n=5, n_agents=10, n_rounds=5, n_sims=2),
        discovery=lambda _n: (markets, "injected"),
        engine_factory=lambda: FlakyEngine(),
    )
    result = scanner.scan()
    # 3 markets in, 2 successful sims out (1 raised), all surviving sort to top.
    assert result.n_scored == 2
    assert len(result.opportunities) == 2


def test_scan_result_to_dict_serializable():
    """Verify ScanResult.to_dict produces JSON-safe output."""
    import json
    scanner, _ = _make_scanner_with([_make_market("X")])
    result = scanner.scan()
    payload = result.to_dict()
    # Should not raise.
    json.dumps(payload)
    assert payload["source"] == "injected"
    assert "opportunities" in payload


# ── Discovery (mock fallback) ─────────────────────────────────────────────


def test_mock_universe_nonempty_and_well_formed():
    markets = KalshiDiscovery._mock_markets()
    assert len(markets) >= 10
    for m in markets:
        assert m.exchange == "kalshi"
        assert 0.0 <= m.yes_price <= 1.0
        assert m.liquidity > 0
        assert m.volume_24h > 0
        assert m.active is True
        assert m.end_date  # non-empty ISO string


def test_discovery_fetch_falls_back_to_mock_when_offline(monkeypatch):
    """When direct API + pmxt both fail, fetch returns the mock universe."""
    disc = KalshiDiscovery()
    monkeypatch.setattr(disc, "_fetch_via_api", lambda max_markets: [])
    monkeypatch.setattr(disc, "_fetch_via_pmxt", lambda max_markets: [])
    markets, source = disc.fetch(max_markets=50)
    assert source == "mock"
    assert len(markets) > 0


def test_discovery_prefers_api_when_available(monkeypatch):
    disc = KalshiDiscovery()
    fake_markets = [_make_market("API-1")]
    monkeypatch.setattr(disc, "_fetch_via_api", lambda max_markets: fake_markets)
    monkeypatch.setattr(disc, "_fetch_via_pmxt", lambda max_markets: [_make_market("PMXT-1")])
    markets, source = disc.fetch(max_markets=50)
    assert source == "kalshi_api"
    assert [m.market_id for m in markets] == ["API-1"]


def test_normalize_kalshi_handles_cents_pricing():
    raw = {
        "ticker": "FOO",
        "title": "Will foo?",
        "yes_ask": 65,        # cents
        "no_ask": 35,
        "volume_24h": 1000,
        "liquidity": 500,
        "close_time": _iso_days_from_now(5),
        "status": "open",
    }
    m = KalshiDiscovery._normalize_kalshi(raw)
    assert m is not None
    assert m.yes_price == pytest.approx(0.65)
    assert m.market_question == "Will foo?"


def test_normalize_kalshi_skips_when_no_ticker():
    assert KalshiDiscovery._normalize_kalshi({"title": "orphan"}) is None
