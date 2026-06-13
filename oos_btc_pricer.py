#!/usr/bin/env python3
"""
oos_btc_pricer.py — does the BTC pricer edge survive OOS + the spread?

Calibration says the pricer is slightly better than the market. This asks the
harder question: trading the pricer's edge, crossing the real bid/ask, does it
make money OUT OF SAMPLE?

Rigor:
  - ONE trade per market (enter at the first signal, hold to settlement) so each
    market is an independent observation — no autocorrelation inflation.
  - Walk-forward: tune edge_min on train markets, score on held-out markets.
  - Deflated Sharpe corrects for the grid search.
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
from src.signals.pricing import make_pricing_brain
from src.signals.walkforward import _simulate_trade, deflated_sharpe_ratio

MK = json.loads(Path("data/raw/btc_markets.json").read_text())
GRID = [0.03, 0.05, 0.08, 0.10]
brain = make_pricing_brain()


def market_pnl(market, edge_min):
    """One trade per market: first signal in the contested band, held to settle."""
    for tk in market["ticks"]:
        if not (0.10 <= tk["price"] <= 0.90):
            continue
        ev = {"spot": tk["spot"], "strike": tk["strike"], "sigma_annual": tk["sigma"],
              "t_years": tk["t_years"], "market_prob": tk["price"]}
        sig = brain(ev, {"edge_min": edge_min})
        if sig["side"] is None:
            continue
        trade = _simulate_trade(
            {"id": market["ticker"], "timestamp": tk["ts"], "market_prob": tk["price"],
             "bid": tk["bid"], "ask": tk["ask"], "actual_outcome": market["outcome"]},
            sig["side"], size_pct=1.0)
        return trade["pnl_per_dollar"] if trade else None
    return None


def score(markets, edge_min):
    return [p for p in (market_pnl(m, edge_min) for m in markets) if p is not None]


def main():
    n = len(MK)
    train_n, test_n, step = max(20, n // 3), max(10, n // 6), max(10, n // 6)
    oos, chosen = [], []
    for start in range(0, n - train_n - test_n + 1, step):
        train = MK[start:start + train_n]
        test = MK[start + train_n:start + train_n + test_n]
        best = max(GRID, key=lambda e: (lambda r: sum(r) / len(r) if r else -9)(score(train, e)))
        chosen.append(best)
        oos.extend(score(test, best))

    print(f"BTC near-money markets: {n} | walk-forward (train {train_n}, test {test_n})")
    print(f"edge_min chosen per fold: {chosen}")
    if len(oos) < 10:
        print("Too few OOS trades."); return
    m, sd = st.mean(oos), (st.pstdev(oos) or 1e-9)
    sharpe = m / sd * (len(oos) ** 0.5)
    dsr = deflated_sharpe_ratio(sharpe, n_trials=len(GRID), n_obs=len(oos))
    wins = sum(1 for x in oos if x > 0) / len(oos)
    print("\n=== OUT-OF-SAMPLE, ONE TRADE/MARKET, SPREAD CROSSED ===")
    print(f"  OOS trades (markets): {len(oos)}")
    print(f"  Avg PnL/$:            {m:+.4f}")
    print(f"  Win rate:             {wins:.1%}")
    print(f"  Sharpe (per trade):   {sharpe:.2f}")
    print(f"  Deflated Sharpe:      {dsr:.3f}  "
          f"({'REAL edge' if dsr > 0.95 else 'not convincing'} after {len(GRID)} trials)")
    print("\n  Each trade crosses the real bid/ask, so this is net of spread.")
    print("  Vol input is realized (prior hour), not Deribit IV — a better vol")
    print("  estimate could only help.")


if __name__ == "__main__":
    main()
