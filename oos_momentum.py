#!/usr/bin/env python3
"""
oos_momentum.py — out-of-sample validation of the momentum thesis on real games.

Walk-forward across games: tune (thesis, window, threshold) on TRAIN games, score
on the held-out TEST games, roll. Reports OOS performance with a deflated Sharpe
that corrects for how many parameter combos were tried. This is the test that
tells us if the in-sample FOLLOW>FADE result is real or a mirage.
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
import numpy as np

from src.sports.momentum import backtest_price_momentum
from src.signals.walkforward import deflated_sharpe_ratio

GAMES = json.loads(Path("data/raw/nba_games.json").read_text())

GRID = [
    {"thesis": th, "window": w, "momentum_threshold": t}
    for th in ("fade", "follow")
    for w in (5, 10, 15)
    for t in (0.03, 0.05, 0.08)
]
FILTER = {"min_price": 0.15, "max_price": 0.85}


def trade_returns(games, params):
    out = []
    for g in games:
        for tr in backtest_price_momentum(g["prices"], g["outcome"], **params, **FILTER):
            out.append(tr["pnl_per_dollar"])
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    n = len(GAMES)
    train_n, test_n, step = 24, 12, 12
    oos = []
    chosen = []
    for start in range(0, n - train_n - test_n + 1, step):
        train = GAMES[start:start + train_n]
        test = GAMES[start + train_n:start + train_n + test_n]
        best = max(GRID, key=lambda p: mean(trade_returns(train, p)))
        chosen.append((best["thesis"], best["window"], best["momentum_threshold"]))
        oos.extend(trade_returns(test, best))

    print(f"Games: {n} | walk-forward folds tuned on {train_n}, scored on {test_n}")
    print(f"Params chosen per fold (OOS): {chosen}")
    if not oos:
        print("No OOS trades."); return
    m, sd = mean(oos), (st.pstdev(oos) or 1e-9)
    sharpe = m / sd * (len(oos) ** 0.5)
    dsr = deflated_sharpe_ratio(sharpe, n_trials=len(GRID), n_obs=len(oos))
    wins = sum(1 for x in oos if x > 0) / len(oos)
    print("\n=== OUT-OF-SAMPLE (held-out games only) ===")
    print(f"  OOS trades:      {len(oos)}")
    print(f"  Avg PnL/$:       {m:+.4f}")
    print(f"  Win rate:        {wins:.1%}")
    print(f"  OOS Sharpe:      {sharpe:.2f}")
    print(f"  Deflated Sharpe: {dsr:.3f}  "
          f"({'REAL — survives' if dsr > 0.95 else 'NOT convincing'} after {len(GRID)} trials)")
    print("\n  Reminder: even a positive OOS number here is partly mechanical —")
    print("  a rising win-price rises because the team is winning. Step 3 (the")
    print("  short-horizon overshoot test) isolates true market inefficiency.")


if __name__ == "__main__":
    main()
