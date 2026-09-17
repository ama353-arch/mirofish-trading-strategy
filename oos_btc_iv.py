#!/usr/bin/env python3
"""
oos_btc_iv.py — BTC pricer OOS with Deribit implied vol, multi-day sample.

Uses Deribit DVOL (implied vol) instead of realized vol, on a sample spread
across the date range (regime diversity). Reports both bounds:
  - TAKER  (cross the real bid/ask)  -> realistic lower bound
  - MAKER  (fill at mid, no adverse selection) -> optimistic upper bound
The truth is between. If even the optimistic maker bound is not significant,
the edge is not tradeable.
"""
from __future__ import annotations
import json, bisect, statistics as st
from pathlib import Path
import requests
from src.signals.pricing import prob_finish_above
from src.signals.walkforward import deflated_sharpe_ratio

MK = json.loads(Path("data/raw/btc_markets.json").read_text())
GRID = [0.03, 0.05, 0.08, 0.10]


def load_dvol(lo, hi):
    """Deribit DVOL (implied vol index), chunked over the time range."""
    pts = []
    cur = lo - 7200
    while cur < hi + 3600:
        end = min(cur + 1000 * 3600, hi + 3600)
        d = requests.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                         params={"currency": "BTC", "start_timestamp": cur * 1000,
                                 "end_timestamp": end * 1000, "resolution": 3600},
                         headers={"User-Agent": "mf"}, timeout=25).json()
        for x in d.get("result", {}).get("data", []):
            pts.append((x[0] // 1000, x[4] / 100.0))
        cur = end
    pts.sort()
    return [t for t, _ in pts], [v for _, v in pts]


ALL_TS = [tk["ts"] for m in MK for tk in m["ticks"]]
DTS, DVOL = load_dvol(min(ALL_TS), max(ALL_TS))


def iv_at(t):
    if not DTS:
        return None
    return DVOL[max(0, bisect.bisect_right(DTS, t) - 1)]


def market_pnl(m, edge_min, cross):
    for tk in m["ticks"]:
        if not (0.10 <= tk["price"] <= 0.90):
            continue
        sig = iv_at(tk["ts"]) or tk["sigma"]
        fv = prob_finish_above(tk["spot"], tk["strike"], sig, tk["t_years"])
        edge = fv - tk["price"]
        if abs(edge) < edge_min:
            continue
        side = "buy" if edge > 0 else "sell"
        o = m["outcome"]
        if cross:
            entry = tk["ask"] if side == "buy" else (1 - tk["bid"])
        else:
            entry = tk["price"] if side == "buy" else (1 - tk["price"])
        payoff = o if side == "buy" else (1 - o)
        if entry <= 0 or entry >= 1:
            continue
        return payoff / entry - 1
    return None


def oos(cross):
    n = len(MK); tr = max(20, n // 3); te = max(10, n // 6); step = te
    res = []
    for s in range(0, n - tr - te + 1, step):
        train, test = MK[s:s + tr], MK[s + tr:s + tr + te]
        def mean(e):
            r = [x for x in (market_pnl(m, e, cross) for m in train) if x is not None]
            return sum(r) / len(r) if r else -9
        best = max(GRID, key=mean)
        res += [x for x in (market_pnl(m, best, cross) for m in test) if x is not None]
    if len(res) < 8:
        return None
    m, sd = st.mean(res), (st.pstdev(res) or 1e-9)
    sh = m / sd * len(res) ** 0.5
    return len(res), m, sum(1 for x in res if x > 0) / len(res), sh, deflated_sharpe_ratio(sh, len(GRID), len(res))


def main():
    days = len({tk["ts"] // 86400 for m in MK for tk in m["ticks"]})
    print(f"BTC markets: {len(MK)} across ~{days} distinct days | DVOL points: {len(DTS)}")
    for label, cross in (("TAKER (cross spread, realistic)", True),
                         ("MAKER (mid, optimistic bound)", False)):
        r = oos(cross)
        if r:
            n, m, wr, sh, d = r
            print(f"  {label:34}: n={n} avgPnL/$ {m:+.4f} win {wr:.1%} Sharpe {sh:+.2f} DSR {d:.3f}")


if __name__ == "__main__":
    main()
