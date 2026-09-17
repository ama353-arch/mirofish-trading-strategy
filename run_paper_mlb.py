#!/usr/bin/env python3
"""
Paper-trade the frozen MLB in-game model against live Kalshi markets.

READ-ONLY and fake money. Game state comes from ESPN play-by-play, prices and depth from one Kalshi
order-book snapshot, fees are the real ones, and every position settles against the exchange's result.

The threshold is declared here, in advance, and not tuned: the backtest was positive at every
threshold, so picking the best one after the fact would be fooling ourselves. Every observation is
logged so the whole threshold curve can be checked later without re-running anything.

    python3 run_paper_mlb.py --contracts 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.paper.ledger import PaperLedger, PaperTrade
from src.paper.live import ReadOnlyKalshi, book_depth, cap_to_depth, quote_from_book
from src.paper.mlb_live import (game_state, has_game_state, matchup_key, parse_ticker,
                                 scoreboard)
from src.paper.mlb_model import MLBModel
from src.paper.rules import model_disagreement

THRESHOLD = 0.05          # declared in advance, never tuned
OUT = Path(os.environ.get("PAPER_DATA_DIR") or (Path(__file__).resolve().parent / "data" / "paper"))


def open_markets(api: ReadOnlyKalshi) -> list[dict]:
    d = api.get("/trade-api/v2/markets", series_ticker="KXMLBGAME", status="open", limit=200)
    out = []
    for m in d.get("markets", []) or []:
        t = parse_ticker(m.get("ticker", ""))
        if t:
            out.append({"ticker": m["ticker"], "parsed": t})
    return out


def pregame_price(api: ReadOnlyKalshi, ticker: str, cache: dict) -> float | None:
    """The market's earliest quoted price — the crowd's view before the game told us anything."""
    if ticker in cache:
        return cache[ticker]
    m = api.get(f"/trade-api/v2/markets/{ticker}").get("market", {})
    try:
        close_ts = int(datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp())
    except (KeyError, TypeError, ValueError):
        return None
    # The backtest anchored on the earliest quote in the market's final 4 hours — i.e. near first
    # pitch. Anchoring at market open instead (days earlier) is a different, staler number and would
    # make staleness look like model insight.
    # A live game's close_time is in the FUTURE, so close-4h can land after now and the API rejects
    # the range. Anchor on the last four hours up to now, which for a game in progress reaches back
    # before first pitch — the same intent as the backtest's anchor.
    now_ts = int(time.time())
    # The backtest anchored on the earliest quote in the market's final 4 hours, which lands near
    # first pitch. Only fall back to a now-relative window if that range has not opened yet.
    start = close_ts - 4 * 3600
    if start >= now_ts:
        start = now_ts - 4 * 3600
    cs = []
    for interval in (1, 60):
        try:
            cs = api.get(f"/trade-api/v2/series/KXMLBGAME/markets/{ticker}/candlesticks",
                         start_ts=start, end_ts=now_ts, period_interval=interval).get("candlesticks", [])
            if cs:
                break
        except Exception:
            continue
    for x in cs:
        b = (x.get("yes_bid") or {}).get("close_dollars")
        a = (x.get("yes_ask") or {}).get("close_dollars")
        if b is None or a is None:
            continue
        b, a = float(b), float(a)
        if 0 < b <= a < 1:
            cache[ticker] = (a + b) / 2
            return cache[ticker]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", type=float, default=100.0)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    api = ReadOnlyKalshi()
    model = MLBModel.load()
    OUT.mkdir(parents=True, exist_ok=True)
    ledger, traded, pending, anchors = PaperLedger(), set(), {}, {}
    lf = OUT / "mlb_ledger.json"
    if lf.exists():
        for row in json.loads(lf.read_text()):
            row.pop("fee", None)
            ledger.add(PaperTrade(**row))
        traded = {t.ticker.rsplit("-", 1)[0] for t in ledger.trades}
        pending = {t.ticker: 1 for t in ledger.trades if t.result is None}
    print(f"MLB paper harness up — model fitted on {model.fitted_games} games, "
          f"threshold {args.threshold*100:.0f}c, {args.contracts:g} contracts "
          f"(read-only; cannot place orders)", flush=True)

    boards, last_board, last_beat = {}, 0.0, 0.0
    while True:
        now = time.time()
        if now - last_board > 300:
            boards = {}
            for d in (datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(days=1)):
                day = d.strftime("%Y%m%d")
                for k, v in scoreboard(day).items():
                    boards[(day, k)] = v
            last_board = now

        try:
            markets = open_markets(api)
        except Exception as e:
            print(f"  [warn] markets: {type(e).__name__}", flush=True)
            markets = []

        for mk in markets:
            tk, p = mk["ticker"], mk["parsed"]
            if tk.rsplit("-", 1)[0] in traded:
                continue   # a game lists one market per team; both sides are the same bet
            g = boards.get((p.date, matchup_key(p.matchup)))
            if not g or g.get("state") != "in":
                continue
            try:
                st = game_state(g["espn_id"])
                if not st:
                    continue
                inning, is_bottom, away, home, outs, bases = st
                pre = pregame_price(api, tk, anchors)
                if pre is None:
                    continue
                book = api.get(f"/trade-api/v2/markets/{tk}/orderbook")
                q = quote_from_book(book, tk, time.time())
                if not q.valid:
                    continue
                mp = model.win_probability(pre, p.yes_team == g["home"], inning, is_bottom,
                                           home, away, bases, outs)
                with (OUT / "mlb_observations.jsonl").open("a") as fo:
                    fo.write(json.dumps({"ts": now, "ticker": tk, "model": mp, "bid": q.yes_bid,
                                         "ask": q.yes_ask, "pre": pre, "inning": inning,
                                         "bot": is_bottom, "away": away, "home": home,
                                         "outs": outs, "bases": bases}) + "\n")
                if not has_game_state(inning, away, home):
                    continue   # no game state yet: any gap is anchor drift, and it loses historically
                order = model_disagreement(q, mp, args.threshold)
                if order is None:
                    continue
                at, better = book_depth(book, order.side, order.price)
                filled = cap_to_depth(args.contracts, better)
                if filled <= 0:
                    continue
                ledger.add(PaperTrade(
                    ticker=tk, series="KXMLBGAME", decision_ts=now, side=order.side,
                    price=order.price, contracts=filled, rule=order.rule,
                    meta={"model": round(mp, 4), "pregame": round(pre, 4), "bid": q.yes_bid,
                          "ask": q.yes_ask, "inning": inning, "bottom": is_bottom,
                          "score": f"{away}-{home}", "outs": outs, "bases": bases,
                          "depth": better, "requested": args.contracts}))
                traded.add(tk.rsplit("-", 1)[0]); pending[tk] = 1
                lf.write_text(json.dumps([t.__dict__ for t in ledger.trades], indent=1))
                print(f"  PAPER BUY {order.side} {tk} @ {order.price:.3f} x{filled:g} | "
                      f"model {mp:.3f} vs {q.yes_bid:.2f}/{q.yes_ask:.2f} | "
                      f"inn {inning}{'B' if is_bottom else 'T'} {away}-{home}", flush=True)
            except Exception as e:
                print(f"  [warn] {tk}: {type(e).__name__}", flush=True)

        for tk in list(pending):
            try:
                m = api.get(f"/trade-api/v2/markets/{tk}").get("market", {})
                r = m.get("result")
                if r in ("yes", "no"):
                    ledger.settle(tk, 1 if r == "yes" else 0)
                    pending.pop(tk)
                    lf.write_text(json.dumps([t.__dict__ for t in ledger.trades], indent=1))
            except Exception:
                pass

        if now - last_beat > 900:
            s = ledger.stats()
            live = sum(1 for mk in markets
                       if (boards.get((mk["parsed"].date, matchup_key(mk["parsed"].matchup))) or {}).get("state") == "in")
            print(f"{datetime.now(timezone.utc):%H:%M:%S} games live {live} | open {s['open']} "
                  f"settled {s['settled']} | net ${s['net']:.2f}"
                  + (f" ({s['net_per_contract']*100:+.2f}c/contract)" if s['settled'] else ""), flush=True)
            last_beat = now
        time.sleep(20)


if __name__ == "__main__":
    main()
