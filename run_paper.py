#!/usr/bin/env python3
"""
Run the live paper-trading harness against Kalshi's 15-minute crypto markets.

READ-ONLY and fake money. It books hypothetical fills at prices that were really
quoted, charges real Kalshi fees, and settles against the exchange's own result.
There is no code path from this process to a live order (see tests/test_paper.py).

    python3 run_paper.py                      # default series, 1 contract per signal
    python3 run_paper.py --series KXBTC15M --contracts 100
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.paper.live import PaperRunner, ReadOnlyKalshi, fifteen_min_crypto, open_markets

# BTC carries the unresolved 99c question; ADA/BCH/TON are newly listed and untested.
DEFAULT_SERIES = ["KXBTC15M", "KXETH15M", "KXXRP15M", "KXADA15M", "KXBCH15M", "KXTON15M"]
WATCH_FROM = 40.0  # start snapshotting quotes this many seconds before close


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="*", default=DEFAULT_SERIES)
    ap.add_argument("--contracts", type=float, default=1.0)
    ap.add_argument("--hours", type=float, default=0.0, help="0 = run until stopped")
    args = ap.parse_args()

    api = ReadOnlyKalshi()
    runner = PaperRunner(api, args.series, contracts=args.contracts)
    stop_at = time.time() + args.hours * 3600 if args.hours else None
    print(f"paper harness up — series={args.series} contracts={args.contracts} "
          f"(read-only; cannot place orders)", flush=True)

    markets: list[dict] = []
    last_refresh = last_beat = last_series_scan = 0.0
    last_poll: dict[str, float] = {}

    started = last_pass = time.time()
    asleep = 0.0
    while stop_at is None or time.time() < stop_at:
        now = time.time()
        if now - last_pass > 60:
            asleep += now - last_pass
            # The process froze — almost always the machine sleeping. Windows were missed.
            print(f"  [GAP] {now - last_pass:.0f}s with no polling — "
                  f"~{(now - last_pass) / 900:.1f} market windows missed", flush=True)
        last_pass = now
        if now - last_series_scan > 1800:
            try:
                found = fifteen_min_crypto(api.get("/trade-api/v2/series", category="Crypto"))
                fresh = [s for s in found if s not in args.series]
                if fresh:
                    args.series += fresh
                    runner.series = args.series
                    print(f"  NEW SERIES LISTED: {fresh}", flush=True)
            except Exception as e:
                print(f"  [warn] series scan: {type(e).__name__}", flush=True)
            last_series_scan = now

        if now - last_refresh > 60:
            markets = []
            for s in args.series:
                try:
                    markets += open_markets(api, s)
                except Exception as e:                      # a dead series must not kill the run
                    print(f"  [warn] {s}: {type(e).__name__}", flush=True)
            settled = runner.settle_pending()
            if settled:
                print(f"  settled {settled}", flush=True)
            last_refresh = now

        for m in markets:
            secs = m["close"] - now
            if not (0 < secs <= WATCH_FROM):
                continue
            if now - last_poll.get(m["ticker"], 0) < 1.0:
                continue
            last_poll[m["ticker"]] = now
            try:
                if runner.check_window(m, window=(2, 15)):
                    t = runner.ledger.trades[-1]
                    print(f"  PAPER BUY {t.side} {t.ticker} @ {t.price:.4f} "
                          f"x{t.contracts:g} ({t.meta['secs_to_close']}s left)", flush=True)
            except Exception as e:
                print(f"  [warn] {m['ticker']}: {type(e).__name__}", flush=True)

        if now - last_beat > 900:
            s = runner.stats_line()
            elapsed = now - started
            duty = 100 * (elapsed - asleep) / elapsed if elapsed else 100
            print(f"{datetime.now(timezone.utc):%H:%M:%S} {s} | awake {duty:.0f}% of {elapsed/3600:.1f}h",
                  flush=True)
            last_beat = now
        time.sleep(0.4)


if __name__ == "__main__":
    main()
