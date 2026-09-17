#!/usr/bin/env python3
"""
fetch_nba_games.py — READ-ONLY: cache real NBA in-game price trajectories.

Pulls settled KXNBAGAME markets, reconstructs each game's live win-price series
from 1-minute candlesticks, and caches them to data/raw/ (gitignored) so the
OOS backtest is reproducible without re-hitting the API. Read-only; never trades.
"""
from __future__ import annotations
import os, time, base64, subprocess, json
from datetime import datetime
from pathlib import Path
import requests

BASE = "https://api.elections.kalshi.com"
OUT = Path("data/raw/nba_games.json")

def _env():
    kv = {}
    for line in Path(".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); kv[k.strip()] = v.strip()
    return kv["KALSHI_API_KEY_ID"], os.path.expanduser(kv["KALSHI_API_PRIVATE_KEY_PATH"])

KID, KEYPATH = _env()

def sign(method, path):
    ts = str(int(time.time() * 1000)); msg = (ts + method + path).encode()
    p = subprocess.run(["openssl","dgst","-sha256","-sign",KEYPATH,
        "-sigopt","rsa_padding_mode:pss","-sigopt","rsa_pss_saltlen:digest"],
        input=msg, capture_output=True)
    return {"KALSHI-ACCESS-KEY":KID,"KALSHI-ACCESS-TIMESTAMP":ts,
            "KALSHI-ACCESS-SIGNATURE":base64.b64encode(p.stdout).decode()}

def get(path, **params):
    r = requests.get(BASE+path, headers=sign("GET",path), params=params, timeout=30)
    return r.status_code, r.json()

def iso2unix(s): return int(datetime.fromisoformat(s.replace("Z","+00:00")).timestamp())

def price_series(ticker, close_iso):
    end = iso2unix(close_iso); start = end - 5*3600
    code, c = get(f"/trade-api/v2/series/KXNBAGAME/markets/{ticker}/candlesticks",
                  start_ts=start, end_ts=end, period_interval=1)
    if code != 200:
        return []
    out = []
    for x in c.get("candlesticks", []):
        cd = (x.get("price") or {}).get("close_dollars")
        if cd is not None:
            out.append(round(float(cd), 4))
    return out

def main(target_games=60):
    by_event = {}
    cursor = None
    # paginate settled markets, keep the most liquid market per game event
    for _ in range(15):
        params = {"series_ticker":"KXNBAGAME","status":"settled","limit":200}
        if cursor: params["cursor"] = cursor
        code, d = get("/trade-api/v2/markets", **params)
        if code != 200: break
        for m in d.get("markets", []):
            ev = m.get("event_ticker")
            if m.get("result") in ("yes","no") and (
                ev not in by_event or m.get("volume_fp",0) > by_event[ev].get("volume_fp",0)):
                by_event[ev] = m
        cursor = d.get("cursor")
        if not cursor: break
    print(f"Found {len(by_event)} settled game-events")

    games = []
    for m in sorted(by_event.values(), key=lambda x: x.get("close_time",""))[:target_games*2]:
        s = price_series(m["ticker"], m["close_time"])
        contested = [p for p in s if 0.15 <= p <= 0.85]
        if len(contested) >= 30:                       # enough live action
            games.append({"ticker": m["ticker"], "close_time": m["close_time"],
                          "outcome": 1.0 if m["result"]=="yes" else 0.0,
                          "prices": s})
        time.sleep(0.15)
        if len(games) >= target_games: break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(games))
    print(f"Cached {len(games)} games with usable in-game price series -> {OUT}")

if __name__ == "__main__":
    main()
