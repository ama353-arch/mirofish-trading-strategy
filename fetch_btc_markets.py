#!/usr/bin/env python3
"""
fetch_btc_markets.py — READ-ONLY: cache near-money hourly BTC market data.

For each settled KXBTCD ('BTC above strike') market, builds per-minute ticks of
{ts, spot, strike, sigma_annual, t_years, market_price, bid, ask} plus the
outcome. Spot from Coinbase (public), vol = realized from the prior hour (no
look-ahead). Cached to data/raw/btc_markets.json (gitignored) so OOS + profit
tests run offline. Read-only; never trades.
"""
from __future__ import annotations
import os, time, base64, subprocess, json, math, statistics as st
from datetime import datetime, timezone
from pathlib import Path
import requests

BASE="https://api.elections.kalshi.com"; OUT=Path("data/raw/btc_markets.json")
YEAR_MIN=365.25*24*60
kv={}
for line in open(".env"):
    line=line.strip()
    if line and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); kv[k.strip()]=v.strip()
KID=kv["KALSHI_API_KEY_ID"]; KEY=os.path.expanduser(kv["KALSHI_API_PRIVATE_KEY_PATH"])
def sign(m,p):
    ts=str(int(time.time()*1000)); msg=(ts+m+p).encode()
    pr=subprocess.run(["openssl","dgst","-sha256","-sign",KEY,"-sigopt","rsa_padding_mode:pss","-sigopt","rsa_pss_saltlen:digest"],input=msg,capture_output=True)
    return {"KALSHI-ACCESS-KEY":KID,"KALSHI-ACCESS-TIMESTAMP":ts,"KALSHI-ACCESS-SIGNATURE":base64.b64encode(pr.stdout).decode()}
def _get(url, **kw):
    """GET with up to 3 retries on transient network errors."""
    last = None
    for attempt in range(3):
        try:
            return requests.get(url, timeout=25, **kw)
        except requests.exceptions.RequestException as e:
            last = e; time.sleep(1.5 * (attempt + 1))
    raise last
def kget(path,**pa): return _get(BASE+path,headers=sign("GET",path),params=pa).json()
def iso2unix(s): return int(datetime.fromisoformat(s.replace("Z","+00:00")).timestamp())
def cb_candles(start,end):
    r=_get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
        params={"granularity":60,"start":datetime.fromtimestamp(start,timezone.utc).isoformat(),
                "end":datetime.fromtimestamp(end,timezone.utc).isoformat()},headers={"User-Agent":"mf"})
    c=r.json(); return sorted([(int(x[0]),float(x[4])) for x in c]) if isinstance(c,list) else []

def _close(cell): return (cell or {}).get("close_dollars")

def main(target=100):
    mkts=[]; cursor=None
    for _ in range(8):
        pa={"series_ticker":"KXBTCD","status":"settled","limit":1000}
        if cursor: pa["cursor"]=cursor
        d=kget("/trade-api/v2/markets",**pa); mkts+=d.get("markets",[]); cursor=d.get("cursor")
        if not cursor: break
    mkts=[m for m in mkts if m.get("result") in ("yes","no") and m.get("floor_strike") and m.get("strike_type")=="greater"]
    mkts.sort(key=lambda m: m.get("close_time",""))
    out=[]; failed=0
    for m in mkts:
        if len(out)>=target: break
        try:
            strike=float(m["floor_strike"]); close=iso2unix(m["close_time"]); opent=iso2unix(m["open_time"])
            btc=cb_candles(opent-3600, close)
            if len(btc)<70: continue
            spot_open=[p for t,p in btc if t>=opent][:1]
            if not spot_open or abs(strike-spot_open[0])/spot_open[0] > 0.015: continue
            pre=[p for t,p in btc if t<opent]
            if len(pre)<30: continue
            rets=[math.log(pre[i]/pre[i-1]) for i in range(1,len(pre))]
            sig=(st.pstdev(rets) or 1e-9)*math.sqrt(YEAR_MIN)
            kc=kget(f"/trade-api/v2/series/KXBTCD/markets/{m['ticker']}/candlesticks",
                    start_ts=opent,end_ts=close,period_interval=1)
            ticks=[]
            for x in kc.get("candlesticks",[]):
                ts=x["end_period_ts"]; price=_close(x.get("price"))
                bid=_close(x.get("yes_bid")); ask=_close(x.get("yes_ask"))
                if price is None: continue
                sp=[p for t,p in btc if t<=ts]
                if not sp: continue
                trem=(close-ts)/(YEAR_MIN*60)
                if trem<=0: continue
                ticks.append({"ts":ts,"spot":sp[-1],"strike":strike,"sigma":sig,"t_years":trem,
                              "price":float(price),
                              "bid":float(bid) if bid is not None else float(price),
                              "ask":float(ask) if ask is not None else float(price)})
            if len(ticks)>=10:
                out.append({"ticker":m["ticker"],"outcome":1.0 if m["result"]=="yes" else 0.0,"ticks":ticks})
            # checkpoint incrementally so a late failure never loses everything
            if len(out)%10==0 and out:
                OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out))
        except Exception:
            failed+=1
        time.sleep(0.12)
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out))
    print(f"Cached {len(out)} near-money BTC markets (skipped {failed}) -> {OUT}")

if __name__=="__main__":
    main()
