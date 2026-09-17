#!/usr/bin/env python3
"""
fetch_aligned_games.py — READ-ONLY: align ESPN win-probability with Kalshi prices.

For each cached NBA game it joins three things on a common wall-clock timeline:
  - ESPN's calibrated win-probability (the fair-value anchor),
  - our diffusion model_wp (a sanity second opinion),
  - the Kalshi live market price.

Caches per-game aligned series to data/raw/nba_aligned.json (gitignored) so the
calibration + divergence backtest run without re-hitting the network. Read-only.
"""
from __future__ import annotations
import os, time, base64, subprocess, json, bisect
from datetime import datetime
from pathlib import Path
import requests
from src.sports.win_probability import nba_win_probability

BASE="https://api.elections.kalshi.com"; HDR={"User-Agent":"Mozilla/5.0"}
OUT=Path("data/raw/nba_aligned.json")
MONTHS={"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
ALIAS={"NYK":"NY","GSW":"GS","NOP":"NO","SAS":"SA","UTA":"UTAH","WAS":"WSH","PHX":"PHX","BKN":"BKN"}

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
def iso2unix(s): return int(datetime.fromisoformat(s.replace("Z","+00:00")).timestamp())
def clk_secs(cd):
    cd=str(cd); return int(float(cd.split(":")[0]))*60+float(cd.split(":")[1]) if ":" in cd else float(cd)
def code_match(kalshi, espn):
    e=espn.upper(); k=kalshi.upper()
    return e==k or k.startswith(e) or e.startswith(k) or ALIAS.get(k)==e

def parse_ticker(t):
    # KXNBAGAME-26APR06NYKATL-NYK
    parts=t.split("-"); dt=parts[1]; team=parts[2]
    yr=2000+int(dt[:2]); mo=MONTHS[dt[2:5]]; day=int(dt[5:7])
    return f"{yr}{mo:02d}{day:02d}", team

def espn_game(date, team):
    sb=requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date}",headers=HDR,timeout=20).json()
    for ev in sb.get("events",[]):
        comp=ev["competitions"][0]["competitors"]
        if any(code_match(team, c["team"]["abbreviation"]) for c in comp):
            home_is_market = any(code_match(team,c["team"]["abbreviation"]) and c["homeAway"]=="home" for c in comp)
            return ev["id"], home_is_market
    return None, None

def espn_series(gid, home_is_market):
    j=requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={gid}",headers=HDR,timeout=25).json()
    wc={p["id"]:p.get("wallclock") for p in j.get("plays",[])}
    espn_wp=[]; model_wp=[]
    for p in j.get("plays",[]):
        w=p.get("wallclock")
        if not w: continue
        per=p["period"]["number"]; secs=max(0,(4-per)*720+clk_secs(p["clock"]["displayValue"])) if per<=4 else clk_secs(p["clock"]["displayValue"])
        lead=(p["homeScore"]-p["awayScore"]) if home_is_market else (p["awayScore"]-p["homeScore"])
        model_wp.append((iso2unix(w), nba_win_probability(lead,secs)))
    for wp in j.get("winprobability",[]):
        w=wc.get(wp.get("playId"))
        if w is None: continue
        hwp=wp.get("homeWinPercentage")
        espn_wp.append((iso2unix(w), hwp if home_is_market else 1.0-hwp))
    espn_wp.sort(); model_wp.sort()
    return espn_wp, model_wp

def kalshi_candles(tkr, close_iso):
    end=iso2unix(close_iso); start=end-5*3600
    path=f"/trade-api/v2/series/KXNBAGAME/markets/{tkr}/candlesticks"
    c=requests.get(BASE+path,headers=sign("GET",path),params={"start_ts":start,"end_ts":end,"period_interval":1},timeout=30).json()
    return [(x["end_period_ts"], float((x.get("price") or {}).get("close_dollars"))) for x in c.get("candlesticks",[]) if (x.get("price") or {}).get("close_dollars") is not None]

def nearest(series_ts, series_v, ts):
    i=bisect.bisect_right(series_ts,ts)-1
    return series_v[i] if i>=0 else None

def main(limit=40):
    games=json.loads(Path("data/raw/nba_games.json").read_text())
    out=[]; skipped=0
    for g in games[:limit*2]:
        try:
            date,team=parse_ticker(g["ticker"])
            gid,home=espn_game(date,team)
            if gid is None: skipped+=1; continue
            espn_wp,model_wp=espn_series(gid,home)
            if not espn_wp: skipped+=1; continue
            cand=kalshi_candles(g["ticker"], g["close_time"])
            ets=[t for t,_ in espn_wp]; evs=[v for _,v in espn_wp]
            mts=[t for t,_ in model_wp]; mvs=[v for _,v in model_wp]
            pts=[]
            for ts,price in cand:
                ew=nearest(ets,evs,ts); mw=nearest(mts,mvs,ts)
                if ew is not None and mw is not None:
                    pts.append([round(price,4),round(ew,4),round(mw,4)])
            contested=[p for p in pts if 0.15<=p[0]<=0.85]
            if len(contested)>=20:
                out.append({"ticker":g["ticker"],"outcome":g["outcome"],"points":pts})
        except Exception:
            skipped+=1
        time.sleep(0.2)
        if len(out)>=limit: break
    OUT.write_text(json.dumps(out))
    print(f"Aligned {len(out)} games (skipped {skipped}) -> {OUT}")
    print("each point = [market_price, espn_wp, model_wp]")

if __name__=="__main__":
    main()
