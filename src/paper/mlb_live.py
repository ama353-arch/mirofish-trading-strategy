"""
Live MLB game state, and matching it to Kalshi markets.

Read-only. State comes from ESPN's public play-by-play: the most recent real half-inning play carries
the inning, the score, the outs and who is on base — everything the frozen model needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"
MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
TICKER = re.compile(r"^KXMLBGAME-(\d{2})([A-Z]{3})(\d{2})(\d{4})([A-Z]+)-([A-Z]+)$")


@dataclass(frozen=True)
class Ticker:
    date: str          # YYYYMMDD
    matchup: str       # concatenated team abbreviations, exchange order
    yes_team: str


def parse_ticker(ticker: str) -> Ticker | None:
    m = TICKER.match(ticker or "")
    if not m:
        return None
    yy, mon, dd, _hhmm, matchup, yes_team = m.groups()
    if mon not in MONTHS:
        return None
    return Ticker(f"20{yy}{MONTHS[mon]:02d}{dd}", matchup, yes_team)


def matchup_key(matchup: str) -> str:
    """Order-independent key: the exchange and ESPN do not agree on home/away ordering."""
    return "".join(sorted(matchup))


def _get(url: str, **params):
    try:
        r = requests.get(url, params=params, timeout=20)
        return r.json() if r.status_code == 200 else {}
    except requests.exceptions.RequestException:
        return {}


def scoreboard(date: str) -> dict:
    """matchup_key -> {espn_id, home, state} for that day's games."""
    out = {}
    for e in _get(SCOREBOARD, dates=date).get("events", []) or []:
        try:
            comp = e["competitions"][0]
            abbrs = [t["team"]["abbreviation"] for t in comp["competitors"]]
            home = next(t["team"]["abbreviation"] for t in comp["competitors"]
                        if t.get("homeAway") == "home")
            out[matchup_key("".join(abbrs))] = {
                "espn_id": e["id"], "home": home,
                "state": ((comp.get("status") or {}).get("type") or {}).get("state"),
            }
        except (KeyError, IndexError, StopIteration):
            continue
    return out


def game_state(espn_id: str) -> tuple | None:
    """(inning, is_bottom, away_score, home_score, outs, bases) from the latest real play."""
    plays = _get(SUMMARY, event=espn_id).get("plays") or []
    best = None
    for p in plays:
        per = p.get("period") or {}
        half = str((per.get("type") or "")).strip().lower()
        if half not in ("top", "bottom"):
            continue            # 'Mid'/'End' markers carry the post-half score, not a half-inning
        if p.get("awayScore") is None or p.get("homeScore") is None or per.get("number") is None:
            continue
        wc = p.get("wallclock") or ""
        if best is None or wc >= best[0]:
            bases = ((1 if p.get("onFirst") else 0) | (2 if p.get("onSecond") else 0)
                     | (4 if p.get("onThird") else 0))
            best = (wc, int(per["number"]), half == "bottom", int(p["awayScore"]),
                    int(p["homeScore"]), int(p.get("outs") or 0), bases)
    return best[1:] if best else None


def has_game_state(inning: int, away: int, home: int) -> bool:
    """Whether the game has yet told the model anything the pre-game price did not already hold.

    Before the third inning with nobody scoring, it has not: the model's state update is ~zero, so a
    disagreement with the market is stale-anchor drift, not insight. That subset loses historically
    in both halves of the backtest.
    """
    return inning >= 3 or (away + home) > 0
