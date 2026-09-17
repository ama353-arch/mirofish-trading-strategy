#!/usr/bin/env python3
"""
Fit the MLB in-game win-probability model and freeze it to disk.

Frozen on purpose: the live paper harness must score games with a model that cannot drift, or a
forward test proves nothing. Re-run this only to deliberately refit, and note the date when you do.

    python3 fit_mlb_model.py            # -> data/paper/mlb_model.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "raw" / "mlb_pbp.jsonl"
OUT = ROOT / "models" / "mlb_model.json"
RNG = np.random.default_rng(23)
SIMS, MAXD = 40000, 12


def timeline(plays: list) -> list:
    """[(ts, inning, is_bottom, away, home, outs, bases)] — real half-innings only.

    ESPN also emits 'Mid'/'End' markers whose score is the state AFTER the bottom half; counting
    them as half-innings makes the top absorb the whole inning's runs.
    """
    out = []
    for row in plays:
        ts, inn, typ, a, h = row[0], row[1], row[2], row[3], row[4]
        outs = row[5] if len(row) > 5 else None
        bases = row[6] if len(row) > 6 else 0
        if inn is None or a is None or h is None:
            continue
        t = str(typ or "").strip().lower()
        if t not in ("top", "bottom"):
            continue
        out.append((float(ts), int(inn), t == "bottom", int(a), int(h),
                    int(outs) if outs is not None else 0, int(bases or 0)))
    out.sort(key=lambda x: x[0])
    return out


def fit_distributions(games: list) -> tuple[np.ndarray, dict]:
    """Runs in a whole half-inning, and runs from a given (bases, outs) to the end of the half."""
    full, rest = [], defaultdict(list)
    for g in games:
        tl = timeline(g["plays"])
        if not tl:
            continue
        halves = defaultdict(list)
        for row in tl:
            halves[(row[1], row[2])].append(row)
        prev_a = prev_h = 0
        for key in sorted(halves, key=lambda x: (x[0], x[1])):
            rows, bot = halves[key], key[1]
            end = max((r[4] if bot else r[3]) for r in rows)
            full.append(max(0, end - (prev_h if bot else prev_a)))
            for r in rows:
                rest[(r[6], min(r[5], 2))].append(max(0, end - (r[4] if bot else r[3])))
            prev_a = max(prev_a, max(r[3] for r in rows))
            prev_h = max(prev_h, max(r[4] for r in rows))
    pool = np.array([x for x in full if 0 <= x <= 15], dtype=np.int64)
    rest_d = {k: [int(x) for x in v if 0 <= x <= 15] for k, v in rest.items() if len(v) >= 200}
    return pool, rest_d


def mc_table(pool: np.ndarray) -> dict:
    """P(home wins) at the START of each half-inning. Inning 10 means 'extra innings'."""
    tab = {}

    def extras(d0: int) -> float:
        diff = np.full(SIMS, d0, dtype=np.int64)
        for _ in range(8):
            tied = diff == 0
            if not tied.any():
                break
            n = int(tied.sum())
            diff[tied] += RNG.choice(pool, n) - RNG.choice(pool, n)
        return float((diff > 0).mean() + 0.5 * (diff == 0).mean())

    for d in range(-MAXD, MAXD + 1):
        tab[f"10|0|{d}"] = extras(d)
    for inn in range(9, 0, -1):
        for bot in (True, False):
            halves, i, b = [], inn, bot
            while i <= 9:
                halves.append(b)
                if b:
                    i, b = i + 1, False
                else:
                    b = True
            for d in range(-MAXD, MAXD + 1):
                diff = np.full(SIMS, d, dtype=np.int64)
                for is_home in halves:
                    r = RNG.choice(pool, SIMS)
                    diff = diff + r if is_home else diff - r
                for _ in range(8):
                    tied = diff == 0
                    if not tied.any():
                        break
                    n = int(tied.sum())
                    diff[tied] += RNG.choice(pool, n) - RNG.choice(pool, n)
                tab[f"{inn}|{int(bot)}|{d}"] = float((diff > 0).mean() + 0.5 * (diff == 0).mean())
    return tab


def main() -> None:
    games = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    games = [g for g in games if len(g.get("plays") or []) > 20]
    pool, rest = fit_distributions(games)
    print(f"fitted on {len(games)} games: {pool.mean():.3f} runs per half-inning, "
          f"{pool.sum()/len(games):.2f} per game, {len(rest)} of 24 base-out buckets")
    tab = mc_table(pool)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "fitted_games": len(games),
        "max_diff": MAXD,
        "half_inning_runs": [int(x) for x in pool],
        "rest_by_state": {f"{b}|{o}": v for (b, o), v in rest.items()},
        "table": tab,
    }))
    print(f"frozen -> {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
