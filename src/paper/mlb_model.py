"""
Frozen MLB in-game win-probability model.

Two ideas make this worth trusting more than a generic forecast:

1. It does NOT re-forecast team quality. The pre-game market price already holds that, and beating the
   crowd on which team is better is a losing game. The model anchors on that price and applies only the
   GAME-STATE update, so the single thing under test is whether the market updates correctly on what
   just happened.
2. It reads the real base-out state. An earlier version knew only inning and score and lost to the
   market worst exactly where the market moved most — because bases-loaded-nobody-out looked the same
   as an empty inning.

The numbers come from `data/paper/mlb_model.json`, frozen by `fit_mlb_model.py`. Frozen on purpose: a
forward test against a model that drifts proves nothing.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "models" / "mlb_model.json"
EXTRA_INNINGS = 10


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(min(x, 30), -30)))


class MLBModel:
    """P(home wins) from game state, and P(yes) once anchored to the pre-game price."""

    def __init__(self, table: dict, rest: dict, max_diff: int, fitted_games: int = 0) -> None:
        self.table = table
        self.rest = rest                      # (bases, outs) -> (values, weights)
        self.max_diff = max_diff
        self.fitted_games = fitted_games
        self.start = table["1|0|0"]

    @classmethod
    def load(cls, path: Path | None = None) -> "MLBModel":
        raw = json.loads((path or DEFAULT_PATH).read_text())
        rest = {}
        for key, vals in raw["rest_by_state"].items():
            b, o = key.split("|")
            counts = Counter(vals)
            total = sum(counts.values())
            items = sorted(counts.items())
            rest[(int(b), int(o))] = ([k for k, _ in items], [v / total for _, v in items])
        return cls(raw["table"], rest, int(raw["max_diff"]), int(raw.get("fitted_games", 0)))

    # ── internals ───────────────────────────────────────────────────────────
    def _clamp(self, d: int) -> int:
        return min(max(int(d), -self.max_diff), self.max_diff)

    def _at(self, inning: int, is_bottom: bool, diff: int) -> float:
        inn = EXTRA_INNINGS if inning >= EXTRA_INNINGS and not is_bottom else min(max(inning, 1), 9)
        return self.table[f"{inn}|{int(bool(is_bottom))}|{self._clamp(diff)}"]

    def _after_half(self, inning: int, is_bottom: bool, diff: int) -> float:
        """P(home wins) once the current half-inning has ended with this score difference."""
        inn = min(max(inning, 1), 9)
        if is_bottom:
            if inn >= 9:
                if diff > 0:
                    return 1.0            # walk-off
                if diff < 0:
                    return 0.0            # home batted last and lost
                return self.table[f"{EXTRA_INNINGS}|0|0"]
            return self._at(inn + 1, False, diff)
        if inn >= 9 and diff > 0:
            return 1.0                    # home already ahead; there is no bottom half
        return self._at(inn, True, diff)

    # ── public ──────────────────────────────────────────────────────────────
    def state_probability(self, inning: int, is_bottom: bool, score_diff: int,
                          bases: int, outs: int) -> float:
        """P(home wins) with league-average teams, from mid-inning.

        Draws the remainder of the CURRENT half-inning from the empirical distribution for this
        (runners, outs), then hands over to the start-of-half table.
        """
        dist = self.rest.get((int(bases), min(int(outs), 2)))
        if dist is None:
            return self._at(inning, is_bottom, score_diff)
        total = 0.0
        for runs, weight in zip(*dist):
            after = score_diff + runs if is_bottom else score_diff - runs
            total += weight * self._after_half(inning, is_bottom, after)
        return min(max(total, 0.0), 1.0)

    def win_probability(self, pregame_yes_price: float, yes_is_home: bool, inning: int,
                        is_bottom: bool, home_score: int, away_score: int,
                        bases: int, outs: int) -> float:
        """P(the YES side of the Kalshi market wins), anchored to the pre-game price."""
        base = _logit(pregame_yes_price if yes_is_home else 1 - pregame_yes_price)
        mc = self.state_probability(inning, is_bottom, home_score - away_score, bases, outs)
        p_home = _sigmoid(base + _logit(mc) - _logit(self.start))
        return p_home if yes_is_home else 1 - p_home
