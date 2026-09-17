"""
win_probability.py — live in-game win-probability models (the fair value for
match-momentum trading).

NBA: the score differential behaves like a random walk, so the probability the
home team is still ahead at the buzzer is a diffusion probability:

    P(win) = Phi( (lead + drift * t) / (sigma * sqrt(t)) )

where sigma is the standard deviation of scoring per second. This is the same
mathematics as the option pricer's N(d2) — a lead is a "spot above strike 0."

Soccer lives in soccer_win_probability (Poisson / Dixon-Coles family).
"""

from __future__ import annotations

import math

from src.signals.walkforward import _norm_cdf

# Final-margin std in the NBA is ~13 points over a 48-minute (2880s) game:
#   sigma * sqrt(2880) ~= 13  ->  sigma ~= 0.242 points / sqrt(second)
_NBA_SIGMA_DEFAULT = 0.242


def nba_win_probability(
    lead: float,
    seconds_remaining: float,
    sigma: float = _NBA_SIGMA_DEFAULT,
    drift: float = 0.0,
) -> float:
    """P(home team wins) given the current `lead` and `seconds_remaining`.

    `drift` is expected points-per-second edge (home-court / team strength);
    default 0. At the buzzer the game is decided: a positive lead is a win, a
    negative lead a loss, a tie a coin-flip (overtime).
    """
    if seconds_remaining <= 0 or sigma <= 0:
        if lead > 0:
            return 1.0
        if lead < 0:
            return 0.0
        return 0.5
    vol = sigma * math.sqrt(seconds_remaining)
    z = (lead + drift * seconds_remaining) / vol
    return _norm_cdf(z)


_FULL_MATCH_MIN = 90.0
_MAX_GOALS = 15   # truncation for the remaining-goals sums (Poisson tail is tiny)


def _poisson_pmf(k: int, mu: float) -> float:
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-mu) * mu**k / math.factorial(k)


def soccer_win_probability(
    home_goals: int,
    away_goals: int,
    minutes_remaining: float,
    lambda_home: float = 1.4,
    lambda_away: float = 1.4,
) -> float:
    """P(home team WINS) from the current score, via a Poisson goal model.

    Each side scores additional goals over the remaining time as a Poisson
    process (rate = season scoring rate scaled by the fraction of match left).
    A draw resolves the "home win" market NO, so only a strictly higher final
    score counts as a win. At full time the result is settled.
    """
    if minutes_remaining <= 0:
        return 1.0 if home_goals > away_goals else 0.0

    frac = min(1.0, minutes_remaining / _FULL_MATCH_MIN)
    mu_home = lambda_home * frac
    mu_away = lambda_away * frac

    p_home_add = [_poisson_pmf(k, mu_home) for k in range(_MAX_GOALS + 1)]
    p_away_add = [_poisson_pmf(k, mu_away) for k in range(_MAX_GOALS + 1)]

    win = 0.0
    for h, ph in enumerate(p_home_add):
        for a, pa in enumerate(p_away_add):
            if (home_goals + h) > (away_goals + a):
                win += ph * pa
    return win
