"""
Tests for live sports win-probability models — the fair-value engines for
in-game match-momentum trading.

NBA: score differential is a random walk, so P(home leads at the buzzer) is a
diffusion probability, the same shape as the option pricer. Soccer: goals are
Poisson arrivals, so in-game win probability comes from the remaining-time goal
distribution (Dixon-Coles family).

These are the `prob_fn` fair values; momentum trading is the divergence between
this fair value and the live market price.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sports.win_probability import nba_win_probability


# ── NBA diffusion win probability ────────────────────────────────────────────

def test_tied_game_with_time_left_is_a_cointoss():
    p = nba_win_probability(lead=0, seconds_remaining=600)
    assert p == pytest.approx(0.5, abs=0.02)


def test_big_lead_late_is_nearly_certain():
    assert nba_win_probability(lead=15, seconds_remaining=60) > 0.99
    assert nba_win_probability(lead=-15, seconds_remaining=60) < 0.01


def test_win_probability_is_monotonic_in_the_lead():
    probs = [nba_win_probability(lead=l, seconds_remaining=300)
             for l in (-6, -3, 0, 3, 6)]
    assert probs == sorted(probs)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_same_lead_is_safer_with_less_time_remaining():
    """A 5-point lead is more secure with 1 minute left than with 20 minutes
    left — less time for the trailing team to come back."""
    late = nba_win_probability(lead=5, seconds_remaining=60)
    early = nba_win_probability(lead=5, seconds_remaining=1200)
    assert late > early > 0.5


def test_buzzer_is_a_hard_settlement():
    assert nba_win_probability(lead=1, seconds_remaining=0) == 1.0
    assert nba_win_probability(lead=-1, seconds_remaining=0) == 0.0
    assert nba_win_probability(lead=0, seconds_remaining=0) == 0.5   # tie -> OT cointoss


# ── Soccer Poisson / Dixon-Coles in-game win probability ─────────────────────

def test_stronger_attack_is_more_likely_to_win_from_kickoff():
    from src.sports.win_probability import soccer_win_probability
    strong = soccer_win_probability(0, 0, minutes_remaining=90,
                                    lambda_home=2.0, lambda_away=1.0)
    weak = soccer_win_probability(0, 0, minutes_remaining=90,
                                  lambda_home=1.0, lambda_away=2.0)
    assert strong > weak
    assert 0.0 <= strong <= 1.0


def test_equal_teams_from_kickoff_leave_draw_mass():
    """P(home win) for evenly matched teams over a full match is well below 0.5
    because a draw carries real probability mass (and resolves the market NO)."""
    from src.sports.win_probability import soccer_win_probability
    p = soccer_win_probability(0, 0, minutes_remaining=90,
                               lambda_home=1.4, lambda_away=1.4)
    assert 0.30 < p < 0.45


def test_two_goal_lead_late_is_nearly_certain():
    from src.sports.win_probability import soccer_win_probability
    p = soccer_win_probability(2, 0, minutes_remaining=2,
                               lambda_home=1.4, lambda_away=1.4)
    assert p > 0.95


def test_win_probability_rises_with_the_current_lead():
    from src.sports.win_probability import soccer_win_probability
    probs = [soccer_win_probability(h, 1, minutes_remaining=30,
                                    lambda_home=1.4, lambda_away=1.4)
             for h in (0, 1, 2, 3)]
    assert probs == sorted(probs)


def test_full_time_only_a_lead_counts_as_a_win():
    """A draw resolves the 'home win' market NO, so at full time only a strictly
    higher score is a win."""
    from src.sports.win_probability import soccer_win_probability
    assert soccer_win_probability(1, 0, minutes_remaining=0) == 1.0
    assert soccer_win_probability(0, 0, minutes_remaining=0) == 0.0   # draw
    assert soccer_win_probability(0, 1, minutes_remaining=0) == 0.0
