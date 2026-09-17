"""
Tests for the frozen MLB in-game win-probability model.

The model exists to answer one question the market might get wrong: given the true game state, how
likely is this team to win? v1 knew only inning and score and lost to the market worst exactly where
the market moved most — because a bases-loaded, nobody-out situation looked identical to an empty
inning. These tests pin the behaviour that fixed it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paper.mlb_model import MLBModel

BASES_EMPTY, BASES_LOADED = 0, 7


@pytest.fixture(scope="module")
def model():
    return MLBModel.load()


# ── the state model (league-average teams) ──────────────────────────────────

def test_being_ahead_is_always_better(model):
    probs = [model.state_probability(5, True, d, BASES_EMPTY, 0) for d in (-3, -1, 0, 1, 3)]
    assert probs == sorted(probs)


def test_a_lead_is_safer_later_in_the_game(model):
    early = model.state_probability(2, True, 2, BASES_EMPTY, 0)
    late = model.state_probability(8, True, 2, BASES_EMPTY, 0)
    assert late > early


def test_runners_on_base_matter_at_the_same_score(model):
    """THE v1 BLIND SPOT. Same inning, same score — the situation is not the same."""
    loaded = model.state_probability(5, True, 0, BASES_LOADED, 0)
    empty = model.state_probability(5, True, 0, BASES_EMPTY, 0)
    assert loaded > empty + 0.10


def test_outs_matter_at_the_same_score(model):
    none_out = model.state_probability(5, True, 0, BASES_LOADED, 0)
    two_out = model.state_probability(5, True, 0, BASES_LOADED, 2)
    assert none_out > two_out


def test_a_home_lead_in_the_bottom_of_the_ninth_is_over(model):
    assert model.state_probability(9, True, 1, BASES_EMPTY, 0) == pytest.approx(1.0)


def test_a_big_late_deficit_is_nearly_hopeless(model):
    assert model.state_probability(9, True, -5, BASES_EMPTY, 2) < 0.02


def test_an_unknown_base_out_state_falls_back_instead_of_crashing(model):
    assert 0.0 <= model.state_probability(5, True, 0, 99, 9) <= 1.0


# ── anchoring to the market's pre-game price ────────────────────────────────

def test_at_the_starting_state_the_model_returns_the_pregame_price(model):
    """The model does not re-forecast team quality — the pre-game price already holds it.
    It only applies the game-state update, so at the opening state it must agree with the price."""
    for price in (0.35, 0.50, 0.72):
        p = model.win_probability(price, True, inning=1, is_bottom=False,
                                  home_score=0, away_score=0, bases=BASES_EMPTY, outs=0)
        assert p == pytest.approx(price, abs=0.02)


def test_the_yes_side_can_be_the_away_team(model):
    """A Kalshi ticker names which team is YES; it is often the away side."""
    home = model.win_probability(0.60, True, 5, True, 3, 1, BASES_EMPTY, 0)
    away = model.win_probability(0.40, False, 5, True, 3, 1, BASES_EMPTY, 0)
    assert home + away == pytest.approx(1.0, abs=0.03)


def test_a_favourite_that_falls_behind_loses_probability(model):
    ahead = model.win_probability(0.65, True, 6, True, 4, 1, BASES_EMPTY, 0)
    behind = model.win_probability(0.65, True, 6, True, 1, 4, BASES_EMPTY, 0)
    assert behind < 0.35 < ahead


def test_probabilities_stay_inside_zero_and_one(model):
    for price in (0.02, 0.5, 0.98):
        for d in (-9, 0, 9):
            p = model.win_probability(price, True, 7, False, 5 + d, 5, BASES_LOADED, 1)
            assert 0.0 <= p <= 1.0


# ── turning a Kalshi ticker into something matchable ────────────────────────

def test_a_kalshi_ticker_names_the_date_matchup_and_which_team_is_yes():
    from src.paper.mlb_live import parse_ticker
    t = parse_ticker("KXMLBGAME-26SEP011840NYMTB-TB")
    assert t.date == "20260901" and t.matchup == "NYMTB" and t.yes_team == "TB"


def test_a_ticker_that_is_not_a_game_market_is_rejected():
    from src.paper.mlb_live import parse_ticker
    assert parse_ticker("KXBTC15M-26SEP120030-30") is None
    assert parse_ticker("nonsense") is None


def test_matching_ignores_the_order_the_teams_are_listed_in():
    from src.paper.mlb_live import matchup_key
    assert matchup_key("NYMTB") == matchup_key("TBNYM")


# ── the rule that turns a disagreement into a paper trade ───────────────────

def test_we_buy_yes_only_when_the_model_beats_the_ask_by_the_threshold():
    from src.paper.rules import Quote, model_disagreement
    q = Quote("T", 0, 0, yes_bid=0.60, yes_ask=0.62)
    assert model_disagreement(q, model_p=0.75, threshold=0.08).side == "yes"
    assert model_disagreement(q, model_p=0.68, threshold=0.08) is None


def test_we_buy_no_when_the_model_is_far_below_the_bid():
    from src.paper.rules import Quote, model_disagreement
    q = Quote("T", 0, 0, yes_bid=0.60, yes_ask=0.62)
    o = model_disagreement(q, model_p=0.40, threshold=0.08)
    assert o.side == "no" and o.price == pytest.approx(0.40)   # 1 - yes_bid


def test_the_threshold_is_net_of_the_fee_not_gross():
    """A 5c gross gap is not a 5c edge — the fee comes out of it first."""
    from src.paper.rules import Quote, model_disagreement
    q = Quote("T", 0, 0, yes_bid=0.48, yes_ask=0.50)
    assert model_disagreement(q, model_p=0.567, threshold=0.05) is None   # gross 6.7c, fee ~1.7c
    assert model_disagreement(q, model_p=0.60, threshold=0.05) is not None


def test_a_crossed_or_missing_quote_produces_no_trade():
    from src.paper.rules import Quote, model_disagreement
    assert model_disagreement(Quote("T", 0, 0, 0.62, 0.60), 0.9, 0.05) is None
    assert model_disagreement(Quote("T", 0, 0, None, 0.60), 0.9, 0.05) is None


# ── The model may only speak once the game has told it something ────────────

def test_the_model_stands_down_before_the_game_provides_any_state():
    """At 0-0 in the first inning a game-state model knows nothing the pre-game price doesn't, so any
    disagreement there is stale-anchor drift or noise. Backtested, that subset loses in BOTH halves
    (-2.6c and -3.8c) while the rest is positive — and the live harness traded almost nothing else."""
    from src.paper.mlb_live import has_game_state
    assert has_game_state(inning=1, away=0, home=0) is False
    assert has_game_state(inning=2, away=0, home=0) is False
    assert has_game_state(inning=3, away=0, home=0) is True      # time elapsed is information
    assert has_game_state(inning=1, away=1, home=0) is True      # a run is information
    assert has_game_state(inning=2, away=0, home=3) is True
