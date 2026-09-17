"""
Tests for the live paper-trading harness.

This harness is a MEASUREMENT INSTRUMENT, not a profit engine: all nine Kalshi
15-minute crypto series measured efficient, so its job is to settle the one
question history could not — the true loss rate on 99c+ favourites in the final
seconds — and to watch newly-listed series. It never places a live order.

The pricing tests below are not ceremony. On 2026-09-11 a spike priced the
longshot at `1 - yes_ask` instead of `1 - yes_bid`, crediting the buyer with a
price on the wrong side of the spread. It manufactured a +5c/contract "edge" in
exactly the two widest-spread markets. These tests pin the correct side.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paper.ledger import PaperLedger, PaperTrade, maker_fee, poisson_cdf, taker_fee
from src.paper.rules import Quote, favourite_at_ask, final_seconds_favourite, longshot_at_ask


# ── Fees (Kalshi: taker 0.07*C*P*(1-P), maker 0.0175*C*P*(1-P)) ──────────────

def test_taker_fee_peaks_at_fifty_cents():
    assert taker_fee(0.50) == pytest.approx(0.0175)
    assert taker_fee(0.90) < taker_fee(0.50)
    assert taker_fee(0.99) < taker_fee(0.90)


def test_taker_fee_is_symmetric_about_fifty_cents():
    assert taker_fee(0.10) == pytest.approx(taker_fee(0.90))


def test_fee_scales_with_contract_count():
    assert taker_fee(0.50, 100) == pytest.approx(1.75)


def test_maker_fee_is_a_quarter_of_taker():
    assert maker_fee(0.50, 100) == pytest.approx(taker_fee(0.50, 100) / 4)


def test_fee_rounds_up_never_down():
    """Kalshi rounds the fee up to a centicent, so a fee is never understated."""
    raw = 0.07 * 0.99 * 0.01
    assert taker_fee(0.99) >= raw
    assert taker_fee(0.99) == pytest.approx(0.0007)


# ── Which side is the favourite, and what does it actually cost ──────────────

def test_favourite_yes_is_bought_at_the_yes_ask():
    side, price = favourite_at_ask(Quote("T", 0, 10, yes_bid=0.97, yes_ask=0.98))
    assert side == "yes"
    assert price == pytest.approx(0.98)


def test_favourite_no_is_bought_at_one_minus_the_BID_not_the_ask():
    """THE REGRESSION TEST. YES is 2/3c, so NO is the favourite. Buying NO lifts
    the NO ask, which is 1 - yes_bid = 0.98 — not 1 - yes_ask = 0.97."""
    side, price = favourite_at_ask(Quote("T", 0, 10, yes_bid=0.02, yes_ask=0.03))
    assert side == "no"
    assert price == pytest.approx(0.98)
    assert price != pytest.approx(0.97)


def test_longshot_is_also_bought_at_its_own_ask():
    side, price = longshot_at_ask(Quote("T", 0, 10, yes_bid=0.97, yes_ask=0.98))
    assert side == "no"
    assert price == pytest.approx(0.03)


def test_buying_both_sides_costs_one_dollar_plus_the_spread():
    """The invariant the bug broke: you cannot buy both sides for less than $1."""
    q = Quote("T", 0, 10, yes_bid=0.40, yes_ask=0.44)
    _, fav = favourite_at_ask(q)
    _, dog = longshot_at_ask(q)
    assert fav + dog == pytest.approx(1.0 + (q.yes_ask - q.yes_bid))
    assert fav + dog > 1.0


# ── The rule under test ──────────────────────────────────────────────────────

def test_rule_fires_on_a_dear_favourite_inside_the_window():
    order = final_seconds_favourite(Quote("T", 0, 10, yes_bid=0.990, yes_ask=0.995))
    assert order is not None and order.side == "yes" and order.price == pytest.approx(0.995)


def test_rule_stands_down_when_the_favourite_is_too_cheap():
    assert final_seconds_favourite(Quote("T", 0, 10, yes_bid=0.90, yes_ask=0.92)) is None


def test_rule_stands_down_outside_the_seconds_window():
    q_early = Quote("T", 0, 300, yes_bid=0.990, yes_ask=0.995)
    q_late = Quote("T", 0, 1, yes_bid=0.990, yes_ask=0.995)
    assert final_seconds_favourite(q_early) is None
    assert final_seconds_favourite(q_late) is None


def test_rule_stands_down_on_a_crossed_or_missing_quote():
    assert final_seconds_favourite(Quote("T", 0, 10, yes_bid=0.99, yes_ask=0.98)) is None
    assert final_seconds_favourite(Quote("T", 0, 10, yes_bid=None, yes_ask=0.99)) is None


# ── The ledger, and the statistic the harness exists to produce ──────────────

def _trade(price, contracts=1, side="yes", ticker="T"):
    return PaperTrade(ticker=ticker, series="KXBTC15M", decision_ts=0.0, side=side,
                      price=price, contracts=contracts, rule="final_seconds_favourite")


def test_a_winning_favourite_pays_the_rest_of_the_dollar_minus_fee():
    t = _trade(0.99)
    assert t.pnl(result=1) == pytest.approx(0.01 - taker_fee(0.99))


def test_a_losing_favourite_costs_the_stake_plus_fee():
    t = _trade(0.99)
    assert t.pnl(result=0) == pytest.approx(-0.99 - taker_fee(0.99))


def test_a_no_side_trade_wins_when_the_market_resolves_no():
    t = _trade(0.98, side="no")
    assert t.pnl(result=0) == pytest.approx(0.02 - taker_fee(0.98))
    assert t.pnl(result=1) == pytest.approx(-0.98 - taker_fee(0.98))


def test_unsettled_trades_are_not_counted():
    led = PaperLedger()
    led.add(_trade(0.99, ticker="A"))
    assert led.stats()["settled"] == 0
    led.settle("A", result=1)
    assert led.stats()["settled"] == 1


def test_settling_an_unknown_ticker_is_a_no_op():
    led = PaperLedger()
    led.add(_trade(0.99, ticker="A"))
    led.settle("ZZZ", result=1)
    assert led.stats()["settled"] == 0


def test_break_even_losses_is_the_number_of_losses_we_could_afford():
    """100 trades at 99c: each can absorb (1 - 0.99 - fee) of a loss and break even."""
    led = PaperLedger()
    for i in range(100):
        led.add(_trade(0.99, ticker=f"T{i}")); led.settle(f"T{i}", result=1)
    s = led.stats()
    assert s["break_even_losses"] == pytest.approx(100 * (0.01 - taker_fee(0.99)))
    assert s["losses"] == 0


def test_zero_losses_alone_does_not_prove_an_edge():
    """The trap this harness exists to avoid: a perfect record on a small sample
    is not significant. 100 clean trades at 99c must NOT report p < 0.05."""
    led = PaperLedger()
    for i in range(100):
        led.add(_trade(0.99, ticker=f"T{i}")); led.settle(f"T{i}", result=1)
    assert led.stats()["exact_p"] > 0.05


def test_enough_clean_trades_eventually_does_prove_it():
    led = PaperLedger()
    for i in range(2000):
        led.add(_trade(0.99, ticker=f"T{i}")); led.settle(f"T{i}", result=1)
    assert led.stats()["exact_p"] < 0.01


def test_net_pnl_is_gross_minus_fees():
    led = PaperLedger()
    led.add(_trade(0.99, contracts=10, ticker="A")); led.settle("A", result=1)
    led.add(_trade(0.99, contracts=10, ticker="B")); led.settle("B", result=0)
    s = led.stats()
    assert s["net"] == pytest.approx(s["gross"] - s["fees"])
    assert s["fees"] == pytest.approx(2 * taker_fee(0.99, 10))


def test_poisson_cdf_matches_hand_computation():
    assert poisson_cdf(0, 2.0) == pytest.approx(math.exp(-2.0))
    assert poisson_cdf(1, 2.0) == pytest.approx(math.exp(-2.0) * 3.0)
    assert poisson_cdf(5, 0.0) == pytest.approx(1.0)


# ── The harness must be structurally incapable of trading ────────────────────

def test_the_client_refuses_every_verb_that_could_place_an_order():
    from src.paper.live import ReadOnlyKalshi
    api = ReadOnlyKalshi.__new__(ReadOnlyKalshi)  # no credentials needed to prove the guard
    for verb in ("POST", "post", "PUT", "DELETE", "PATCH"):
        with pytest.raises(PermissionError):
            api.request(verb, "/trade-api/v2/portfolio/orders")


def test_the_allowed_verb_set_is_exactly_get():
    from src.paper.live import ReadOnlyKalshi
    assert set(ReadOnlyKalshi.ALLOWED) == {"GET"}


# ── A quoted price with nothing behind it is not a fill ──────────────────────

def test_depth_to_buy_yes_comes_from_the_resting_NO_bids():
    """Buying YES at 0.998 matches someone bidding 0.002 for NO. The book quotes
    bids on each side, so our fill size lives on the OPPOSITE side of the book."""
    from src.paper.live import book_depth
    book = {"orderbook_fp": {"no_dollars": [["0.0020", "500"], ["0.0030", "250"]],
                             "yes_dollars": [["0.9000", "10"]]}}
    at, better = book_depth(book, side="yes", price=0.998)
    assert at == pytest.approx(500)
    assert better == pytest.approx(750)  # the 0.003 NO bid is YES at 0.997 — cheaper for us


def test_depth_to_buy_no_comes_from_the_resting_YES_bids():
    from src.paper.live import book_depth
    book = {"orderbook_fp": {"yes_dollars": [["0.0200", "400"], ["0.0300", "100"]],
                             "no_dollars": [["0.5000", "9"]]}}
    at, better = book_depth(book, side="no", price=0.98)
    assert at == pytest.approx(400)
    assert better == pytest.approx(500)


def test_depth_is_zero_when_nobody_is_there():
    from src.paper.live import book_depth
    assert book_depth({"orderbook_fp": {"no_dollars": [], "yes_dollars": []}},
                      side="yes", price=0.99) == (0.0, 0.0)


def test_depth_reads_the_legacy_key_too():
    """Kalshi's `*_fp` migration left legacy keys behind that read empty or stale."""
    from src.paper.live import book_depth
    book = {"orderbook": {"no_dollars": [["0.0100", "77"]], "yes_dollars": []}}
    at, _ = book_depth(book, side="yes", price=0.99)
    assert at == pytest.approx(77)


def test_fill_is_capped_by_what_was_actually_resting():
    from src.paper.live import cap_to_depth
    assert cap_to_depth(requested=100, available=30) == 30
    assert cap_to_depth(requested=100, available=250) == 100
    assert cap_to_depth(requested=100, available=0) == 0


def test_new_fifteen_minute_series_are_discovered_and_others_ignored():
    from src.paper.live import fifteen_min_crypto
    payload = {"series": [{"ticker": "KXADA15M", "frequency": "fifteen_min"},
                          {"ticker": "KXBTCD", "frequency": "hourly"},
                          {"ticker": "KXNEW15M", "frequency": "fifteen_min"},
                          {"frequency": "fifteen_min"}]}
    assert fifteen_min_crypto(payload) == ["KXADA15M", "KXNEW15M"]


# ── Price and size must come from the SAME instant ───────────────────────────

def test_quote_is_derived_from_the_book_so_price_and_depth_agree():
    """Best YES bid is the top of the yes side; the YES ask is 1 - the best NO bid.
    Taking the price from /markets and the size from /orderbook reads two different
    instants — in the closing seconds the book moves between them."""
    from src.paper.live import quote_from_book
    book = {"orderbook_fp": {"yes_dollars": [["0.9600", "5"], ["0.9700", "10"]],
                             "no_dollars": [["0.0100", "7"], ["0.0200", "50"]]}}
    q = quote_from_book(book, ticker="T", close_ts=100.0, now=90.0)
    assert q.yes_bid == pytest.approx(0.97)
    assert q.yes_ask == pytest.approx(0.98)
    assert q.seconds_to_close == pytest.approx(10.0)
    assert q.valid


def test_quote_from_an_empty_book_is_invalid_not_a_guess():
    from src.paper.live import quote_from_book
    q = quote_from_book({"orderbook_fp": {"yes_dollars": [], "no_dollars": []}},
                        ticker="T", close_ts=100.0, now=90.0)
    assert not q.valid


def test_book_derived_ask_has_depth_behind_it_by_construction():
    """The invariant the two-call version broke: the ask we quote always has size."""
    from src.paper.live import book_depth, quote_from_book
    book = {"orderbook_fp": {"yes_dollars": [["0.9900", "12"]], "no_dollars": [["0.0050", "33"]]}}
    q = quote_from_book(book, ticker="T", close_ts=100.0, now=90.0)
    at, _ = book_depth(book, side="yes", price=q.yes_ask)
    assert q.yes_ask == pytest.approx(0.995)
    assert at == pytest.approx(33)


# ── Dust fills and correlated trades both corrupt the loss statistic ─────────

def test_a_fill_below_one_contract_is_not_a_trade():
    """Kalshi allows fractional size and rounds the fee UP to a centicent, so on a
    dust fill the minimum fee can exceed the maximum profit — a trade that cannot
    break even at any loss rate, which poisons the break-even total."""
    from src.paper.live import cap_to_depth
    assert cap_to_depth(requested=100, available=0.5) == 0
    assert cap_to_depth(requested=100, available=1.0) == 1.0


def test_break_even_loss_rate_stays_positive_at_the_minimum_size():
    t = PaperTrade(ticker="T", series="S", decision_ts=0.0, side="yes", price=0.999,
                   contracts=1.0, rule="r")
    assert t.break_even_loss_rate > 0


def test_trades_are_deduplicated_by_EVENT_not_by_market():
    """A coin-race event lists one market per coin. Booking several legs of the same
    event gives correlated outcomes, and the Poisson test assumes independence."""
    from src.paper.live import event_of
    assert event_of("KXBTC15M-26SEP120030-30") == "KXBTC15M-26SEP120030"
    assert event_of("KXCRYPTOLEAD15M-26SEP120030-BTC") == "KXCRYPTOLEAD15M-26SEP120030"
    assert event_of("KXCRYPTOLEAD15M-26SEP120030-ETH") == event_of("KXCRYPTOLEAD15M-26SEP120030-BTC")


# ── An order takes time to arrive, and the book moves while it travels ───────

def test_a_limit_order_fills_only_at_or_better_than_the_price_we_sent():
    from src.paper.live import limit_fill
    assert limit_fill(decision_price=0.995, arrival_ask=0.995) == pytest.approx(0.995)
    assert limit_fill(decision_price=0.995, arrival_ask=0.990) == pytest.approx(0.990)  # improved
    assert limit_fill(decision_price=0.995, arrival_ask=0.997) is None                  # ran away


def test_a_vanished_ask_is_not_a_fill():
    from src.paper.live import limit_fill
    assert limit_fill(decision_price=0.995, arrival_ask=None) is None


# ── Credentials: a file path on a laptop, an env var in a container ─────────

def test_container_credentials_come_from_the_environment():
    from src.paper.live import resolve_credentials
    c = resolve_credentials({"KALSHI_API_KEY_ID": "abc", "KALSHI_PRIVATE_KEY": "-----BEGIN-----\nX\n"})
    assert c.key_id == "abc" and c.private_key_pem.startswith("-----BEGIN-----")


def test_escaped_newlines_in_an_env_var_are_restored():
    """Pasting a PEM into a hosting dashboard usually turns real newlines into backslash-n.
    Left alone, the key silently fails to parse."""
    from src.paper.live import resolve_credentials
    c = resolve_credentials({"KALSHI_API_KEY_ID": "abc",
                             "KALSHI_PRIVATE_KEY": "-----BEGIN-----\\nX\\n-----END-----"})
    assert "\\n" not in c.private_key_pem
    assert c.private_key_pem.count("\n") == 2


def test_missing_credentials_fail_loudly(tmp_path):
    from src.paper.live import resolve_credentials
    with pytest.raises(RuntimeError, match="KALSHI"):
        resolve_credentials({}, env_path=tmp_path / "nope.env")


def test_a_local_dotenv_still_works(tmp_path):
    from src.paper.live import resolve_credentials
    key = tmp_path / "k.pem"; key.write_text("-----BEGIN RSA-----\nzz\n-----END RSA-----")
    env = tmp_path / ".env"
    env.write_text(f"KALSHI_API_KEY_ID=fromfile\nKALSHI_API_PRIVATE_KEY_PATH={key}\n")
    c = resolve_credentials({}, env_path=env)
    assert c.key_id == "fromfile" and "BEGIN RSA" in c.private_key_pem


def test_signing_produces_a_verifiable_rsa_pss_signature():
    """Kalshi requires RSA-PSS with a digest-length salt. If the container signs differently from
    the laptop, every request 401s — so pin it rather than trust that openssl and the library agree."""
    crypto = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from src.paper.live import sign_message

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM,
                            serialization.PrivateFormat.TraditionalOpenSSL,
                            serialization.NoEncryption()).decode()
    sig = sign_message(pem, b"hello")
    key.public_key().verify(
        sig, b"hello",
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256.digest_size),
        hashes.SHA256())
