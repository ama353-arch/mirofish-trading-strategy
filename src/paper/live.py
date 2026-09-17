"""
Live, read-only Kalshi client and the paper-trading runner.

The runner watches 15-minute crypto markets, snapshots real quotes as each window
closes, applies the candidate rule, and books hypothetical fills at prices that were
genuinely quoted, with real Kalshi fees. It settles them against the exchange's own
published result.

It cannot trade. `ReadOnlyKalshi` refuses any verb but GET, so there is no code path
from this process to an order — see tests/test_paper.py.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.paper.ledger import PaperLedger, PaperTrade
from src.paper.rules import Quote, _sides, final_seconds_favourite

BASE = "https://api.elections.kalshi.com"
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("PAPER_DATA_DIR") or (ROOT / "data" / "paper"))


@dataclass(frozen=True)
class Credentials:
    key_id: str
    private_key_pem: str


def resolve_credentials(env: dict | None = None, env_path: Path | None = None) -> Credentials:
    """Credentials from the environment (a container) or a local .env plus key file (a laptop).

    Hosting dashboards turn the newlines in a pasted PEM into literal backslash-n, which parses as a
    corrupt key and 401s every request, so those are restored here.
    """
    env = os.environ if env is None else env
    key_id, pem = env.get("KALSHI_API_KEY_ID"), env.get("KALSHI_PRIVATE_KEY")
    if key_id and pem:
        return Credentials(key_id, pem.replace("\\n", "\n"))
    path = env_path or (ROOT / ".env")
    if path.exists():
        kv = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip()
        key_id = key_id or kv.get("KALSHI_API_KEY_ID")
        key_file = kv.get("KALSHI_API_PRIVATE_KEY_PATH")
        if key_id and key_file:
            p = Path(os.path.expanduser(key_file))
            if p.exists():
                return Credentials(key_id, p.read_text())
    raise RuntimeError(
        "No Kalshi credentials. Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY (the PEM contents), "
        "or provide a .env with KALSHI_API_KEY_ID and KALSHI_API_PRIVATE_KEY_PATH.")


def sign_message(private_key_pem: str, message: bytes) -> bytes:
    """RSA-PSS with a digest-length salt, as Kalshi requires.

    Prefers signing in-process; falls back to openssl so a machine without `cryptography` still works.
    The fallback writes the key to a private temp file, so it is the second choice, not the first.
    """
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
        return key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256.digest_size),
            hashes.SHA256(),
        )
    except ImportError:
        import stat
        import tempfile

        fd, tmp = tempfile.mkstemp()
        try:
            os.write(fd, private_key_pem.encode())
            os.close(fd)
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
            proc = subprocess.run(
                ["openssl", "dgst", "-sha256", "-sign", tmp,
                 "-sigopt", "rsa_padding_mode:pss", "-sigopt", "rsa_pss_saltlen:digest"],
                input=message, capture_output=True)
            return proc.stdout
        finally:
            os.path.exists(tmp) and os.unlink(tmp)


class ReadOnlyKalshi:
    """Signed Kalshi client that can only read. Any non-GET verb raises."""

    ALLOWED = frozenset({"GET"})

    def __init__(self, env_path: Path | None = None) -> None:
        creds = resolve_credentials(env_path=env_path)
        self.key_id = creds.key_id
        self._pem = creds.private_key_pem

    def _headers(self, method: str, path: str) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        sig = sign_message(self._pem, (ts + method + path).encode())
        return {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}

    def request(self, method: str, path: str, **params):
        if method.upper() not in self.ALLOWED:
            raise PermissionError(
                f"{method} is blocked: the paper harness is read-only and cannot place orders"
            )
        for attempt in range(4):
            try:
                r = requests.get(BASE + path, headers=self._headers("GET", path),
                                 params=params, timeout=20)
                if r.status_code == 429:
                    time.sleep(1.5 * (attempt + 1)); continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))

    def get(self, path: str, **params):
        return self.request("GET", path, **params)


def iso_to_unix(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def open_markets(api: ReadOnlyKalshi, series: str) -> list[dict]:
    d = api.get("/trade-api/v2/markets", series_ticker=series, status="open", limit=100)
    out = []
    for m in d.get("markets", []):
        try:
            out.append({"ticker": m["ticker"], "series": series, "close": iso_to_unix(m["close_time"])})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def quote_now(api: ReadOnlyKalshi, ticker: str, close_ts: float) -> Quote:
    m = api.get(f"/trade-api/v2/markets/{ticker}").get("market", {})
    f = lambda k: float(m[k]) if m.get(k) not in (None, "") else None
    now = time.time()
    return Quote(ticker=ticker, ts=now, seconds_to_close=close_ts - now,
                 yes_bid=f("yes_bid_dollars"), yes_ask=f("yes_ask_dollars"))


def settle_result(api: ReadOnlyKalshi, ticker: str) -> int | None:
    m = api.get(f"/trade-api/v2/markets/{ticker}").get("market", {})
    r = m.get("result")
    return 1 if r == "yes" else 0 if r == "no" else None


def _sides_for(q: Quote, side: str) -> tuple[str, float]:
    """The (side, price) pair for `side` at its own ask, from this quote."""
    fav, dog = _sides(q)
    return fav if fav[0] == side else dog


def _best_bid(levels) -> float | None:
    best = None
    for level in levels or []:
        try:
            p = float(level[0])
        except (TypeError, ValueError, IndexError):
            continue
        if best is None or p > best:
            best = p
    return best


def quote_from_book(book: dict, ticker: str, close_ts: float, now: float | None = None) -> Quote:
    """Build the quote from the order book itself, so price and depth share one instant.

    The book quotes bids per side: the YES ask is 1 - the best NO bid. Reading the price
    from /markets and the size from /orderbook samples two moments, and in the closing
    seconds the book moves between them.
    """
    ob = book.get("orderbook_fp") or book.get("orderbook") or {}
    yes_bid = _best_bid(ob.get("yes_dollars"))
    no_bid = _best_bid(ob.get("no_dollars"))
    return Quote(ticker=ticker, ts=now if now is not None else time.time(),
                 seconds_to_close=close_ts - (now if now is not None else time.time()),
                 yes_bid=yes_bid, yes_ask=None if no_bid is None else 1 - no_bid)


def book_depth(book: dict, side: str, price: float) -> tuple[float, float]:
    """Contracts resting that would fill a buy of `side` at `price`: (at that price, at or better).

    The book quotes BIDS on each side, so buying YES at p matches NO bids at (1 - p).
    A NO bid at a HIGHER price is a CHEAPER yes for us, so "at or better" sums every
    opposite-side bid at price >= (1 - p).
    """
    ob = book.get("orderbook_fp") or book.get("orderbook") or {}
    levels = ob.get("no_dollars" if side == "yes" else "yes_dollars") or []
    target = 1 - price
    at = better = 0.0
    for level in levels:
        try:
            p, size = float(level[0]), float(level[1])
        except (TypeError, ValueError, IndexError):
            continue
        if abs(p - target) < 5e-5:
            at += size
        if p >= target - 5e-5:
            better += size
    return at, better


MIN_CONTRACTS = 1.0


def cap_to_depth(requested: float, available: float, min_size: float = MIN_CONTRACTS) -> float:
    """A price on the screen with nothing behind it is not a fill, and neither is dust.

    Kalshi rounds the fee up to a centicent, so below ~1 contract the minimum fee can
    exceed the maximum profit: a trade that cannot break even at any loss rate.
    """
    filled = max(0.0, min(requested, available))
    return filled if filled >= min_size else 0.0


LATENCY_S = 0.5  # an order does not reach the exchange instantly


def limit_fill(decision_price: float, arrival_ask: float | None) -> float | None:
    """What a limit order sent at `decision_price` actually pays when it arrives.

    Fills at the arrival ask when that is at or better than the price we sent; if the
    market ran away in the meantime, the order simply does not fill.
    """
    if arrival_ask is None:
        return None
    return arrival_ask if arrival_ask <= decision_price else None


def event_of(ticker: str) -> str:
    """The event a market belongs to. One event can list many markets (a coin race),
    and their outcomes are correlated, so we take at most one trade per event."""
    return ticker.rsplit("-", 1)[0]


def fifteen_min_crypto(series_payload: dict) -> list[str]:
    """Every 15-minute crypto series Kalshi currently lists, so new ones get picked up."""
    return sorted(s["ticker"] for s in (series_payload.get("series") or [])
                  if s.get("frequency") == "fifteen_min" and s.get("ticker"))


class PaperRunner:
    """Polls live markets, books paper trades on rule hits, settles them, saves state."""

    def __init__(self, api: ReadOnlyKalshi, series: list[str], contracts: float = 1.0,
                 out_dir: Path = OUT) -> None:
        self.api, self.series, self.contracts = api, series, contracts
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.ledger = PaperLedger()
        self.traded: set[str] = set()
        self.pending: dict[str, str] = {}   # ticker -> series, awaiting settlement
        self._load()

    # ── persistence ─────────────────────────────────────────────────────────
    def _load(self) -> None:
        f = self.out / "ledger.json"
        if not f.exists():
            return
        for row in json.loads(f.read_text()):
            row.pop("fee", None)
            self.ledger.add(PaperTrade(**row))
        self.traded = {event_of(t.ticker) for t in self.ledger.trades}
        self.pending = {t.ticker: t.series for t in self.ledger.trades if t.result is None}

    def save(self) -> None:
        (self.out / "ledger.json").write_text(json.dumps([asdict(t) for t in self.ledger.trades], indent=1))
        (self.out / "stats.json").write_text(json.dumps(self.ledger.stats(), indent=1))

    def log_quote(self, q: Quote, series: str) -> None:
        with (self.out / "quotes.jsonl").open("a") as fh:
            fh.write(json.dumps({"series": series, **asdict(q)}) + "\n")

    # ── one pass ────────────────────────────────────────────────────────────
    def check_window(self, mkt: dict, window: tuple[float, float]) -> bool:
        book = self.api.get(f"/trade-api/v2/markets/{mkt['ticker']}/orderbook")
        q = quote_from_book(book, mkt["ticker"], mkt["close"])
        self.log_quote(q, mkt["series"])
        if event_of(mkt["ticker"]) in self.traded:
            return False
        order = final_seconds_favourite(q, window=window)
        if order is None:
            return False
        # The order has to travel. Re-read the book on arrival and fill like a limit order.
        time.sleep(LATENCY_S)
        book2 = self.api.get(f"/trade-api/v2/markets/{mkt['ticker']}/orderbook")
        q2 = quote_from_book(book2, mkt["ticker"], mkt["close"])
        arrival = None
        if q2.valid:
            side2, price2 = _sides_for(q2, order.side)
            arrival = price2 if side2 == order.side else None
        fill_price = limit_fill(order.price, arrival)
        if fill_price is None:
            return False  # the market moved away before the order landed
        at, better = book_depth(book2, order.side, fill_price)
        filled = cap_to_depth(self.contracts, better)
        if filled <= 0:
            return False  # the price was on screen but nothing was resting behind it
        self.ledger.add(PaperTrade(
            ticker=mkt["ticker"], series=mkt["series"], decision_ts=q.ts, side=order.side,
            price=fill_price, contracts=filled, rule=order.rule,
            meta={"yes_bid": q.yes_bid, "yes_ask": q.yes_ask, "secs_to_close": round(q.seconds_to_close, 2),
                  "requested": self.contracts, "depth_at_price": at, "depth_at_or_better": better,
                  "decision_price": order.price, "slippage": round(fill_price - order.price, 6),
                  "method": "book_snapshot_limit_v2"},
        ))
        self.traded.add(event_of(mkt["ticker"]))
        self.pending[mkt["ticker"]] = mkt["series"]
        self.save()
        return True

    def settle_pending(self) -> int:
        done = 0
        for ticker in list(self.pending):
            try:
                r = settle_result(self.api, ticker)
            except requests.exceptions.RequestException:
                continue
            if r is not None:
                self.ledger.settle(ticker, r)
                self.pending.pop(ticker)
                done += 1
        if done:
            self.save()
        return done

    def stats_line(self) -> str:
        s = self.ledger.stats()
        if not s["settled"]:
            return f"open={s['open']} settled=0 (nothing to report yet)"
        return (f"open={s['open']} settled={s['settled']} losses={s['losses']} "
                f"vs break-even {s['break_even_losses']:.2f} | net ${s['net']:.4f} "
                f"({s['net_per_contract']*100:+.3f}c/contract) | exact p={s['exact_p']:.3f}")
