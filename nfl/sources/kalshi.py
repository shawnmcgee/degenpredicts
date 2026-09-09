"""Kalshi event-contract prices for the NFL.

Series tickers and rules wordings were confirmed against the live API on 2026-09-08. That run
also corrected a real assumption: Kalshi names NFL teams by CITY in the spread ladder
("If Kansas City wins by more than 7.5 points") and with a one-letter disambiguator in the
moneyline subtitle ("New York G"), not by full club name. The matcher in ``sources/odds.py``
carries all those forms; without them the ladder matched nothing.

The module is still built to be wrong safely rather than to assume it is right, because a
series can be renamed at any time:

* series tickers come from :mod:`nfl.config` and are environment-overridable;
* the rules regexes accept the wordings Kalshi is known to use across leagues
  ("NFL football game", "pro football game", or a bare "football game") rather than pinning
  one phrasing;
* the market shape is read from ``floor_strike`` / ``strike_type`` where possible and only
  falls back to the regex for the strike;
* every entry point returns an empty frame instead of raising, and :mod:`nfl.predict` wraps
  the calls, so a wrong ticker costs the exchange columns and nothing else.

Run this module directly to confirm the tickers against the live API::

    python -m nfl.sources.kalshi --discover

That prints every series whose ticker or title mentions the NFL, plus a parsed sample of the
configured series, which is enough to either confirm the defaults or tell you what to set
``DEGEN_KALSHI_ML_SERIES`` and friends to.

Two things this module is careful about, for the same reasons as the college version:

* **Liquidity.** Many listed markets have never traded. EV is computed against the ask, never
  the mid, and a market is only ``tradeable`` if the quote is tight with real size and volume.
* **Coherence.** P(win by >10) can never exceed P(win by >6). When the asks say otherwise at
  least one quote is stale, and the whole event's ladder is suspect.

Read-only market data needs no authentication.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

import pandas as pd

from .. import config
from ..http import get_json

log = logging.getLogger(__name__)

BASE = os.environ.get("DEGEN_KALSHI_BASE", "https://api.elections.kalshi.com/trade-api/v2")


def series(kind: str) -> str:
    """The configured series ticker, read at call time so a `DEGEN_KALSHI_*_SERIES` override
    applies whenever it is set rather than only if it was set before this module was first
    imported. Confirm the defaults with `python -m nfl.sources.kalshi --discover`."""
    return {"moneyline": config.KALSHI_SERIES_MONEYLINE,
            "spread": config.KALSHI_SERIES_SPREAD,
            "total": config.KALSHI_SERIES_TOTAL}[kind]

# Liquidity gates. A quote wider than this is a placeholder, not a market. The NFL board is
# far more liquid than the college one, so these can afford to be tighter.
MAX_SPREAD = float(os.environ.get("DEGEN_KALSHI_MAX_SPREAD", "0.04"))
MIN_ASK_SIZE = float(os.environ.get("DEGEN_KALSHI_MIN_SIZE", "50"))
MIN_VOLUME = float(os.environ.get("DEGEN_KALSHI_MIN_VOLUME", "100"))

# Kalshi's sport phrase varies by series ("college football game", "NFL football game",
# "pro football game"). Matching an explicit alternation rather than one fixed string keeps
# this working if the NFL series words it differently from the NCAAF one.
#
# The qualifier list is deliberately closed. An earlier version used a permissive
# `[A-Za-z ]{0,20}football` and it silently ate team names: with a lazy `(?P<b>.+?)` before
# it, "the Dallas vs New York Giants NFL football game" parsed the home team as "New", because
# "York Giants NFL football" is a legal match for a permissive sport phrase. Every team whose
# name has more than one word would have been truncated, and nothing about the output would
# have looked wrong.
_SPORT = r"(?:NFL |pro |professional |American |college )?football"
_DATE = r"(?P<date>[A-Za-z]{3} \d{1,2}, \d{4})"

RULES_RE = re.compile(
    rf"If (?P<winner>.+?) wins the (?P<a>.+?) vs (?P<b>.+?) {_SPORT} game "
    rf".*?scheduled for {_DATE}", re.I)
TOTAL_RE = re.compile(
    rf"score more than (?P<strike>[\d.]+) points in the (?P<a>.+?) vs (?P<b>.+?) {_SPORT} "
    rf"game .*?scheduled for {_DATE}", re.I)
SPREAD_RE = re.compile(
    rf"If (?P<team>.+?) wins by more than (?P<strike>[\d.]+) points in the (?P<a>.+?) vs "
    rf"(?P<b>.+?) {_SPORT} game .*?scheduled for {_DATE}", re.I)


def _f(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _date_from(text: str):
    try:
        return datetime.strptime(text, "%b %d, %Y").date()
    except ValueError:
        return None


def _liquidity(m: dict) -> dict:
    yes_bid, yes_ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
    spread = yes_ask - yes_bid if yes_ask == yes_ask and yes_bid == yes_bid else float("nan")
    ask_size = _f(m.get("yes_ask_size_fp"), 0.0)
    volume = _f(m.get("volume_fp"), 0.0)
    return {"yes_bid": yes_bid, "yes_ask": yes_ask, "quote_spread": spread,
            "ask_size": ask_size, "bid_size": _f(m.get("yes_bid_size_fp"), 0.0),
            "volume": volume, "open_interest": _f(m.get("open_interest_fp"), 0.0),
            "tradeable": bool(spread == spread and spread <= MAX_SPREAD
                              and ask_size >= MIN_ASK_SIZE and volume >= MIN_VOLUME)}


def fetch_markets(series_ticker: str = None, status: str = "open",
                  max_pages: int = 20) -> list[dict]:
    """Page through /markets for a series. Kalshi paginates with an opaque `cursor`."""
    series_ticker = series_ticker or series("moneyline")
    out, cursor, pages = [], None, 0
    while pages < max_pages:
        params = {"limit": 200, "status": status, "series_ticker": series_ticker}
        if cursor:
            params["cursor"] = cursor
        payload = get_json(f"{BASE}/markets", params=params)
        if not payload:
            break
        batch = payload.get("markets") or []
        out.extend(batch)
        cursor = payload.get("cursor")
        pages += 1
        if not cursor or not batch:
            break
    if not out:
        log.warning("kalshi %s: no markets returned. If this persists, confirm the ticker "
                    "with `python -m nfl.sources.kalshi --discover`.", series_ticker)
    else:
        log.info("kalshi %s: %d markets over %d page(s)", series_ticker, len(out), pages)
    return out


def list_series(keyword: str = "NFL") -> list[dict]:
    """Every series whose ticker or title mentions `keyword`. Use this to confirm tickers."""
    payload = get_json(f"{BASE}/series", params={"limit": 200})
    if not payload:
        return []
    k = keyword.lower()
    return [{"ticker": s.get("ticker"), "title": s.get("title")}
            for s in payload.get("series", [])
            if k in str(s.get("ticker", "")).lower() or k in str(s.get("title", "")).lower()]


def parse_ladder(m: dict, kind: str) -> dict | None:
    """Parse one rung of a spread or total ladder.

    Both series carry the handicap in ``floor_strike`` with ``strike_type: "greater"``, so a
    market resolves YES when the quantity exceeds the strike. The matchup is phrased away-first
    in ``rules_primary``.
    """
    rules = m.get("rules_primary") or ""
    hit = (TOTAL_RE if kind == "total" else SPREAD_RE).search(rules)
    if not hit:
        return None
    strike = _f(m.get("floor_strike"), _f(hit.group("strike")))
    if strike != strike:
        return None
    return {
        "kind": kind, "ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"),
        "strike": strike,
        "team_raw": hit.group("team").strip() if kind == "spread" else None,
        "away_raw": hit.group("a").strip(), "home_raw": hit.group("b").strip(),
        "date": _date_from(hit.group("date")),
        **_liquidity(m),
        "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def ladder_board(kind: str, matcher=None) -> pd.DataFrame:
    """Every strike of every open spread/total market, normalised to our team codes."""
    rows = [parse_ladder(m, kind) for m in fetch_markets(series(kind))]
    rows = [r for r in rows if r and r["date"]]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if matcher is not None:
        df["home_team"] = df["home_raw"].map(lambda x: matcher(x) if x else None)
        df["away_team"] = df["away_raw"].map(lambda x: matcher(x) if x else None)
        df["team"] = df["team_raw"].map(lambda x: matcher(x) if x else None)
    else:
        df["home_team"], df["away_team"], df["team"] = df["home_raw"], df["away_raw"], df["team_raw"]
    log.info("kalshi %s ladder: %d rungs over %d events, %d tradeable",
             kind, len(df), df["event_ticker"].nunique(), int(df["tradeable"].sum()))
    return df


def monotonicity_breaks(ladder: pd.DataFrame) -> list[dict]:
    """Find rungs that contradict each other.

    P(win by >10) can never exceed P(win by >6). When the asks say otherwise at least one
    quote is stale, and the whole event's ladder is suspect.

    The check itself is venue-agnostic and lives in `venues`, which groups by venue so that
    two exchanges quoting the same strike differently reads as cross-venue disagreement rather
    than as one book contradicting itself.
    """
    from . import venues
    return venues.monotonicity_breaks(ladder)


def parse_market(m: dict) -> dict | None:
    """Parse one side of a moneyline market ("<team> wins")."""
    rules = m.get("rules_primary") or ""
    hit = RULES_RE.search(rules)
    team = m.get("yes_sub_title") or str(m.get("title", "")).replace(" wins", "")
    gdate, away, home = None, None, None
    if hit:
        gdate = _date_from(hit.group("date"))
        away, home = hit.group("a").strip(), hit.group("b").strip()  # phrased away-first
    if gdate is None:
        # fall back to the event ticker's embedded date: ...-26SEP13DALNYG
        mt = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", m.get("event_ticker", "") or "")
        if mt:
            try:
                gdate = datetime.strptime(f"{mt.group(2)} {mt.group(3)} 20{mt.group(1)}",
                                          "%b %d %Y").date()
            except ValueError:
                gdate = None
    return {
        "ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"),
        "kalshi_team": team, "away_raw": away, "home_raw": home, "date": gdate,
        "last_price": _f(m.get("last_price_dollars")),
        **_liquidity(m),
        "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def moneyline_board(matcher=None) -> pd.DataFrame:
    """One row per (game, side) with the market's price, normalised to our team codes."""
    rows = [parse_market(m) for m in fetch_markets(series("moneyline"))]
    rows = [r for r in rows if r and r["kalshi_team"] and r["date"]]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if matcher is not None:
        df["team"] = df["kalshi_team"].map(matcher)
        df["home_team"] = df["home_raw"].map(lambda x: matcher(x) if x else None)
        df["away_team"] = df["away_raw"].map(lambda x: matcher(x) if x else None)
    else:
        df["team"] = df["kalshi_team"]
        df["home_team"], df["away_team"] = df["home_raw"], df["away_raw"]
    log.info("kalshi board: %d sides, %d tradeable", len(df), int(df["tradeable"].sum()))
    return df


VENUE = "kalshi"          # the name this venue is known by in `venues.py` and the fee table


def fee(price: float, coef: float | None = None) -> float:
    """Kalshi's published taker fee per contract: 0.07 * P * (1-P) (maker is a quarter of it).

    Verify at kalshi.com/fee-schedule - they revise it periodically. The maths lives in
    `venues.fee`, shared with Polymarket because both schedules have this same shape; this
    wrapper pins the venue so a bare `kalshi.fee(ask)` can never be charged at another
    exchange's rate.
    """
    from . import venues
    return venues.fee(price, VENUE, coef)


def contract_ev(p_win: float, ask: float) -> tuple[float, float]:
    """Buying one YES contract at `ask` costs ask + fee and pays 1.00 if it hits.

    Returns (ev_per_contract, roi_fraction).
    """
    from . import venues
    return venues.contract_ev(p_win, ask, VENUE)


def _discover(keyword: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    keyword = keyword or os.environ.get("DEGEN_KALSHI_KEYWORD", "").strip() or "NFL"
    print(f"Series matching {keyword!r}:")
    for s in list_series(keyword) or [{"ticker": "(none found or API unreachable)", "title": ""}]:
        print(f"  {s['ticker']:<24} {s['title']}")
    for label, ticker, kind in (("moneyline", series("moneyline"), None),
                                ("spread", series("spread"), "spread"),
                                ("total", series("total"), "total")):
        markets = fetch_markets(ticker)
        print(f"\n{label} series {ticker}: {len(markets)} open markets")
        for m in markets[:2]:
            parsed = parse_market(m) if kind is None else parse_ladder(m, kind)
            print(f"  rules: {str(m.get('rules_primary'))[:140]}")
            print(f"  parsed: {parsed}")
        if markets and all((parse_market(m) if kind is None else parse_ladder(m, kind)) is None
                           for m in markets[:5]):
            print("  !! markets returned but none parsed - the rules wording has changed; "
                  "update the regexes at the top of this module.")


if __name__ == "__main__":
    import sys
    if "--discover" in sys.argv:
        # optional: --keyword FOOTBALL, when the series is not named for the league
        kw = None
        if "--keyword" in sys.argv:
            i = sys.argv.index("--keyword")
            kw = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
        _discover(kw)
    else:
        print(__doc__)
