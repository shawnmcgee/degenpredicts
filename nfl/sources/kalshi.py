"""Kalshi event-contract prices for the NFL.

The college module in ``cfb/sources/kalshi.py`` was written against live payloads captured
from the NCAAF series. **These NFL series tickers and rules wordings are not confirmed the
same way** - the exchange was unreachable from the machine this was written on - so this
module is built to be wrong safely rather than to assume it is right:

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
from datetime import datetime

import pandas as pd

from .. import config
from core.http import get_json

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
        "pulled_at": datetime.utcnow().isoformat(timespec="seconds"),
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

    P(X > 10) can never exceed P(X > 6). When the asks say otherwise, at least one quote is
    stale or a placeholder. With real liquidity behind it that is either a mispricing or a
    warning that the whole event's quotes are junk - either way, worth surfacing.
    """
    out = []
    for (ev, team), grp in ladder.groupby(["event_ticker", ladder["team"].fillna("")]):
        g = grp.dropna(subset=["yes_ask"]).sort_values("strike")
        asks = g["yes_ask"].tolist()
        for i in range(len(asks) - 1):
            if asks[i + 1] > asks[i] + 0.01:      # a higher strike should not cost more
                out.append({"event_ticker": ev, "team": team or None,
                            "lower_strike": g["strike"].iloc[i], "lower_ask": asks[i],
                            "higher_strike": g["strike"].iloc[i + 1], "higher_ask": asks[i + 1]})
    if out:
        log.info("kalshi: %d monotonicity breaks across ladders", len(out))
    return out


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
        "pulled_at": datetime.utcnow().isoformat(timespec="seconds"),
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


def fee(price: float, coef: float | None = None) -> float:
    """Kalshi's published taker fee per contract: 0.07 * P * (1-P) (maker is a quarter of it).

    Verify at kalshi.com/fee-schedule - they revise it periodically.
    """
    coef = config.FEE_COEF.get(config.VENUE, 0.07) if coef is None else coef
    if coef is None:
        return 0.0
    return coef * price * (1 - price)


def contract_ev(p_win: float, ask: float) -> tuple[float, float]:
    """Buying one YES contract at `ask` costs ask + fee and pays 1.00 if it hits.

    Returns (ev_per_contract, roi_fraction).
    """
    if p_win is None or p_win != p_win or ask != ask or ask <= 0 or ask >= 1:
        return float("nan"), float("nan")
    cost = ask + fee(ask)
    return p_win - cost, (p_win - cost) / cost


def _discover() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Series matching 'NFL':")
    for s in list_series("NFL") or [{"ticker": "(none found or API unreachable)", "title": ""}]:
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
        _discover()
    else:
        print(__doc__)
