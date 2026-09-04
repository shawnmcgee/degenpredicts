"""Kalshi event-contract prices for college football.

Structure, confirmed against a live response on 2026-09-04:

    series       KXNCAAFGAME                            (moneyline: "<team> wins")
    event        KXNCAAFGAME-26SEP17SYRPITT             one game
    market       KXNCAAFGAME-26SEP17SYRPITT-SYR         one side of that game

Each market is binary and settles at $1.00 or $0.00, so ``yes_ask_dollars`` reads directly as
the market's probability. That is what makes this comparable to our model: the margin model
plus its fitted sigma gives P(team wins), and the ask gives the market's price for the same
event.

Two things this module is careful about:

* **Liquidity.** Many listed markets have never traded. The live sample had one game quoting
  0.08 bid / 0.81 ask with zero volume - an "edge" against that ask is imaginary. Markets are
  flagged tradeable only if the spread is tight and there is real size and volume.
* **Team names.** Kalshi's ``yes_sub_title`` is sometimes bare ("Syracuse", "Cal Poly") and
  sometimes carries the mascot ("Central Washington Wildcats"). We reuse the Odds API matcher
  to reduce to CFBD names, and drop anything that isn't FBS.

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

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_MONEYLINE = "KXNCAAFGAME"      # "<team> wins"
SERIES_SPREAD = "KXNCAAFSPREAD"       # "<team> wins by over X points"   (ladder of strikes)
SERIES_TOTAL = "KXNCAAFTOTAL"         # "Over X points scored"           (ladder of strikes)

# "If Syracuse wins the Syracuse vs Pittsburgh college football game originally scheduled for
#  Sep 17, 2026, then the market resolves to Yes."
RULES_RE = re.compile(
    r"If (?P<winner>.+?) wins the (?P<a>.+?) vs (?P<b>.+?) college football game "
    r"originally scheduled for (?P<date>[A-Za-z]{3} \d{1,2}, \d{4})", re.I)

# Liquidity gates. A quote wider than this is a placeholder, not a market.
MAX_SPREAD = float(os.environ.get("DEGEN_KALSHI_MAX_SPREAD", "0.06"))
MIN_ASK_SIZE = float(os.environ.get("DEGEN_KALSHI_MIN_SIZE", "25"))
MIN_VOLUME = float(os.environ.get("DEGEN_KALSHI_MIN_VOLUME", "50"))


def _f(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def fetch_markets(series_ticker: str = SERIES_MONEYLINE, status: str = "open",
                  max_pages: int = 20) -> list[dict]:
    """Page through /markets for a series. Kalshi paginates with an opaque `cursor`."""
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
    log.info("kalshi %s: %d markets over %d page(s)", series_ticker, len(out), pages)
    return out


def list_series(keyword: str = "NCAA") -> list[str]:
    """Helper for finding spread/total series tickers, which differ from the moneyline one."""
    payload = get_json(f"{BASE}/series", params={"limit": 200})
    if not payload:
        return []
    hits = [s.get("ticker") for s in payload.get("series", [])
            if keyword.lower() in str(s.get("ticker", "")).lower()
            or keyword.lower() in str(s.get("title", "")).lower()]
    return [h for h in hits if h]


# "If the teams collectively score more than 78.5 points in the Sacred Heart vs Monmouth
#  college football game originally scheduled for Sep 5, 2026, ..."
TOTAL_RE = re.compile(
    r"score more than (?P<strike>[\d.]+) points in the (?P<a>.+?) vs (?P<b>.+?) college "
    r"football game .*?scheduled for (?P<date>[A-Za-z]{3} \d{1,2}, \d{4})", re.I)

# "If San Diego wins by more than 8.5 points in the UC Davis vs San Diego college football
#  game originally scheduled for Sep 5, 2026, ..."
SPREAD_RE = re.compile(
    r"If (?P<team>.+?) wins by more than (?P<strike>[\d.]+) points in the (?P<a>.+?) vs "
    r"(?P<b>.+?) college football game .*?scheduled for (?P<date>[A-Za-z]{3} \d{1,2}, \d{4})", re.I)


def _date_from(text: str):
    try:
        return datetime.strptime(text, "%b %d, %Y").date()
    except ValueError:
        return None


def parse_ladder(m: dict, kind: str) -> dict | None:
    """Parse one rung of a spread or total ladder.

    Both series expose the handicap in ``floor_strike`` with ``strike_type: "greater"``, so a
    market always resolves YES when the quantity exceeds the strike. The matchup is phrased
    away-first in ``rules_primary``, same as the moneyline series.
    """
    rules = m.get("rules_primary") or ""
    hit = (TOTAL_RE if kind == "total" else SPREAD_RE).search(rules)
    if not hit:
        return None
    strike = _f(m.get("floor_strike"), _f(hit.group("strike")))
    if strike != strike:
        return None
    yes_bid, yes_ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
    spread = yes_ask - yes_bid if yes_ask == yes_ask and yes_bid == yes_bid else float("nan")
    ask_size = _f(m.get("yes_ask_size_fp"), 0.0)
    volume = _f(m.get("volume_fp"), 0.0)
    return {
        "kind": kind, "ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"),
        "strike": strike,
        "team_raw": hit.group("team").strip() if kind == "spread" else None,
        "away_raw": hit.group("a").strip(), "home_raw": hit.group("b").strip(),
        "date": _date_from(hit.group("date")),
        "yes_bid": yes_bid, "yes_ask": yes_ask, "quote_spread": spread,
        "ask_size": ask_size, "volume": volume,
        "open_interest": _f(m.get("open_interest_fp"), 0.0),
        "tradeable": bool(spread == spread and spread <= MAX_SPREAD
                          and ask_size >= MIN_ASK_SIZE and volume >= MIN_VOLUME),
        "pulled_at": datetime.utcnow().isoformat(timespec="seconds"),
    }


def ladder_board(kind: str, matcher=None) -> pd.DataFrame:
    """Every strike of every open spread/total market, normalised to CFBD names."""
    series = SERIES_SPREAD if kind == "spread" else SERIES_TOTAL
    rows = [parse_ladder(m, kind) for m in fetch_markets(series)]
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

    P(X > 8.5) can never exceed P(X > 4.5). When the asks say otherwise, at least one quote is
    stale or a placeholder. The live sample had San Diego "wins by over 8.5" asking 90c while
    "over 4.5" asked 17c - on an untraded book. Worth surfacing: with real liquidity it is
    either a mispricing or a warning that the whole event's quotes are junk.
    """
    out = []
    for (ev, team), grp in ladder.groupby(["event_ticker", ladder["team"].fillna("")]):
        g = grp.dropna(subset=["yes_ask"]).sort_values("strike")
        asks = g["yes_ask"].tolist()
        for i in range(len(asks) - 1):
            if asks[i + 1] > asks[i] + 0.01:      # higher strike should not cost more
                out.append({"event_ticker": ev, "team": team or None,
                            "lower_strike": g["strike"].iloc[i], "lower_ask": asks[i],
                            "higher_strike": g["strike"].iloc[i + 1], "higher_ask": asks[i + 1]})
    if out:
        log.info("kalshi: %d monotonicity breaks across ladders", len(out))
    return out


def parse_market(m: dict) -> dict | None:
    rules = m.get("rules_primary") or ""
    hit = RULES_RE.search(rules)
    team = m.get("yes_sub_title") or m.get("title", "").replace(" wins", "")
    gdate = None
    if hit:
        try:
            gdate = datetime.strptime(hit.group("date"), "%b %d, %Y").date()
        except ValueError:
            gdate = None
        # Kalshi phrases the matchup away-first.
        away, home = hit.group("a").strip(), hit.group("b").strip()
    else:
        away = home = None
    if gdate is None:
        # fall back to the event ticker's embedded date: ...-26SEP17SYRPITT
        mt = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", m.get("event_ticker", ""))
        if mt:
            try:
                gdate = datetime.strptime(f"{mt.group(2)} {mt.group(3)} 20{mt.group(1)}",
                                          "%b %d %Y").date()
            except ValueError:
                gdate = None

    yes_bid, yes_ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
    spread = yes_ask - yes_bid if yes_ask == yes_ask and yes_bid == yes_bid else float("nan")
    ask_size = _f(m.get("yes_ask_size_fp"), 0.0)
    volume = _f(m.get("volume_fp"), 0.0)
    return {
        "ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"),
        "kalshi_team": team, "away_raw": away, "home_raw": home, "date": gdate,
        "yes_bid": yes_bid, "yes_ask": yes_ask, "quote_spread": spread,
        "ask_size": ask_size, "bid_size": _f(m.get("yes_bid_size_fp"), 0.0),
        "volume": volume, "open_interest": _f(m.get("open_interest_fp"), 0.0),
        "last_price": _f(m.get("last_price_dollars")),
        "tradeable": bool(spread == spread and spread <= MAX_SPREAD
                          and ask_size >= MIN_ASK_SIZE and volume >= MIN_VOLUME),
        "pulled_at": datetime.utcnow().isoformat(timespec="seconds"),
    }


def moneyline_board(matcher=None) -> pd.DataFrame:
    """One row per (game, side) with the market's price, normalised to CFBD team names."""
    rows = [parse_market(m) for m in fetch_markets()]
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
    if p_win != p_win or ask != ask or ask <= 0 or ask >= 1:
        return float("nan"), float("nan")
    cost = ask + fee(ask)
    return p_win - cost, (p_win - cost) / cost
