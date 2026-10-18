"""Kalshi's NFL player-prop ladders.

Confirmed against the live API on 2026-10-18. Five series, one ladder per player per game:

    KXNFLPASSYDS  "Trevor Lawrence: 225+ passing yards"     floor_strike 224.5
    KXNFLRSHYDS   "Chris Rodriguez Jr.: 65+ rushing yards"  floor_strike 64.5
    KXNFLRECYDS   receiving yards, same shape
    KXNFLREC      "Josh Cameron: 3+ receptions"             floor_strike 2.5
    KXNFLANYTD    anytime touchdown

A rung resolves YES when the stat is at least the number in the title, i.e. strictly greater
than ``floor_strike``. The matchup is in ``rules_primary``, away team first, by city
("in the Houston vs Jacksonville Pro Football game originally scheduled for Oct 18, 2026"),
and the ticker carries Kalshi's team code, the player and his jersey number::

    KXNFLPASSYDS-26OCT18HOUJAC-JACTLAWRENCE16-225
                 └ date+teams └ team, name, jersey └ threshold

**Players are matched on team and jersey number, then checked on surname.** Names are the
weak link: Kalshi writes "Chris Rodriguez Jr.", nflverse may not, and two players on one team
can share a surname. Jersey numbers are unique within a team on game day, so the number finds
the player and the surname confirms it. A rung whose number and surname disagree - a stale
roster, a mid-week trade - falls back to a surname match within the two teams playing, and is
left unmatched rather than guessed if that is not unique. An unmatched rung costs one price on
the page; a wrong match would price one player's line off another player's projection.

Read-only market data, no key. Every entry point returns an empty frame instead of raising.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import pandas as pd

from .. import config
from ..props.features import norm_name
from ..teams import TEAM_ALIASES
from . import kalshi

log = logging.getLogger(__name__)

MATCHUP_RE = re.compile(
    r"in the (?P<a>.+?) vs (?P<b>.+?) (?:NFL |pro |professional )?football game"
    r".*?scheduled for (?P<date>[A-Za-z]{3} \d{1,2}, \d{4})", re.I)
THRESH_RE = re.compile(r"(\d+)\s*\+")
PLAYER_SEG_RE = re.compile(r"^(?P<letters>[A-Z]+?)(?P<jersey>\d{1,2})$")

# Props quote thinner than game lines, so the liquidity gate is looser than kalshi.MAX_SPREAD:
# a rung is "tradeable" with a quote no wider than this and real size on the ask.
PROP_MAX_SPREAD = 0.06
PROP_MIN_SIZE = 20.0


def _f(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _aliases_for(team: str) -> list[str]:
    """Every spelling that maps to `team`, longest first, so 'JAC' wins over 'JA'-anything."""
    return sorted({a for a, c in TEAM_ALIASES.items() if c == team}, key=len, reverse=True)


def parse_rung(m: dict, market: str) -> dict | None:
    """One market -> one rung, or None if it is not a player prop we can read."""
    title = str(m.get("title") or m.get("yes_sub_title") or "")
    rules = str(m.get("rules_primary") or "")
    if ":" not in title:
        return None
    player = title.split(":", 1)[0].strip()
    strike = _f(m.get("floor_strike"))
    if strike == strike:
        k = strike + 0.5
    else:
        hit = THRESH_RE.search(title)
        k = float(hit.group(1)) if hit else (1.0 if market == "anytime_td" else float("nan"))
    if k != k:
        return None
    mu = MATCHUP_RE.search(rules)
    ticker = str(m.get("ticker") or "")
    parts = ticker.split("-")
    seg = parts[2] if len(parts) >= 4 else ""
    ps = PLAYER_SEG_RE.match(seg)
    yes_bid, yes_ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
    spread = yes_ask - yes_bid if yes_ask == yes_ask and yes_bid == yes_bid else float("nan")
    ask_size = _f(m.get("yes_ask_size_fp"), 0.0)
    date = None
    if mu:
        try:
            date = datetime.strptime(mu.group("date"), "%b %d, %Y").date()
        except ValueError:
            date = None
    return {
        "market": market, "ticker": ticker, "event_ticker": m.get("event_ticker"),
        "player_raw": player, "k": k,
        "seg_letters": ps.group("letters") if ps else "",
        "jersey": int(ps.group("jersey")) if ps else None,
        "away_raw": mu.group("a").strip() if mu else "", "home_raw": mu.group("b").strip() if mu else "",
        "date": date,
        "yes_bid": yes_bid, "yes_ask": yes_ask, "quote_spread": spread, "ask_size": ask_size,
        "volume": _f(m.get("volume_fp"), 0.0), "open_interest": _f(m.get("open_interest_fp"), 0.0),
        "tradeable": bool(spread == spread and spread <= PROP_MAX_SPREAD and ask_size >= PROP_MIN_SIZE),
        "close_time": m.get("close_time"),
        "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def fetch(series: dict[str, str] | None = None) -> pd.DataFrame:
    series = series or config.KALSHI_PROP_SERIES
    rows = []
    for market, ticker in series.items():
        try:
            ms = kalshi.fetch_markets(ticker, max_pages=40)
        except Exception as e:               # never fatal: the board runs without prices
            log.warning("kalshi props %s: %s", ticker, e)
            ms = []
        parsed = [parse_rung(m, market) for m in ms]
        ok = [r for r in parsed if r]
        if ms and not ok:
            log.warning("kalshi props %s: %d markets returned, none parsed - wording changed?",
                        ticker, len(ms))
        rows.extend(ok)
    return pd.DataFrame(rows)


def attach_games(rungs: pd.DataFrame, games: pd.DataFrame, matcher) -> pd.DataFrame:
    """Resolve each rung's game from its date and the two clubs named in the rules."""
    if rungs.empty:
        return rungs
    r = rungs.copy()
    r["away_team"] = r["away_raw"].map(lambda x: matcher(x) if x else None)
    r["home_team"] = r["home_raw"].map(lambda x: matcher(x) if x else None)
    g = games[["game_id", "date", "home_team", "away_team"]].copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    out = r.merge(g, on=["date", "home_team", "away_team"], how="left")
    # Kalshi phrases away-first, but a neutral-site or reversed listing should still join.
    miss = out["game_id"].isna()
    if miss.any():
        flip = r.loc[miss.values].merge(
            g.rename(columns={"home_team": "away_team", "away_team": "home_team"}),
            on=["date", "home_team", "away_team"], how="left")
        out.loc[miss, "game_id"] = flip["game_id"].values
    n_miss = int(out["game_id"].isna().sum())
    if n_miss:
        log.info("kalshi props: %d rungs are not on the board (already under way, or no "
                 "scheduled game matched)", n_miss)
    return out


def _team_from_segment(letters: str, home: str, away: str) -> str | None:
    best = None
    for team in (home, away):
        for alias in _aliases_for(team):
            if letters.startswith(alias) and (best is None or len(alias) > best[1]):
                best = (team, len(alias))
    return best[0] if best else None


def _surname(name: str) -> str:
    parts = norm_name(name).split()
    return parts[-1] if parts else ""


def match_players(rungs: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Attach `player_id` (GSIS) and `team` to every rung: jersey first, surname to confirm."""
    if rungs.empty:
        return rungs
    r = rungs.copy()
    ros = roster.copy() if roster is not None else pd.DataFrame()
    if ros.empty:
        r["player_id"], r["team"], r["match"] = None, None, "no roster"
        return r
    ros["jersey"] = pd.to_numeric(ros["jersey_number"], errors="coerce")
    ros["surname"] = ros["full_name"].map(_surname)
    ros["key"] = ros["full_name"].map(norm_name)
    by_num = {(t, int(j)): g for (t, j), g in ros.dropna(subset=["jersey"]).groupby(["team", "jersey"])}
    by_team = {t: g for t, g in ros.groupby("team")}

    ids, teams, how = [], [], []
    cache: dict[tuple, tuple] = {}
    for row in r.itertuples():
        key = (row.player_raw, row.seg_letters, row.jersey, row.home_team, row.away_team)
        if key in cache:
            pid, team, h = cache[key]
        else:
            pid, team, h = None, None, "unmatched"
            want_sur, want_key = _surname(row.player_raw), norm_name(row.player_raw)
            seg_team = _team_from_segment(row.seg_letters or "", row.home_team, row.away_team) \
                if isinstance(row.home_team, str) and isinstance(row.away_team, str) else None
            if seg_team is not None and row.jersey is not None:
                cand = by_num.get((seg_team, int(row.jersey)))
                if cand is not None:
                    ok = cand[(cand["surname"] == want_sur) | (cand["key"] == want_key)]
                    if len(ok) == 1:
                        pid, team, h = ok["gsis_id"].iloc[0], seg_team, "jersey"
            if pid is None:
                pool = pd.concat([by_team[t] for t in (row.home_team, row.away_team)
                                  if isinstance(t, str) and t in by_team]) \
                    if any(isinstance(t, str) and t in by_team for t in (row.home_team, row.away_team)) \
                    else pd.DataFrame()
                if len(pool):
                    exact = pool[pool["key"] == want_key]
                    if len(exact) == 1:
                        pid, team, h = exact["gsis_id"].iloc[0], exact["team"].iloc[0], "name"
                    else:
                        sur = pool[pool["surname"] == want_sur]
                        if seg_team is not None:
                            sur = sur[sur["team"] == seg_team]
                        if len(sur) == 1:
                            pid, team, h = sur["gsis_id"].iloc[0], sur["team"].iloc[0], "surname"
                        elif len(sur) > 1:
                            h = "ambiguous"
            cache[key] = (pid, team, h)
        ids.append(pid)
        teams.append(team)
        how.append(h)
    r["player_id"], r["team"], r["match"] = ids, teams, how
    bad = r[r["player_id"].isna()]
    if len(bad):
        log.warning("kalshi props: %d rungs (%d players) unmatched: %s", len(bad),
                    bad["player_raw"].nunique(), sorted(bad["player_raw"].unique())[:15])
    return r
