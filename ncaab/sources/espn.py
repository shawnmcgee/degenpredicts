"""ESPN's men's college basketball scoreboard: finals, the slate, and the book's line.

    python -m ncaab.sources.espn               # refresh data/ncaab/games.csv
    python -m ncaab.sources.espn --day 2026-11-03   # print one day's parsed scoreboard

One public endpoint, no key, no quota - ``scoreboard?dates=YYYYMMDD&groups=50`` returns every
Division I game that day with:

| Field | Used for |
|---|---|
| ``competitors[].team.id`` | the rating key - ESPN's ids are stable, so no name matching at all |
| ``competitors[].score``, ``status.type`` | finals, and whether a game is still to tip |
| ``competitions[].odds`` | ESPN's partner book (DraftKings): spread, total and prices. It stays on finished games, which makes it the closing line the market-aware models train against |
| ``curatedRank.current`` | the AP rank at the time (99 = unranked) |
| ``neutralSite``, ``conferenceCompetition``, ``team.conferenceId`` | context the ratings cannot see |
| ``season.type``, ``notes[].headline`` | postseason, and which tournament |

The odds are read by :mod:`ncaab.sources.espn_odds`, the same parser the football, hockey and
soccer boards already run live, so both of ESPN's odds shapes are covered.

**Written without reaching ESPN from the machine it was built on.** Every field above is read
defensively and the tests parse payloads of both shapes, but the first Actions run is the real
test: look for ``ESPN <date>: N games, M finals, K with a spread`` in the training log. A day
that fails to load is retried on the next run rather than recorded as empty.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .. import config
from ..http import get_json
from . import espn_odds

log = logging.getLogger(__name__)

GAME_COLS = ["game_id", "date", "season", "season_type", "kickoff_utc",
             "home_id", "away_id", "home_team", "away_team", "home_abbr", "away_abbr",
             "home_conf", "away_conf", "home_rank", "away_rank", "neutral_site", "conf_game",
             "note", "home_points", "away_points", "total_points", "home_margin",
             "spread_home", "total_line", "provider"]
DAY_COLS = ["date", "n_games", "n_final", "fetched_at"]
UNRANKED = 99


def _int(x, default=None):
    try:
        v = int(float(x))
    except (TypeError, ValueError):
        return default
    return v


def _team(c: dict) -> dict:
    t = c.get("team") if isinstance(c.get("team"), dict) else {}
    name = t.get("location") or t.get("shortDisplayName") or t.get("displayName") or t.get("name")
    rank = c.get("curatedRank") if isinstance(c.get("curatedRank"), dict) else {}
    r = _int(rank.get("current"), UNRANKED)
    return {"id": str(t.get("id") or c.get("id") or ""),
            "name": str(name or ""),
            "abbr": str(t.get("abbreviation") or ""),
            "conf": str(t.get("conferenceId") or ""),
            "rank": r if r and 1 <= r <= 25 else UNRANKED,
            "score": str(c.get("score") if c.get("score") is not None else "").strip()}


def _note(comp: dict, ev: dict) -> str:
    for n in (comp.get("notes") or ev.get("notes") or []):
        if isinstance(n, dict) and n.get("headline"):
            return str(n["headline"]).strip()
    return ""


def parse_scoreboard(payload) -> pd.DataFrame:
    """Scoreboard JSON -> one row per game. Pure: no network, no clock.

    ``state`` is ESPN's pre / in / post; ``final`` is a completed game with two numeric scores.
    ``spread_home`` is in book convention - negative means the home side is favoured.
    """
    if not isinstance(payload, dict):
        return pd.DataFrame()
    rows = []
    for ev in payload.get("events") or []:
        if not isinstance(ev, dict):
            continue
        comp = (ev.get("competitions") or [{}])[0] or {}
        sides = {c.get("homeAway"): c for c in comp.get("competitors") or [] if isinstance(c, dict)}
        if "home" not in sides or "away" not in sides:
            continue
        h, a = _team(sides["home"]), _team(sides["away"])
        if not h["id"] or not a["id"]:
            continue
        ts = pd.to_datetime(comp.get("date") or ev.get("date"), utc=True, errors="coerce")
        if ts is pd.NaT or ts != ts:
            continue
        st = ((comp.get("status") or ev.get("status") or {}).get("type") or {})
        state = str(st.get("state") or "")
        final = bool(st.get("completed")) and state == "post" \
            and h["score"].isdigit() and a["score"].isdigit()
        season = ev.get("season") if isinstance(ev.get("season"), dict) else {}
        odds = espn_odds.parse_odds(comp.get("odds") or ev.get("odds") or [],
                                    h["abbr"], a["abbr"], config.MAX_SPREAD)
        local = ts.tz_convert(config.ET)
        hp = int(h["score"]) if final else np.nan
        ap = int(a["score"]) if final else np.nan
        rows.append({
            "game_id": str(ev.get("id") or comp.get("id") or ""),
            "date": local.date(), "season": config.season_of(local.date()),
            "season_type": _int(season.get("type"), 2),
            "kickoff_utc": ts.isoformat(), "state": state, "final": final,
            "status": str(st.get("name") or ""),
            "home_id": h["id"], "away_id": a["id"],
            "home_team": h["name"], "away_team": a["name"],
            "home_abbr": h["abbr"], "away_abbr": a["abbr"],
            "home_conf": h["conf"], "away_conf": a["conf"],
            "home_rank": h["rank"], "away_rank": a["rank"],
            "neutral_site": bool(comp.get("neutralSite")),
            "conf_game": bool(comp.get("conferenceCompetition")),
            "note": _note(comp, ev),
            "home_points": hp, "away_points": ap,
            "total_points": hp + ap if final else np.nan,
            "home_margin": hp - ap if final else np.nan,
            "spread_home": odds["spread_home"], "total_line": odds["total_line"],
            "spread_home_price": odds["spread_home_price"],
            "spread_away_price": odds["spread_away_price"],
            "over_price": odds["over_price"], "under_price": odds["under_price"],
            "open_spread_home": odds["open_spread_home"], "open_total": odds["open_total"],
            "provider": odds["provider"],
        })
    return pd.DataFrame(rows)


def fetch_day(day: date) -> pd.DataFrame | None:
    """One day's scoreboard, or None when ESPN did not answer (so the day is retried later)."""
    payload = get_json(config.ESPN_SCOREBOARD,
                       params={**config.ESPN_PARAMS, "dates": day.strftime("%Y%m%d")})
    if payload is None or not isinstance(payload, dict) or "events" not in payload:
        log.warning("ESPN %s: no scoreboard", day)
        return None
    df = parse_scoreboard(payload)
    if len(df):
        log.debug("ESPN %s: %d games, %d finals, %d with a spread", day, len(df),
                  int(df["final"].sum()), int(df["spread_home"].notna().sum()))
    return df


# ---------------------------------------------------------------------------------
# the games cache
# ---------------------------------------------------------------------------------
def load() -> pd.DataFrame:
    if not config.GAMES.exists():
        return pd.DataFrame(columns=GAME_COLS)
    df = pd.read_csv(config.GAMES, dtype={"game_id": str, "home_id": str, "away_id": str,
                                          "home_conf": str, "away_conf": str})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in ("home_conf", "away_conf", "note", "provider"):
        if c in df:
            df[c] = df[c].fillna("")
    for c in ("neutral_site", "conf_game"):
        if c in df:
            df[c] = df[c].astype(str).str.lower().isin(("true", "1", "1.0"))
    return df


def _days() -> pd.DataFrame:
    if not config.DAYS.exists():
        return pd.DataFrame(columns=DAY_COLS)
    d = pd.read_csv(config.DAYS)
    d["date"] = pd.to_datetime(d["date"]).dt.date
    return d


def save(games: pd.DataFrame, days: pd.DataFrame | None = None) -> None:
    config.ensure_dirs()
    out = (games.reindex(columns=GAME_COLS)
                .drop_duplicates("game_id", keep="last")
                .sort_values(["date", "kickoff_utc", "game_id"]))
    out.to_csv(config.GAMES, index=False)
    if days is not None:
        (days.drop_duplicates("date", keep="last").sort_values("date")
             .reindex(columns=DAY_COLS).to_csv(config.DAYS, index=False))


def days_to_fetch(start: date, end: date, done: set, recheck_days: int = 3) -> list[date]:
    """In-season days not fetched for good yet, plus the last few (late finals, stat fixes).

    A day counts as done once it was fetched after it was over - an empty Christmas Eve is
    remembered as empty instead of being asked for again on every run."""
    out, d = [], start
    while d <= end:
        if config.in_season(d) and (d not in done or (end - d).days < recheck_days):
            out.append(d)
        d += timedelta(days=1)
    return out


def update(start: date | None = None, end: date | None = None, recheck_days: int = 3,
           fetch=fetch_day) -> pd.DataFrame:
    """Bring games.csv up to date through yesterday and return every final."""
    games, days = load(), _days()
    end = end or (config.today_et() - timedelta(days=1))
    start = start or config.season_start(config.FIRST_SEASON)
    done = set(days["date"]) if len(days) else set()
    todo = days_to_fetch(start, end, done, recheck_days)
    if not todo:
        log.info("games cache current: %d finals", len(games))
        return games

    log.info("fetching %d ESPN scoreboard days (%s .. %s)", len(todo), todo[0], todo[-1])
    frames, logged = [], []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for i, day in enumerate(todo, 1):
        df = fetch(day)
        if df is None:
            continue                  # not recorded: retried next run
        fin = df[df["final"]] if len(df) else df
        if len(fin):
            frames.append(fin)
        logged.append({"date": day, "n_games": len(df), "n_final": len(fin), "fetched_at": now})
        if i % 50 == 0:
            log.info("  %d/%d days, %d finals so far", i, len(todo), sum(len(f) for f in frames))
        if config.ESPN_SLEEP and fetch is fetch_day:
            time.sleep(config.ESPN_SLEEP)

    if frames:
        new = pd.concat(frames, ignore_index=True).reindex(columns=GAME_COLS)
        games = pd.concat([games, new], ignore_index=True) if len(games) else new
    if logged:
        days = pd.concat([days, pd.DataFrame(logged)], ignore_index=True) if len(days) \
            else pd.DataFrame(logged)
    save(games, days)
    g = load()
    if len(g):
        lined = g["spread_home"].notna().mean() * 100
        log.info("cache: %d finals over seasons %s, %.0f%% with a closing spread",
                 len(g), sorted(int(s) for s in g["season"].unique()), lined)
    return g


def slate(days: int | None = None, fetch=fetch_day, now=None) -> pd.DataFrame:
    """Games from today through ``days`` ahead that have not tipped, with their prices.

    Anything under way is dropped here, before it can reach the board: a price taken during
    the game is an in-play price, and grading against it is how the NFL board once published
    a 28.5 total on a game that closed 42.5.
    """
    horizon = config.BOARD_DAYS if days is None else days
    today = config.today_et()
    frames = [fetch(today + timedelta(days=i)) for i in range(horizon + 1)]
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    s = pd.concat(frames, ignore_index=True).drop_duplicates("game_id", keep="last")
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    tipped = (s["state"] != "pre") | (pd.to_datetime(s["kickoff_utc"], utc=True) <= now)
    gone = s["status"].str.contains("POSTPONED|CANCELED|CANCELLED|SUSPENDED", case=False, na=False)
    if tipped.any():
        log.info("ESPN slate: skipped %d games already under way or over", int(tipped.sum()))
    s = s[~tipped & ~gone].copy()
    log.info("ESPN slate: %d games to come, %d with a spread, %d with a total (%s)", len(s),
             int(s["spread_home"].notna().sum()), int(s["total_line"].notna().sum()),
             ", ".join(sorted(set(s["provider"]) - {""})) or "-")
    return s.reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="print one day's parsed scoreboard, write nothing")
    args = ap.parse_args(argv)
    if args.day:
        df = fetch_day(date.fromisoformat(args.day))
        cols = ["kickoff_utc", "away_team", "home_team", "state", "home_points", "away_points",
                "spread_home", "total_line", "home_rank", "away_rank", "neutral_site"]
        print("no scoreboard" if df is None else df[[c for c in cols if c in df]].to_string())
        return
    update()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
