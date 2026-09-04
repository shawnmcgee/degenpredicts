"""CollegeFootballData.com client.

CFBD is the spine of this pipeline. Crucially its ``/lines`` endpoint returns *historical*
spreads and totals (including opening numbers), so the market-aware model can be trained from
day one instead of after a season of self-logging.

Free tier is 1,000 calls/month. A full 10-season backfill costs roughly 40 calls; the daily
run costs 3-4. Auth is a Bearer token.

Field names differ between CFBD API versions (``home_team`` vs ``homeTeam``), so every read
goes through :func:`pick` which tries both.
"""
from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd

from .. import config as _cfg  # noqa: F401  (keeps import symmetry if reorganised)
from ..config import (CFBD_BASE, CFBD_KEY, GAMES, LINES, PREFERRED_PROVIDERS, RETURNING,
                      SP_PRIOR, ensure_dirs, season_of)
from core.http import get_json

log = logging.getLogger(__name__)

GAME_COLS = ["game_id", "season", "week", "season_type", "date", "start_time_tbd",
             "home_team", "away_team", "home_points", "away_points", "total_points",
             "home_margin", "neutral_site", "conference_game", "home_conf", "away_conf",
             "completed"]


def _headers() -> dict:
    if not CFBD_KEY:
        raise RuntimeError("CFBD_API_KEY is not set")
    return {"Authorization": f"Bearer {CFBD_KEY}", "Accept": "application/json"}


def pick(d: dict, *names, default=None):
    """Read the first present key out of several spellings (snake_case / camelCase)."""
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _parse_dt(s):
    if not s:
        return None, None
    try:
        ts = pd.Timestamp(s)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        et = ts.tz_convert("America/New_York")
        return et.date(), et
    except Exception:
        return None, None


# ---------------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------------
def fetch_games(season: int, season_type: str = "both") -> pd.DataFrame:
    rows = []
    types = ["regular", "postseason"] if season_type == "both" else [season_type]
    for st in types:
        data = get_json(f"{CFBD_BASE}/games",
                        params={"year": season, "seasonType": st, "division": "fbs"},
                        headers=_headers())
        if not data:
            log.warning("no games returned for %s %s", season, st)
            continue
        for g in data:
            gdate, et = _parse_dt(pick(g, "startDate", "start_date"))
            hp = pick(g, "homePoints", "home_points")
            ap = pick(g, "awayPoints", "away_points")
            hp = _num(hp) if hp is not None else float("nan")
            ap = _num(ap) if ap is not None else float("nan")
            home = pick(g, "homeTeam", "home_team")
            away = pick(g, "awayTeam", "away_team")
            if not home or not away or gdate is None:
                continue
            done = hp == hp and ap == ap
            rows.append({
                "game_id": str(pick(g, "id", "gameId")),
                "season": int(pick(g, "season", default=season)),
                "week": int(pick(g, "week", default=0)),
                "season_type": st,
                "date": gdate,
                "start_time_tbd": bool(pick(g, "startTimeTBD", "start_time_tbd", default=False)),
                "tip_et": et.strftime("%a %b %d, %-I:%M %p") if et is not None else "",
                "home_team": str(home), "away_team": str(away),
                "home_points": hp, "away_points": ap,
                "total_points": hp + ap if done else float("nan"),
                "home_margin": hp - ap if done else float("nan"),
                "neutral_site": bool(pick(g, "neutralSite", "neutral_site", default=False)),
                "conference_game": bool(pick(g, "conferenceGame", "conference_game", default=False)),
                "home_conf": pick(g, "homeConference", "home_conference", default=""),
                "away_conf": pick(g, "awayConference", "away_conference", default=""),
                "completed": done,
            })
    return pd.DataFrame(rows)


def load_games() -> pd.DataFrame:
    if GAMES.exists():
        df = pd.read_csv(GAMES, dtype={"game_id": str}, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        return df
    return pd.DataFrame(columns=GAME_COLS)


def update_games(seasons: list[int] | None = None) -> pd.DataFrame:
    """Fetch whole seasons at a time (one call each) and merge into the cache."""
    from ..config import FIRST_SEASON, today_et
    cache = load_games()
    if seasons is None:
        this = season_of(today_et())
        have_complete = set()
        if len(cache):
            # a past season is 'done' once we hold completed postseason games for it
            for s, grp in cache.groupby("season"):
                if s < this and (grp["season_type"] == "postseason").any():
                    have_complete.add(int(s))
        seasons = [s for s in range(FIRST_SEASON, this + 1) if s not in have_complete]
    if not seasons:
        log.info("games cache current: %d games", len(cache))
        return cache

    log.info("fetching CFBD games for seasons %s", seasons)
    frames = [cache] if len(cache) else []
    for s in seasons:
        f = fetch_games(s)
        log.info("  %s: %d games (%d final)", s, len(f), int(f["completed"].sum()))
        if len(f):
            frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "game_id"]).drop_duplicates("game_id", keep="last")
    ensure_dirs()
    out.to_csv(GAMES, index=False)
    return load_games()


# ---------------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------------
LINE_COLS = ["game_id", "season", "week", "date", "home_team", "away_team", "provider",
             "spread_home", "spread_open", "total_line", "total_open", "home_ml", "away_ml"]


def fetch_lines(season: int, week: int | None = None, season_type: str = "regular") -> pd.DataFrame:
    params = {"year": season, "seasonType": season_type}
    if week is not None:
        params["week"] = week
    data = get_json(f"{CFBD_BASE}/lines", params=params, headers=_headers())
    if not data:
        return pd.DataFrame(columns=LINE_COLS)
    rows = []
    for g in data:
        gdate, _ = _parse_dt(pick(g, "startDate", "start_date"))
        base = {
            "game_id": str(pick(g, "id", "gameId")),
            "season": int(pick(g, "season", default=season)),
            "week": int(pick(g, "week", default=0)),
            "date": gdate,
            "home_team": pick(g, "homeTeam", "home_team"),
            "away_team": pick(g, "awayTeam", "away_team"),
        }
        for ln in pick(g, "lines", default=[]) or []:
            rows.append({**base,
                         "provider": pick(ln, "provider", default="?"),
                         # CFBD quotes the spread from the home side: -6.5 = home favoured
                         "spread_home": _num(pick(ln, "spread")),
                         "spread_open": _num(pick(ln, "spreadOpen", "spread_open")),
                         "total_line": _num(pick(ln, "overUnder", "over_under")),
                         "total_open": _num(pick(ln, "overUnderOpen", "over_under_open")),
                         "home_ml": _num(pick(ln, "homeMoneyline", "home_moneyline")),
                         "away_ml": _num(pick(ln, "awayMoneyline", "away_moneyline"))})
    return pd.DataFrame(rows)


def consensus(lines: pd.DataFrame) -> pd.DataFrame:
    """Collapse many providers per game into one row: preferred provider, else median."""
    if lines.empty:
        return lines
    out = []
    for gid, grp in lines.groupby("game_id"):
        have = {r.provider: r for r in grp.itertuples()}
        chosen = next((p for p in PREFERRED_PROVIDERS if p in have), None)
        row = grp.iloc[0][["game_id", "season", "week", "date", "home_team", "away_team"]].to_dict()
        if chosen:
            r = have[chosen]
            row.update({"spread_home": r.spread_home, "spread_open": r.spread_open,
                        "total_line": r.total_line, "total_open": r.total_open,
                        "provider": chosen})
        else:
            row.update({"spread_home": grp["spread_home"].median(),
                        "spread_open": grp["spread_open"].median(),
                        "total_line": grp["total_line"].median(),
                        "total_open": grp["total_open"].median(),
                        "provider": "median"})
        row["n_providers"] = len(grp)
        out.append(row)
    return pd.DataFrame(out)


def load_lines() -> pd.DataFrame:
    if LINES.exists():
        df = pd.read_csv(LINES, dtype={"game_id": str}, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        return df
    return pd.DataFrame(columns=LINE_COLS + ["n_providers"])


def update_lines(seasons: list[int] | None = None, refresh_current: bool = True) -> pd.DataFrame:
    from ..config import FIRST_SEASON, today_et
    cache = load_lines()
    this = season_of(today_et())
    if seasons is None:
        have = set(cache["season"].unique()) if len(cache) else set()
        seasons = [s for s in range(FIRST_SEASON, this + 1) if s not in have or s == this]
    frames = [cache[~cache["season"].isin(seasons)]] if len(cache) else []
    for s in seasons:
        for st in ("regular", "postseason"):
            f = fetch_lines(s, season_type=st)
            if len(f):
                frames.append(consensus(f))
        log.info("  lines %s fetched", s)
    if not frames:
        return cache
    out = pd.concat(frames, ignore_index=True).drop_duplicates("game_id", keep="last")
    ensure_dirs()
    out.to_csv(LINES, index=False)
    log.info("lines cache: %d games", len(out))
    return load_lines()


# ---------------------------------------------------------------------------------
# Preseason-known context: returning production and PRIOR-season SP+
# ---------------------------------------------------------------------------------
def fetch_returning(season: int) -> pd.DataFrame:
    data = get_json(f"{CFBD_BASE}/player/returning", params={"year": season}, headers=_headers())
    if not data:
        return pd.DataFrame()
    return pd.DataFrame([{
        "season": season, "team": pick(r, "team"),
        "ret_ppa": _num(pick(r, "totalPPA", "total_ppa")),
        "ret_pass_pct": _num(pick(r, "percentPassingPPA", "percent_passing_ppa")),
        "ret_rush_pct": _num(pick(r, "percentRushingPPA", "percent_rushing_ppa")),
        "ret_usage": _num(pick(r, "usage")),
    } for r in data if pick(r, "team")])


def fetch_sp(season: int) -> pd.DataFrame:
    data = get_json(f"{CFBD_BASE}/ratings/sp", params={"year": season}, headers=_headers())
    if not data:
        return pd.DataFrame()
    rows = []
    for r in data:
        team = pick(r, "team")
        if not team:
            continue
        off = pick(r, "offense", default={}) or {}
        dfn = pick(r, "defense", default={}) or {}
        rows.append({"season": season, "team": team,
                     "sp_overall": _num(pick(r, "rating")),
                     "sp_off": _num(pick(off, "rating")),
                     "sp_def": _num(pick(dfn, "rating"))})
    return pd.DataFrame(rows)


def _update_seasonal(path, fetcher, label) -> pd.DataFrame:
    from ..config import FIRST_SEASON, today_et
    this = season_of(today_et())
    cache = pd.read_csv(path) if path.exists() else pd.DataFrame()
    have = set(cache["season"].unique()) if len(cache) else set()
    want = [s for s in range(FIRST_SEASON - 1, this + 1) if s not in have or s >= this - 1]
    frames = [cache[~cache["season"].isin(want)]] if len(cache) else []
    for s in want:
        f = fetcher(s)
        if len(f):
            frames.append(f)
    if not frames:
        log.warning("no %s data retrieved", label)
        return cache
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["season", "team"], keep="last")
    ensure_dirs()
    out.to_csv(path, index=False)
    log.info("%s: %d team-seasons", label, len(out))
    return out


def update_returning() -> pd.DataFrame:
    return _update_seasonal(RETURNING, fetch_returning, "returning production")


def update_sp() -> pd.DataFrame:
    """SP+ is an END-OF-SEASON rating. It is only ever joined from season-1, never the
    current season, so it cannot leak."""
    return _update_seasonal(SP_PRIOR, fetch_sp, "SP+ ratings")


def load_returning() -> pd.DataFrame:
    return pd.read_csv(RETURNING) if RETURNING.exists() else pd.DataFrame()


def load_sp() -> pd.DataFrame:
    return pd.read_csv(SP_PRIOR) if SP_PRIOR.exists() else pd.DataFrame()
