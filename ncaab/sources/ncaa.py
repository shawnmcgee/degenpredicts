"""Final scores from ncaa.com via the henrygd/ncaa-api wrapper.

The cache in data/games.csv is the source of truth. We only fetch days we don't have, plus a
short re-check window so games that were live last time get their final score.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

import pandas as pd

from .. import config
from ..http import get_json, session
from ..team_names import normalize_team_name

log = logging.getLogger(__name__)

COLUMNS = ["game_id", "date", "season", "home_team", "away_team", "home_points", "away_points",
           "total_points", "home_margin", "neutral_site", "fetched_at"]


def _headers() -> dict:
    return {"x-ncaa-key": config.NCAA_API_KEY} if config.NCAA_API_KEY else {}


def fetch_day(day: date) -> list[dict]:
    url = f"{config.NCAA_API_BASE}/scoreboard/basketball-men/d1/{day:%Y/%m/%d}/all-conf"
    payload = get_json(url, headers=_headers())
    if not payload:
        return []
    out = []
    for obj in payload.get("games", []):
        g = obj.get("game") or {}
        if str(g.get("gameState", "")).lower() != "final":
            continue
        home, away = g.get("home") or {}, g.get("away") or {}
        hs, as_ = str(home.get("score", "")).strip(), str(away.get("score", "")).strip()
        if not (hs.isdigit() and as_.isdigit()):
            continue
        try:
            gdate = datetime.strptime(g.get("startDate", ""), "%m-%d-%Y").date()
        except ValueError:
            gdate = day
        h = normalize_team_name((home.get("names") or {}).get("full"))
        a = normalize_team_name((away.get("names") or {}).get("full"))
        if not h or not a:
            continue
        hp, ap = int(hs), int(as_)
        out.append({
            "game_id": str(g.get("gameID")), "date": gdate, "season": config.season_of(gdate),
            "home_team": h, "away_team": a, "home_points": hp, "away_points": ap,
            "total_points": hp + ap, "home_margin": hp - ap,
            "neutral_site": _neutral(g),
            "fetched_at": datetime.utcnow().isoformat(timespec="seconds"),
        })
    return out


def _neutral(g: dict) -> bool:
    """ncaa.com exposes this inconsistently; check the handful of shapes it has used."""
    for key in ("neutralSite", "neutralsite", "isNeutralSite"):
        if key in g:
            return str(g[key]).lower() in ("true", "1", "yes")
    venue = g.get("venue") or {}
    for key in ("neutral", "neutralSite"):
        if key in venue:
            return str(venue[key]).lower() in ("true", "1", "yes")
    return False


def load() -> pd.DataFrame:
    if config.GAMES.exists():
        df = pd.read_csv(config.GAMES, dtype={"game_id": str}, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        return df
    return pd.DataFrame(columns=COLUMNS)


def save(df: pd.DataFrame) -> None:
    config.ensure_dirs()
    (df.drop_duplicates("game_id", keep="last")
       .sort_values(["date", "game_id"])
       .to_csv(config.GAMES, index=False))


def update(start: date | None = None, end: date | None = None, recheck_days: int = 3) -> pd.DataFrame:
    cache = load()
    end = end or config.today_et()
    start = start or config.season_start(config.FIRST_SEASON)
    have = set(cache["date"]) if len(cache) else set()

    days, d = [], start
    while d <= end:
        s = config.season_of(d)
        if config.season_start(s) <= d <= config.season_end(s):
            if d not in have or (end - d).days <= recheck_days:
                days.append(d)
        d += timedelta(days=1)

    if not days:
        log.info("games cache current: %d games", len(cache))
        return cache

    log.info("fetching %d scoreboard days (%s .. %s)", len(days), days[0], days[-1])
    session()  # warm the pool
    new = []
    for i, day in enumerate(days, 1):
        new.extend(fetch_day(day))
        if i % 50 == 0:
            log.info("  %d/%d days, %d games", i, len(days), len(new))
        time.sleep(config.NCAA_RATE_LIMIT)

    if new:
        cache = pd.concat([cache, pd.DataFrame(new)], ignore_index=True)
    save(cache)
    log.info("cache: %d games, seasons %s", len(cache), sorted(cache["season"].unique()))
    return load()
