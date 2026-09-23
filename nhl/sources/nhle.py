"""Games, starting goalies and shots from the NHL Stats REST API. No key, no quota.

    python -m nhl.sources.nhle            # refresh the cache and print what was parsed

Three endpoints are the whole data spine:

| Endpoint | What it gives |
|---|---|
| ``/game`` | every game ever scheduled: id, season, type, date, teams, final score, state and the period it ended in - so regulation, overtime and shootout are known without a box score |
| ``/team`` | team id -> tri-code, for mapping the numeric ids ``/game`` uses |
| ``/goalie/summary?isGame=true`` | one row per goalie per game: started, shots against, goals against |

The goalie rows are what make this better than a scores feed. Summed per team they give shots on
goal for the other side, goals scored against a goalie (so empty-net goals can be separated from
real ones), and who started - and ``/game`` already includes the schedule for games not yet
played, so the board needs no second source.

Parsing is defensive by design. This module was written without being able to reach the API
from the machine it was written on, so field names follow code that uses these endpoints
successfully (see README), unknown teams are skipped and logged rather than invented, and
:func:`update_games` never overwrites a completed game with an incomplete one or a cached
season with an empty response. A bad morning degrades to yesterday's cache, loudly.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from .. import config
from ..http import get_json
from ..teams import UnknownTeam, canon

log = logging.getLogger(__name__)

GAME_COLS = ["game_id", "season", "game_type", "date", "start_et", "home_team", "away_team",
             "home_goals", "away_goals", "decided_in", "completed", "home_goalie_id",
             "home_goalie", "away_goalie_id", "away_goalie", "home_shots", "away_shots",
             "home_goalie_ga", "away_goalie_ga", "source"]
GAME_TYPES = {2: "R", 3: "P"}      # regular season and playoffs; preseason and all-star dropped
# gameStateId: 1 scheduled ... 5 game over, 6 final, 7 official. Scores can still be corrected
# at 5, so only 6 and 7 count as done.
FINAL_STATES = {6, 7}
SORT_GAME = '[{"property":"gameId","direction":"ASC"},{"property":"playerId","direction":"ASC"}]'


def _url(path: str) -> str:
    return f"{config.NHL_STATS_API}/{path.lstrip('/')}"


def _all_rows(path: str, params: dict, paged: bool = True) -> list[dict] | None:
    """Every row of a stats endpoint.

    The report endpoints (``goalie/summary``) page, so they are asked for ``limit=-1`` and then
    walked 100 rows at a time if that comes up short. The list endpoints (``game``, ``team``) are
    called exactly as the working reference code calls them - with no paging parameters - and
    only fall back to paging if that fails. Guessing wrong here would not raise: it would return
    nothing, keep yesterday's cache, and leave the board empty all season.
    """
    j = get_json(_url(path), params={**params, "start": 0, "limit": -1} if paged else params)
    if (j is None or "data" not in j) and not paged:
        j = get_json(_url(path), params={**params, "start": 0, "limit": -1})
    if j is None or "data" not in j:
        return None
    data, total = list(j["data"]), j.get("total", len(j["data"]))
    start = len(data)
    while start < total:
        page = get_json(_url(path), params={**params, "start": start, "limit": 100})
        if not page or not page.get("data"):
            log.warning("%s: stopped paging at %d of %d rows", path, start, total)
            break
        data.extend(page["data"])
        start += len(page["data"])
    return data


def fetch_teams() -> dict[int, str]:
    """NHL team id -> tri-code."""
    rows = _all_rows("team", {}, paged=False) or []
    return {int(r["id"]): str(r.get("triCode") or r.get("rawTricode") or "")
            for r in rows if r.get("id") is not None}


def fetch_games(from_season: int | None = None) -> list[dict]:
    """Game records, optionally only from `from_season` on.

    The filtered call keeps a daily refresh to one small request. If the filter is refused or
    comes back empty, the unfiltered list (every game since 1917, a few megabytes) is fetched
    and filtered here instead - slower, but it cannot silently return nothing.
    """
    rows = None
    if from_season is not None:
        rows = _all_rows("game", {"cayenneExp": f"season>={config.season_id(from_season)}"},
                         paged=False)
        if not rows:
            log.warning("filtered /game call returned nothing - falling back to the full list")
    if not rows:
        rows = _all_rows("game", {}, paged=False) or []
    # filtered here as well, in case the filter was ignored rather than refused
    if from_season is not None:
        rows = [r for r in rows if int(r.get("season") or 0) >= config.season_id(from_season)]
    return rows


def fetch_goalie_games(season: int) -> list[dict]:
    return _all_rows("goalie/summary", {
        "isAggregate": "false", "isGame": "true", "sort": SORT_GAME,
        "cayenneExp": f"seasonId={config.season_id(season)} and gameTypeId>=2"}) or []


def _decided_in(period, game_type: str, season: int) -> str:
    try:
        p = int(period)
    except (TypeError, ValueError):
        return "REG"
    if p <= 3:
        return "REG"
    return "SO" if game_type == "R" and p >= 5 and season >= 2005 else "OT"


def parse_games(rows: list[dict], teams: dict[int, str]) -> pd.DataFrame:
    """Game records -> this pipeline's schema, goalie columns still empty. Pure."""
    out, unknown = [], set()
    for r in rows:
        gt = GAME_TYPES.get(int(r.get("gameType") or 0))
        if gt is None:
            continue
        try:
            home = canon(teams.get(int(r["homeTeamId"]), ""))
            away = canon(teams.get(int(r["visitingTeamId"]), ""))
        except (UnknownTeam, KeyError, TypeError, ValueError):
            unknown.add((r.get("homeTeamId"), r.get("visitingTeamId")))
            continue
        season = int(r["season"]) // 10000
        hs, vs = r.get("homeScore"), r.get("visitingScore")
        done = int(r.get("gameStateId") or 0) in FINAL_STATES and hs is not None and vs is not None
        start = str(r.get("easternStartTime") or "")
        out.append({
            "game_id": str(r["id"]), "season": season, "game_type": gt,
            "date": str(r.get("gameDate") or start[:10])[:10], "start_et": start,
            "home_team": home, "away_team": away,
            "home_goals": float(hs) if done else np.nan,
            "away_goals": float(vs) if done else np.nan,
            "decided_in": _decided_in(r.get("period"), gt, season) if done else "",
            "completed": bool(done), "source": "nhle"})
    if unknown:
        log.error("skipped games with unmapped team ids: %s", sorted(unknown, key=str)[:10])
    return pd.DataFrame(out, columns=[c for c in GAME_COLS if c not in (
        "home_goalie_id", "home_goalie", "away_goalie_id", "away_goalie", "home_shots",
        "away_shots", "home_goalie_ga", "away_goalie_ga")])


def goalie_table(rows: list[dict]) -> pd.DataFrame:
    """Per game and team: the starter, and shots/goals against summed over every goalie used."""
    if not rows:
        return pd.DataFrame(columns=["game_id", "team", "goalie_id", "goalie", "sa", "ga"])
    g = pd.DataFrame(rows).drop_duplicates(["gameId", "playerId"])
    teams = []
    for t in g["teamAbbrev"]:
        try:
            teams.append(canon(t))
        except UnknownTeam:
            teams.append(None)
    g["team"] = teams
    g = g[g["team"].notna()].copy()
    g["game_id"] = g["gameId"].astype(str)
    g["started"] = pd.to_numeric(g.get("gamesStarted"), errors="coerce").fillna(0)
    g["toi"] = pd.to_numeric(g.get("timeOnIce"), errors="coerce").fillna(0)
    agg = g.groupby(["game_id", "team"]).agg(
        sa=("shotsAgainst", lambda s: pd.to_numeric(s, errors="coerce").sum()),
        ga=("goalsAgainst", lambda s: pd.to_numeric(s, errors="coerce").sum())).reset_index()
    st = (g.sort_values(["started", "toi"], ascending=False)
           .drop_duplicates(["game_id", "team"])[["game_id", "team", "playerId", "goalieFullName"]]
           .rename(columns={"playerId": "goalie_id", "goalieFullName": "goalie"}))
    return agg.merge(st, on=["game_id", "team"], how="left")


def attach_goalies(games: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Starter and shot columns for both sides. Shots FOR a side are what the other side's
    goalies faced; goalie goals-against are goals the OTHER side scored on a goalie."""
    g = games.copy()
    for side in ("home", "away"):
        s = gt.rename(columns={"goalie_id": f"{side}_goalie_id", "goalie": f"{side}_goalie",
                               "sa": f"_{side}_sa", "ga": f"{side}_goalie_ga"})
        g = g.merge(s, left_on=["game_id", f"{side}_team"], right_on=["game_id", "team"],
                    how="left").drop(columns="team")
    g["home_shots"] = g.pop("_away_sa")
    g["away_shots"] = g.pop("_home_sa")
    return g


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Upsert by game_id, never letting a worse row replace a better one.

    A completed game is not replaced by a version that is not completed, and goalie columns
    are not replaced by blanks - an API hiccup on one morning must not erase history.
    """
    if old is None or old.empty:
        return new.reindex(columns=GAME_COLS)
    if new is None or new.empty:
        return old.reindex(columns=GAME_COLS)
    old = old.reindex(columns=GAME_COLS).set_index("game_id")
    new = new.reindex(columns=GAME_COLS).set_index("game_id")
    both = old.index.intersection(new.index)
    regress = both[old.loc[both, "completed"].astype(bool).values
                   & ~new.loc[both, "completed"].astype(bool).values]
    new = new.drop(index=regress)
    for c in ("home_goalie_id", "home_goalie", "away_goalie_id", "away_goalie", "home_shots",
              "away_shots", "home_goalie_ga", "away_goalie_ga"):
        keep = new.index.intersection(old.index)
        new.loc[keep, c] = new.loc[keep, c].where(new.loc[keep, c].notna(), old.loc[keep, c])
    out = pd.concat([old.drop(index=new.index.intersection(old.index)), new])
    return out.reset_index().sort_values(["date", "game_id"]).reset_index(drop=True)


def load_games() -> pd.DataFrame:
    if not config.GAMES.exists():
        return pd.DataFrame(columns=GAME_COLS)
    df = pd.read_csv(config.GAMES, dtype={"game_id": str, "start_et": str, "decided_in": str},
                     low_memory=False)
    df["completed"] = df["completed"].astype(str).str.lower().isin(["true", "1", "1.0"])
    df["decided_in"] = df["decided_in"].fillna("")
    return df


def update_games(full: bool = False) -> pd.DataFrame:
    """Refresh the cache from the API and return every game, scheduled ones included.

    A normal run refreshes the previous, current and next season - three seasons, because in
    September the "current" season by date is the one that just ended and the one about to start
    is the one with a schedule. An empty cache, or ``full=True``, backfills from
    LOAD_FROM_SEASON.
    """
    config.ensure_dirs()
    cached = load_games()
    now = config.season_of(config.today_et())
    backfill = full or cached.empty or int(cached["season"].min()) > config.LOAD_FROM_SEASON
    first = config.LOAD_FROM_SEASON if backfill else now - 1
    teams = fetch_teams()
    if not teams:
        log.warning("NHL /team unavailable - keeping the cached games")
        return cached
    # always filtered: a backfill wants 2005 on, not every game since 1917 and a log full of
    # defunct franchises that map to nothing
    rows = fetch_games(first)
    games = parse_games(rows, teams)
    games = games[games["season"] >= config.LOAD_FROM_SEASON]
    if games.empty:
        log.warning("NHL /game returned no usable games - keeping the cached games")
        return cached
    seasons = sorted(int(s) for s in games.loc[games["completed"], "season"].unique()
                     if int(s) >= first)
    gts = []
    for s in seasons:
        gt = goalie_table(fetch_goalie_games(s))
        if gt.empty:
            log.warning("no goalie logs for %d - its games keep whatever goalie data is cached", s)
        gts.append(gt)
    gt = pd.concat(gts, ignore_index=True) if gts else goalie_table([])
    games = attach_goalies(games, gt)
    merged = _merge(cached, games)
    merged.to_csv(config.GAMES, index=False)
    done = merged[merged["completed"]]
    log.info("NHL API: %d games parsed (%d completed, %d scheduled), seasons %s; cache now %d "
             "games, %d with a starting goalie", len(games), int(games["completed"].sum()),
             int((~games["completed"]).sum()), f"{min(seasons, default='-')}-{max(seasons, default='-')}",
             len(merged), int(done["home_goalie_id"].notna().sum()))
    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    g = update_games()
    print(json.dumps({"games": len(g), "completed": int(g["completed"].sum()),
                      "seasons": sorted(int(s) for s in g["season"].unique())}, indent=2))
