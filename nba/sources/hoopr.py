"""Schedule, results, possessions and player minutes from hoopR. No key, no quota.

    python -m nba.sources.hoopr              # refresh the cache and print what was parsed
    python -m nba.sources.hoopr --full       # rebuild every season from LOAD_FROM_SEASON

hoopR (the sportsdataverse's NBA arm) scrapes ESPN every morning of the season and republishes
it as static files on GitHub releases - the NBA's nflverse. One file per season per feed:

| Feed | What it gives |
|---|---|
| ``espn_nba_schedules/nba_schedule_<Y>.csv`` | every game, **including unplayed ones**: id, date, tip time, teams, score, state, periods (overtime), venue city, neutral-site flag, regular season / play-in / playoffs |
| ``espn_nba_team_boxscores/team_box_<Y>.csv`` | per team: FGA, FTA, offensive rebounds, turnovers - possessions |
| ``espn_nba_player_boxscores/player_box_<Y>.csv`` | per player: minutes and the box score - who played, and what he did |
| ``espn_nba_rosters/rosters_<Y>.csv`` | the current roster of every club |

``<Y>`` is the year a season ENDS (hoopR's 2027 is 2026-27). This module converts to the repo's
convention, the year it starts, exactly once - in :func:`_season`.

**Every id is ESPN's**, so a game here, a player on the injury report and a price on the ESPN
scoreboard all join on ids with no name matching at all.

Parsing is defensive. The All-Star game ("EAST" v "WEST", "Team Shaq" v "Team Chuck") and
unfilled knockout slots ("TBD") are in the schedule and map to no club, so they are skipped and
logged rather than rated. A completed game is never replaced by an incomplete version of itself,
and a season's player file is never replaced by a smaller one - a bad morning degrades to
yesterday's cache, loudly.
"""
from __future__ import annotations

import argparse
import io
import json
import logging

import numpy as np
import pandas as pd

from .. import config
from ..http import get
from ..players import game_score, offence_score
from ..teams import UnknownTeam, canon

log = logging.getLogger(__name__)

GAME_COLS = ["game_id", "season", "game_type", "date", "tip_utc", "home_team", "away_team",
             "home_points", "away_points", "completed", "neutral_site", "periods", "venue_city",
             "home_poss", "away_poss", "poss", "source"]
PLAYER_COLS = ["game_id", "team", "athlete_id", "minutes", "gs", "os"]
GAME_TYPES = {2: "R", 3: "P", 5: "I"}      # regular season, playoffs, play-in; preseason dropped


def _season(hoopr_season) -> int:
    """hoopR names a season by the year it ends; this repo by the year it starts."""
    return int(hoopr_season) - 1


def _url(feed: str, name: str) -> str:
    return f"{config.HOOPR_RELEASES}/{feed}/{name}"


def _csv(feed: str, name: str) -> pd.DataFrame:
    r = get(_url(feed, name))
    if r is None or r.status_code != 200 or not r.content:
        log.info("hoopR: %s/%s unavailable (%s)", feed, name, getattr(r, "status_code", "no response"))
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(r.content), low_memory=False)


def _code(x):
    try:
        return canon(x)
    except UnknownTeam:
        return None


def parse_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """hoopR schedule -> this pipeline's game rows (possessions still empty). Pure."""
    if df.empty:
        return pd.DataFrame(columns=GAME_COLS)
    d = df.copy()
    d["game_type"] = pd.to_numeric(d["season_type"], errors="coerce").map(GAME_TYPES)
    d = d[d["game_type"].notna()]
    d["home_team"] = d["home_abbreviation"].map(_code)
    d["away_team"] = d["away_abbreviation"].map(_code)
    skipped = d[d["home_team"].isna() | d["away_team"].isna()]
    if len(skipped):
        log.info("hoopR: skipped %d games between non-NBA sides (All-Star, TBD): %s", len(skipped),
                 sorted(set(skipped["home_abbreviation"].astype(str))
                        | set(skipped["away_abbreviation"].astype(str)))[:8])
    d = d[d["home_team"].notna() & d["away_team"].notna()]
    done = d["status_type_completed"].astype(str).str.lower().isin(["true", "1", "1.0"])
    tip = pd.to_datetime(d["date"], utc=True, errors="coerce")
    local = pd.to_datetime(d["game_date"], errors="coerce")
    out = pd.DataFrame({
        "game_id": d["game_id"].astype("int64").astype(str),
        "season": d["season"].map(_season),
        "game_type": d["game_type"],
        # the game's own local (Eastern) date where hoopR gives one, else the UTC tip in Eastern
        "date": local.dt.date.where(local.notna(), tip.dt.tz_convert(config.ET).dt.date),
        "tip_utc": tip.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "home_team": d["home_team"], "away_team": d["away_team"],
        "home_points": pd.to_numeric(d["home_score"], errors="coerce").where(done),
        "away_points": pd.to_numeric(d["away_score"], errors="coerce").where(done),
        "completed": done.values,
        "neutral_site": d["neutral_site"].astype(str).str.lower().isin(["true", "1", "1.0"]),
        "periods": pd.to_numeric(d.get("status_period"), errors="coerce").where(done),
        "venue_city": d.get("venue_address_city"),
        "source": "hoopr"})
    return out.reset_index(drop=True)


def parse_team_box(df: pd.DataFrame) -> pd.DataFrame:
    """Possessions per team per game: FGA + 0.44 FTA - offensive rebounds + turnovers.

    Player turnovers only. Some seasons' ``total_turnovers`` count every team turnover twice
    (2010 reads 30 for a side that committed 15), so the column that is right in every season
    is the one used.
    """
    if df.empty:
        return pd.DataFrame(columns=["game_id", "side", "poss"])
    f = lambda c: pd.to_numeric(df[c], errors="coerce")
    poss = f("field_goals_attempted") + 0.44 * f("free_throws_attempted") \
        - f("offensive_rebounds") + f("turnovers")
    return pd.DataFrame({"game_id": df["game_id"].astype("int64").astype(str),
                         "side": df["team_home_away"].astype(str).str.lower(),
                         "poss": poss}).drop_duplicates(["game_id", "side"])


def attach_possessions(games: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    g = games.drop(columns=[c for c in ("home_poss", "away_poss", "poss") if c in games])
    for side in ("home", "away"):
        s = box[box["side"] == side][["game_id", "poss"]].rename(columns={"poss": f"{side}_poss"})
        g = g.merge(s, on="game_id", how="left")
    g["poss"] = g[["home_poss", "away_poss"]].mean(axis=1)
    return g.reindex(columns=GAME_COLS)


def parse_player_box(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Player box -> (compact player-games for everyone who logged minutes, names)."""
    if df.empty:
        return pd.DataFrame(columns=PLAYER_COLS), pd.DataFrame(columns=["athlete_id", "name"])
    d = df.rename(columns={"field_goals_made": "fgm", "field_goals_attempted": "fga",
                           "free_throws_made": "ftm", "free_throws_attempted": "fta",
                           "offensive_rebounds": "oreb", "defensive_rebounds": "dreb",
                           "assists": "ast", "steals": "stl", "blocks": "blk", "fouls": "pf",
                           "turnovers": "tov"})
    d["minutes"] = pd.to_numeric(d["minutes"], errors="coerce")
    d = d[d["minutes"].fillna(0) > 0].copy()
    d["team"] = d["team_abbreviation"].map(_code)
    d = d[d["team"].notna() & d["athlete_id"].notna()]
    out = pd.DataFrame({"game_id": d["game_id"].astype("int64").astype(str), "team": d["team"],
                        "athlete_id": d["athlete_id"].astype("int64"),
                        "minutes": d["minutes"].round(1),
                        "gs": game_score(d).round(1), "os": offence_score(d).round(1)})
    names = d[["athlete_id", "athlete_display_name"]].dropna().rename(
        columns={"athlete_display_name": "name"}).astype({"athlete_id": "int64"})
    return out.drop_duplicates(["game_id", "athlete_id"]), names.drop_duplicates("athlete_id",
                                                                                   keep="last")


def parse_rosters(df: pd.DataFrame) -> dict[str, set]:
    """Club -> set of player ids on its current roster."""
    if df.empty or "athlete_id" not in df:
        return {}
    d = df.copy()
    d["team"] = d["team_abbreviation"].map(_code)
    d = d[d["team"].notna() & d["athlete_id"].notna()]
    return {t: set(g["athlete_id"].astype("int64")) for t, g in d.groupby("team")}


# ---------------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------------
def load_games() -> pd.DataFrame:
    if not config.GAMES.exists():
        return pd.DataFrame(columns=GAME_COLS)
    df = pd.read_csv(config.GAMES, dtype={"game_id": str}, low_memory=False)
    df["completed"] = df["completed"].astype(str).str.lower().isin(["true", "1", "1.0"])
    df["neutral_site"] = df["neutral_site"].astype(str).str.lower().isin(["true", "1", "1.0"])
    return df


def players_path(season: int):
    return config.PLAYERS / f"{season}.csv"


def load_players(seasons=None) -> pd.DataFrame:
    """Every committed player-game, or only those of `seasons`."""
    if not config.PLAYERS.exists():
        return pd.DataFrame(columns=PLAYER_COLS)
    files = sorted(config.PLAYERS.glob("*.csv"))
    if seasons is not None:
        want = {int(s) for s in seasons}
        files = [f for f in files if f.stem.isdigit() and int(f.stem) in want]
    frames = [pd.read_csv(f, dtype={"game_id": str}) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PLAYER_COLS)


def load_names() -> dict[int, str]:
    if not config.PLAYER_NAMES.exists():
        return {}
    df = pd.read_csv(config.PLAYER_NAMES)
    return dict(zip(df["athlete_id"].astype("int64"), df["name"].astype(str)))


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Upsert by game_id, never letting a worse row replace a better one.

    A completed game is not replaced by a version that is not completed, and possessions are not
    replaced by blanks - an upstream hiccup on one morning must not erase history.
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
    keep = new.index.intersection(old.index)
    for c in ("home_poss", "away_poss", "poss", "venue_city", "periods"):
        new.loc[keep, c] = new.loc[keep, c].where(new.loc[keep, c].notna(), old.loc[keep, c])
    out = pd.concat([old.drop(index=new.index.intersection(old.index)), new])
    out = out.reset_index().rename(columns={"index": "game_id"})
    return out.sort_values(["date", "game_id"]).reset_index(drop=True).reindex(columns=GAME_COLS)


def _save_players(season: int, rows: pd.DataFrame) -> bool:
    """Write a season's player file unless it would shrink what is already there."""
    path = players_path(season)
    if rows.empty:
        return False
    if path.exists():
        old = pd.read_csv(path, dtype={"game_id": str})
        if len(rows) < len(old):
            log.warning("hoopR: %s player rows for %d, fewer than the %d cached - keeping the "
                        "cache", len(rows), season, len(old))
            return False
    config.ensure_dirs()
    rows.sort_values(["game_id", "team", "athlete_id"]).to_csv(path, index=False)
    return True


def _save_names(names: pd.DataFrame) -> None:
    if names.empty:
        return
    old = pd.read_csv(config.PLAYER_NAMES) if config.PLAYER_NAMES.exists() else \
        pd.DataFrame(columns=["athlete_id", "name"])
    allx = pd.concat([old, names], ignore_index=True).drop_duplicates("athlete_id", keep="last")
    allx.sort_values("athlete_id").to_csv(config.PLAYER_NAMES, index=False)


def fetch_season(season: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(games, player rows, names) for one season, from all three feeds."""
    y = season + 1
    games = parse_schedule(_csv("espn_nba_schedules", f"nba_schedule_{y}.csv"))
    if games.empty:
        return games, pd.DataFrame(columns=PLAYER_COLS), pd.DataFrame()
    if games["completed"].any():
        games = attach_possessions(games, parse_team_box(_csv("espn_nba_team_boxscores",
                                                              f"team_box_{y}.csv")))
        players, names = parse_player_box(_csv("espn_nba_player_boxscores", f"player_box_{y}.csv"))
    else:
        games = games.reindex(columns=GAME_COLS)
        players, names = pd.DataFrame(columns=PLAYER_COLS), pd.DataFrame()
    return games, players, names


def rosters(season: int | None = None) -> dict[str, set]:
    """Current rosters, or {} if hoopR has not published them."""
    season = config.season_of(config.today_et()) if season is None else season
    return parse_rosters(_csv("espn_nba_rosters", f"rosters_{season + 1}.csv"))


def update(full: bool = False) -> pd.DataFrame:
    """Refresh the cache and return every game, scheduled ones included.

    A normal run refreshes the current and next season, and the previous one until six weeks
    after its Finals - late stat corrections land in that window, and after it a finished
    season's twenty megabytes of box scores are not worth fetching twice a day. An empty cache,
    or ``full=True``, backfills from LOAD_FROM_SEASON.
    """
    config.ensure_dirs()
    cached = load_games()
    today = config.today_et()
    now = config.season_of(today)
    backfill = full or cached.empty or int(cached["season"].min()) > config.LOAD_FROM_SEASON
    recent = (today - config.season_end(now - 1)).days <= 45
    seasons = range(config.LOAD_FROM_SEASON if backfill else (now - 1 if recent else now), now + 2)
    merged = cached
    for s in seasons:
        games, players, names = fetch_season(s)
        if games.empty:
            log.info("hoopR: nothing for %s", config.season_label(s))
            continue
        merged = _merge(merged, games)
        wrote = _save_players(s, players)
        _save_names(names)
        log.info("hoopR %s: %d games (%d completed), %d player rows%s", config.season_label(s),
                 len(games), int(games["completed"].sum()), len(players),
                 "" if wrote or players.empty else " (cache kept)")
    merged.to_csv(config.GAMES, index=False)
    done = merged[merged["completed"].astype(bool)]
    log.info("hoopR: cache now %d games (%d completed, %d with possessions)", len(merged),
             len(done), int(done["poss"].notna().sum()))
    return load_games()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args(argv)
    g = update(full=a.full)
    print(json.dumps({"games": len(g), "completed": int(g["completed"].sum()),
                      "seasons": sorted(int(s) for s in g["season"].unique())}, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
