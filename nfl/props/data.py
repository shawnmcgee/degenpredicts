"""Player-week history for the props models, from nflverse. No key, no quota.

    python -m nfl.props.data                 # refresh this season (and backfill any missing)
    python -m nfl.props.data --all           # rebuild every season from PROPS_FIRST_SEASON

One trimmed CSV per season in ``data/nfl/props/players/``: offensive skill players only, the
columns the models read, and **one row for every game a player was on the field** - which the
source file does not give you.

**The zero games.** nflverse's ``stats_player_week`` only has a row when a player recorded a
stat. A tight end who ran 30 routes and was never targeted has no row for that game, so he
looks like a player who always catches something. In 2025 that hid 16% of the games played by
tight ends on 40-60% of snaps and 11% of games by receivers on 20-40%. Those are precisely the
games where a receptions or receiving-yards *under* wins, so leaving them out would have
tilted every projection toward the over and nothing about it would have looked wrong. Snap
counts know who was on the field; they are joined through the ``players`` crosswalk (snap
counts key on PFR ids, stats on GSIS ids - the same join ``nflverse.season_continuity``
documents) and every on-field game with no stat line becomes a row of zeros.

``snap_pct`` comes along for free and is one of the best inputs there is: a player's share of
his team's snaps is the stickiest measure of his role.
"""
from __future__ import annotations

import argparse
import io
import logging

import numpy as np
import pandas as pd

from .. import config
from ..http import get
from ..teams import UnknownTeam, canon

log = logging.getLogger(__name__)

SKILL = ("QB", "RB", "FB", "WR", "TE")
STAT_COLS = ["completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
             "sacks_suffered", "passing_air_yards", "carries", "rushing_yards", "rushing_tds",
             "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_air_yards",
             "target_share", "air_yards_share"]
COLS = ["season", "week", "season_type", "game_id", "team", "opponent_team", "player_id",
        "player_name", "position", *STAT_COLS, "offense_snaps", "snap_pct", "zero_filled"]


def _csv(url: str) -> pd.DataFrame:
    r = get(url)
    if r is None or r.status_code != 200:
        log.info("props: %s unavailable (%s)", url.rsplit("/", 1)[-1],
                 getattr(r, "status_code", "no response"))
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(r.text), low_memory=False)
    except (ValueError, pd.errors.ParserError) as e:
        log.warning("props: could not parse %s (%s)", url, e)
        return pd.DataFrame()


def _release(asset: str, name: str) -> pd.DataFrame:
    return _csv(f"{config.NFLVERSE_RELEASES}/{asset}/{name}")


def _canon(code):
    try:
        return canon(str(code))
    except UnknownTeam:
        return None


def trim(stats: pd.DataFrame, snaps: pd.DataFrame | None, xwalk: dict[str, str]) -> pd.DataFrame:
    """One season: skill players' stat rows, plus a zero row for every on-field game the stats
    file left out, with each row's snap share attached."""
    if stats.empty:
        return pd.DataFrame(columns=COLS)
    s = stats[stats["position"].isin(SKILL)].copy()
    s["player_name"] = s.get("player_display_name", s.get("player_name"))
    for c in STAT_COLS:
        # A column nflverse drops or renames reads as zeros rather than killing the job.
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0.0) if c in s else 0.0
    s["zero_filled"] = 0

    snap = pd.DataFrame(columns=["game_id", "player_id", "offense_snaps", "snap_pct"])
    zeros = pd.DataFrame(columns=COLS)
    if snaps is not None and len(snaps) and xwalk:
        sn = snaps[snaps["position"].isin(SKILL)].copy()
        sn["player_id"] = sn["pfr_player_id"].map(xwalk)
        sn = sn.dropna(subset=["player_id"])
        sn["offense_snaps"] = pd.to_numeric(sn["offense_snaps"], errors="coerce").fillna(0.0)
        sn["snap_pct"] = pd.to_numeric(sn["offense_pct"], errors="coerce").fillna(0.0)
        sn = sn[sn["offense_snaps"] > 0]
        snap = sn[["game_id", "player_id", "offense_snaps", "snap_pct"]].drop_duplicates(
            ["game_id", "player_id"])
        have = set(zip(s["game_id"], s["player_id"]))
        miss = sn[[(g, p) not in have for g, p in zip(sn["game_id"], sn["player_id"])]].copy()
        if len(miss):
            miss["season_type"] = np.where(miss["game_type"].astype(str).str.upper() == "REG",
                                           "REG", "POST")
            miss["opponent_team"] = miss["opponent"]
            miss["player_name"] = miss["player"]
            for c in STAT_COLS:
                miss[c] = 0.0
            miss["zero_filled"] = 1
            zeros = miss[[c for c in COLS if c in miss.columns]]
    out = s.drop(columns=[c for c in ("offense_snaps", "snap_pct") if c in s.columns])
    out = out.merge(snap, on=["game_id", "player_id"], how="left")
    out = pd.concat([out, zeros], ignore_index=True) if len(zeros) else out
    out["team"] = out["team"].map(_canon)
    out["opponent_team"] = out["opponent_team"].map(_canon)
    out = out.dropna(subset=["team", "opponent_team", "game_id", "player_id"])
    for c in COLS:
        if c not in out:
            out[c] = np.nan
    out = out[COLS].sort_values(["week", "game_id", "team", "player_id"])
    for c in ("target_share", "air_yards_share", "snap_pct"):
        out[c] = out[c].round(4)
    log.info("props %s: %d player-games (%d rebuilt from snap counts)",
             out["season"].iloc[0] if len(out) else "?", len(out), int(out["zero_filled"].sum()))
    return out.reset_index(drop=True)


def update(seasons: list[int] | None = None, rebuild: bool = False) -> None:
    """Refresh the current season, and backfill any season that has no file yet."""
    from ..sources.nflverse import player_crosswalk

    this = config.season_of(config.today_et())
    config.PROPS_PLAYERS.mkdir(parents=True, exist_ok=True)
    if seasons is None:
        seasons = [s for s in range(config.PROPS_FIRST_SEASON, this + 1)
                   if rebuild or s == this or not (config.PROPS_PLAYERS / f"{s}.csv").exists()]
    xw = player_crosswalk()
    for s in seasons:
        stats = _release("stats_player", f"stats_player_week_{s}.csv")
        snaps = _release("snap_counts", f"snap_counts_{s}.csv")
        df = trim(stats, snaps, xw)
        path = config.PROPS_PLAYERS / f"{s}.csv"
        # A bad refresh never erases history: a season file is never replaced by a smaller one.
        if path.exists():
            old = pd.read_csv(path, usecols=["game_id"])
            if len(df) < len(old):
                log.warning("props %s: refresh returned %d rows against %d cached - keeping "
                            "the cache", s, len(df), len(old))
                continue
        if len(df):
            df.to_csv(path, index=False)
    update_current()


def update_current() -> None:
    """This season's roster (jersey numbers, for matching Kalshi's tickers) and injury report."""
    this = config.season_of(config.today_et())
    config.PROPS.mkdir(parents=True, exist_ok=True)
    r = _release("rosters", f"roster_{this}.csv")
    if len(r):
        keep = ["season", "team", "position", "jersey_number", "status", "full_name",
                "first_name", "last_name", "football_name", "gsis_id"]
        r = r[[c for c in keep if c in r.columns]].copy()
        r["team"] = r["team"].map(_canon)
        r.dropna(subset=["team", "gsis_id"]).to_csv(config.PROPS_ROSTER, index=False)
    i = _release("injuries", f"injuries_{this}.csv")
    if len(i):
        keep = ["season", "week", "team", "gsis_id", "position", "full_name", "report_status",
                "report_primary_injury", "practice_status"]
        i = i[[c for c in keep if c in i.columns]].copy()
        i["team"] = i["team"].map(_canon)
        i.dropna(subset=["team"]).to_csv(config.PROPS_INJURIES, index=False)


def load(first: int | None = None, last: int | None = None) -> pd.DataFrame:
    first = config.PROPS_FIRST_SEASON if first is None else first
    frames = []
    for p in sorted(config.PROPS_PLAYERS.glob("*.csv")):
        s = int(p.stem)
        if s < first or (last is not None and s > last):
            continue
        frames.append(pd.read_csv(p, dtype={"game_id": str, "player_id": str}))
    if not frames:
        return pd.DataFrame(columns=COLS)
    return pd.concat(frames, ignore_index=True)


def load_roster() -> pd.DataFrame:
    if not config.PROPS_ROSTER.exists():
        return pd.DataFrame()
    return pd.read_csv(config.PROPS_ROSTER, dtype={"gsis_id": str})


def load_injuries() -> pd.DataFrame:
    if not config.PROPS_INJURIES.exists():
        return pd.DataFrame()
    return pd.read_csv(config.PROPS_INJURIES, dtype={"gsis_id": str})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="rebuild every season")
    a = ap.parse_args(argv)
    update(rebuild=a.all)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
