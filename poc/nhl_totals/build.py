"""Reduce the raw season files to four small per-game tables.

    python -m poc.nhl_totals.build

  team_games.parquet    one row per (game, team): goals, xG and shot attempts split by strength,
                        penalties taken, empty-net goals - everything a team rating learns from
  goalie_games.parquet  one row per (game, goalie): xG faced and goals allowed (no empty net)
  player_games.parquet  one row per (game, skater): TOI, points, individual xG, 5v5 on-ice xG
  refs.parquet          one row per game: the two referees

Conventions worth knowing (checked against the files, not assumed):
  * a BLOCKED_SHOT event belongs to the BLOCKING team, so its shooter is the other side;
  * PENALTY events belong to the penalized team;
  * xG is only on unblocked attempts; shootout attempts are dropped everywhere;
  * strength is read from the skater counts, so it is the shooting side's view.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhl.teams import canon

from .fetch import FIRST_SEASON, LAST_SEASON, cache_dir, path_for

SHOTS = {"GOAL", "SHOT", "MISSED_SHOT", "BLOCKED_SHOT"}
PBP_COLS = ["event_type", "event_team_type", "period_type", "empty_net", "xg", "home_skaters",
            "away_skaters", "event_goalie_id", "event_player_1_id", "event_player_2_id",
            "event_player_3_id", "penalty_minutes", "penalty_severity", "game_id", "season_type",
            "home_abbr", "away_abbr", "home_goalie_id", "away_goalie_id"] + \
           [f"{s}_on_{i}_id" for s in ("home", "away") for i in range(1, 7)]


def _toi_min(s: pd.Series) -> pd.Series:
    mm = s.fillna("0:00").astype(str).str.split(":", expand=True)
    return pd.to_numeric(mm[0], errors="coerce").fillna(0) + \
        pd.to_numeric(mm[1], errors="coerce").fillna(0) / 60


def _season_pbp(season: int):
    d = pd.read_parquet(path_for("pbp", season), columns=PBP_COLS)
    d = d[d["period_type"] != "SHOOTOUT"]
    d["home_team"] = d["home_abbr"].map(canon)
    d["away_team"] = d["away_abbr"].map(canon)

    sh = d[d["event_type"].isin(SHOTS)].copy()
    blocked = sh["event_type"] == "BLOCKED_SHOT"
    # side that took the shot
    sh["shoot"] = np.where(blocked, np.where(sh["event_team_type"] == "home", "away", "home"),
                           sh["event_team_type"])
    own = np.where(sh["shoot"] == "home", sh["home_skaters"], sh["away_skaters"])
    opp = np.where(sh["shoot"] == "home", sh["away_skaters"], sh["home_skaters"])
    sh["str"] = np.select([own > opp, own < opp], ["pp", "sh"], "ev")
    sh["is_5v5"] = (sh["home_skaters"] == 5) & (sh["away_skaters"] == 5)
    sh["en"] = sh["empty_net"].fillna(False).astype(bool)
    sh["goal"] = (sh["event_type"] == "GOAL").astype(int)
    sh["sog"] = sh["event_type"].isin(["GOAL", "SHOT"]).astype(int)
    sh["unbl"] = (~blocked).astype(int)
    sh["xg"] = sh["xg"].fillna(0.0)
    sh["xg_ne"] = np.where(sh["en"], 0.0, sh["xg"])  # xG on a goalie

    rows = []
    for side in ("home", "away"):
        s = sh[sh["shoot"] == side]
        g = s.groupby("game_id")
        t = pd.DataFrame({
            "cf": g.size(), "ff": g["unbl"].sum(), "sf": g["sog"].sum(),
            "gf_all": g["goal"].sum(),
            "gf_en": s[s["en"]].groupby("game_id")["goal"].sum(),
            "xgf": g["xg_ne"].sum(),
            "xgf_ev": s[s["str"] == "ev"].groupby("game_id")["xg_ne"].sum(),
            "xgf_pp": s[s["str"] == "pp"].groupby("game_id")["xg_ne"].sum(),
            "xgf_sh": s[s["str"] == "sh"].groupby("game_id")["xg_ne"].sum(),
            "xgf_5v5": s[s["is_5v5"]].groupby("game_id")["xg_ne"].sum(),
            "cf_5v5": s[s["is_5v5"]].groupby("game_id").size(),
            "gf_pp": s[(s["str"] == "pp") & ~s["en"]].groupby("game_id")["goal"].sum(),
        })
        t["side"] = side
        rows.append(t)
    tg = pd.concat(rows).fillna(0.0).reset_index()

    pen = d[(d["event_type"] == "PENALTY") & d["penalty_minutes"].isin([2, 4, 5])
            & (d["penalty_severity"].isin(["Minor", "Major", "Bench Minor"]) | d["penalty_severity"].isna())]
    pc = pen.groupby(["game_id", "event_team_type"]).size().rename("pen_taken").reset_index() \
        .rename(columns={"event_team_type": "side"})
    tg = tg.merge(pc, on=["game_id", "side"], how="left").fillna({"pen_taken": 0})

    meta = d.groupby("game_id").agg(home_team=("home_team", "first"), away_team=("away_team", "first"),
                                    game_type=("season_type", "first")).reset_index()
    tg = tg.merge(meta, on="game_id")
    tg["team"] = np.where(tg["side"] == "home", tg["home_team"], tg["away_team"])
    tg["opp"] = np.where(tg["side"] == "home", tg["away_team"], tg["home_team"])
    tg = tg.drop(columns=["home_team", "away_team"])

    # goalies: unblocked attempts on a goalie, credited to the goalie in net
    on_g = sh[(sh["unbl"] == 1) & ~sh["en"] & sh["event_goalie_id"].notna()]
    gg = on_g.groupby(["game_id", "event_goalie_id"]).agg(
        xga=("xg", "sum"), ga=("goal", "sum"), sa=("sog", "sum"), fa=("unbl", "sum")).reset_index() \
        .rename(columns={"event_goalie_id": "goalie_id"})

    # skaters: 5v5 on-ice xG for / against, individual xG
    five = sh[sh["is_5v5"] & (sh["unbl"] == 1)]
    parts = []
    for side in ("home", "away"):
        cols = [f"{side}_on_{i}_id" for i in range(1, 7)]
        m = five[["game_id", "shoot", "xg_ne", "goal"] + cols].melt(
            id_vars=["game_id", "shoot", "xg_ne", "goal"], value_name="player_id").dropna(subset=["player_id"])
        m["for"] = m["shoot"] == side
        parts.append(m)
    m = pd.concat(parts)
    m["xgf"] = np.where(m["for"], m["xg_ne"], 0.0)
    m["xga"] = np.where(~m["for"], m["xg_ne"], 0.0)
    m["gf"] = np.where(m["for"], m["goal"], 0)
    m["ga"] = np.where(~m["for"], m["goal"], 0)
    oi = m.groupby(["game_id", "player_id"])[["xgf", "xga", "gf", "ga"]].sum() \
        .add_prefix("oi5_").reset_index()
    ind = sh[(sh["unbl"] == 1) & sh["event_player_1_id"].notna()].groupby(
        ["game_id", "event_player_1_id"])["xg"].sum().rename("ixg").reset_index() \
        .rename(columns={"event_player_1_id": "player_id"})
    pg = oi.merge(ind, on=["game_id", "player_id"], how="outer").fillna(0.0)
    pg["player_id"] = pg["player_id"].astype("int64")
    gg["goalie_id"] = gg["goalie_id"].astype("int64")
    return tg, gg, pg


def build():
    out = cache_dir() / "built"
    out.mkdir(exist_ok=True)
    T, G, P, R = [], [], [], []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        tg, gg, pg = _season_pbp(season)

        # who dressed, and for how long (box score), merged onto the PBP skater rows
        sk = pd.read_parquet(path_for("skater_box", season))
        sk["toi"] = _toi_min(sk["toi"])
        sk["team"] = sk["team_abbrev"].map(canon)
        sk = sk[["game_id", "player_id", "team", "position", "toi", "goals", "assists",
                 "shots_on_goal", "power_play_goals"]]
        pg = sk.merge(pg, on=["game_id", "player_id"], how="left").fillna(
            {c: 0.0 for c in ["oi5_xgf", "oi5_xga", "oi5_gf", "oi5_ga", "ixg"]})

        gb = pd.read_parquet(path_for("goalie_box", season))
        gb["toi"] = _toi_min(gb["toi"])
        gb["team"] = gb["team_abbrev"].map(canon)
        gb = gb[["game_id", "player_id", "player_name", "team", "starter", "toi"]].rename(
            columns={"player_id": "goalie_id", "player_name": "goalie"})
        gg = gb.merge(gg, on=["game_id", "goalie_id"], how="left").fillna(
            {"xga": 0.0, "ga": 0, "sa": 0, "fa": 0})

        of = pd.read_parquet(path_for("officials", season))
        refs = of[of["role"] == "referee"].sort_values(["game_id", "name"]).groupby("game_id")["name"] \
            .agg(lambda s: list(s)[:2]).reset_index()
        refs["ref1"] = refs["name"].str[0]
        refs["ref2"] = refs["name"].str[1]
        refs = refs.drop(columns="name")

        for df in (tg, gg, pg, refs):
            df["sdv_season"] = season
        T.append(tg); G.append(gg); P.append(pg); R.append(refs)
        print(season, f"games={tg.game_id.nunique()} goalies={len(gg)} skaters={len(pg)} refs={len(refs)}")
    pd.concat(T).to_parquet(out / "team_games.parquet", index=False)
    pd.concat(G).to_parquet(out / "goalie_games.parquet", index=False)
    pd.concat(P).to_parquet(out / "player_games.parquet", index=False)
    pd.concat(R).to_parquet(out / "refs.parquet", index=False)


if __name__ == "__main__":
    build()
