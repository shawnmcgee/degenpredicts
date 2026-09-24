"""Feature construction for the NBA, replayed in date order.

A feature on game N only ever sees games 1..N-1. That guarantee is what the project rests on,
and it is verified in CI by rebuilding from a truncated schedule and asserting identical rows.
Two NBA-specific ways to break it are pinned there too: the availability delta may not move when
only the bench's garbage-time minutes change, and nothing in a row may move when only that
game's score does.

The features, in the order they earn their keep:

* **Ratings** (:mod:`nba.ratings`) - expected possessions and points per 100 for each side, and
  the components behind them.
* **Who is playing** (:mod:`nba.players`) - the availability delta for each side, net and
  offensive, and the single largest absence. Worth 0.21 points of margin error walk-forward,
  the largest thing found. Training rows use who actually played; the board uses the injury
  report and the current roster.
* **Schedule** - rest, back-to-backs, three-in-four and four-in-six, the distance and time zones
  travelled since the last game, road-trip length and the venue's altitude (Denver and Salt Lake
  City). Measured from each game's actual venue, so London, Paris, Mexico City and the bubble
  need no special case.
* **Context** - playoffs, the play-in, neutral sites, the no-crowd 2020 seasons, early season.
* **The market** - for the market-aware models, the closing number itself and the no-market
  model's disagreement with it: the model learns how the market errs, not basketball again.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .players import PlayerBook, PlayerConfig
from .ratings import RatingBook, RatingConfig
from .teams import city, haversine_km

log = logging.getLogger(__name__)

REST_CAP = 7
EARLY_GAMES = 10

RATING_FEATURES = ["e_margin0", "e_total0", "e_poss", "H", "L", "h_off", "h_def", "a_off",
                   "a_def", "h_pace", "a_pace"]
AVAIL_FEATURES = ["h_d_net", "a_d_net", "h_d_off", "a_d_off", "h_miss_top", "a_miss_top"]
SCHEDULE_FEATURES = ["h_rest", "a_rest", "rest_diff", "h_b2b", "a_b2b", "h_3in4", "a_3in4",
                     "h_4in6", "a_4in6", "h_km", "a_km", "h_tz_abs", "a_tz_abs", "h_road",
                     "a_road", "altitude"]
CONTEXT_FEATURES = ["playoff", "playin", "no_crowd", "neutral", "h_games", "a_games", "early"]
BASE_FEATURES = RATING_FEATURES + AVAIL_FEATURES + SCHEDULE_FEATURES + CONTEXT_FEATURES
# The market-aware models explain the RESIDUAL against the closing number, so the no-market
# model's disagreement with it is the feature that carries the ratings.
MARKET_FEATURES = {
    "margin": ["dev_margin", "mkt_margin"] + AVAIL_FEATURES + SCHEDULE_FEATURES + CONTEXT_FEATURES,
    "total": ["dev_total", "mkt_total", "e_poss"] + AVAIL_FEATURES + SCHEDULE_FEATURES
    + CONTEXT_FEATURES,
}


def in_bubble(d) -> bool:
    return config.BUBBLE[0] <= d <= config.BUBBLE[1]


def schedule_features(games: pd.DataFrame, today=None) -> pd.DataFrame:
    """Rest, back-to-backs, 3-in-4, 4-in-6, travel, time zones and road trips for every game.

    Built over the schedule, not just completed games, so tomorrow's game knows tonight's makes
    it a back-to-back. Past games that never happened (postponed, cancelled) are skipped.
    """
    today = today or config.today_et()
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    g = g[g["completed"].astype(bool) | (g["date"] >= today)].sort_values(["date", "game_id"])
    last: dict[str, dict] = {}
    rows = []
    for r in g.itertuples(index=False):
        here = city(getattr(r, "venue_city", None), r.home_team)
        out = {"game_id": r.game_id, "altitude": float(here[3]) if here else 0.0}
        for side, team, home in (("h", r.home_team, True), ("a", r.away_team, False)):
            prev = last.get(team)
            if prev is None or (r.date - prev["date"]).days > 20:
                out.update({f"{side}_rest": float(REST_CAP), f"{side}_b2b": 0, f"{side}_3in4": 0,
                            f"{side}_4in6": 0, f"{side}_km": 0.0, f"{side}_tz": 0.0,
                            f"{side}_road": 0})
                continue
            rest = (r.date - prev["date"]).days
            recent = [d for d in prev["dates"] if (r.date - d).days <= 5]
            out[f"{side}_rest"] = float(min(rest, REST_CAP))
            out[f"{side}_b2b"] = int(rest == 1)
            out[f"{side}_3in4"] = int(sum(1 for d in recent if (r.date - d).days <= 3) >= 2)
            out[f"{side}_4in6"] = int(len(recent) >= 3)
            out[f"{side}_km"] = haversine_km(prev["loc"], here) if prev["loc"] and here else 0.0
            out[f"{side}_tz"] = float(here[2] - prev["loc"][2]) if prev["loc"] and here else 0.0
            out[f"{side}_road"] = 0 if home else prev["road"] + 1
        rows.append(out)
        for team, home in ((r.home_team, True), (r.away_team, False)):
            prev = last.get(team)
            dates = ([d for d in prev["dates"] if (r.date - d).days <= 6] if prev else []) + [r.date]
            last[team] = {"date": r.date, "loc": here, "dates": dates,
                          "road": 0 if home else ((prev["road"] + 1) if prev else 1)}
    return pd.DataFrame(rows)


def _row(rb: RatingBook, pb: PlayerBook, home: str, away: str, neutral: bool,
         d_h: dict, d_a: dict) -> dict:
    """One game's pre-game features from both books' state before it."""
    adj = pb.adjustment(d_h, d_a)
    e = rb.expect(home, away, neutral, adj)
    e0 = rb.expect(home, away, neutral)
    return {"e_margin": e["margin"], "e_total": e["total"], "e_home": e["pts_h"],
            "e_away": e["pts_a"], "e_margin0": e0["margin"], "e_total0": e0["total"],
            "e_poss": e["poss"], "H": 0.0 if neutral else rb.H, "L": rb.L,
            "h_off": rb.off.get(home, 0.0), "h_def": rb.dfn.get(home, 0.0),
            "a_off": rb.off.get(away, 0.0), "a_def": rb.dfn.get(away, 0.0),
            "h_pace": rb.pace.get(home, 0.0), "a_pace": rb.pace.get(away, 0.0),
            "h_d_net": d_h["net"], "a_d_net": d_a["net"], "h_d_off": d_h["off"],
            "a_d_off": d_a["off"], "h_miss_top": d_h["top"], "a_miss_top": d_a["top"],
            "h_games": rb.games.get(home, 0), "a_games": rb.games.get(away, 0)}


def _player_rows(players: pd.DataFrame | None) -> dict:
    if players is None or players.empty:
        return {}
    p = players[pd.to_numeric(players["minutes"], errors="coerce").fillna(0) > 0]
    out = {}
    for gid, grp in p.groupby(p["game_id"].astype(str), sort=False):
        out[gid] = {team: list(zip(t["athlete_id"].astype(int), t["minutes"].astype(float),
                                   t["gs"].astype(float), t["os"].astype(float)))
                    for team, t in grp.groupby("team", sort=False)}
    return out


def build(games: pd.DataFrame, players: pd.DataFrame | None = None,
          lines: pd.DataFrame | None = None, upcoming: pd.DataFrame | None = None,
          cfg: RatingConfig | None = None, pcfg: PlayerConfig | None = None,
          first_train_season: int | None = None, today=None, board_news: dict | None = None,
          names: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, RatingBook, PlayerBook]:
    """Replay every completed game; emit one pre-game row per game from the training window.

    Seasons before `first_train_season` are warm-up: they advance both books but emit no rows.
    ``upcoming`` rows (the board) are priced off the final state, with availability from
    ``board_news`` - ``{game_id: {"h": (report, roster), "a": (report, roster)}}`` - where
    report maps player id -> status and roster is the set of player ids on the team.
    """
    first = config.FIRST_SEASON if first_train_season is None else first_train_season
    today = today or config.today_et()
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    g["completed"] = g["completed"].astype(bool)
    g["game_id"] = g["game_id"].astype(str)
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    sched = {r["game_id"]: r for r in schedule_features(g, today=today).to_dict("records")}
    by_game = _player_rows(players)

    rb, pb = RatingBook(cfg), PlayerBook(pcfg)
    if names:
        pb.names.update(names)
    rows = []
    for r in g[g["completed"]].itertuples(index=False):
        rb.rollover(int(r.season))
        pb.rollover(int(r.season))
        neutral = bool(r.neutral_site) or in_bubble(r.date)
        teams = by_game.get(r.game_id, {})
        hp, ap = teams.get(r.home_team, []), teams.get(r.away_team, [])
        # a game with no player log (a handful, mostly 2012 and 2020) gets no availability
        # delta rather than a guess: better a zero than a delta built on nobody playing
        have = bool(hp) and bool(ap)
        d_h = pb.delta(r.home_team, pb.played_target(p for p, *_ in hp)) if have else \
            {"net": 0.0, "off": 0.0, "top": 0.0}
        d_a = pb.delta(r.away_team, pb.played_target(p for p, *_ in ap)) if have else \
            {"net": 0.0, "off": 0.0, "top": 0.0}
        if int(r.season) >= first:
            row = _row(rb, pb, r.home_team, r.away_team, neutral, d_h, d_a)
            row.update({"game_id": r.game_id, "season": int(r.season), "date": r.date,
                        "game_type": r.game_type, "home_team": r.home_team,
                        "away_team": r.away_team, "home_points": r.home_points,
                        "away_points": r.away_points, "neutral": int(neutral),
                        "has_players": int(have)})
            rows.append(row)
        # State advances only AFTER the row is recorded. This ordering IS the leak-free
        # guarantee, and test_leak_free rebuilds from a truncated schedule to prove it.
        adj = pb.adjustment(d_h, d_a)
        rb.update(r.home_team, r.away_team, r.home_points, r.away_points, r.poss, r.periods,
                  neutral, adj)
        for team, pr in ((r.home_team, hp), (r.away_team, ap)):
            pb.update(team, pr, r.date)
    train = pd.DataFrame(rows)

    up_rows = []
    if upcoming is not None and len(upcoming):
        up = upcoming.copy()
        up["date"] = pd.to_datetime(up["date"]).dt.date
        up["game_id"] = up["game_id"].astype(str)
        season_now = int(up["season"].max()) if "season" in up else config.season_of(today)
        rb.rollover(season_now)
        pb.rollover(season_now)
        news = board_news or {}
        for r in up.sort_values(["date", "game_id"]).to_dict("records"):
            n = news.get(r["game_id"], {})
            t_h, notes_h = pb.board_target(r["home_team"], *n.get("h", (None, None)))
            t_a, notes_a = pb.board_target(r["away_team"], *n.get("a", (None, None)))
            d_h, d_a = pb.delta(r["home_team"], t_h), pb.delta(r["away_team"], t_a)
            neutral = bool(r.get("neutral_site")) or in_bubble(r["date"])
            row = _row(rb, pb, r["home_team"], r["away_team"], neutral, d_h, d_a)
            row.update({k: v for k, v in r.items() if k not in row})
            row.update({"neutral": int(neutral), "h_notes": notes_h, "a_notes": notes_a,
                        "h_pending": pb.pending_points(notes_h),
                        "a_pending": pb.pending_points(notes_a)})
            up_rows.append(row)
    board = pd.DataFrame(up_rows)

    for frame in (train, board):
        if len(frame):
            finish(frame, sched)
    if lines is not None and len(lines) and len(train):
        train = attach_market(train, lines)
    return train, board, rb, pb


def finish(frame: pd.DataFrame, sched: dict) -> pd.DataFrame:
    """Schedule and context columns, in place."""
    cols = ["h_rest", "a_rest", "h_b2b", "a_b2b", "h_3in4", "a_3in4", "h_4in6", "a_4in6",
            "h_km", "a_km", "h_tz", "a_tz", "h_road", "a_road", "altitude"]
    for c in cols:
        frame[c] = [sched.get(str(g), {}).get(c, np.nan) for g in frame["game_id"]]
    frame["rest_diff"] = frame["h_rest"] - frame["a_rest"]
    frame["h_km"] = frame["h_km"] / 1000
    frame["a_km"] = frame["a_km"] / 1000
    frame["h_tz_abs"] = frame["h_tz"].abs()
    frame["a_tz_abs"] = frame["a_tz"].abs()
    gt = frame["game_type"].astype(str)
    frame["playoff"] = (gt == "P").astype(int)
    frame["playin"] = (gt == "I").astype(int)
    bubble = np.array([in_bubble(d) for d in pd.to_datetime(frame["date"]).dt.date], dtype=bool)
    frame["no_crowd"] = (frame["season"].astype(int).isin(config.NO_CROWD_SEASONS).to_numpy()
                         | bubble).astype(int)
    frame["early"] = (np.minimum(frame["h_games"], frame["a_games"]) < EARLY_GAMES).astype(int)
    return frame


LINE_COLS = ["spread_home", "total_line", "p_home", "home_dec", "away_dec", "source", "ml_source",
             "is_closing"]


def attach_market(train: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    keep = ["game_id"] + [c for c in LINE_COLS if c in lines.columns]
    ln = lines[keep].copy()
    ln["game_id"] = ln["game_id"].astype(str)
    ln = ln.drop_duplicates("game_id", keep="last")
    out = train.merge(ln, on="game_id", how="left")
    return market_view(out)


def market_view(df: pd.DataFrame) -> pd.DataFrame:
    """The market's expected margin and total, on the model's own scale.

    ``spread_home`` is in book convention - -6.5 means home favoured by 6.5 - so the market's
    expected home margin is its negative. That sign is flipped here and nowhere else.
    """
    df = df.copy()
    for c in LINE_COLS:
        if c not in df:
            df[c] = np.nan
    df["mkt_margin"] = -pd.to_numeric(df["spread_home"], errors="coerce")
    df["mkt_total"] = pd.to_numeric(df["total_line"], errors="coerce")
    return df
