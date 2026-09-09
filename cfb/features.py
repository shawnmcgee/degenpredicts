"""Feature construction for college football.

Built by replaying the schedule in order, so a feature attached to game N only ever saw games
1..N-1. Two extra ingredients that matter enormously in weeks 1-4, when in-season data barely
exists, and which are both legitimately known before kickoff:

* ``prev_sp_*``   - last season's SP+ overall/offence/defence. Joined from season-1 only.
* ``ret_*``       - returning production (share of last year's PPA coming back). Published
                    in the offseason, so it's known at week 1.

FCS opponents *are* in the raw CFBD game file whatever classification the request asked for,
so :func:`fbs_membership` resolves which teams played FBS football in each season and every
row carries an ``fbs`` flag. What that flag is used for is deliberately asymmetric:

* **The board is filtered by it** (see ``predict.build_board``). We do not publish a number on
  a game the model has no business pricing, and the site claims FBS coverage.
* **Training and the rating replay are not.** That was measured rather than assumed: dropping
  the ~68% non-FBS rows costs 0.05-0.10 points of holdout MAE and 0.5-1.8 points of ATS on
  FBS-vs-FBS games, stable across seeds. Replaying eleven seasons rates the repeat FCS
  visitors properly (2025: non-FBS mean margin -2.0 vs FBS +10.8), so they are not
  "average FBS" by the time they matter, and the extra 18k rows are worth more than the
  contamination costs.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .ratings import RatingBook, RatingConfig

log = logging.getLogger(__name__)

BASE_FEATURES = [
    "neutral_site", "conference_game", "week", "is_early_season",
    "exp_margin", "exp_total", "exp_home_points", "exp_away_points",
    "h_margin", "a_margin", "h_off", "a_off", "h_def", "a_def",
    "h_games", "a_games", "h_rest", "a_rest", "h_bye", "a_bye",
    "h_form_margin", "a_form_margin", "h_form_total", "a_form_total",
    "h_prev_sp", "a_prev_sp", "h_prev_sp_off", "a_prev_sp_off",
    "h_prev_sp_def", "a_prev_sp_def", "sp_margin_est", "sp_total_est",
    "h_ret_ppa", "a_ret_ppa", "h_ret_pass", "a_ret_pass",
]
MARKET_FEATURES = ["total_line", "spread_home", "total_vs_model", "spread_vs_model",
                   "total_move", "spread_move"]

REST_CAP = 21
EARLY_WEEKS = 4


def fbs_membership(games: pd.DataFrame | None = None,
                   sp: pd.DataFrame | None = None) -> dict[int, set[str]]:
    """Which teams played FBS football in each season.

    Two independent sources, unioned because each is positive evidence on its own:

    * ``sp_ratings.csv`` - SP+ only rates FBS teams, so its roster *is* the FBS membership
      list for that season (129-139 teams). This is membership, not performance, and it is
      known before a ball is snapped, so using the current season's roster leaks nothing.
    * a ``home_classification`` / ``away_classification`` column, when the games cache
      carries one. Older caches predate it; newer ones get it straight from CFBD.

    A season with neither source resolves to an empty set, which :func:`is_fbs_game` reads as
    "don't filter" - a missing roster must not silently delete a whole season.
    """
    out: dict[int, set[str]] = {}
    if sp is not None and len(sp):
        for r in sp.itertuples():
            out.setdefault(int(r.season), set()).add(r.team)
    if games is not None and len(games):
        for side in ("home", "away"):
            col = f"{side}_classification"
            if col not in games.columns:
                continue
            m = games[col].astype(str).str.lower().eq("fbs")
            for s, t in zip(games.loc[m, "season"], games.loc[m, f"{side}_team"]):
                out.setdefault(int(s), set()).add(t)
    return out


def is_fbs_game(members: dict[int, set[str]], season, home, away) -> bool:
    """True when both sides were FBS that season, or when we hold no roster to judge by."""
    roster = members.get(int(season))
    if not roster:
        return True
    return home in roster and away in roster


def hfa_suppressed(season, neutral) -> bool:
    """True when home-field advantage should be treated as absent for this game.

    Two separate causes, one consequence: an actual neutral site, and a season played without
    crowds. ``neutral_site`` stays a feature in its own right - this only governs what the
    rating engine credits.
    """
    return bool(neutral) or int(season) in config.NO_CROWD_SEASONS


def _rest(last, gdate):
    if last is None or gdate is None:
        return np.nan
    return float(min((gdate - last).days, REST_CAP))


def _row(book, sp_prev, ret, season, home, away, gdate, week, no_hfa, conf_game,
         neutral=None) -> dict:
    """`no_hfa` governs the rating maths; `neutral` is the venue as a feature. They differ
    only in a no-crowd season, where the game was at a real home venue with no home edge."""
    neutral = no_hfa if neutral is None else neutral
    h, a = book.get(season, home), book.get(season, away)
    e = book.expect(season, home, away, no_hfa)
    hs, as_ = sp_prev.get((season - 1, home), {}), sp_prev.get((season - 1, away), {})
    hr, ar = ret.get((season, home), {}), ret.get((season, away), {})

    h_sp, a_sp = hs.get("sp_overall", np.nan), as_.get("sp_overall", np.nan)
    h_spo, a_spo = hs.get("sp_off", np.nan), as_.get("sp_off", np.nan)
    h_spd, a_spd = hs.get("sp_def", np.nan), as_.get("sp_def", np.nan)
    sp_margin = (h_sp - a_sp) + (0 if no_hfa else book.cfg.hfa) if h_sp == h_sp and a_sp == a_sp else np.nan
    sp_total = (h_spo + a_spd) / 2 + (a_spo + h_spd) / 2 \
        if not any(x != x for x in (h_spo, a_spd, a_spo, h_spd)) else np.nan

    h_rest, a_rest = _rest(h.last_date, gdate), _rest(a.last_date, gdate)
    return {
        "neutral_site": int(bool(neutral)), "conference_game": int(bool(conf_game)),
        "week": int(week), "is_early_season": int(week <= EARLY_WEEKS),
        "exp_margin": e["exp_margin"], "exp_total": e["exp_total"],
        "exp_home_points": e["exp_home_points"], "exp_away_points": e["exp_away_points"],
        "h_margin": h.margin, "a_margin": a.margin,
        "h_off": h.off, "a_off": a.off, "h_def": h.deff, "a_def": a.deff,
        "h_games": h.games, "a_games": a.games,
        "h_rest": h_rest, "a_rest": a_rest,
        "h_bye": int(h_rest >= 12) if h_rest == h_rest else 0,
        "a_bye": int(a_rest >= 12) if a_rest == a_rest else 0,
        "h_form_margin": float(np.mean(h.recent_margin)) if h.recent_margin else 0.0,
        "a_form_margin": float(np.mean(a.recent_margin)) if a.recent_margin else 0.0,
        "h_form_total": float(np.mean(h.recent_total)) if h.recent_total else 0.0,
        "a_form_total": float(np.mean(a.recent_total)) if a.recent_total else 0.0,
        "h_prev_sp": h_sp, "a_prev_sp": a_sp,
        "h_prev_sp_off": h_spo, "a_prev_sp_off": a_spo,
        "h_prev_sp_def": h_spd, "a_prev_sp_def": a_spd,
        "sp_margin_est": sp_margin, "sp_total_est": sp_total,
        "h_ret_ppa": hr.get("ret_ppa", np.nan), "a_ret_ppa": ar.get("ret_ppa", np.nan),
        "h_ret_pass": hr.get("ret_pass_pct", np.nan), "a_ret_pass": ar.get("ret_pass_pct", np.nan),
    }


def _market(row: dict, total_line, spread_home, total_open=np.nan, spread_open=np.nan) -> dict:
    tl = float(total_line) if total_line == total_line and total_line is not None else np.nan
    sh = float(spread_home) if spread_home == spread_home and spread_home is not None else np.nan
    row["total_line"] = tl
    row["spread_home"] = sh
    row["total_vs_model"] = row["exp_total"] - tl
    # CFBD quotes spreads from the home side: -6.5 means home favoured by 6.5
    row["spread_vs_model"] = row["exp_margin"] - (-sh)
    # NaN, never 0.0. `spread_open` is absent for 100% of 2015-2020 and ~45% of 2022-2025 but
    # present on 96% of live rows, so filling the gap with a real number teaches the model that
    # "the line has not moved" describes six whole seasons - a train/serve mismatch, and the
    # same one NFL was deliberately built to avoid. XGBoost handles missing natively, which is
    # what the rest of these features already rely on.
    row["total_move"] = tl - total_open if total_open == total_open else np.nan
    row["spread_move"] = sh - spread_open if spread_open == spread_open else np.nan
    return row


def build(games: pd.DataFrame, upcoming: pd.DataFrame | None = None,
          lines: pd.DataFrame | None = None, sp: pd.DataFrame | None = None,
          returning: pd.DataFrame | None = None,
          cfg: RatingConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, RatingBook]:
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"]).dt.date
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)

    sp_prev, ret = {}, {}
    if sp is not None and len(sp):
        sp_prev = {(int(r.season), r.team): {"sp_overall": r.sp_overall, "sp_off": r.sp_off,
                                             "sp_def": r.sp_def} for r in sp.itertuples()}
    if returning is not None and len(returning):
        ret = {(int(r.season), r.team): {"ret_ppa": r.ret_ppa, "ret_pass_pct": r.ret_pass_pct}
               for r in returning.itertuples()}

    line_map, n_prov = {}, {}
    if lines is not None and len(lines):
        for r in lines.itertuples():
            line_map[str(r.game_id)] = (r.spread_home, r.total_line,
                                        getattr(r, "spread_open", np.nan),
                                        getattr(r, "total_open", np.nan))
            n_prov[str(r.game_id)] = getattr(r, "n_providers", np.nan)

    members = fbs_membership(games, sp)

    book = RatingBook(cfg)
    rows = []
    for g in games.itertuples(index=False):
        if not bool(getattr(g, "completed", False)):
            continue  # scheduled-but-unplayed rows live in `upcoming`, not training
        season, week = int(g.season), int(g.week)
        neutral = bool(getattr(g, "neutral_site", False))
        no_hfa = hfa_suppressed(season, neutral)
        r = _row(book, sp_prev, ret, season, g.home_team, g.away_team, g.date, week,
                 no_hfa, bool(getattr(g, "conference_game", False)), neutral=neutral)
        sh, tl, so, to = line_map.get(str(g.game_id), (np.nan, np.nan, np.nan, np.nan))
        r = _market(r, tl, sh, to, so)
        r.update({"game_id": g.game_id, "date": g.date, "season": season, "week": week,
                  "home_team": g.home_team, "away_team": g.away_team,
                  "total_points": g.total_points, "home_margin": g.home_margin,
                  "home_conf": getattr(g, "home_conf", ""), "away_conf": getattr(g, "away_conf", ""),
                  "n_providers": n_prov.get(str(g.game_id), np.nan)})
        rows.append(r)
        book.update(season, g.home_team, g.away_team, g.home_points, g.away_points, g.date, no_hfa)

    train_rows = pd.DataFrame(rows)
    if len(train_rows):
        train_rows["fbs"] = [is_fbs_game(members, s, h, a) for s, h, a in
                             zip(train_rows.season, train_rows.home_team, train_rows.away_team)]
        log.info("%d training rows, %d of them FBS vs FBS",
                 len(train_rows), int(train_rows["fbs"].sum()))

    up_rows = []
    if upcoming is not None and len(upcoming):
        for rec in upcoming.to_dict("records"):
            gdate = pd.Timestamp(rec["date"]).date()
            season = int(rec.get("season") or config.season_of(gdate))
            neutral = bool(rec.get("neutral_site", False))
            r = _row(book, sp_prev, ret, season, rec["home_team"], rec["away_team"], gdate,
                     int(rec.get("week", 1)), hfa_suppressed(season, neutral),
                     bool(rec.get("conference_game", False)), neutral=neutral)
            r = _market(r, rec.get("total_line", np.nan), rec.get("spread_home", np.nan),
                        rec.get("total_open", np.nan), rec.get("spread_open", np.nan))
            r.update(rec)
            r["fbs"] = is_fbs_game(members, season, rec["home_team"], rec["away_team"])
            up_rows.append(r)
    return train_rows, pd.DataFrame(up_rows), book


def fbs_teams(games: pd.DataFrame, season: int, sp: pd.DataFrame | None = None) -> set[str]:
    """The name list the odds and Kalshi matchers resolve feed spellings against.

    Built from the FBS roster when one is available. Taking every name in the games file
    instead put 699 teams in the 2025 table, which is not belt-and-braces: ``difflib`` at
    cutoff 0.80 gets hundreds of extra collision targets to mis-resolve a name onto.
    """
    members = fbs_membership(games, sp)
    named = members.get(int(season), set()) | members.get(int(season) - 1, set())
    if named:
        return named
    if games is None or not len(games) or "season" not in games.columns:
        return set()
    m = games["season"].isin([season, season - 1])
    return set(games.loc[m, "home_team"]) | set(games.loc[m, "away_team"])
