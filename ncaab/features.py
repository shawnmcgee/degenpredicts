"""Build model features by replaying the game log chronologically.

The same function produces training rows and prediction rows, so the model can never see a
feature distribution at predict time that it wasn't trained on.

Two targets:
    total_points  -> the totals model
    home_margin   -> the spread model  (positive = home won by that much)

Market features (the line itself) are optional. When present the models learn to predict the
*residual* against the market, which is a far easier and better-calibrated target than the raw
number. train.py fits both a market-aware and a market-free model; predict.py uses whichever
is available for a given game.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config
from .ratings import LEAGUE_PACE, LEAGUE_PPP, RatingBook, RatingConfig
from .sources import torvik as torvik_src

log = logging.getLogger(__name__)

BASE_FEATURES = [
    "neutral_site", "days_into_season",
    "exp_margin", "exp_total", "exp_pace",
    "h_margin", "a_margin", "h_off", "a_off", "h_def", "a_def", "h_pace", "a_pace",
    "h_games", "a_games", "h_rest", "a_rest", "h_b2b", "a_b2b",
    "h_form_margin", "a_form_margin", "h_form_total", "a_form_total",
    "h_adj_o", "h_adj_d", "h_adj_t", "a_adj_o", "a_adj_d", "a_adj_t",
    "adj_total_est", "adj_margin_est",
]
MARKET_FEATURES = ["total_line", "spread_home", "total_vs_model", "spread_vs_model"]

REST_CAP = 14


def _rest(last, gdate) -> float:
    if last is None or gdate is None:
        return np.nan
    d = (gdate - last).days
    return float(min(d, REST_CAP))


def _row(book: RatingBook, tor: dict, season: int, home: str, away: str, gdate: date,
         neutral: bool) -> dict:
    h = book.get(season, home)
    a = book.get(season, away)
    e = book.expect(season, home, away, neutral)
    th, ta = tor.get(home, {}), tor.get(away, {})

    adj_o_h, adj_d_h, adj_t_h = th.get("adj_o", np.nan), th.get("adj_d", np.nan), th.get("adj_t", np.nan)
    adj_o_a, adj_d_a, adj_t_a = ta.get("adj_o", np.nan), ta.get("adj_d", np.nan), ta.get("adj_t", np.nan)
    pace_est = np.nanmean([adj_t_h, adj_t_a]) if not (np.isnan(adj_t_h) and np.isnan(adj_t_a)) else np.nan
    adj_total = ((adj_o_h + adj_d_a) / 2 + (adj_o_a + adj_d_h) / 2) * pace_est / 100 \
        if not any(np.isnan(x) for x in (adj_o_h, adj_d_a, adj_o_a, adj_d_h)) and pace_est == pace_est else np.nan
    adj_margin = ((adj_o_h - adj_d_h) - (adj_o_a - adj_d_a)) * (pace_est or LEAGUE_PACE) / 100 \
        if not any(np.isnan(x) for x in (adj_o_h, adj_d_h, adj_o_a, adj_d_a)) else np.nan

    h_rest, a_rest = _rest(h.last_date, gdate), _rest(a.last_date, gdate)
    return {
        "neutral_site": int(bool(neutral)),
        "days_into_season": (gdate - config.season_start(season)).days,
        "exp_margin": e["exp_margin"], "exp_total": e["exp_total"], "exp_pace": e["exp_pace"],
        "h_margin": h.margin, "a_margin": a.margin,
        "h_off": h.off, "a_off": a.off, "h_def": h.deff, "a_def": a.deff,
        "h_pace": h.pace, "a_pace": a.pace, "h_games": h.games, "a_games": a.games,
        "h_rest": h_rest, "a_rest": a_rest,
        "h_b2b": int(h_rest == 1), "a_b2b": int(a_rest == 1),
        "h_form_margin": float(np.mean(h.recent_margin)) if h.recent_margin else 0.0,
        "a_form_margin": float(np.mean(a.recent_margin)) if a.recent_margin else 0.0,
        "h_form_total": float(np.mean(h.recent_total)) if h.recent_total else 0.0,
        "a_form_total": float(np.mean(a.recent_total)) if a.recent_total else 0.0,
        "h_adj_o": adj_o_h, "h_adj_d": adj_d_h, "h_adj_t": adj_t_h,
        "a_adj_o": adj_o_a, "a_adj_d": adj_d_a, "a_adj_t": adj_t_a,
        "adj_total_est": adj_total, "adj_margin_est": adj_margin,
    }


def _add_market(row: dict, total_line, spread_home) -> dict:
    row["total_line"] = float(total_line) if total_line == total_line and total_line is not None else np.nan
    row["spread_home"] = float(spread_home) if spread_home == spread_home and spread_home is not None else np.nan
    row["total_vs_model"] = row["exp_total"] - row["total_line"]
    # book spreads are quoted from the home side: -6.5 means home favoured by 6.5
    row["spread_vs_model"] = row["exp_margin"] - (-row["spread_home"])
    return row


def build(games: pd.DataFrame, upcoming: pd.DataFrame | None = None,
          torvik_cache: pd.DataFrame | None = None, lines: pd.DataFrame | None = None,
          cfg: RatingConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, RatingBook]:
    """Replay `games` and return (train_rows, upcoming_rows, rating_book)."""
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"]).dt.date
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)

    torvik_cache = torvik_src.load() if torvik_cache is None else torvik_cache
    # historical lines keyed by (date, home, away) so training rows can carry the market
    line_map = {}
    if lines is not None and len(lines):
        for r in lines.itertuples():
            line_map[(r.date, r.home_team, r.away_team)] = (
                getattr(r, "total_line", np.nan), getattr(r, "spread_home", np.nan))

    book = RatingBook(cfg)
    rows, tor_day, tor_cached = [], None, {}

    for g in games.itertuples(index=False):
        season = int(g.season)
        if g.date != tor_day:
            tor_day, tor_cached = g.date, torvik_src.as_of(torvik_cache, g.date)
        neutral = bool(getattr(g, "neutral_site", False))
        r = _row(book, tor_cached, season, g.home_team, g.away_team, g.date, neutral)
        tl, sp = line_map.get((g.date, g.home_team, g.away_team), (np.nan, np.nan))
        r = _add_market(r, tl, sp)
        r.update({"game_id": g.game_id, "date": g.date, "season": season,
                  "home_team": g.home_team, "away_team": g.away_team,
                  "total_points": g.total_points, "home_margin": g.home_margin})
        rows.append(r)
        book.update(season, g.home_team, g.away_team, g.home_points, g.away_points, g.date, neutral)

    train_rows = pd.DataFrame(rows)

    up_rows = []
    if upcoming is not None and len(upcoming):
        for rec in upcoming.to_dict("records"):
            gdate = pd.Timestamp(rec["date"]).date()
            season = config.season_of(gdate)
            tor = torvik_src.as_of(torvik_cache, gdate)
            r = _row(book, tor, season, rec["home_team"], rec["away_team"], gdate,
                     bool(rec.get("neutral_site", False)))
            r = _add_market(r, rec.get("total_line", np.nan), rec.get("spread_home", np.nan))
            r.update(rec)
            up_rows.append(r)
    return train_rows, pd.DataFrame(up_rows), book


def known_teams(games: pd.DataFrame, season: int) -> set[str]:
    m = games["season"].isin([season, season - 1])
    return set(games.loc[m, "home_team"]) | set(games.loc[m, "away_team"])
