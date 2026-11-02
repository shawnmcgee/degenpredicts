"""Model features, built by replaying the game log in order.

The same pass produces training rows and today's rows, so the models never see a feature at
predict time that they were not trained on. Every rating attached to game N was built from
games 1..N-1 only.

Two targets:
    total_points  -> the totals models
    home_margin   -> the spread models (positive = home won by that much)

The market-aware models also get the line. They predict the line's *error* (actual minus line)
rather than the score itself - see :mod:`ncaab.models` for why that matters to tree models.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from . import config
from .ratings import RatingBook, RatingConfig

log = logging.getLogger(__name__)

BASE_FEATURES = [
    "neutral_site", "conf_game", "postseason", "days_into_season",
    "exp_margin", "exp_total", "exp_pace",
    "h_margin", "a_margin", "h_off", "a_off", "h_def", "a_def", "h_pace", "a_pace",
    "h_games", "a_games", "h_rest", "a_rest", "h_b2b", "a_b2b",
    "h_form_margin", "a_form_margin", "h_form_total", "a_form_total",
    "h_rank_pts", "a_rank_pts", "h_new", "a_new", "h_power", "a_power",
]
MARKET_FEATURES = ["total_line", "spread_home", "total_vs_ratings", "margin_vs_ratings"]
TARGETS = {"margin": "home_margin", "total": "total_points"}
LINE_COLS = {"margin": "spread_home", "total": "total_line"}

REST_CAP = 14


def market_line(df: pd.DataFrame, kind: str) -> pd.Series:
    """The line in the target's own units: a home spread of -6.5 expects a home margin of +6.5.
    This is the only place the spread's sign is flipped."""
    if kind == "margin":
        return -pd.to_numeric(df["spread_home"], errors="coerce")
    return pd.to_numeric(df["total_line"], errors="coerce")


def _rest(last, gdate) -> float:
    if last is None or gdate is None:
        return float(REST_CAP)
    return float(min((gdate - last).days, REST_CAP))


def _rank_pts(rank) -> float:
    """AP rank as strength: No. 1 is 25 points, No. 25 is 1, unranked is 0."""
    try:
        r = int(rank)
    except (TypeError, ValueError):
        return 0.0
    return float(26 - r) if 1 <= r <= 25 else 0.0


def _flag(v) -> int:
    if isinstance(v, str):
        return int(v.strip().lower() in ("true", "1", "1.0", "yes"))
    try:
        return int(bool(v) and v == v)
    except (TypeError, ValueError):
        return 0


def _row(book: RatingBook, season: int, rec: dict, gdate: date) -> dict:
    home, away = str(rec["home_id"]), str(rec["away_id"])
    neutral = bool(_flag(rec.get("neutral_site")))
    h, a = book.get(season, home), book.get(season, away)
    e = book.expect(season, home, away, neutral)
    h_rest, a_rest = _rest(h.last_date, gdate), _rest(a.last_date, gdate)
    return {
        "neutral_site": int(neutral), "conf_game": _flag(rec.get("conf_game")),
        "postseason": int(str(rec.get("season_type", 2)) in ("3", "3.0")),
        "days_into_season": (gdate - config.season_start(season)).days,
        "exp_margin": e["exp_margin"], "exp_total": e["exp_total"], "exp_pace": e["exp_pace"],
        "h_margin": h.margin, "a_margin": a.margin,
        "h_off": h.off, "a_off": a.off, "h_def": h.deff, "a_def": a.deff,
        "h_pace": h.pace, "a_pace": a.pace, "h_games": h.games, "a_games": a.games,
        "h_rest": h_rest, "a_rest": a_rest,
        "h_b2b": int(h_rest <= 1), "a_b2b": int(a_rest <= 1),
        "h_form_margin": float(np.mean(h.recent_margin)) if h.recent_margin else 0.0,
        "a_form_margin": float(np.mean(a.recent_margin)) if a.recent_margin else 0.0,
        "h_form_total": float(np.mean(h.recent_total)) if h.recent_total else 0.0,
        "a_form_total": float(np.mean(a.recent_total)) if a.recent_total else 0.0,
        "h_rank_pts": _rank_pts(rec.get("home_rank")), "a_rank_pts": _rank_pts(rec.get("away_rank")),
        "h_new": int(h.new), "a_new": int(a.new),
        "h_power": int(str(rec.get("home_conf", "")).split(".")[0] in config.POWER),
        "a_power": int(str(rec.get("away_conf", "")).split(".")[0] in config.POWER),
    }


def _add_market(row: dict, total_line, spread_home) -> dict:
    tl = pd.to_numeric(total_line, errors="coerce")
    sp = pd.to_numeric(spread_home, errors="coerce")
    row["total_line"] = float(tl) if tl == tl else np.nan
    row["spread_home"] = float(sp) if sp == sp else np.nan
    row["total_vs_ratings"] = row["exp_total"] - row["total_line"]
    row["margin_vs_ratings"] = row["exp_margin"] + row["spread_home"]   # exp - (-spread)
    return row


def build(games: pd.DataFrame, upcoming: pd.DataFrame | None = None,
          cfg: RatingConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, RatingBook]:
    """Replay every final in ``games`` and return (train_rows, upcoming_rows, rating_book)."""
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    sort = ["date", "kickoff_utc", "game_id"] if "kickoff_utc" in g else ["date", "game_id"]
    g = g.sort_values(sort).reset_index(drop=True)

    book = RatingBook(cfg)
    rows = []
    for rec in g.to_dict("records"):
        season = int(rec["season"])
        r = _row(book, season, rec, rec["date"])
        r = _add_market(r, rec.get("total_line"), rec.get("spread_home"))
        r.update({"game_id": str(rec["game_id"]), "date": rec["date"], "season": season,
                  "home_id": str(rec["home_id"]), "away_id": str(rec["away_id"]),
                  "home_team": rec.get("home_team"), "away_team": rec.get("away_team"),
                  "total_points": rec["total_points"], "home_margin": rec["home_margin"]})
        rows.append(r)
        book.update(season, str(rec["home_id"]), str(rec["away_id"]), rec["home_points"],
                    rec["away_points"], rec["date"], bool(_flag(rec.get("neutral_site"))))
    train = pd.DataFrame(rows)
    if len(train):
        # games each team played that season - in hindsight, so used ONLY to decide which
        # games are D1-vs-D1 training rows, never as a feature
        n = pd.concat([train[["season", "home_id"]].rename(columns={"home_id": "t"}),
                       train[["season", "away_id"]].rename(columns={"away_id": "t"})])
        counts = n.groupby(["season", "t"]).size()
        key_h = list(zip(train["season"], train["home_id"]))
        key_a = list(zip(train["season"], train["away_id"]))
        train["h_season_games"] = counts.reindex(key_h).to_numpy()
        train["a_season_games"] = counts.reindex(key_a).to_numpy()

    up = []
    if upcoming is not None and len(upcoming):
        for rec in upcoming.to_dict("records"):
            gdate = pd.Timestamp(rec["date"]).date()
            r = _row(book, config.season_of(gdate), rec, gdate)
            r = _add_market(r, rec.get("total_line"), rec.get("spread_home"))
            keep = {k: v for k, v in rec.items() if k not in r}
            r.update(keep)
            up.append(r)
    return train, pd.DataFrame(up), book


def d1_rows(train: pd.DataFrame) -> pd.DataFrame:
    """Training rows between two Division I teams, past the warm-up season."""
    if train.empty:
        return train
    first = int(train["season"].min()) + config.WARMUP_SEASONS
    keep = (train["season"] >= first) & (train["h_season_games"] >= config.D1_MIN_GAMES) & \
        (train["a_season_games"] >= config.D1_MIN_GAMES)
    return train[keep].reset_index(drop=True)
