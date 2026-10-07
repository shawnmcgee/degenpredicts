"""Six-point NFL teasers through 3 and 7: which legs qualify, how they grade, how they did.

A teaser moves the line in your favour on every leg in exchange for having to win them all, and
books price every teased point the same. NFL margins are not uniform - 3 and 7 are the commonest
results by far - so six points that cross both are worth more than six that cross neither. An
underdog at +1.5 or +2.5 teased to +7.5 or +8.5 is the leg that crosses both; a high total makes
those margins matter less, so legs in games totalled above 49 are left out.

The favourite version (-7.5/-8.5 teased to -1.5/-2.5) is deliberately absent: it held up until
2020 and has covered 64.8% since, against a 73.9% break-even.

This is a SPORTSBOOK product. Exchanges do not sell teasers - their +7.5 is a separate contract
priced on its own, so the mispricing a teaser exploits does not exist there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

ERAS = [(2010, 2015), (2016, 2020), (2021, 2025)]


def break_even(price: float | None = None, legs: int = 2) -> float:
    """The win rate each leg needs for a ``legs``-team teaser at an American ``price``."""
    price = config.TEASER_PRICE if price is None else price
    pay = price / 100 if price > 0 else 100 / -price
    return (1 / (1 + pay)) ** (1 / legs)


def two_team_ev(leg_win: float, price: float | None = None) -> float:
    """EV per unit of a two-team teaser whose legs each win ``leg_win``, independently."""
    price = config.TEASER_PRICE if price is None else price
    pay = price / 100 if price > 0 else 100 / -price
    return leg_win ** 2 * (1 + pay) - 1


def legs(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, the qualifying leg if there is one: ``teaser_team``, ``teaser_line`` (the
    number it was at) and ``teaser_to`` (the teased number). Rows that do not qualify get NaN.
    Needs ``home_team``, ``away_team``, ``spread_home`` (negative = home favoured) and
    ``total_line``."""
    sh = pd.to_numeric(df["spread_home"], errors="coerce")
    tl = pd.to_numeric(df["total_line"], errors="coerce")
    ok_total = tl.notna() & (tl <= config.TEASER_MAX_TOTAL)
    home_dog = sh.isin(config.TEASER_DOG_LINES) & ok_total
    away_dog = (-sh).isin(config.TEASER_DOG_LINES) & ok_total
    team = np.where(home_dog, df["home_team"], np.where(away_dog, df["away_team"], None))
    line = np.where(home_dog, sh, np.where(away_dog, -sh, np.nan))
    return pd.DataFrame({"teaser_team": team, "teaser_line": line,
                         "teaser_to": line + config.TEASER_POINTS}, index=df.index)


def grade(df: pd.DataFrame) -> pd.Series:
    """"win"/"loss" for each qualifying leg, NaN elsewhere. Teased lines are half-points, so a
    leg cannot push. Needs the columns :func:`legs` reads plus ``home_margin``."""
    lg = legs(df)
    margin = pd.to_numeric(df["home_margin"], errors="coerce")
    side = np.where(lg["teaser_team"] == df["home_team"], margin, -margin)
    won = side + lg["teaser_to"] > 0
    out = pd.Series(np.where(won, "win", "loss"), index=df.index, dtype=object)
    return out.where(lg["teaser_team"].notna() & margin.notna())


def record(results: pd.Series) -> dict:
    """Wins, losses and what they are worth, for a set of graded legs."""
    w, l = int((results == "win").sum()), int((results == "loss").sum())
    rate = w / (w + l) if w + l else None
    return {"legs": w + l, "wins": w, "losses": l,
            "win_pct": round(100 * rate, 1) if rate is not None else None,
            "break_even": round(100 * break_even(), 1),
            "two_team_ev": round(100 * two_team_ev(rate), 1) if rate is not None else None}


def backtest(games: pd.DataFrame, lines: pd.DataFrame) -> list[dict]:
    """The rule against closing lines and final scores, one row per era. Closing lines are the
    number nflverse keeps; a live bettor takes an earlier one, so read this as the rule's
    long-run shape, not a promise."""
    g = games.drop(columns=[c for c in ("spread_home", "total_line") if c in games]).copy()
    g["game_id"] = g["game_id"].astype(str)
    ln = lines[["game_id", "spread_home", "total_line"]].copy()
    ln["game_id"] = ln["game_id"].astype(str)
    d = g.merge(ln, on="game_id", how="inner")
    d = d[d["completed"].astype(bool) & d["home_points"].notna()].copy()
    d["home_margin"] = d["home_points"] - d["away_points"]
    d["teaser_result"] = grade(d)
    out = []
    for lo, hi in ERAS:
        r = d.loc[d["season"].between(lo, hi), "teaser_result"]
        if r.notna().any():
            out.append({"era": f"{lo}-{str(hi)[-2:]}", **record(r)})
    return out
