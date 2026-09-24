"""Grade published NBA picks and write metrics.json.

    python -m nba.grade

Grading joins on ESPN's game id, so there is no team-name matching to go wrong, and it settles
exactly the way a book does:

* **Spread** - the side's final margin plus its line, overtime included. A whole-number line
  pushes on an exact hit and the stake comes back.
* **Total** - final points against the line, overtime included; a whole-number total pushes.
* **Moneyline** - the side won or it did not. There are no ties in basketball.

Units are settled at the price the pick was published at. Closing-line value is measured from
the number we FIRST published to the last snapshot before tip: in points for spreads and totals
(signed so that positive means the market moved toward our side), and in the market's own
de-vigged win probability for the moneyline. CLV says something within weeks; a win rate needs
seasons.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import config
from .sources import hoopr, odds

log = logging.getLogger("nba.grade")
EDGE_BUCKETS = {"spread": [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 99)],
                "total": [(0, 1), (1, 2), (2, 3), (3, 4.5), (4.5, 99)],
                "ml": [(-1.0, 0.0), (0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.0)]}


def _load(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"game_id": str}, low_memory=False)
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def settle(side_margin, line):
    """'win' / 'loss' / 'push' for a side's final margin against its line."""
    adj = np.asarray(side_margin, float) + np.asarray(line, float)
    return np.select([adj > 0, adj < 0], ["win", "loss"], default="push")


def _units(result, stake, dec):
    stake = pd.Series(stake).fillna(0.0).values
    dec = pd.Series(dec).fillna(1.0).values
    # "+ 0.0" folds the -0.0 an unstaked loss produces back into a plain zero
    return np.select([result == "win", result == "loss"], [stake * (dec - 1), -stake],
                     default=0.0).round(3) + 0.0


def grade() -> pd.DataFrame:
    picks = _load(config.PICKS)
    if picks.empty:
        log.warning("no picks yet")
        return picks
    done = _load(config.RESULTS)
    already = set(done["game_id"].astype(str)) if len(done) else set()
    pending = picks[~picks["game_id"].astype(str).isin(already)]
    if pending.empty:
        log.info("nothing new to grade")
        return done

    games = hoopr.update()
    finals = games[games["completed"].astype(bool)][["game_id", "home_points", "away_points"]].copy()
    finals["game_id"] = finals["game_id"].astype(str)
    m = pending.merge(finals, on="game_id", how="inner")
    if m.empty:
        log.info("%d picks still awaiting final scores", len(pending))
        return done
    m = m.copy()
    m["home_margin"] = m["home_points"] - m["away_points"]
    m["total_points"] = m["home_points"] + m["away_points"]

    home = m["spread_side"] == "home"
    side_margin = np.where(home, m["home_margin"], -m["home_margin"])
    has_spread = m["spread_line"].notna()
    m["spread_result"] = np.where(has_spread, settle(side_margin, m["spread_line"].fillna(0)), "")
    over = m["total_side"] == "over"
    tot_margin = np.where(over, m["total_points"] - m["total_line_used"],
                          m["total_line_used"] - m["total_points"])
    has_total = m["total_line_used"].notna()
    m["total_result"] = np.where(has_total, settle(np.nan_to_num(tot_margin), 0.0), "")
    ml_home = m["ml_side"] == "home"
    has_ml = m["ml_side"].isin(["home", "away"])
    won = np.where(ml_home, m["home_margin"] > 0, m["home_margin"] < 0)
    m["ml_result"] = np.where(has_ml, np.where(won, "win", "loss"), "")
    for k in ("spread", "total", "ml"):
        m[f"{k}_units"] = _units(m[f"{k}_result"], m[f"{k}_stake"], m[f"{k}_dec"])
    m["margin_abs_err"] = (m["pub_margin"] - m["home_margin"]).abs()
    m["total_abs_err"] = (m["pub_total"] - m["total_points"]).abs()

    # ---- closing-line value ------------------------------------------------------------
    close = odds.last_before_start(odds.load_snapshots())
    for c in ("spread_clv", "total_clv", "ml_clv"):
        m[c] = np.nan
    if len(close):
        c = close[["game_id", "spread_home", "total_line", "p_home"]].rename(
            columns={"spread_home": "close_spread_home", "total_line": "close_total",
                     "p_home": "close_p_home"})
        c["game_id"] = c["game_id"].astype(str)
        m = m.merge(c, on="game_id", how="left")
        first_sh = m["first_spread_home"] if "first_spread_home" in m else m["spread_home"]
        # "+ 0.0" again: an unchanged line must read 0.0, not the -0.0 rounding noise leaves
        m["spread_clv"] = (np.where(home, first_sh - m["close_spread_home"],
                                    m["close_spread_home"] - first_sh)).round(2) + 0.0
        first_t = m["first_total_line"] if "first_total_line" in m else m["total_line_used"]
        m["total_clv"] = (np.where(over, m["close_total"] - first_t,
                                   first_t - m["close_total"])).round(2) + 0.0
        first_p = m["first_mkt_p_home"] if "first_mkt_p_home" in m else m["mkt_p_home"]
        m["ml_clv"] = (100 * np.where(ml_home, m["close_p_home"] - first_p,
                                      first_p - m["close_p_home"])).round(2) + 0.0
    m["graded_at"] = str(config.today_et())

    done = pd.concat([done, m], ignore_index=True) if len(done) else m
    config.ensure_dirs()
    done.to_csv(config.RESULTS, index=False)
    log.info("graded %d | spreads %s | totals %s | moneylines %s", len(m),
             m["spread_result"].value_counts().to_dict(), m["total_result"].value_counts().to_dict(),
             m["ml_result"].value_counts().to_dict())
    return done


def _rec(df: pd.DataFrame, kind: str) -> dict:
    empty = {"n": 0, "wins": 0, "losses": 0, "pushes": 0, "win_pct": 0.0, "units": 0.0,
             "roi": 0.0, "clv": None, "expected_pct": None}
    if df.empty or f"{kind}_result" not in df:
        return empty
    r = df[f"{kind}_result"].astype(str)
    w, l, p = int((r == "win").sum()), int((r == "loss").sum()), int((r == "push").sum())
    if not w + l + p:
        return empty
    staked = float(df[f"{kind}_stake"].fillna(0).sum())
    units = float(df[f"{kind}_units"].fillna(0).sum())
    clv = f"{kind}_clv"
    return {"n": w + l + p, "wins": w, "losses": l, "pushes": p,
            "win_pct": round(100 * w / (w + l), 1) if w + l else 0.0,
            "units": round(units, 2), "roi": round(100 * units / staked, 1) if staked else 0.0,
            "clv": round(float(df[clv].mean()), 2) if clv in df and df[clv].notna().any() else None,
            # the rate the market's own prices expected for the same picks - the bar to beat,
            # because a moneyline favourite "winning" 70% of the time is what its price said
            "expected_pct": round(100 * float(df[f"{kind}_mkt_p"].mean()), 1)
            if f"{kind}_mkt_p" in df and df[f"{kind}_mkt_p"].notna().any() else None}


def metrics(done: pd.DataFrame) -> dict:
    today = config.today_et()
    out = {"updated": str(today), "sport": "nba", "break_even": config.BREAK_EVEN,
           "venue": config.VENUE, "spreads": {}, "totals": {}, "moneyline": {}, "by_week": []}
    if done.empty:
        return out
    season = done[done["season"] == done["season"].max()]
    for key, kind, err, edge in (("spreads", "spread", "margin_abs_err", "spread_disagree"),
                                 ("totals", "total", "total_abs_err", "total_disagree"),
                                 ("moneyline", "ml", None, "ml_ev")):
        played = season[season[f"{kind}_strength"].isin(["play", "bold"])]
        out[key] = {"season": _rec(played, kind), "all_games": _rec(season, kind),
                    "mae": round(float(season[err].mean()), 2) if err and err in season else None,
                    "by_edge": []}
        for lo, hi in EDGE_BUCKETS[kind]:
            v = season[edge] if kind == "ml" else season[edge].abs()
            b = season[(v >= lo) & (v < hi)]
            if len(b):
                label = (f"{lo:+.2f}-{hi:+.2f}" if hi < 1 else f"{lo:+.2f}-+") if kind == "ml" \
                    else (f"{lo:g}-{hi:g}" if hi < 99 else f"{lo:g}-+")
                out[key]["by_edge"].append({"bucket": label, "lo": lo, "hi": hi, **_rec(b, kind)})
    # ISO weeks, for the cumulative-units line; basketball has no rounds to number
    wk = pd.to_datetime(season["date"]).dt.strftime("%G-W%V")
    cum = {"spread": 0.0, "total": 0.0, "ml": 0.0}
    for w, grp in season.groupby(wk):
        row = {"week": w, "games": int(len(grp))}
        for k in cum:
            u = float(grp[f"{k}_units"].sum())
            cum[k] += u
            row[f"{k}_units"] = round(u, 2)
            row[f"cum_{k}"] = round(cum[k], 2)
        out["by_week"].append(row)
    return out


def main():
    done = grade()
    m = metrics(done)
    config.ensure_dirs()
    config.METRICS.write_text(json.dumps(m, indent=2, default=str))
    log.info("spreads   %s", m["spreads"].get("all_games"))
    log.info("totals    %s", m["totals"].get("all_games"))
    log.info("moneyline %s", m["moneyline"].get("all_games"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
