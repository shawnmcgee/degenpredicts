"""Grade published picks and write metrics.json.

    python -m cfb.grade

Grading joins on CFBD's ``game_id``, so there's no team-name matching to go wrong - a big
simplification over the old pipeline, which merged on (date, home, away) strings.
"""
from __future__ import annotations

import json
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config
from .sources import cfbd, odds

log = logging.getLogger("cfb.grade")
BREAK_EVEN = config.BREAK_EVEN  # venue-aware; see config.break_even_pct()


def _load(path, dates=("date",)):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"game_id": str})
    for c in dates:
        if c in df:
            df[c] = pd.to_datetime(df[c]).dt.date
    return df


def grade() -> pd.DataFrame:
    picks = _load(config.PICKS)
    if picks.empty:
        log.warning("no picks yet")
        return picks
    done = _load(config.RESULTS)
    already = set(done["game_id"]) if len(done) else set()
    pending = picks[~picks["game_id"].isin(already)]
    if pending.empty:
        log.info("nothing new to grade")
        return done

    games = cfbd.update_games()
    finals = games[games["completed"]][["game_id", "home_points", "away_points",
                                        "total_points", "home_margin"]]
    m = pending.merge(finals, on="game_id", how="inner")
    if m.empty:
        log.info("%d picks still awaiting final scores", len(pending))
        return done

    m["total_result"] = "push"
    m.loc[(m.total_pick == "Over") & (m.total_points > m.total_line), "total_result"] = "win"
    m.loc[(m.total_pick == "Over") & (m.total_points < m.total_line), "total_result"] = "loss"
    m.loc[(m.total_pick == "Under") & (m.total_points < m.total_line), "total_result"] = "win"
    m.loc[(m.total_pick == "Under") & (m.total_points > m.total_line), "total_result"] = "loss"

    # home covers when home_margin + spread_home > 0
    cover = m["home_margin"] + m["spread_home"]
    took_home = m["margin_edge"] > 0
    m["spread_result"] = "push"
    m.loc[(took_home & (cover > 0)) | (~took_home & (cover < 0)), "spread_result"] = "win"
    m.loc[(took_home & (cover < 0)) | (~took_home & (cover > 0)), "spread_result"] = "loss"

    for k in ("total", "spread"):
        payout = m[f"{k}_payout"].fillna(100 / 110)
        stake = m[f"{k}_stake"].fillna(0)
        m[f"{k}_units"] = np.select(
            [m[f"{k}_result"] == "win", m[f"{k}_result"] == "loss"],
            [stake * payout, -stake], default=0.0).round(3)
    m["total_abs_err"] = (m["total_pred"] - m["total_points"]).abs()
    m["margin_abs_err"] = (m["margin_pred"] - m["home_margin"]).abs()

    # closing-line value: our number vs the last line we saw before kickoff
    closing = cfbd.load_lines()[["game_id", "spread_home", "total_line"]].rename(
        columns={"spread_home": "close_spread", "total_line": "close_total"})
    m = m.merge(closing, on="game_id", how="left")
    m["total_clv"] = np.where(m.total_pick == "Over",
                              m.close_total - m.total_line, m.total_line - m.close_total)
    m["spread_clv"] = np.where(took_home, m.close_spread - m.spread_home,
                               m.spread_home - m.close_spread)
    m["graded_at"] = str(config.today_et())

    done = pd.concat([done, m], ignore_index=True) if len(done) else m
    config.ensure_dirs()
    done.to_csv(config.RESULTS, index=False)
    log.info("graded %d | totals %s | spreads %s", len(m),
             m.total_result.value_counts().to_dict(), m.spread_result.value_counts().to_dict())
    return done


def _rec(df, kind) -> dict:
    if df.empty:
        return {"n": 0, "wins": 0, "losses": 0, "pushes": 0, "win_pct": 0.0,
                "units": 0.0, "roi": 0.0, "clv": None}
    r = df[f"{kind}_result"]
    w, l, p = int((r == "win").sum()), int((r == "loss").sum()), int((r == "push").sum())
    staked = float(df[f"{kind}_stake"].fillna(0).sum())
    units = float(df[f"{kind}_units"].fillna(0).sum())
    clv = f"{kind}_clv"
    return {"n": w + l + p, "wins": w, "losses": l, "pushes": p,
            "win_pct": round(100 * w / (w + l), 1) if w + l else 0.0,
            "units": round(units, 2), "roi": round(100 * units / staked, 1) if staked else 0.0,
            "clv": round(float(df[clv].mean()), 2) if clv in df and df[clv].notna().any() else None}


def metrics(done: pd.DataFrame) -> dict:
    today = config.today_et()
    out = {"updated": str(today), "break_even": round(BREAK_EVEN, 2), "venue": config.VENUE,
           "totals": {}, "spreads": {}, "by_week": []}
    if done.empty:
        return out
    season = done[done["season"] == config.season_of(today)]
    if season.empty:
        season = done[done["season"] == done["season"].max()]

    for kind, edge_col, err_col, strength_col in (
            ("total", "total_disagree", "total_abs_err", "total_strength"),
            ("spread", "margin_disagree", "margin_abs_err", "spread_strength")):
        key = "totals" if kind == "total" else "spreads"
        played = season[season[strength_col].isin(["play", "bold"])]
        out[key] = {
            "season": _rec(played, kind),
            "all_games": _rec(season, kind),
            "last2weeks": _rec(played[played["week"] >= played["week"].max() - 1], kind)
            if len(played) else _rec(played, kind),
            "mae": round(float(season[err_col].mean()), 2) if len(season) else None,
            "by_edge": [],
        }
        for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 999)]:
            b = season[(season[edge_col].abs() >= lo) & (season[edge_col].abs() < hi)]
            if len(b):
                out[key]["by_edge"].append({"bucket": f"{lo}-{hi if hi < 999 else '+'}", **_rec(b, kind)})

    for wk, grp in season.groupby("week"):
        played = grp[grp["total_strength"].isin(["play", "bold"]) |
                     grp["spread_strength"].isin(["play", "bold"])]
        out["by_week"].append({
            "week": int(wk), "games": int(len(grp)),
            "total_units": round(float(grp["total_units"].sum()), 2),
            "spread_units": round(float(grp["spread_units"].sum()), 2),
            "played": int(len(played)),
        })
    cum_t = cum_s = 0.0
    for row in out["by_week"]:
        cum_t += row["total_units"]; cum_s += row["spread_units"]
        row["cum_total"] = round(cum_t, 2); row["cum_spread"] = round(cum_s, 2)
    return out


def main():
    done = grade()
    m = metrics(done)
    config.ensure_dirs()
    config.METRICS.write_text(json.dumps(m, indent=2, default=str))
    log.info("totals %s", m["totals"].get("season"))
    log.info("spreads %s", m["spreads"].get("season"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
