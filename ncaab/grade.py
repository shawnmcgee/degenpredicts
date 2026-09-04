"""Grade published picks and write metrics.json.

    python -m ncaab.grade

Tracks, separately for totals and spreads:
    W-L-P record, units won/lost at the actual prices, ROI
    accuracy by |edge| bucket  -> this is how you set the edge thresholds
    closing-line value          -> the least noisy evidence that the model has an edge
"""
from __future__ import annotations

import json
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config
from .sources import ncaa, odds

log = logging.getLogger("ncaab.grade")

BREAK_EVEN = 52.38


def _load(path, dates=("date",)):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in dates:
        if c in df:
            df[c] = pd.to_datetime(df[c]).dt.date
    return df


def _match_results(picks: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    res = results[["date", "home_team", "away_team", "total_points", "home_margin"]]
    frames = []
    for shift in (0, 1, -1):
        r = res.copy()
        r["date"] = r["date"] + timedelta(days=shift)
        frames.append(r)
    res = pd.concat(frames).drop_duplicates(["date", "home_team", "away_team"])
    merged = picks.merge(res, on=["date", "home_team", "away_team"], how="left")
    miss = merged["total_points"].isna()
    if miss.any():  # try the swapped orientation (neutral-site listings disagree)
        swap = res.rename(columns={"home_team": "away_team", "away_team": "home_team"})
        swap["home_margin"] = -swap["home_margin"]
        alt = picks.loc[miss.values].merge(swap, on=["date", "home_team", "away_team"], how="left")
        merged.loc[miss, "total_points"] = alt["total_points"].values
        merged.loc[miss, "home_margin"] = alt["home_margin"].values
    return merged


def grade() -> pd.DataFrame:
    picks = _load(config.PICKS)
    if picks.empty:
        log.warning("no picks yet")
        return picks
    done = _load(config.RESULTS)
    graded = set(done["event_id"]) if len(done) else set()
    pending = picks[~picks["event_id"].isin(graded) & (picks["date"] < config.today_et())]
    if pending.empty:
        log.info("nothing new to grade")
        return done

    results = ncaa.update(start=pending["date"].min() - timedelta(days=2), recheck_days=5)
    m = _match_results(pending, results).dropna(subset=["total_points"])
    if m.empty:
        log.warning("%d pending picks, none matched to a final score", len(pending))
        return done

    # totals
    m["total_result"] = "push"
    m.loc[(m.total_pick == "Over") & (m.total_points > m.total_line), "total_result"] = "win"
    m.loc[(m.total_pick == "Over") & (m.total_points < m.total_line), "total_result"] = "loss"
    m.loc[(m.total_pick == "Under") & (m.total_points < m.total_line), "total_result"] = "win"
    m.loc[(m.total_pick == "Under") & (m.total_points > m.total_line), "total_result"] = "loss"

    # spreads: home covers when home_margin + spread_home > 0
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

    # closing-line value: did the line move our way after we picked?
    close = odds.closing_lines()
    if len(close):
        c = close[["event_id", "close_total", "close_spread_home"]]
        m = m.merge(c, on="event_id", how="left")
        m["total_clv"] = np.where(m.total_pick == "Over",
                                  m.close_total - m.total_line, m.total_line - m.close_total)
        m["spread_clv"] = np.where(m.margin_edge > 0,
                                   m.close_spread_home - m.spread_home,
                                   m.spread_home - m.close_spread_home)

    m["graded_at"] = str(config.today_et())
    done = pd.concat([done, m], ignore_index=True) if len(done) else m
    config.ensure_dirs()
    done.to_csv(config.RESULTS, index=False)
    log.info("graded %d (totals %s / spreads %s)", len(m),
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
    clv_col = f"{kind}_clv"
    return {"n": w + l + p, "wins": w, "losses": l, "pushes": p,
            "win_pct": round(100 * w / (w + l), 1) if w + l else 0.0,
            "units": round(units, 2), "roi": round(100 * units / staked, 1) if staked else 0.0,
            "clv": round(float(df[clv_col].mean()), 2) if clv_col in df and df[clv_col].notna().any() else None}


def metrics(done: pd.DataFrame) -> dict:
    today = config.today_et()
    out = {"updated": str(today), "break_even": BREAK_EVEN, "totals": {}, "spreads": {}}
    if done.empty:
        return out
    season = done[done["date"] >= config.season_start(config.season_of(today))]
    if season.empty:
        season = done[done["date"] >= config.season_start(config.season_of(done["date"].max()))]

    for kind, edge_col in (("total", "total_edge"), ("spread", "margin_edge")):
        key = "totals" if kind == "total" else "spreads"
        played = season[season[f"{'total' if kind == 'total' else 'spread'}_strength"].isin(["play", "bold"])]
        out[key] = {
            "season": _rec(played, kind),
            "all_games": _rec(season, kind),
            "last7": _rec(played[played["date"] >= today - timedelta(days=7)], kind),
            "last30": _rec(played[played["date"] >= today - timedelta(days=30)], kind),
            "mae": round(float(season[f"{'total' if kind == 'total' else 'margin'}_abs_err"].mean()), 2),
            "by_edge": [],
        }
        for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 99)]:
            b = season[(season[edge_col].abs() >= lo) & (season[edge_col].abs() < hi)]
            if len(b):
                out[key]["by_edge"].append({"bucket": f"{lo}-{hi if hi < 99 else '+'}", **_rec(b, kind)})

    daily = (season.groupby("date")
             .apply(lambda d: pd.Series({"total_units": d["total_units"].sum(),
                                         "spread_units": d["spread_units"].sum()}))
             .reset_index())
    daily["date"] = daily["date"].astype(str)
    daily["cum_total"] = daily["total_units"].cumsum().round(2)
    daily["cum_spread"] = daily["spread_units"].cumsum().round(2)
    out["daily"] = daily.to_dict("records")
    return out


def main():
    done = grade()
    m = metrics(done)
    config.ensure_dirs()
    config.METRICS.write_text(json.dumps(m, indent=2, default=str))
    log.info("totals season %s", m["totals"].get("season"))
    log.info("spreads season %s", m["spreads"].get("season"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
