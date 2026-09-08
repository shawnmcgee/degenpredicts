"""Grade published EPL picks and write metrics.json.

    python -m epl.grade

Grading joins on ``game_id``, which is season-plus-clubs rather than anything date-based, so a
postponed and rearranged fixture still grades against the pick that was published for it. In a
league that rearranges as many matches as this one, a date-keyed id would leave picks stranded
in picks.csv forever.

Three results are graded per match, because three markets were published off one model:

* **Asian handicap** - with pushes, including the half-push a quarter line produces.
* **Over/under** - with a push when the goal line is a whole number and the match lands on it.
* **1X2** - win or lose, no push.

Closing-line value is measured against the closing price football-data records after the match.
For the Asian handicap that is a genuinely sharp instrument and it is the metric to watch: a
cover rate needs hundreds of matches before it says anything, CLV says something within weeks.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import config
from .sources import footballdata

log = logging.getLogger("epl.grade")
BREAK_EVEN = config.BREAK_EVEN


def _load(path, dates=("date",)):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"game_id": str}, low_memory=False)
    for c in dates:
        if c in df:
            df[c] = pd.to_datetime(df[c]).dt.date
    return df


def _ah_result(handicap, supremacy, took_home) -> tuple[str, float]:
    """Settle one side of an Asian handicap. Returns (result, fraction of stake won).

    Quarter lines settle as two half-stakes, so the honest outcomes include "half win" and
    "half loss" - a -0.75 backer whose team wins by exactly one goal wins half the stake and
    gets the other half back. Collapsing that to a plain win or loss, which is what a
    two-outcome grader would do, misstates the return on a line that is quoted on most matches
    in this league.
    """
    if handicap != handicap or supremacy != supremacy:
        return "", 0.0
    h = float(handicap) if took_home else -float(handicap)
    margin = float(supremacy) if took_home else -float(supremacy)
    adj = margin + h
    q = round(h * 4) / 4
    if abs(q * 2 - round(q * 2)) > 1e-9:          # quarter line: two half stakes
        legs = [margin + q - 0.25, margin + q + 0.25]
        won = sum(1 for a in legs if a > 0) / 2
        push = sum(1 for a in legs if a == 0) / 2
        lost = 1 - won - push
        if won == 1:
            return "win", 1.0
        if lost == 1:
            return "loss", -1.0
        if won and push:
            return "half win", 0.5
        if lost and push:
            return "half loss", -0.5
        return "push", 0.0
    if adj > 0:
        return "win", 1.0
    if adj < 0:
        return "loss", -1.0
    return "push", 0.0


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

    games = footballdata.update_games()
    finals = games[games["completed"].astype(bool)][
        ["game_id", "home_goals", "away_goals", "supremacy", "total_goals", "result"]]
    m = pending.merge(finals, on="game_id", how="inner", suffixes=("", "_final"))
    if m.empty:
        log.info("%d picks still awaiting final scores", len(pending))
        return done
    m = m.copy()

    # ---- Asian handicap ---------------------------------------------------------
    took_home = (m["sup_side_val"] if "sup_side_val" in m else m.get("sup_edge", 0)) > 0
    res, frac = [], []
    for hc, sup, th in zip(m.get("ah_home", pd.Series(np.nan, index=m.index)),
                           m["supremacy"], took_home):
        r, f = _ah_result(hc, sup, bool(th))
        res.append(r)
        frac.append(f)
    m["ah_result"] = res
    m["ah_frac"] = frac

    # ---- over / under -----------------------------------------------------------
    took_over = (m["total_side_val"] if "total_side_val" in m else m.get("total_edge", 0)) > 0
    line = m.get("total_line", pd.Series(np.nan, index=m.index))
    m["total_result"] = "push"
    m.loc[(took_over) & (m.total_goals > line), "total_result"] = "win"
    m.loc[(took_over) & (m.total_goals < line), "total_result"] = "loss"
    m.loc[(~took_over) & (m.total_goals < line), "total_result"] = "win"
    m.loc[(~took_over) & (m.total_goals > line), "total_result"] = "loss"
    m["total_frac"] = np.select([m.total_result == "win", m.total_result == "loss"],
                                [1.0, -1.0], default=0.0)

    # ---- 1X2 ---------------------------------------------------------------------
    actual = np.select([m.supremacy > 0, m.supremacy == 0], ["home", "draw"], default="away")
    m["x2_result"] = np.where(m.get("x2_side", "").astype(str) == actual, "win", "loss")
    m.loc[m.get("x2_side", pd.Series([None] * len(m))).isna(), "x2_result"] = ""
    m["x2_frac"] = np.select([m.x2_result == "win", m.x2_result == "loss"],
                             [1.0, -1.0], default=0.0)

    for k in ("ah", "total", "x2"):
        stake = m.get(f"{k}_stake", pd.Series(0.0, index=m.index)).fillna(0)
        price = m.get(f"{k}_price", pd.Series(np.nan, index=m.index))
        mult = (price - 1).fillna(config.SPORTSBOOK_DECIMAL - 1)
        f = m[f"{k}_frac"]
        # A half win returns half the stake at the price; a half loss gives half back.
        m[f"{k}_units"] = np.where(f > 0, stake * mult * f, stake * f).round(3)

    m["sup_abs_err"] = (m["sup_pred"] - m["supremacy"]).abs()
    m["total_abs_err"] = (m["total_pred"] - m["total_goals"]).abs()

    # ---- closing-line value ------------------------------------------------------
    closing = footballdata.load_lines()
    if len(closing):
        c = closing[["game_id", "ah_home", "mkt_sup", "mkt_total"]].rename(
            columns={"ah_home": "close_ah", "mkt_sup": "close_sup", "mkt_total": "close_total"})
        m = m.merge(c, on="game_id", how="left")
        # Positive CLV means the market moved toward the side we took after we took it.
        m["ah_clv"] = np.where(took_home, m.close_sup - m.mkt_sup, m.mkt_sup - m.close_sup)
        m["total_clv"] = np.where(took_over, m.close_total - m.mkt_total,
                                  m.mkt_total - m.close_total)
    m["graded_at"] = str(config.today_uk())

    done = pd.concat([done, m], ignore_index=True) if len(done) else m
    config.ensure_dirs()
    done.to_csv(config.RESULTS, index=False)
    log.info("graded %d | handicap %s | totals %s | 1X2 %s", len(m),
             m.ah_result.value_counts().to_dict(), m.total_result.value_counts().to_dict(),
             m.x2_result.value_counts().to_dict())
    return done


def _rec(df, kind) -> dict:
    if df.empty or f"{kind}_result" not in df:
        return {"n": 0, "wins": 0, "losses": 0, "pushes": 0, "win_pct": 0.0,
                "units": 0.0, "roi": 0.0, "clv": None}
    r = df[f"{kind}_result"]
    w = float((r == "win").sum() + 0.5 * (r == "half win").sum())
    l = float((r == "loss").sum() + 0.5 * (r == "half loss").sum())
    p = int((r == "push").sum())
    staked = float(df.get(f"{kind}_stake", pd.Series(dtype=float)).fillna(0).sum())
    units = float(df.get(f"{kind}_units", pd.Series(dtype=float)).fillna(0).sum())
    clv = f"{kind}_clv"
    return {"n": int((r != "").sum()), "wins": round(w, 1), "losses": round(l, 1), "pushes": p,
            "win_pct": round(100 * w / (w + l), 1) if w + l else 0.0,
            "units": round(units, 2), "roi": round(100 * units / staked, 1) if staked else 0.0,
            "clv": round(float(df[clv].mean()), 3)
            if clv in df and df[clv].notna().any() else None}


def metrics(done: pd.DataFrame) -> dict:
    today = config.today_uk()
    out = {"updated": str(today), "sport": "epl", "league": config.LEAGUE,
           "league_name": config.LEAGUE_NAME, "break_even": round(BREAK_EVEN, 2),
           "venue": config.VENUE, "spreads": {}, "totals": {}, "x2": {}, "by_week": []}
    if done.empty:
        return out
    season = done[done["season"] == config.season_of(today)]
    if season.empty:
        season = done[done["season"] == done["season"].max()]

    # `spreads` carries the handicap, under the shared name the landing page reads.
    for key, kind, edge_col, err_col, strength_col in (
            ("spreads", "ah", "sup_disagree", "sup_abs_err", "ah_strength"),
            ("totals", "total", "total_disagree", "total_abs_err", "total_strength"),
            ("x2", "x2", "sup_disagree", "sup_abs_err", "ah_strength")):
        played = season[season[strength_col].isin(["play", "bold"])] \
            if strength_col in season else season.iloc[0:0]
        out[key] = {
            "season": _rec(played, kind),
            # `all_games` is the one to read early. With thresholds deliberately set above
            # every disagreement bucket that cleared break-even, `season` will often be empty
            # by design - but every match is still predicted, graded and CLV-tracked, so the
            # evidence accumulates whether or not anything is staked.
            "all_games": _rec(season, kind),
            "mae": round(float(season[err_col].mean()), 3)
            if err_col in season and len(season) else None,
            "by_edge": [],
        }
        for lo, hi in [(0, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.75), (0.75, 999)]:
            if edge_col not in season:
                break
            b = season[(season[edge_col].abs() >= lo) & (season[edge_col].abs() < hi)]
            if len(b):
                out[key]["by_edge"].append(
                    {"bucket": f"{lo}-{hi if hi < 999 else '+'}", **_rec(b, kind)})

    if "matchweek" in season:
        for wk, grp in season.groupby("matchweek"):
            out["by_week"].append({
                "week": int(wk), "games": int(len(grp)),
                "ah_units": round(float(grp.get("ah_units", pd.Series(dtype=float)).sum()), 2),
                "total_units": round(float(grp.get("total_units", pd.Series(dtype=float)).sum()), 2),
                "x2_units": round(float(grp.get("x2_units", pd.Series(dtype=float)).sum()), 2),
            })
        cum = {"ah": 0.0, "total": 0.0, "x2": 0.0}
        for row in out["by_week"]:
            for k in cum:
                cum[k] += row[f"{k}_units"]
                row[f"cum_{k}"] = round(cum[k], 2)
    # The landing page reads spread_units/total_units; keep the shared spelling.
    for row in out["by_week"]:
        row["spread_units"] = row["ah_units"]
        row["cum_spread"] = row.get("cum_ah", 0.0)
    return out


def main():
    done = grade()
    m = metrics(done)
    config.ensure_dirs()
    config.METRICS.write_text(json.dumps(m, indent=2, default=str))
    log.info("handicap %s", m["spreads"].get("all_games"))
    log.info("totals    %s", m["totals"].get("all_games"))
    log.info("1X2       %s", m["x2"].get("all_games"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
