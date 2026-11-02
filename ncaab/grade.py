"""Grade the board and write metrics.json.

    python -m ncaab.grade

Every game on the board is graded on both markets: the side the model took, against the line it
was published at, by ESPN's event id - so a final can never be matched to the wrong game.

Tracked, for spreads and totals separately:
    W-L-P, season and last seven days     every pick, since every game gets one; nothing staked
    hit rate by size of disagreement      does a bigger gap mean a likelier side?
    closing-line value                    did the line move toward the number we took?
    model error                           the raw projection against the final

A game ESPN never finalises (postponed, cancelled) is closed out as ``void`` a week after its
date, so it stops being asked for and counts in no record.
"""
from __future__ import annotations

import json
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config
from .sources import espn, lines

log = logging.getLogger("ncaab.grade")

VOID_AFTER_DAYS = 7
GAP_BUCKETS = {"spread": [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)],
               "total": [(0, 1.5), (1.5, 3), (3, 4.5), (4.5, 7), (7, 99)]}


def _read(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"game_id": str, "home_id": str, "away_id": str})
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def results_for(m: pd.DataFrame) -> pd.DataFrame:
    """Win/loss/push on both markets for rows that carry a final. Pure."""
    m = m.copy()
    tp, tl = m["total_points"], m["total_line"]
    over = m["total_pick"] == "Over"
    m["total_result"] = np.select(
        [tl.isna(), tp == tl, over & (tp > tl), ~over & (tp < tl)],
        ["none", "push", "win", "win"], default="loss")
    # home covers when home_margin + spread_home > 0 (spread_home -6.5: home must win by 7)
    cover = m["home_margin"] + m["spread_home"]
    took_home = m["spread_side"].astype(str) == m["home_team"].astype(str)
    m["spread_result"] = np.select(
        [m["spread_home"].isna(), cover == 0, took_home & (cover > 0), ~took_home & (cover < 0)],
        ["none", "push", "win", "win"], default="loss")
    m["total_abs_err"] = (m["total_raw"] - m["total_points"]).abs().round(2)
    m["margin_abs_err"] = (m["margin_raw"] - m["home_margin"]).abs().round(2)
    # CLV from the number first published. spread_home is negative for a home favourite, so
    # the market moving toward the home side makes it MORE negative: first - close for a home
    # pick, close - first for an away pick. Totals: close - first for an over.
    fs = m.get("first_seen_spread", m["spread_home"]).fillna(m["spread_home"])
    ft = m.get("first_seen_total", m["total_line"]).fillna(m["total_line"])
    m["spread_clv"] = np.where(took_home, fs - m["close_spread"], m["close_spread"] - fs)
    m["total_clv"] = np.where(over, m["close_total"] - ft, ft - m["close_total"])
    m.loc[m["spread_result"] == "none", "spread_clv"] = np.nan
    m.loc[m["total_result"] == "none", "total_clv"] = np.nan
    return m


def grade(fetch: bool = True, fetch_day=espn.fetch_day, get_odds=None) -> pd.DataFrame:
    picks, done = _read(config.PICKS), _read(config.RESULTS)
    if picks.empty:
        log.warning("no picks yet")
        return done
    picks = picks.drop_duplicates("game_id", keep="last")
    graded = set(done["game_id"]) if len(done) else set()
    today = config.today_et()
    pending = picks[~picks["game_id"].isin(graded) & (picks["date"] < today)]
    if pending.empty:
        log.info("nothing new to grade")
        return done

    games = espn.update(start=min(pending["date"]) - timedelta(days=1), recheck_days=3,
                        fetch=fetch_day) if fetch else espn.load()
    if fetch:
        # the closing line for CLV: ESPN's own, asked for the handful of games just finished
        ask = games[games["game_id"].isin(pending["game_id"])]
        if len(ask):
            lines.backfill(budget_min=5, only=ask,
                           **({"get": get_odds} if get_odds is not None else {}))
    # closes come from lines.csv only - attach's fallback to our own pick would make CLV zero
    games = lines.attach(games, picks=pd.DataFrame())
    fin = games[["game_id", "home_points", "away_points", "total_points", "home_margin",
                 "spread_home", "total_line"]].rename(columns={"spread_home": "close_spread",
                                                               "total_line": "close_total"})
    m = pending.merge(fin, on="game_id", how="left")
    have = m["total_points"].notna()
    stale = ~have & (m["date"] < today - timedelta(days=VOID_AFTER_DAYS))
    rows = []
    if have.any():
        g = m[have].copy()
        # no close on the final scoreboard: the last number we published is the best we have
        g["close_spread"] = g["close_spread"].fillna(g["spread_home"])
        g["close_total"] = g["close_total"].fillna(g["total_line"])
        rows.append(results_for(g))
    if stale.any():
        v = m[stale].copy()
        v["total_result"] = v["spread_result"] = "void"
        rows.append(v)
    if not rows:
        log.warning("%d pending games, none final yet", len(pending))
        return done
    new = pd.concat(rows, ignore_index=True)
    new["graded_at"] = str(today)
    done = pd.concat([done, new], ignore_index=True) if len(done) else new
    config.ensure_dirs()
    done.to_csv(config.RESULTS, index=False)
    log.info("graded %d games (spreads %s / totals %s)", len(new),
             new["spread_result"].value_counts().to_dict(),
             new["total_result"].value_counts().to_dict())
    return done


# ---------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------
def wlp(results: pd.Series) -> dict:
    w, l, p = (int((results == k).sum()) for k in ("win", "loss", "push"))
    return {"n": w + l + p, "wins": w, "losses": l, "pushes": p,
            "win_pct": round(100 * w / (w + l), 1) if w + l else None}


def _clv(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return round(float(s.mean()), 2) if len(s) else None


def _block(df: pd.DataFrame, mk: str, today) -> dict:
    res, gap = f"{mk}_result", f"{'margin' if mk == 'spread' else 'total'}_disagree"
    rows = df[df[res].isin(["win", "loss", "push"])]
    allg = {**wlp(rows[res]), "clv": _clv(rows.get(f"{mk}_clv", pd.Series(dtype=float)))}
    last7 = rows[rows["date"] >= today - timedelta(days=7)]
    err = rows[f"{'margin' if mk == 'spread' else 'total'}_abs_err"]
    buckets = []
    g = pd.to_numeric(rows.get(gap), errors="coerce").abs()
    for lo, hi in GAP_BUCKETS[mk]:
        b = rows[(g >= lo) & (g < hi)]
        if len(b):
            buckets.append({"gap": f"{lo:g}-{hi:g}" if hi < 99 else f"{lo:g}+", **wlp(b[res])})
    # "season" is the STAKED record the chooser shows units for; nothing is staked here, so it
    # is empty and the chooser shows the model's side record, labelled unstaked.
    return {"season": {"n": 0, "wins": 0, "losses": 0, "pushes": 0, "win_pct": None,
                       "units": 0.0, "roi": None, "clv": None},
            "all_games": allg, "last7": wlp(last7[res]),
            "mae": round(float(err.mean()), 2) if err.notna().any() else None,
            "by_gap": buckets}


def week_of(d) -> pd.Timestamp:
    """Monday of the game's week."""
    d = pd.Timestamp(d)
    return (d - pd.Timedelta(days=d.weekday())).normalize()


def metrics(done: pd.DataFrame, today=None) -> dict:
    today = today or config.today_et()
    out = {"updated": str(today), "sport": "ncaab", "break_even": config.BREAK_EVEN,
           "spreads": {}, "totals": {}, "by_week": []}
    if done.empty:
        return out
    season = config.season_of(today)
    cur = done[done["date"].map(config.season_of) == season]
    if cur.empty:
        cur = done[done["date"].map(config.season_of) == max(done["date"].map(config.season_of))]
    out["season"] = int(config.season_of(max(cur["date"])))
    out["spreads"] = _block(cur, "spread", today)
    out["totals"] = _block(cur, "total", today)
    wk = cur.assign(_wk=cur["date"].map(week_of))
    for start, g in sorted(wk.groupby("_wk"), key=lambda kv: kv[0], reverse=True):
        end = start + pd.Timedelta(days=6)
        out["by_week"].append({"week": start.date().isoformat(),
                               "label": f"{start:%b} {start.day} – {end:%b} {end.day}",
                               "spread": wlp(g["spread_result"]), "total": wlp(g["total_result"])})
    return out


def main():
    done = grade()
    m = metrics(done)
    config.ensure_dirs()
    config.METRICS.write_text(json.dumps(m, indent=2, default=str) + "\n")
    log.info("spreads %s | totals %s", (m.get("spreads") or {}).get("all_games"),
             (m.get("totals") or {}).get("all_games"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
