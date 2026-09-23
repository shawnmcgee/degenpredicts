"""Grade published NHL picks and write metrics.json.

    python -m nhl.grade

Grading joins on the NHL's own game id, so there is no team-name matching to go wrong, and it
settles exactly the way a book does:

* **Puck line** - the side's margin plus its line, overtime and shootout included; the shootout
  winner is credited one goal, as the NHL publishes it and as books settle it. A +/-1.5 line
  cannot push; an alternate whole-number line can, and is handled.
* **Total** - final goals against the line, shootout goal included; a whole-number line pushes.

Units are settled at the price the pick was published at. Closing-line value is measured from
the number we FIRST published to the last snapshot before puck drop, in the market's implied
goals - supremacy for the puck line, total goals for the total - signed so that positive means
the market moved toward us. CLV says something within weeks; a win rate needs seasons.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import config
from .sources import nhle, odds

log = logging.getLogger("nhl.grade")
EDGE_BUCKETS = [(-1.0, 0.0), (0.0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 1.0)]


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

    games = nhle.update_games()
    finals = games[games["completed"].astype(bool)][
        ["game_id", "home_goals", "away_goals", "decided_in"]].copy()
    finals["game_id"] = finals["game_id"].astype(str)
    m = pending.merge(finals, on="game_id", how="inner")
    if m.empty:
        log.info("%d picks still awaiting final scores", len(pending))
        return done
    m = m.copy()
    m["home_margin"] = m["home_goals"] - m["away_goals"]
    m["total_goals"] = m["home_goals"] + m["away_goals"]

    home = m["spread_side"] == "home"
    side_margin = np.where(home, m["home_margin"], -m["home_margin"])
    m["spread_result"] = settle(side_margin, m["spread_line"])
    over = m["total_side"] == "over"
    tot_margin = np.where(over, m["total_goals"] - m["total_line_used"],
                          m["total_line_used"] - m["total_goals"])
    m["total_result"] = settle(tot_margin, 0.0)
    m["spread_units"] = _units(m["spread_result"], m["spread_stake"], m["spread_dec"])
    m["total_units"] = _units(m["total_result"], m["total_stake"], m["total_dec"])
    m["total_abs_err"] = (m["exp_total"] - m["total_goals"]).abs()
    m["margin_abs_err"] = (m["exp_margin"] - m["home_margin"]).abs()

    # ---- closing-line value --------------------------------------------------------------
    close = odds.last_before_start(odds.load_snapshots())
    m["spread_clv"] = np.nan
    m["total_clv"] = np.nan
    if len(close) and "first_m_lh" in m:
        from .features import market_view
        from .scoreline import Table
        from .train import load_models
        try:
            theta = load_models()[1].get("theta")
        except (OSError, ValueError, KeyError):
            theta = None
        c = market_view(close, Table(theta))[["game_id", "m_lh", "m_la"]]
        c = c.rename(columns={"m_lh": "close_lh", "m_la": "close_la"})
        c["game_id"] = c["game_id"].astype(str)
        m = m.merge(c, on="game_id", how="left")
        first_sup = m["first_m_lh"] - m["first_m_la"]
        close_sup = m["close_lh"] - m["close_la"]
        # "+ 0.0" again: an unchanged line must read 0.0, not the -0.0 rounding noise leaves
        m["spread_clv"] = np.where(home, close_sup - first_sup,
                                   first_sup - close_sup).round(3) + 0.0
        first_tot = m["first_m_lh"] + m["first_m_la"]
        close_tot = m["close_lh"] + m["close_la"]
        m["total_clv"] = np.where(over, close_tot - first_tot,
                                  first_tot - close_tot).round(3) + 0.0
    m["graded_at"] = str(config.today_et())

    done = pd.concat([done, m], ignore_index=True) if len(done) else m
    config.ensure_dirs()
    done.to_csv(config.RESULTS, index=False)
    log.info("graded %d | puck line %s | totals %s", len(m),
             m["spread_result"].value_counts().to_dict(), m["total_result"].value_counts().to_dict())
    return done


def _rec(df: pd.DataFrame, kind: str) -> dict:
    if df.empty or f"{kind}_result" not in df:
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
            "clv": round(float(df[clv].mean()), 3) if clv in df and df[clv].notna().any() else None,
            # the rate the market's own prices expected for the same picks - the bar to beat,
            # because a +1.5 dog "winning" 65% of the time is what the price already said
            "expected_pct": round(100 * float(df[f"{kind}_mkt_p"].mean()), 1)
            if f"{kind}_mkt_p" in df and df[f"{kind}_mkt_p"].notna().any() else None}


def metrics(done: pd.DataFrame) -> dict:
    today = config.today_et()
    out = {"updated": str(today), "sport": "nhl", "break_even": config.BREAK_EVEN,
           "venue": config.VENUE, "spreads": {}, "totals": {}, "by_week": []}
    if done.empty:
        return out
    season = done[done["season"] == done["season"].max()]
    for key, kind, err in (("spreads", "spread", "margin_abs_err"), ("totals", "total", "total_abs_err")):
        played = season[season[f"{kind}_strength"].isin(["play", "bold"])]
        out[key] = {"season": _rec(played, kind), "all_games": _rec(season, kind),
                    "mae": round(float(season[err].mean()), 3) if err in season else None,
                    "by_edge": []}
        for lo, hi in EDGE_BUCKETS:
            evs = season[f"{kind}_ev"]
            b = season[(evs >= lo) & (evs < hi)]
            if len(b):
                out[key]["by_edge"].append({"bucket": f"{lo:+.2f}-{hi:+.2f}" if hi < 1 else f"{lo:+.2f}-+",
                                            "lo": lo, "hi": hi, **_rec(b, kind)})
    # ISO weeks, for the cumulative-units line; hockey has no rounds to number
    wk = pd.to_datetime(season["date"]).dt.strftime("%G-W%V")
    cum_t = cum_s = 0.0
    for w, grp in season.groupby(wk):
        cum_t += float(grp["total_units"].sum())
        cum_s += float(grp["spread_units"].sum())
        out["by_week"].append({"week": w, "games": int(len(grp)),
                               "spread_units": round(float(grp["spread_units"].sum()), 2),
                               "total_units": round(float(grp["total_units"].sum()), 2),
                               "cum_spread": round(cum_s, 2), "cum_total": round(cum_t, 2)})
    return out


def main():
    done = grade()
    m = metrics(done)
    config.ensure_dirs()
    config.METRICS.write_text(json.dumps(m, indent=2, default=str))
    log.info("puck line %s", m["spreads"].get("all_games"))
    log.info("totals    %s", m["totals"].get("all_games"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
