"""Walk-forward backtest of every model family on both evaluation tracks.

    python -m poc.nhl_totals.backtest

For each test season the models are fitted on every earlier season only (recent seasons weighted
up, half-life four seasons) and then score the whole test season blind. Nothing about a test
season - not its games, not its lines, not a tuning choice - reaches the model that scores it.

Two tracks, because the two archives of lines know different things:

  close  2012-13 -> 2022-23, SportsbookReviewsOnline CLOSING totals (no prices). A closing line
         has seen the confirmed starters and lineups, so the models may too ("CLOSE" features).
         Only a hit rate can be measured - there is no price to say what that hit rate was worth.
  noon   2023-24 -> 2025-26 (to Jan 9), The Odds API NOON prices, plus this season's pre-game
         snapshots. At noon most starters and lineups are not confirmed, so the honest models see
         only the usual starter and the usual lineup ("NOON" features). Here the price exists, so
         the model is scored against the market's own probability and in units won.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import cache_dir
from .models import CLOSE, NOON, RESID, Juice, Resid, _weights, engineer, fit_predict, logit

OUT = Path(__file__).resolve().parent / "results"
CLOSE_TEST = list(range(2012, 2023))
NOON_TEST = [2023, 2024, 2025, 2026]


def load() -> pd.DataFrame:
    df = pd.read_parquet(cache_dir() / "built" / "features.parquet")
    df = df[df["completed"] & df["total_line"].notna() & (df["season"] >= 2011)].copy()
    df = df.reset_index(drop=True)
    X = engineer(df)
    X["season"] = df["season"].values
    X["total"] = df["total"].values
    X["y"] = np.where(df["total"] > df["total_line"], 1.0, np.where(df["total"] < df["total_line"], 0.0, np.nan))
    for c in ("game_id", "date", "home_team", "away_team", "over_dec", "under_dec", "p_over_mkt", "line_source"):
        X[c] = df[c].values
    X["priced"] = X["p_over_mkt"].notna()
    return X


def walk_forward(X: pd.DataFrame, seasons, cols, tag) -> pd.DataFrame:
    parts = []
    for s in seasons:
        train, test = X[X["season"] < s], X[X["season"] == s]
        if test.empty:
            continue
        preds = fit_predict(train, test, cols)
        t = test[["game_id", "season", "date", "line", "total", "y", "over_dec", "under_dec", "p_over_mkt",
                  "priced"]].copy()
        for k, v in preds.items():
            t[f"p_{k}"] = v
        parts.append(t)
        print(f"  [{tag}] {s}: {len(test)} games")
    return pd.concat(parts, ignore_index=True)


# --- scoring -------------------------------------------------------------------------------

def hit_table(p: pd.DataFrame, col: str, thresholds=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08)):
    """Pick the side the model favours; hit rate on decided games, for rising confidence."""
    d = p[p["y"].notna()]
    rows = []
    for t in thresholds:
        s = d[(d[col] - 0.5).abs() >= t]
        if len(s) == 0:
            rows.append({"min_edge": t, "n": 0})
            continue
        hit = ((s[col] > 0.5) == (s["y"] == 1)).astype(float)
        n, h = len(s), hit.mean()
        se = math.sqrt(0.25 / n)
        rows.append({"min_edge": t, "n": n, "share_of_games": round(n / len(d), 3), "hit_pct": round(100 * h, 2),
                     "se_pct": round(100 * se, 2), "z_vs_50": round((h - 0.5) / se, 2)})
    return rows


def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def price_table(p: pd.DataFrame, col: str, evs=(0.0, 0.02, 0.04, 0.06)):
    """Priced games: bet the side whose EV at the posted price clears the bar; units and ROI."""
    d = p[p["priced"]].copy()
    q = d[col].values
    ev_o = q * d["over_dec"] - 1
    ev_u = (1 - q) * d["under_dec"] - 1
    side_over = ev_o >= ev_u
    ev = np.where(side_over, ev_o, ev_u)
    dec = np.where(side_over, d["over_dec"], d["under_dec"])
    won = np.where(d["y"].isna(), np.nan, np.where(side_over, d["y"] == 1, d["y"] == 0))
    units = np.where(np.isnan(won), 0.0, np.where(won == 1, dec - 1, -1.0))
    implied = np.where(side_over, d["p_over_mkt"], 1 - d["p_over_mkt"])
    rows = []
    for t in evs:
        m = ev >= t
        dec_m = m & ~np.isnan(won)
        n = int(m.sum())
        if n == 0:
            rows.append({"min_ev": t, "bets": 0})
            continue
        roi = units[m].sum() / n
        sd = np.std(units[m]) / math.sqrt(n)
        rows.append({"min_ev": t, "bets": n, "hit_pct": round(100 * np.nanmean(won[dec_m]), 2),
                     "breakeven_pct": round(100 * np.mean(1 / dec[dec_m]), 2),
                     "market_fair_pct": round(100 * np.mean(implied[dec_m]), 2),
                     "units": round(float(units[m].sum()), 2), "roi_pct": round(100 * roi, 2),
                     "roi_se_pct": round(100 * sd, 2)})
    return rows


def blend_with_market(p: pd.DataFrame, col: str, beta: float = 1.0) -> np.ndarray:
    """Market's fair P(over), moved by what the model knows beyond the line's structure."""
    return 1 / (1 + np.exp(-(logit(p["p_over_mkt"]) + beta * (logit(p[col]) - logit(p["p_base"])))))


def summarize(p: pd.DataFrame, models, priced: bool) -> dict:
    d = p[p["y"].notna()]
    res = {"games": int(len(p)), "decided": int(len(d)), "over_rate_pct": round(100 * d["y"].mean(), 2),
           "models": {}}
    for m in models:
        col = f"p_{m}"
        r = {"log_loss": round(log_loss(d["y"].values, d[col].values), 5),
             "hit": hit_table(p, col),
             "by_season_hit_pct": {int(s): round(100 * ((g[col] > 0.5) == (g["y"] == 1)).mean(), 1)
                                   for s, g in d.groupby("season")}}
        if priced:
            r["vs_market_log_loss"] = round(log_loss(d["y"].values, d["p_over_mkt"].values), 5)
            p[f"pm_{m}"] = blend_with_market(p, col)
            dd = p[p["y"].notna()]
            r["blend_log_loss"] = round(log_loss(dd["y"].values, dd[f"pm_{m}"].values), 5)
            r["blend_hit"] = hit_table(p, f"pm_{m}")
            r["blend_price"] = price_table(p, f"pm_{m}")
        res["models"][m] = r
    return res


def anchored(X: pd.DataFrame) -> dict:
    """Market-anchored models: the price (real, or modelled where missing) plus what features add.

    The juice model is fitted on the priced games' prices only - no results - so using it in every
    fold leaks nothing. Two readings come out of this:
      * unpriced 2016-22: does the modelled price ALONE already predict the results as well as the
        feature models? If so, the features' apparent skill there was reading the juice;
      * priced 2023-26: with the real price as the anchor, do the features add anything?
    """
    J = Juice().fit(X)
    X = X.assign(off=J.logit_price(X))
    out = {}
    variants = {"resid_logit": dict(C=0.002, kind="logit"), "resid_gbm": dict(C=None, kind="gbm")}
    for era, seasons in (("unpriced_2016_22", range(2016, 2023)), ("priced_2023_26", NOON_TEST)):
        parts = []
        for s in seasons:
            tr = X[(X["season"] < s) & X["y"].notna()]
            te = X[X["season"] == s].copy()
            te["p_price"] = 1 / (1 + np.exp(-te["off"]))
            for name, kw in variants.items():
                m = Resid(RESID, **kw).fit(tr, tr["y"].values, _weights(tr["season"]).values, tr["off"].values)
                te[f"p_{name}"] = m.predict(te, te["off"].values)
            parts.append(te)
        p = pd.concat(parts, ignore_index=True)
        d = p[p["y"].notna()]
        r = {"decided": int(len(d)),
             "log_loss_price": round(log_loss(d["y"].values, d["p_price"].values), 5),
             "price_is": "modelled from features (no results used)" if era.startswith("un") else "real, devigged"}
        for name in variants:
            r[name] = {"log_loss": round(log_loss(d["y"].values, d[f"p_{name}"].values), 5),
                       "hit": hit_table(p, f"p_{name}")}
            if era.startswith("priced"):
                r[name]["price"] = price_table(p, f"p_{name}", evs=(0.0, 0.01, 0.02, 0.03))
        r["price_alone_hit"] = hit_table(p, "p_price")
        out[era] = r
    return out


def main():
    OUT.mkdir(exist_ok=True)
    X = load()
    models = ["base", "logit", "gbm", "pois", "ens"]
    results = {}

    print("close track (closing lines, real starters + lineups)")
    pc = walk_forward(X, CLOSE_TEST, CLOSE, "close")
    results["close"] = summarize(pc, models, priced=False)

    print("noon track (noon prices, projected starters, usual lineups)")
    pn = walk_forward(X, NOON_TEST, NOON, "noon")
    results["noon"] = summarize(pn, models, priced=True)

    print("noon track, upper bound (noon prices, but tonight's real starters + lineups)")
    pu = walk_forward(X, NOON_TEST, CLOSE, "noon+news")
    results["noon_with_news"] = summarize(pu, models, priced=True)

    print("market-anchored models (price + what the features add)")
    results["anchored"] = anchored(X)

    (OUT / "backtest.json").write_text(json.dumps(results, indent=1))
    pc.to_csv(OUT / "preds_close.csv.gz", index=False)
    pn.to_csv(OUT / "preds_noon.csv.gz", index=False)
    pu.to_csv(OUT / "preds_noon_news.csv.gz", index=False)
    report(results)


def report(results):
    for era, r in results.get("anchored", {}).items():
        print(f"\n=== anchored, {era}: {r['decided']} decided, price ({r['price_is']}) log loss {r['log_loss_price']}")
        for name in ("resid_logit", "resid_gbm"):
            v = r[name]
            h = {x["min_edge"]: x for x in v["hit"]}
            print(f"  {name}: log loss {v['log_loss']}  hit all {h[0.0]['hit_pct']}%/{h[0.0]['n']}")
            for b in v.get("price", []):
                if b.get("bets"):
                    print(f"         EV>={b['min_ev']:.2f}: {b['bets']} bets hit {b['hit_pct']}% "
                          f"(break-even {b['breakeven_pct']}%) ROI {b['roi_pct']}% +/- {b['roi_se_pct']}")
    for track, r in results.items():
        if track == "anchored":
            continue
        print(f"\n=== {track}: {r['decided']} decided games, overs {r['over_rate_pct']}%")
        for m, v in r["models"].items():
            h = {x["min_edge"]: x for x in v["hit"]}
            line = f"  {m:6s} logloss {v['log_loss']:.5f}"
            if "vs_market_log_loss" in v:
                line += f" (market {v['vs_market_log_loss']:.5f}, blend {v['blend_log_loss']:.5f})"
            line += "  hit all " + f"{h[0.0]['hit_pct']}%/{h[0.0]['n']}"
            for t in (0.02, 0.04, 0.06):
                if h[t].get("n"):
                    line += f" | edge>={t:.2f}: {h[t]['hit_pct']}%/{h[t]['n']}"
            print(line)
            if "blend_price" in v:
                for b in v["blend_price"]:
                    if b.get("bets"):
                        print(f"         EV>={b['min_ev']:.2f}: {b['bets']} bets hit {b['hit_pct']}% "
                              f"(break-even {b['breakeven_pct']}%) ROI {b['roi_pct']}% +/- {b['roi_se_pct']}")


if __name__ == "__main__":
    main()
