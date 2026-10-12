"""Tests of the market itself - where a totals edge would have to come from if not from a model.

    python -m poc.nhl_totals.market_checks        # writes results/market_checks.json

  juiced_side      how often the side the price already favours wins: the hit rate a "model" gets
                   for free by reading the juice, and why a raw hit rate is not an edge
  calibration      the devigged price against results, by price bucket and by line
  shrink           does the market spread its totals too wide (bet toward the league mean)?
  cross_market     does the puck-line price know something about goals the total does not?
  forecast_power   how well anything - the line, the market, the model - predicts total goals
  feature_scan     every feature against (result - price) on priced games, and against the
                   market-structure baseline on closing lines; with ~40 features scanned, a lone
                   |z| of 2-2.5 is what chance alone produces
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .backtest import OUT, load


def _units(d, over: np.ndarray) -> np.ndarray:
    dec = np.where(over, d["over_dec"], d["under_dec"])
    win = np.where(over, d["y"] == 1, d["y"] == 0)
    return np.where(d["y"].isna(), 0.0, np.where(win, dec - 1, -1.0))


def _roi(u):
    return {"bets": int(len(u)), "roi_pct": round(100 * float(np.mean(u)), 2),
            "se_pct": round(100 * float(np.std(u) / math.sqrt(len(u))), 2)}


def _p_over_nb(mu, line, k=30.0):
    n, q, fl = k, k / (k + mu), np.floor(line)
    cdf, pmf = stats.nbinom.cdf(fl, n, q), stats.nbinom.pmf(fl, n, q)
    po, pu = 1 - cdf, np.where(np.isclose(line, fl), cdf - pmf, cdf)
    return po / (po + pu)


def main():
    X = load()
    p = X[X["priced"]].copy()
    d = p[p["y"].notna()]
    res = {}

    fav_over = d["p_over_mkt"] > 0.5
    res["juiced_side"] = []
    for t in (0.0, 0.01, 0.02, 0.03, 0.04):
        m = (d["p_over_mkt"] - 0.5).abs() >= t
        hit = (fav_over[m] == (d["y"][m] == 1)).mean()
        fair = np.where(fav_over[m], d["p_over_mkt"][m], 1 - d["p_over_mkt"][m]).mean()
        res["juiced_side"].append({"min_gap": t, "n": int(m.sum()), "hit_pct": round(100 * hit, 2),
                                   "fair_pct": round(100 * fair, 2),
                                   **_roi(_units(d[m], fav_over[m].values))})

    p["bucket"] = pd.cut(p["p_over_mkt"], [0, 0.47, 0.49, 0.50, 0.51, 0.53, 1])
    res["calibration_by_price"] = [
        {"bucket": str(b), "n": int(len(g)), "fair_over_pct": round(100 * g["p_over_mkt"].mean(), 1),
         "actual_over_pct": round(100 * g["y"].mean(), 1),
         "over": _roi(_units(g, np.ones(len(g), bool))), "under": _roi(_units(g, np.zeros(len(g), bool)))}
        for b, g in p.groupby("bucket", observed=True)]
    res["calibration_by_line"] = [
        {"line": float(L), "n": int(len(g)), "fair_over_pct": round(100 * g["p_over_mkt"].mean(), 1),
         "actual_over_pct": round(100 * g["y"].mean(), 1),
         "over": _roi(_units(g, np.ones(len(g), bool))), "under": _roi(_units(g, np.zeros(len(g), bool)))}
        for L, g in p.groupby("line") if len(g) >= 30]

    # market-implied mean total: the negative binomial mean whose P(over) is the fair price
    p["mu_mkt"] = [optimize.brentq(lambda m: _p_over_nb(m, L) - q, 3, 10)
                   for L, q in zip(p["line"], p["p_over_mkt"])]
    p["dev"] = p["mu_mkt"] - p.groupby("season")["mu_mkt"].transform("mean")
    p["q"] = pd.qcut(p["dev"], 5, labels=False)
    res["shrink"] = {
        "slope_total_on_market_mean": round(float(np.polyfit(p["mu_mkt"], p["total"], 1)[0]), 3),
        "by_quintile": [{"quintile": int(q), "n": int(len(g)), "market_mean": round(g["mu_mkt"].mean(), 2),
                         "actual_mean": round(g["total"].mean(), 2),
                         "under": _roi(_units(g, np.zeros(len(g), bool))),
                         "over": _roi(_units(g, np.ones(len(g), bool)))} for q, g in p.groupby("q")],
        "top_quintile_under_roi_by_season": {
            int(s): round(100 * float(_units(g[g["q"] == 4], np.zeros((g["q"] == 4).sum(), bool)).mean()), 1)
            for s, g in p.groupby("season")}}

    lines = pd.read_csv(OUT.parents[2] / "data/nhl/lines.csv")[["game_id", "pl_home", "p_pl_home", "p_home"]]
    c = p.merge(lines, on="game_id")
    c["p_fav"] = np.where(c["p_home"] >= 0.5, c["p_home"], 1 - c["p_home"])
    c["p_fav_pl"] = np.where((c["p_home"] >= 0.5) & (c["pl_home"] == -1.5), c["p_pl_home"],
                             np.where((c["p_home"] < 0.5) & (c["pl_home"] == 1.5), 1 - c["p_pl_home"], np.nan))
    c = c[c["p_fav_pl"].notna()]
    A = np.c_[np.ones(len(c)), c["p_fav"], c["p_fav"] ** 2, c["p_over_mkt"], c["line"], c["p_fav"] * c["p_over_mkt"]]
    b, *_ = np.linalg.lstsq(A, c["p_fav_pl"], rcond=None)
    c["pl_resid"] = c["p_fav_pl"] - A @ b
    cd = c[c["y"].notna()]
    r = cd["pl_resid"].corr(cd["y"] - cd["p_over_mkt"])
    res["cross_market"] = {"n": int(len(cd)), "corr_pl_residual_vs_total_result": round(float(r), 4),
                           "z": round(float(r * math.sqrt(len(cd))), 2)}

    res["forecast_power"] = {k: round(float(np.corrcoef(p[k], p["total"])[0, 1]), 4)
                             for k in ("line", "mu_mkt", "exp_tot_proj", "exp_tot_act")}
    A = np.c_[np.ones(len(p)), p["mu_mkt"], p["exp_tot_proj"] - p["mu_mkt"]]
    b, *_ = np.linalg.lstsq(A, p["total"], rcond=None)
    e = p["total"] - A @ b
    se = np.sqrt(np.diag(np.var(e) * np.linalg.inv(A.T @ A)))
    res["forecast_power"]["model_beyond_market_coef"] = round(float(b[2]), 3)
    res["forecast_power"]["model_beyond_market_se"] = round(float(se[2]), 3)

    feats = [c for c in X.columns if X[c].dtype.kind in "fi" and c not in
             ("season", "total", "y", "game_id", "over_dec", "under_dec", "p_over_mkt")
             and not c.startswith("line_")]
    pr = d.assign(r=d["y"] - d["p_over_mkt"])
    scan = pd.DataFrame({"corr": pr[feats].corrwith(pr["r"])})
    scan["z"] = scan["corr"] * math.sqrt(len(pr))
    pc = pd.read_csv(OUT / "preds_close.csv.gz")[["game_id", "p_base"]]
    cl = X.merge(pc, on="game_id")
    cl = cl[cl["y"].notna()].assign(r=lambda x: x["y"] - x["p_base"])
    scan["corr_close"] = cl[feats].corrwith(cl["r"])
    scan["z_close"] = scan["corr_close"] * math.sqrt(len(cl))
    scan = scan.dropna().sort_values("z", key=abs, ascending=False).round(4)
    res["feature_scan"] = {"n_priced": int(len(pr)), "n_close": int(len(cl)),
                           "rows": scan.reset_index(names="feature").to_dict("records")}

    (OUT / "market_checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "feature_scan"}, indent=1)[:6000])
    print(scan.head(12).to_string())


if __name__ == "__main__":
    main()
