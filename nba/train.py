"""Train the NBA models and write the report that says whether to bet any of it.

    python -m nba.train                # refresh data, fit, evaluate, save
    python -m nba.train --no-fetch     # use the committed cache (offline / CI)

Four models, the NFL's four in shape - margin and total, each with and without the market:

    margin_nomarket / total_nomarket   ratings, availability, schedule and context only
    margin_market   / total_market     the closing line, plus how the no-market model and the
                                       situation say the line errs (a residual model)

All four are ridge regressions saved as JSON coefficients (:mod:`nba.linear`), weighted toward
recent seasons (half-life four seasons) because the sport keeps moving. The published number is
the line moved part of the way to the market-aware model, by a shrink fitted walk-forward.

**Everything is evaluated walk-forward** - train on every season before S, predict S - over the
last seven complete seasons, and the shrink applied to season S is fitted only on the seasons
before it, so no number in the report was fitted on the games it scores. The numbers that
decide whether to bet, in data/nba/models/meta.json:

    eval.margin_market.beats_market / ats_rate     against the CLOSING spread
    eval.margin_market.ats_by_disagreement         does it get better when it disagrees more?
    eval.*.significance                            what |z| a bucket needs given the looks taken
    eval.moneyline.log_loss_edge                   the proper scoring rule, against the close

**What the first full run found** (2019-20 to 2025-26, ~8,800 closing lines scored out of
sample, nothing tuned or fitted on the season it scores):

* the no-market model's margin error is 10.74 points against the closing spread's 10.49, and
  its total error 14.63 against the closing total's 14.33 - the gap is information the line
  has and a public box-score model does not;
* the market-aware models tie the close (10.494 against 10.494; 14.330 against 14.329), cover
  50.4% of spreads and 50.4% of totals, and no disagreement bucket or situation clears the
  52.4% break-even once corrected for the looks taken;
* on the moneyline the published probabilities tie the de-vigged close (log loss 0.6057
  against 0.6058). The one bucket that showed anything is 10%+ EV - underdogs the closing
  spread rated better than their moneyline did - at about +16% on ~770 bets, positive in six
  of seven seasons: 1.8 standard errors, suggestive and unproven, like the NHL's puck line;
* in the research, against OPENING numbers the model went 50.6% and did not anticipate which
  way the line would move.

That is the NFL and Premier League result again, and the page says so rather than burying it.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .features import BASE_FEATURES, MARKET_FEATURES, build
from .linear import Ridge, recency_weights
from .odds_math import ev as expected_value
from .odds_math import win_prob
from .sources import history, hoopr, odds

log = logging.getLogger("nba.train")

TARGETS = ("margin", "total")
HALF_LIFE = 4.0
ALPHA_NOMARKET = 10.0
# Heavily regularised: there are ~1,300 closing lines a season and the market is right about
# nearly all of them. A light ridge here learns noise and prices it as insight.
ALPHA_MARKET = 1000.0
WALK_FORWARD_SEASONS = config._env_int("DEGEN_WALK_SEASONS", 7)
MIN_TRAIN_GAMES = 3000
MIN_MARKET_TRAIN = 2000
MIN_TEST_GAMES = 400
MIN_SHRINK_GAMES = 1500     # fewer than this and the shrink stays at its default
MIN_BUCKET = 60
MIN_SEGMENT = 150
MIN_SEASON_ROWS = 30
FAMILY_ALPHA = 0.05
BUCKETS = {"margin": [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 99)],
           "total": [(0, 1), (1, 2), (2, 3), (3, 4.5), (4.5, 99)]}
VIEW_BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 99)]
EV_BUCKETS = [(-1.0, 0.0), (0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.0)]


# ---------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------
def assemble(fetch: bool = True):
    games = hoopr.update() if fetch else hoopr.load_games()
    if games.empty:
        raise SystemExit("no games cached - run `python -m nba.sources.hoopr` with network access")
    if fetch:
        odds.consolidate(games)
    return games, hoopr.load_players(), history.load_lines(), hoopr.load_names()


def complete_seasons(seasons) -> list[int]:
    """Seasons that have finished. The in-progress season must never be the holdout."""
    today = config.today_et()
    return sorted(int(s) for s in set(seasons) if config.season_end(int(s)) < today)


def outcomes(D: pd.DataFrame) -> pd.DataFrame:
    D = D.copy()
    D["margin"] = D["home_points"] - D["away_points"]
    D["total"] = D["home_points"] + D["away_points"]
    return D


# ---------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------
def fit_nomarket(tr: pd.DataFrame, target: str, season: int) -> Ridge:
    return Ridge(BASE_FEATURES, ALPHA_NOMARKET).fit(
        tr, target, recency_weights(tr["season"].values, season, HALF_LIFE))


def fit_market(tr: pd.DataFrame, target: str, season: int) -> Ridge:
    """A residual model: what the closing number gets wrong, not the score itself."""
    return Ridge(MARKET_FEATURES[target], ALPHA_MARKET).fit(
        tr, tr[target] - tr[f"mkt_{target}"],
        recency_weights(tr["season"].values, season, HALF_LIFE))


def nomarket_oos(D: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample no-market predictions for every season that has two before it.

    The market models take the no-market model's disagreement with the line as a feature, and
    that disagreement has to be out of sample on the rows they learn from too - an in-sample
    prediction is too good, and a residual model trained on it learns to trust it too much.
    """
    D = D.copy()
    for t in TARGETS:
        D[f"nm_{t}"] = np.nan
    seasons = sorted(int(s) for s in D["season"].unique())
    for s in seasons[2:]:
        tr, te = D[D["season"] < s], D["season"] == s
        if len(tr) < MIN_TRAIN_GAMES:
            continue
        for t in TARGETS:
            D.loc[te, f"nm_{t}"] = fit_nomarket(tr, t, s).predict(D[te])
    for t in TARGETS:
        D[f"dev_{t}"] = D[f"nm_{t}"] - D[f"mkt_{t}"]
    return D


def best_shrink(pred, line, actual) -> float:
    """How far from the closing line toward the model: minimises pooled absolute error."""
    pred, line, actual = (np.asarray(x, float) for x in (pred, line, actual))
    ok = np.isfinite(pred) & np.isfinite(line) & np.isfinite(actual)
    grid = np.round(np.arange(0, config.SHRINK_CAP + 1e-9, 0.05), 2)
    maes = [np.abs(line[ok] + w * (pred[ok] - line[ok]) - actual[ok]).mean() for w in grid]
    return float(grid[int(np.argmin(maes))])


def shrink_for(pool: pd.DataFrame, target: str) -> tuple[float, float, int]:
    """(shrink, raw fit, games) from a pool of out-of-sample market-model predictions."""
    p = pool[pool[f"mkt_{target}"].notna() & pool[f"raw_{target}"].notna()]
    if p.empty:
        return config.DEFAULT_SHRINK, config.DEFAULT_SHRINK, 0
    raw = best_shrink(p[f"raw_{target}"], p[f"mkt_{target}"], p[target])
    return (raw if len(p) >= MIN_SHRINK_GAMES else min(raw, config.DEFAULT_SHRINK)), raw, len(p)


# ---------------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------------
def walk_forward(D: pd.DataFrame, n_seasons: int = WALK_FORWARD_SEASONS) -> pd.DataFrame:
    """Out-of-sample predictions from all four models for the last n complete seasons."""
    comp = complete_seasons(D["season"].unique())
    closing = D[D["mkt_margin"].notna()]
    first_close = int(closing["season"].min()) if len(closing) else 9999
    tests = [s for s in comp[-n_seasons:] if s >= first_close + 2]
    chunks = []
    for s in tests:
        te = D[D["season"] == s].copy()
        if len(te) < MIN_TEST_GAMES:
            continue
        for t in TARGETS:
            te[f"raw_{t}"] = np.nan
            tr = D[(D["season"] < s) & D[f"mkt_{t}"].notna() & D[f"dev_{t}"].notna()]
            if len(tr) < MIN_MARKET_TRAIN:
                continue
            has = te[f"mkt_{t}"].notna() & te[f"dev_{t}"].notna()
            m = fit_market(tr, t, s)
            te.loc[has, f"raw_{t}"] = te.loc[has, f"mkt_{t}"] + m.predict(te[has])
        chunks.append(te)
        log.info("walk-forward %d: %d games, %d with a closing line", s, len(te),
                 int(te["mkt_margin"].notna().sum()))
    if not chunks:
        return pd.DataFrame()
    pool = pd.concat(chunks, ignore_index=True)
    # strict shrinks: the one applied to season S is fitted only on test seasons before it
    for t in TARGETS:
        pool[f"shrink_{t}"] = np.nan
        pool[f"pub_{t}"] = np.nan
        for s in sorted(pool["season"].unique()):
            prior = pool[pool["season"] < s]
            w = shrink_for(prior, t)[0] if len(prior) else config.DEFAULT_SHRINK
            sel = pool["season"] == s
            pool.loc[sel, f"shrink_{t}"] = w
            line, raw = pool.loc[sel, f"mkt_{t}"], pool.loc[sel, f"raw_{t}"]
            pool.loc[sel, f"pub_{t}"] = np.where(line.notna() & raw.notna(),
                                                 line + w * (raw - line), pool.loc[sel, f"nm_{t}"])
    return pool


# ---------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------
def _roi(rate: float) -> float:
    """Return per unit risked at -110."""
    return rate * (100 / 110) - (1 - rate)


def _seg(b: pd.DataFrame, label: str, min_n: int = MIN_SEGMENT) -> dict | None:
    """One slice's record, with the two things needed to judge whether it is real: its
    distance from break-even in standard errors, and how it behaved season by season."""
    if len(b) < min_n:
        return None
    rate = float(b["right"].mean())
    se = (rate * (1 - rate) / len(b)) ** 0.5
    be = config.BREAK_EVEN / 100
    out = {"segment": label, "n": int(len(b)), "cover_pct": round(100 * rate, 1),
           "stderr": round(100 * se, 2),
           "vs_break_even_se": round((rate - be) / se, 2) if se else None,
           "roi_pct": round(100 * _roi(rate), 2)}
    by = b.groupby("season")["right"].agg(["size", "mean"])
    by = by[by["size"] >= MIN_SEASON_ROWS]
    if len(by):
        out["season_cover_pct"] = {int(s): round(100 * float(m), 1) for s, m in by["mean"].items()}
        out["seasons_measured"] = int(len(by))
        out["seasons_above_break_even"] = int((by["mean"] > be).sum())
    return out


def _decided(pool: pd.DataFrame, target: str, pred_col: str) -> pd.DataFrame:
    """Games with a line that did not push, and whether `pred_col`'s side of it won."""
    p = pool[pool[f"mkt_{target}"].notna() & pool[pred_col].notna()].copy()
    p = p[p[target] != p[f"mkt_{target}"]]
    p["dis"] = p[pred_col] - p[f"mkt_{target}"]
    p = p[p["dis"] != 0]
    p["right"] = np.where(p["dis"] > 0, p[target] > p[f"mkt_{target}"],
                          p[target] < p[f"mkt_{target}"])
    return p


def _buckets(p: pd.DataFrame, edges) -> list[dict]:
    out = []
    for lo, hi in edges:
        row = _seg(p[(p["dis"].abs() >= lo) & (p["dis"].abs() < hi)],
                   f"{lo:g}-{hi:g}" if hi < 99 else f"{lo:g}+", min_n=MIN_BUCKET)
        if row:
            row["disagreement"] = row.pop("segment")
            out.append(row)
    return out


def _softness(p: pd.DataFrame, target: str) -> dict:
    """Where, if anywhere, is the NBA market soft? Cut on the situations a public model has a
    structural reason to see differently from the book - and each one reported with its
    per-season record, because a real edge shows up in most seasons and a fluke in one."""
    out: dict[str, list] = {}

    def add(key, rows):
        rows = [r for r in rows if r]
        if rows:
            out[key] = rows

    b2b = (p["h_b2b"] == 1) | (p["a_b2b"] == 1)
    add("by_back_to_back", [_seg(p[b2b], "a side on a back-to-back"),
                            _seg(p[~b2b], "no back-to-back")])
    rd = p["rest_diff"].abs()
    add("by_rest", [_seg(p[rd <= 1], "even rest"), _seg(p[rd >= 2], "a rest mismatch")])
    early = p["early"] == 1
    post = (p["playoff"] == 1) | (p["playin"] == 1)
    add("by_stage", [_seg(p[early], "first 10 games"), _seg(p[~early & ~post], "regular season"),
                     _seg(p[post], "play-in and playoffs", min_n=100)])
    fav = p["mkt_margin"].abs()
    add("by_spread_size", [_seg(p[fav <= 3], "pick'em (<=3)"), _seg(p[(fav > 3) & (fav <= 7)], "3.5-7"),
                           _seg(p[(fav > 7) & (fav <= 10)], "7.5-10"), _seg(p[fav > 10], "10.5+")])
    key = np.maximum(p["h_miss_top"], p["a_miss_top"]) * 0.45 >= 2.0
    add("by_absence", [_seg(p[key], "a key player out"), _seg(p[~key], "no key absence")])
    far = p["a_tz_abs"] >= 2
    add("by_travel", [_seg(p[far], "away side crossed 2+ time zones"),
                      _seg(p[~far], "shorter trip")])
    add("by_altitude", [_seg(p[p["altitude"] >= 1000], "at altitude (Denver, Utah)")])
    big = p["dis"].abs() >= (1.0 if target == "margin" else 2.0)
    add("soft_and_loud", [_seg(p[key & big], "key absence & a real disagreement", min_n=100),
                          _seg(p[~key & big], "full strength & a real disagreement", min_n=100)])
    return out


def _significance(*groups) -> dict:
    """Stamp every reported slice with whether it survives the number of looks taken.

    These tables exist to be scanned for something to bet, which is exactly when an uncorrected
    z-score misleads: across k slices the chance one clears z >= 2 by luck grows fast with k.
    Sidak turns the family-wise error rate back into the 5% a reader assumes.

    **One-sided, on purpose.** The only finding worth flagging is a slice that beats break-even.
    The NFL report tests |z| because its slices hold a few hundred games; an NBA slice holds
    thousands, and at that size a slice covering a plain 50% is "significantly" below the 52.4%
    break-even - which is just the vig, not a discovery. The first run of this report called six
    totals slices significant for exactly that reason.
    """
    segs = [r for g in groups for r in (g or []) if r]
    if not segs:
        return {}
    from scipy.stats import norm
    k = len(segs)
    crit = float(norm.isf(1 - (1 - FAMILY_ALPHA) ** (1 / k)))
    for r in segs:
        z = r.get("vs_break_even_se", r.get("vs_zero_se"))
        r["significant"] = bool(z is not None and z >= crit)
    survivors = [r.get("segment") or r.get("disagreement") for r in segs if r["significant"]]
    return {"comparisons": k, "family_alpha": FAMILY_ALPHA, "z_required": round(crit, 2),
            "significant_segments": survivors,
            "verdict": ("no segment survives correction for the number of looks taken"
                        if not survivors else f"{len(survivors)} of {k} segments survive correction")}


def _mae(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return round(float(np.abs(a[ok] - b[ok]).mean()), 3) if ok.any() else None


def evaluate_target(pool: pd.DataFrame, target: str) -> tuple[dict, dict]:
    """(no-market block, market block) for one target."""
    rating = "e_margin" if target == "margin" else "e_total"
    seasons = sorted(int(s) for s in pool["season"].unique())
    nm = {"test_seasons": seasons, "n_test_total": int(len(pool)),
          "mae_model": _mae(pool[f"nm_{target}"], pool[target]),
          "mae_rating_baseline": _mae(pool[rating], pool[target])}
    cl = pool[pool[f"mkt_{target}"].notna()]
    nm["n_closing"] = int(len(cl))
    nm["mae_model_on_closing_games"] = _mae(cl[f"nm_{target}"], cl[target])
    nm["mae_market_baseline"] = _mae(cl[f"mkt_{target}"], cl[target])
    view = _decided(pool, target, f"nm_{target}")
    nm["ats_rate"] = round(100 * float(view["right"].mean()), 1) if len(view) else None
    nm["ats_n"] = int(len(view))
    nm["ats_by_disagreement"] = _buckets(view, VIEW_BUCKETS)
    nm["significance"] = _significance(nm["ats_by_disagreement"])

    mk = {"test_seasons": seasons, "n_test_total": int(len(cl)),
          "mae_model": _mae(cl[f"pub_{target}"], cl[target]),
          "mae_raw_model": _mae(cl[f"raw_{target}"], cl[target]),
          "mae_market_baseline": _mae(cl[f"mkt_{target}"], cl[target])}
    resid = (cl[f"pub_{target}"] - cl[target]).dropna()
    mk["sigma"] = round(float(resid.std()), 2) if len(resid) else None
    mk["bias"] = round(float(resid.mean()), 2) if len(resid) else None
    dec = _decided(pool, target, f"raw_{target}")
    if len(dec):
        rate = float(dec["right"].mean())
        mk.update(ats_rate=round(100 * rate, 1), ats_n=int(len(dec)),
                  ats_stderr=round(100 * (0.25 / len(dec)) ** 0.5, 2))
    mk["beats_market"] = bool(mk["mae_model"] is not None and mk["mae_market_baseline"] is not None
                              and mk["mae_model"] < mk["mae_market_baseline"])
    mk["break_even_pct"] = round(config.BREAK_EVEN, 2)
    mk["shrink_by_season"] = {int(s): round(float(w), 2) for s, w in
                              pool.groupby("season")[f"shrink_{target}"].first().items()}
    mk["ats_by_disagreement"] = _buckets(dec, BUCKETS[target])
    mk["market_softness"] = _softness(dec, target) if len(dec) else {}
    mk["significance"] = _significance(mk["ats_by_disagreement"], *mk["market_softness"].values())
    per = []
    for s in seasons:
        c = cl[cl["season"] == s]
        d = dec[dec["season"] == s]
        if len(c):
            per.append({"season": s, "n": int(len(c)), "mae_model": _mae(c[f"pub_{target}"], c[target]),
                        "mae_market": _mae(c[f"mkt_{target}"], c[target]),
                        "mae_nomarket": _mae(c[f"nm_{target}"], c[target]),
                        "ats_rate": round(100 * float(d["right"].mean()), 1) if len(d) else None})
    mk["per_season"] = per
    return nm, mk


def _logloss(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def evaluate_moneyline(pool: pd.DataFrame, sigma: float) -> dict:
    """The moneyline: log loss against the de-vigged close, and ROI at the closing price."""
    m = pool[pool["p_home"].notna() & pool["pub_margin"].notna()].copy()
    if m.empty:
        return {"skipped": True, "reason": "no closing moneylines in the walk-forward"}
    won = (m["margin"] > 0).astype(float).values
    p_model = win_prob(m["pub_margin"].values, sigma)
    p_nm = win_prob(m["nm_margin"].values, sigma)
    out = {"n": int(len(m)), "sigma": sigma,
           "log_loss_model": round(float(_logloss(p_model, won).mean()), 5),
           "log_loss_market": round(float(_logloss(m["p_home"].values, won).mean()), 5),
           "log_loss_nomarket_model": round(float(_logloss(p_nm, won).mean()), 5)}
    out["log_loss_edge"] = round(out["log_loss_market"] - out["log_loss_model"], 5)
    out["beats_market"] = bool(out["log_loss_edge"] > 0)
    out["per_season"] = [{"season": int(s), "n": int(sel.sum()),
                          "log_loss_edge": round(float(_logloss(m["p_home"].values[sel], won[sel]).mean()
                                                       - _logloss(p_model[sel], won[sel]).mean()), 5)}
                         for s in sorted(m["season"].unique())
                         for sel in [(m["season"] == s).values]]
    priced = m["home_dec"].notna().values & m["away_dec"].notna().values
    if priced.sum() >= MIN_TEST_GAMES:
        mm, pm, wn = m[priced], p_model[priced], won[priced]
        ev_h = expected_value(pm, 0.0, mm["home_dec"].values)
        ev_a = expected_value(1 - pm, 0.0, mm["away_dec"].values)
        home = ev_h >= ev_a
        evv = np.where(home, ev_h, ev_a)
        dec = np.where(home, mm["home_dec"].values, mm["away_dec"].values)
        hit = np.where(home, wn == 1, wn == 0)
        profit = np.where(hit, dec - 1, -1.0)
        b = pd.DataFrame({"season": mm["season"].values, "ev": evv, "profit": profit, "won": hit})
        rows = []
        for lo, hi in EV_BUCKETS:
            x = b[(b["ev"] >= lo) & (b["ev"] < hi)]
            if len(x) >= MIN_BUCKET:
                r = x["profit"].to_numpy(float)
                se = float(r.std(ddof=1) / np.sqrt(len(r)))
                row = {"ev": f"{lo:+.2f} to {hi:+.2f}" if hi < 1 else f"{lo:+.2f}+", "lo": lo, "hi": hi,
                       "n": int(len(x)), "hit_pct": round(100 * float(x["won"].mean()), 1),
                       "roi_pct": round(100 * float(r.mean()), 2), "stderr": round(100 * se, 2),
                       "vs_zero_se": round(float(r.mean()) / se, 2) if se else None,
                       "candidate": lo >= 0}
                by = x.groupby("season")["profit"].agg(["size", "mean"])
                by = by[by["size"] >= MIN_SEASON_ROWS]
                if len(by):
                    row["season_roi_pct"] = {int(s): round(100 * float(v), 1) for s, v in by["mean"].items()}
                    row["seasons_measured"] = int(len(by))
                    row["seasons_positive"] = int((by["mean"] > 0).sum())
                rows.append(row)
        out["roi_by_ev"] = rows
        out["significance"] = _significance([r for r in rows if r["candidate"]])
    return out


def evaluate(pool: pd.DataFrame) -> dict:
    out = {}
    for t in TARGETS:
        out[f"{t}_nomarket"], out[f"{t}_market"] = evaluate_target(pool, t)
    return out


def win_sigma(pool: pd.DataFrame) -> float:
    """The spread of the margin that makes win probabilities honest, by maximum likelihood.

    Not the residual standard deviation. Margins are fat-tailed - garbage time widens blowouts
    without changing who won - so a normal curve with the residual sd (about 13.8 points) puts
    too much probability on big underdogs. The first run of this report found exactly that: its
    "best" moneyline bucket was long-shot underdogs the model gave 25-30%, which won 19.9% of
    the time, as their prices said they would. One parameter, fitted on the pooled walk-forward.
    """
    p = pool[pool["pub_margin"].notna()]
    won = (p["margin"] > 0).astype(float).values
    grid = np.arange(9.0, 16.01, 0.25)
    ll = [float(_logloss(win_prob(p["pub_margin"].values, s), won).mean()) for s in grid]
    return round(float(grid[int(np.argmin(ll))]), 2)


def recent_sigma(pool: pd.DataFrame, target: str, seasons: int = 3) -> float:
    """Spread of the published number's errors over the last few test seasons. NBA margins
    have become more variable - 12 points around the prediction before 2016, 14 since - so an
    all-time figure would overstate how sure a 2026 prediction can be."""
    last = sorted(pool["season"].unique())[-seasons:]
    p = pool[pool["season"].isin(last)]
    r = (p[f"pub_{target}"] - p[target]).dropna()
    default = 13.5 if target == "margin" else 18.5
    return round(float(r.std()), 2) if len(r) > 500 else default


# ---------------------------------------------------------------------------------
def save_models(models: dict) -> None:
    config.ensure_dirs()
    for name, m in models.items():
        m.save(config.MODEL_DIR / f"{name}.json")


def load_models() -> tuple[dict, dict]:
    meta = json.loads((config.MODEL_DIR / "meta.json").read_text())
    models = {n: Ridge.load(config.MODEL_DIR / f"{n}.json") for n in meta["models"]}
    return models, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args(argv)

    games, players, lines, names = assemble(fetch=not args.no_fetch)
    D, _, _, _ = build(games, players, lines, names=names)
    D = nomarket_oos(outcomes(D))
    log.info("%d training games, seasons %d-%d, %d with a closing spread, %d with a closing "
             "moneyline", len(D), int(D["season"].min()), int(D["season"].max()),
             int(D["mkt_margin"].notna().sum()), int(D["p_home"].notna().sum()))

    pool = walk_forward(D)
    if pool.empty:
        raise SystemExit("not enough complete seasons with closing lines to evaluate")
    ev = evaluate(pool)
    sigma = {t: recent_sigma(pool, t) for t in TARGETS}
    sigma["win"] = win_sigma(pool)
    ev["moneyline"] = evaluate_moneyline(pool, sigma["win"])

    last = int(D["season"].max()) + 1
    models = {}
    shrink = {}
    for t in TARGETS:
        models[f"{t}_nomarket"] = fit_nomarket(D, t, last)
        tr = D[D[f"mkt_{t}"].notna() & D[f"dev_{t}"].notna()]
        if len(tr) >= MIN_MARKET_TRAIN:
            models[f"{t}_market"] = fit_market(tr, t, last)
        w, raw, n = shrink_for(pool, t)
        shrink[t], shrink[f"{t}_raw"], shrink[f"n_{t}"] = w, raw, n
    save_models(models)
    meta = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sport": "nba", "models": sorted(models), "base_features": BASE_FEATURES,
            "market_features": MARKET_FEATURES, "n_rows": int(len(D)),
            "seasons": sorted(int(s) for s in D["season"].unique()),
            "half_life_seasons": HALF_LIFE, "shrink": shrink, "sigma": sigma,
            "effects": {n: m.effects() for n, m in models.items()},
            "lines": {"spread_sources": lines["source"].fillna("").value_counts().to_dict(),
                      "moneyline_sources": lines["ml_source"].fillna("").value_counts().to_dict()}
            if len(lines) else {},
            "eval": ev}
    config.ensure_dirs()
    (config.MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    for t in TARGETS:
        nm, mk = ev[f"{t}_nomarket"], ev[f"{t}_market"]
        log.info("%s: no-market MAE %.3f | market-aware %.3f vs the close's %.3f | ATS %s%% on %s "
                 "| shrink %.2f (raw %.2f) | %s", t, nm["mae_model_on_closing_games"],
                 mk["mae_model"], mk["mae_market_baseline"], mk.get("ats_rate"), mk.get("ats_n"),
                 shrink[t], shrink[f"{t}_raw"], (mk.get("significance") or {}).get("verdict"))
    ml = ev["moneyline"]
    if not ml.get("skipped"):
        log.info("moneyline: log loss %.5f vs market %.5f (edge %+.5f)", ml["log_loss_model"],
                 ml["log_loss_market"], ml["log_loss_edge"])
    log.info("saved %s to %s", sorted(models), config.MODEL_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
