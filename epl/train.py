"""Train the Premier League supremacy and total-goals models.

    python -m epl.train                # fetch data, train, evaluate, save
    python -m epl.train --no-fetch     # use the local cache (offline / CI test)
    python -m epl.train --search       # small randomised hyper-parameter search

Four models, same shape as the other two sports:

    total_nomarket / sup_nomarket      - features only (used when no price is available yet)
    total_market   / sup_market        - features + the market (used when we have prices)

The number that decides whether to bet any of this is in data/epl/models/meta.json:

    mae_model            model error, in goals
    mae_rating_baseline  the raw rating expectation, no ML
    mae_market_baseline  the market's own number  <- if you don't beat this, you have no edge
    cover_rate           how often the model's side of the handicap landed, out of sample

**Set your expectations before you read it.** The closing Asian handicap on a Premier League
match is, by most measures, the single most efficient price in world sport: enormous limits,
sharp money, and twenty clubs that thousands of people model full-time. The realistic outcome
is ``beats_market: false`` and a fitted ``shrink`` near zero. That is the model correctly
reporting that the market already knows what it knows.

**What is different here is the probability report.** In the other two sports a point
prediction plus a sigma is the whole model, so mean absolute error tells you most of what you
need. Here the model's actual output is a distribution over scorelines, and a model can have
identical MAE on supremacy while being badly wrong about how often matches are drawn. So
alongside the error tables this run scores the implied 1X2 probabilities directly - log loss
and Brier score against the de-vigged market, plus a calibration table and a specific check on
the draw. That last one matters: the draw is the outcome public models are worst at and the one
where the market has least to gain from being precise, so if there is anything here at all, it
is the most likely place to find it.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from . import config
from .features import BASE_FEATURES, MARKET_FEATURES, build
from .poisson import grid, match_odds, over_under
from .sources import active


def source():
    """The configured history/prices backend, resolved at call time.

    Never bound at import: the tests and the docs both switch DEGEN_EPL_SOURCE, and a module
    captured at import would ignore them - the same trap the config paths already avoid.
    """
    return active()


log = logging.getLogger("epl.train")

# Shallower and more regularised than the college defaults, and close to the NFL's. 380
# matches a season sounds like plenty next to the NFL's 272, but the target has a standard
# deviation of about 1.9 goals against the NFL margin's 13.5 - the signal-to-noise ratio per
# row is what sets how much tree you can afford, and it is not better here.
DEFAULTS = dict(n_estimators=700, max_depth=3, learning_rate=0.02, subsample=0.8,
                colsample_bytree=0.7, min_child_weight=20, reg_lambda=5.0, random_state=42)

TARGETS = {"total": "total_goals", "sup": "supremacy"}
# The market's number for each target, already in the model's units. Unlike the NFL there is no
# sign flip: `mkt_sup` is the market's expected home supremacy (from the Asian handicap, whose
# sign convention is fixed once in sources/source().py) and `mkt_total` is its expected
# goal count (inverted from the over/under price through the same scoreline model the
# predictions come out of).
MARKET_COL = {"total": "mkt_total", "sup": "mkt_sup"}
RATING_COL = {"total": "exp_total", "sup": "exp_sup"}

# Hard ceiling on how far we will move off the market number. The same reasoning as the NFL's
# 0.45, applied to a market that is if anything sharper.
SHRINK_CAP = float(os.environ.get("DEGEN_SHRINK_CAP", "0.40"))

MIN_TRAIN_GAMES = 600
MIN_TEST_GAMES = 200
MIN_SHRINK_GAMES = 700
MIN_BUCKET = 80
MIN_SEGMENT = 150
# Five seasons is 1,900 matches, comparable to the NFL walk-forward's six seasons of 272.
WALK_FORWARD_SEASONS = int(os.environ.get("DEGEN_WALK_SEASONS", "5"))
MIN_SEASON_ROWS = 40


def make_model(params=None):
    p = {**DEFAULTS, **(params or {})}
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:squarederror", n_jobs=-1, **p), "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        log.warning("xgboost missing - using HistGradientBoostingRegressor")
        return HistGradientBoostingRegressor(
            max_iter=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=p["learning_rate"], random_state=42), "sklearn"


def _fit(df, feats, target, params=None):
    m, backend = make_model(params)
    m.fit(df[feats], df[target])
    return m, backend


def _save(model, backend, name):
    config.ensure_dirs()
    if backend == "xgboost":
        model.save_model(config.MODEL_DIR / f"{name}.json")
    else:
        import joblib
        joblib.dump(model, config.MODEL_DIR / f"{name}.joblib")


def load_models() -> tuple[dict, dict]:
    meta = json.loads((config.MODEL_DIR / "meta.json").read_text())
    models = {}
    for name in meta["models"]:
        if meta["backend"] == "xgboost":
            from xgboost import XGBRegressor
            m = XGBRegressor()
            m.load_model(config.MODEL_DIR / f"{name}.json")
        else:
            import joblib
            m = joblib.load(config.MODEL_DIR / f"{name}.joblib")
        models[name] = m
    return models, meta


def assemble(fetch: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    if fetch:
        games = source().update_games()
        source().build_strength(games)
    else:
        games = source().load_games()
    if games.empty:
        raise SystemExit("no matches cached - run once with network access")
    lines = source().load_lines()
    coverage = source().coverage_report(games, lines)
    train_rows, _, _ = build(games, lines=lines, strength=source().load_strength())
    log.info("%d completed matches, seasons %s, %d with a market number",
             len(train_rows), sorted(train_rows["season"].unique()),
             int(train_rows["mkt_sup"].notna().sum()) if len(train_rows) else 0)
    for c in coverage:
        log.info("  %s: %d matches, %d priced (%d closing) ah=%d ou=%d %s",
                 c["season"], c["matches"], c["priced"], c["closing"], c["ah"], c["ou"],
                 ",".join(c["sources"])[:60])
    return train_rows, coverage


def complete_seasons(df: pd.DataFrame) -> list[int]:
    """Seasons that have finished. The in-progress season must never be the holdout."""
    today = config.today_uk()
    return sorted(int(s) for s in df["season"].unique() if config.season_end(int(s)) < today)


def walk_forward(df: pd.DataFrame, kind: str, market: bool, params=None,
                 n_seasons: int = WALK_FORWARD_SEASONS) -> dict:
    """Train on everything before season S, predict S, for each of the last n_seasons
    complete seasons; pool the out-of-sample predictions."""
    target = TARGETS[kind]
    feats = BASE_FEATURES + (MARKET_FEATURES if market else [])
    line_col, rating_col = MARKET_COL[kind], RATING_COL[kind]

    d = df.dropna(subset=[target])
    if market:
        d = d[d[line_col].notna()]
    comp = complete_seasons(d)
    if len(comp) < 2:
        return {"skipped": True, "reason": "need >=2 complete seasons", "n": int(len(d))}

    tests = [s for s in comp[-n_seasons:] if s > comp[0]]
    chunks, per_season = [], []
    for s in tests:
        tr, te = d[d["season"] < s], d[d["season"] == s]
        if len(tr) < MIN_TRAIN_GAMES or len(te) < MIN_TEST_GAMES:
            continue
        model, _ = _fit(tr, feats, target, params)
        pred = model.predict(te[feats])
        line = te[line_col].values if market else np.full(len(te), np.nan)
        chunk = pd.DataFrame({"season": s, "game_id": te["game_id"].values,
                              "pred": pred, "actual": te[target].values,
                              "rating": te[rating_col].values, "line": line})
        chunks.append(chunk)
        per_season.append({"season": int(s), "n": int(len(te)),
                           "mae_model": round(float(mean_absolute_error(chunk.actual, chunk.pred)), 3),
                           "cover_rate": _cover(chunk)[0] if market else None})
    if not chunks:
        return {"skipped": True, "reason": "no season had enough data", "n": int(len(d))}

    pool = pd.concat(chunks, ignore_index=True)
    resid = pool["actual"] - pool["pred"]
    out = {
        "test_seasons": [int(s) for s in pool["season"].unique()],
        "n_test_total": int(len(pool)),
        "mae_model": round(float(mean_absolute_error(pool.actual, pool.pred)), 3),
        "mae_rating_baseline": round(float(mean_absolute_error(pool.actual, pool.rating)), 3),
        "sigma": round(float(resid.std()), 3),
        "bias": round(float(resid.mean()), 3),
        "per_season": per_season,
    }
    if market:
        out["mae_market_baseline"] = round(float(mean_absolute_error(pool.actual, pool.line)), 3)
        rate, n_dec = _cover(pool)
        out["cover_rate"] = rate
        out["cover_n"] = n_dec
        out["cover_stderr"] = round(float(100 * (0.25 / max(n_dec, 1)) ** 0.5), 2)
        out["beats_market"] = bool(out["mae_model"] < out["mae_market_baseline"])
        shrink = _best_shrink(pool.pred.values, pool.line.values, pool.actual.values)
        out["shrink_raw"] = round(float(shrink), 2)
        if n_dec < MIN_SHRINK_GAMES:
            log.warning("%s/mkt: only %d decided matches - clamping shrink to the default",
                        kind, n_dec)
            shrink = min(shrink, config.DEFAULT_SHRINK)
        out["shrink"] = round(float(min(shrink, SHRINK_CAP)), 2)
        out["cover_by_disagreement"] = _by_disagreement(pool, kind)
        out["break_even_pct"] = round(config.BREAK_EVEN, 2)
        out["venue"] = config.VENUE
        out["market_softness"] = _softness(pool, d, kind)
        out["significance"] = _apply_multiple_comparisons(
            out["cover_by_disagreement"], *out["market_softness"].values())
    return out


def _decided(pool: pd.DataFrame) -> pd.DataFrame:
    """Matches where the model's side of the market number actually resolved.

    Pushes are dropped, and in football they are not rare: supremacy is an integer, so a
    handicap quoted at a whole number pushes whenever the match lands exactly on it. Scoring
    those as half-wins - or worse, as losses - would misstate the cover rate by more than any
    edge being measured.
    """
    p = pool.dropna(subset=["line"]).copy()
    p = p[p["actual"] != p["line"]].copy()
    p["right"] = np.where(p["pred"] > p["line"], p["actual"] > p["line"], p["actual"] < p["line"])
    p["disagree"] = (p["pred"] - p["line"]).abs()
    return p


def _roi(rate: float, price: float = 0.50) -> float:
    coef = config.FEE_COEF.get(config.VENUE, None)
    if coef is None:
        return rate * (config.SPORTSBOOK_DECIMAL - 1) - (1 - rate)
    fee = coef * price * (1 - price)
    cost = price + fee
    return (rate * (1 - cost) - (1 - rate) * cost) / cost


def _cover(chunk: pd.DataFrame) -> tuple[float | None, int]:
    decided = chunk["actual"] != chunk["line"]
    if not decided.any():
        return None, 0
    c = chunk[decided]
    right = np.where(c["pred"] > c["line"], c["actual"] > c["line"], c["actual"] < c["line"])
    return round(float(100 * right.mean()), 1), int(len(c))


def _seg(b: pd.DataFrame, label: str, min_n: int = MIN_SEGMENT) -> dict | None:
    """One segment's record, with the two things needed to judge whether it is real."""
    if len(b) < min_n:
        return None
    rate = float(b["right"].mean())
    se = (rate * (1 - rate) / len(b)) ** 0.5
    be = config.BREAK_EVEN / 100
    out = {"segment": label, "n": int(len(b)),
           "cover_pct": round(100 * rate, 1),
           "stderr": round(100 * se, 2),
           "vs_break_even_se": round((rate - be) / se, 2) if se else None,
           "roi_pct": round(100 * _roi(rate), 2)}
    if "season" in b.columns:
        by = b.groupby("season")["right"].agg(["size", "mean"])
        by = by[by["size"] >= MIN_SEASON_ROWS]
        if len(by):
            out["season_cover_pct"] = {int(s): round(100 * float(m), 1)
                                       for s, m in by["mean"].items()}
            out["seasons_measured"] = int(len(by))
            out["seasons_above_break_even"] = int((by["mean"] > be).sum())
    return out


def _softness(pool: pd.DataFrame, meta: pd.DataFrame, kind: str = "sup") -> dict:
    """Where, if anywhere, is this market soft?

    The segments are chosen to be the places where a public model has any structural reason to
    know something the closing price does not, rather than a generic slicing of the data:

    * **Promoted clubs.** The three newcomers are the clubs with the least data, the most
      uncertain priors and the widest disagreement among modellers. If the market misprices
      anything systematically it is most likely to be here, and this pipeline explicitly carries
      a prior up from the division below to have a view.
    * **European commitments.** A club playing Thursday in the Europa League and Sunday in the
      league is a well-known situation - which cuts both ways, because well-known situations
      are priced. Worth measuring rather than assuming.
    * **Behind closed doors.** A genuine natural experiment in home advantage, and a segment
      where the market was demonstrably learning in real time.
    * **Derbies**, **favourite size**, **stage of season**, and **short rest**.

    Every row reports `vs_break_even_se`. Treat anything under +2 as unproven, and remember you
    are looking at ~20 segments, so the best one landing at +2 is roughly what chance produces.
    """
    p = _decided(pool)
    if p.empty:
        return {}
    # NB: no "season" here - the pool already carries it, and merging it again collides into
    # season_x/season_y, which silently empties the per-season stability these are judged on.
    cols = ["game_id", "h_promoted", "a_promoted", "h_in_europe", "a_in_europe", "is_derby",
            "no_crowd", "is_early_season", "h_games", "a_games", "mkt_sup", "mkt_total",
            "h_short_rest", "a_short_rest", "rest_diff", "travel_km", "season_frac"]
    have = [c for c in cols if c in meta.columns]
    p = p.merge(meta[have].astype({"game_id": str}), on="game_id", how="left")
    out: dict[str, list] = {}

    def add(key, rows):
        rows = [r for r in rows if r]
        if rows:
            out[key] = rows

    if "h_promoted" in p:
        promoted = (p.h_promoted == 1) | (p.a_promoted == 1)
        add("by_promotion", [
            _seg(p[promoted], "a promoted club involved"),
            _seg(p[~promoted], "no promoted club"),
        ])

    if "h_in_europe" in p:
        euro = (p.h_in_europe == 1) | (p.a_in_europe == 1)
        add("by_european_load", [
            _seg(p[euro], "a European club involved"),
            _seg(p[~euro], "neither club in Europe"),
        ])

    if "no_crowd" in p:
        add("by_crowd", [
            _seg(p[p.no_crowd == 1], "behind closed doors", min_n=100),
            _seg(p[p.no_crowd == 0], "normal crowd"),
        ])

    if "is_derby" in p:
        add("by_derby", [
            _seg(p[p.is_derby == 1], "local derby", min_n=80),
            _seg(p[p.is_derby == 0], "not a derby"),
        ])

    if "h_games" in p:
        early = p[["h_games", "a_games"]].min(axis=1)
        add("by_stage", [
            _seg(p[early < 6], "first six matches"),
            _seg(p[(early >= 6) & (early < 19)], "matches 6-19"),
            _seg(p[early >= 19], "second half of the season"),
        ])

    if "mkt_sup" in p:
        fav = p.mkt_sup.abs()
        add("by_favourite_size", [
            _seg(p[fav <= 0.25], "near level"),
            _seg(p[(fav > 0.25) & (fav <= 0.75)], "0.25-0.75 goals"),
            _seg(p[(fav > 0.75) & (fav <= 1.5)], "0.75-1.5 goals"),
            _seg(p[fav > 1.5], "1.5+ goals"),
        ])

    if "h_short_rest" in p:
        short = (p.h_short_rest == 1) | (p.a_short_rest == 1)
        add("by_rest", [
            _seg(p[short], "a side on short rest", min_n=100),
            _seg(p[~short], "normal rest"),
        ])

    # The interesting cross: our biggest disagreements on the clubs the market has least
    # history for. If a public-data edge exists anywhere in this league, it is here.
    if "h_promoted" in p:
        promoted = (p.h_promoted == 1) | (p.a_promoted == 1)
        # A "big" disagreement is a different size for the two targets: half a goal about who
        # wins is a far bigger claim than half a goal about how many are scored.
        big = p.disagree >= BIG_DISAGREEMENT.get(kind, 0.4)
        add("soft_and_loud", [
            _seg(p[promoted & big], "promoted club & big disagreement", min_n=80),
            _seg(p[~promoted & big], "established clubs & big disagreement", min_n=100),
        ])
    return out


FAMILY_ALPHA = 0.05


def _apply_multiple_comparisons(*groups) -> dict:
    """Stamp every reported segment with whether it survives the number of looks taken.

    These tables exist to be scanned for something to bet, which is exactly the situation where
    an uncorrected z-score misleads. Across k independent segments the chance at least one
    clears |z| >= 2 is 1 - 0.954**k: about 60% at k=20. The Sidak correction turns the
    family-wise error rate back into the 5% a reader assumes they are getting.
    """
    segs = [r for g in groups for r in (g or []) if r]
    k = len(segs)
    if not k:
        return {}
    from scipy.stats import norm
    per_comparison = 1 - (1 - FAMILY_ALPHA) ** (1 / k)
    crit = float(norm.isf(per_comparison / 2))
    for r in segs:
        z = r.get("vs_break_even_se")
        r["significant"] = bool(z is not None and abs(z) >= crit)
    survivors = [r.get("segment") or r.get("disagreement") for r in segs if r["significant"]]
    return {"comparisons": k,
            "family_alpha": FAMILY_ALPHA,
            "z_required": round(crit, 2),
            "z_required_note": (f"{k} segments were looked at, so a segment needs "
                                f"|z| >= {crit:.2f} - not 2.0 - before it means anything"),
            "significant_segments": survivors,
            "verdict": ("no segment survives correction for the number of looks taken"
                        if not survivors else
                        f"{len(survivors)} of {k} segments survive correction")}


def _best_shrink(pred, line, actual) -> float:
    best, best_mae = 0.0, np.inf
    for w in np.arange(0, 1.01, 0.05):
        mae = mean_absolute_error(actual, line + w * (pred - line))
        if mae < best_mae:
            best, best_mae = float(w), mae
    return best


# Disagreement buckets are in GOALS and are narrower for supremacy than for totals, because a
# 0.5-goal disagreement about who wins is a much bigger claim than a 0.5-goal disagreement
# about how many are scored.
# What counts as a large disagreement, per target, for the cross-segment below.
BIG_DISAGREEMENT = {"sup": 0.4, "total": 0.6}

BUCKETS = {"sup": [(0, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.75), (0.75, 999)],
           "total": [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.9), (0.9, 999)]}


def _by_disagreement(pool: pd.DataFrame, kind: str) -> list[dict]:
    """Does the model get *better* when it disagrees with the market the most?

    If the high-disagreement buckets clear break-even by more than about two standard errors on
    a real sample, there is an edge in a subset. If they do not, larger disagreements are just
    larger errors - which is the more common finding, and against this market it is the
    expected one.
    """
    decided = _decided(pool)
    if decided.empty:
        return []
    out = []
    for lo, hi in BUCKETS.get(kind, BUCKETS["sup"]):
        b = decided[(decided["disagree"] >= lo) & (decided["disagree"] < hi)]
        row = _seg(b, f"{lo}-{hi if hi < 999 else '+'}", min_n=MIN_BUCKET)
        if row:
            row["disagreement"] = row.pop("segment")
            out.append(row)
    return out


# ---------------------------------------------------------------------------------
# The probability report - the part with no analogue in the other two pipelines
# ---------------------------------------------------------------------------------
def probability_report(df: pd.DataFrame, sup_eval: dict, tot_eval: dict,
                       n_seasons: int = WALK_FORWARD_SEASONS) -> dict:
    """Score the model's implied 1X2 probabilities against the market's.

    Point error is not the right test for this sport. Two models can have identical mean
    absolute error on supremacy while disagreeing completely about how often a match ends
    level, and the draw is a quarter of all results and a market in its own right. So the
    walk-forward is re-run for the joint (supremacy, total) pair, pushed through the same
    Dixon-Coles grid the live predictions use, and scored as a probabilistic forecast:

    * **log loss** - the proper scoring rule. Lower is better, and the gap to the market's own
      log loss is the honest answer to "does this model know anything the price does not".
    * **Brier score** - less sensitive to confident mistakes, reported because the two
      disagreeing is itself informative.
    * **draw calibration** - predicted draw rate against actual, which is the specific failure
      mode a Gaussian margin model would have and this one is built to avoid.

    A model that beats the market on log loss while losing on supremacy MAE is not a
    contradiction: it would mean the edge is in the shape of the distribution rather than in
    its centre, which for football is the more plausible of the two.
    """
    if sup_eval.get("skipped") or tot_eval.get("skipped"):
        return {"skipped": True, "reason": "underlying models were skipped"}

    feats = BASE_FEATURES + MARKET_FEATURES
    d = df.dropna(subset=["supremacy", "total_goals", "mkt_sup", "mkt_total"])
    d = d[d["mkt_p_home"].notna()]
    comp = complete_seasons(d)
    if len(comp) < 2:
        return {"skipped": True, "reason": "need >=2 complete seasons"}

    rows = []
    for s in [x for x in comp[-n_seasons:] if x > comp[0]]:
        tr, te = d[d["season"] < s], d[d["season"] == s]
        if len(tr) < MIN_TRAIN_GAMES or len(te) < MIN_TEST_GAMES:
            continue
        m_sup, _ = _fit(tr, feats, "supremacy")
        m_tot, _ = _fit(tr, feats, "total_goals")
        p_sup = m_sup.predict(te[feats])
        p_tot = m_tot.predict(te[feats])
        # Published numbers are the market shrunk toward the model, exactly as predict.py does
        # it - scoring the raw model output would flatter a model nobody is going to bet.
        sh_s = sup_eval.get("shrink", config.DEFAULT_SHRINK)
        sh_t = tot_eval.get("shrink", config.DEFAULT_SHRINK)
        sup = te["mkt_sup"].values + sh_s * (p_sup - te["mkt_sup"].values)
        tot = te["mkt_total"].values + sh_t * (p_tot - te["mkt_total"].values)
        for i, r in enumerate(te.itertuples()):
            ph, pd_, pa = match_odds(grid(float(sup[i]), float(tot[i])))
            if ph != ph:
                continue
            rows.append({"season": int(s),
                         "p_home": ph, "p_draw": pd_, "p_away": pa,
                         "m_home": r.mkt_p_home, "m_draw": r.mkt_p_draw,
                         "m_away": r.mkt_p_away,
                         "outcome": 0 if r.supremacy > 0 else (1 if r.supremacy == 0 else 2)})
    if not rows:
        return {"skipped": True, "reason": "no season had enough data"}

    p = pd.DataFrame(rows)
    eps = 1e-12
    out = {"n": int(len(p)), "seasons": sorted(p.season.unique().tolist())}
    for who, cols in (("model", ("p_home", "p_draw", "p_away")),
                      ("market", ("m_home", "m_draw", "m_away"))):
        probs = p[list(cols)].to_numpy(dtype=float)
        probs = np.clip(probs, eps, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        hit = probs[np.arange(len(p)), p["outcome"].to_numpy()]
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(p)), p["outcome"].to_numpy()] = 1.0
        out[f"log_loss_{who}"] = round(float(-np.log(hit).mean()), 4)
        out[f"brier_{who}"] = round(float(((probs - onehot) ** 2).sum(axis=1).mean()), 4)
    out["log_loss_edge"] = round(out["log_loss_market"] - out["log_loss_model"], 4)
    out["beats_market_log_loss"] = bool(out["log_loss_model"] < out["log_loss_market"])
    actual_draw = float((p["outcome"] == 1).mean())
    out["draw"] = {
        "actual_pct": round(100 * actual_draw, 1),
        "model_pct": round(100 * float(p["p_draw"].mean()), 1),
        "market_pct": round(100 * float(p["m_draw"].mean()), 1),
        "note": ("the draw is the outcome a Gaussian margin model cannot express at all, "
                 "which is why this pipeline models scorelines instead"),
    }
    out["calibration"] = _calibration(p)
    out["interpretation"] = (
        "log_loss_edge > 0 means the model's probabilities scored better than the "
        "de-vigged market's. Expect it to be negative; a small positive number on one "
        "walk-forward is not evidence of anything on its own."
    )
    return out


def _calibration(p: pd.DataFrame) -> list[dict]:
    """Predicted vs actual, bucketed. The check that a probability means what it says."""
    rows = []
    stacked = pd.concat([
        pd.DataFrame({"pred": p.p_home, "hit": (p.outcome == 0).astype(int), "side": "home"}),
        pd.DataFrame({"pred": p.p_draw, "hit": (p.outcome == 1).astype(int), "side": "draw"}),
        pd.DataFrame({"pred": p.p_away, "hit": (p.outcome == 2).astype(int), "side": "away"}),
    ], ignore_index=True)
    for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .65), (.65, 1.01)]:
        b = stacked[(stacked.pred >= lo) & (stacked.pred < hi)]
        if len(b) < 50:
            continue
        rows.append({"bucket": f"{lo:.2f}-{hi:.2f}", "n": int(len(b)),
                     "predicted_pct": round(100 * float(b.pred.mean()), 1),
                     "actual_pct": round(100 * float(b.hit.mean()), 1)})
    return rows


def evaluate(df, kind, market, params=None):
    return walk_forward(df, kind, market, params)


def search(df, kind, market, n_iter=12):
    rng = np.random.default_rng(0)
    space = {"n_estimators": [500, 700, 1000], "max_depth": [2, 3, 4],
             "learning_rate": [0.01, 0.02, 0.04], "subsample": [0.7, 0.8, 1.0],
             "colsample_bytree": [0.6, 0.7, 0.9], "min_child_weight": [15, 20, 30]}
    best, best_mae = None, np.inf
    for _ in range(n_iter):
        p = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        r = evaluate(df, kind, market, p)
        mae = r.get("mae_model", np.inf)
        if mae < best_mae:
            best, best_mae = p, mae
    log.info("%s/%s best params %s (MAE %.3f)", kind, "mkt" if market else "base", best, best_mae)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--search", action="store_true")
    args = ap.parse_args(argv)

    df, coverage = assemble(fetch=not args.no_fetch)
    meta = {"trained_at": datetime.utcnow().isoformat(timespec="seconds"),
            "sport": "epl", "league": config.LEAGUE, "league_name": config.LEAGUE_NAME,
            "base_features": BASE_FEATURES, "market_features": MARKET_FEATURES,
            "n_rows": int(len(df)), "seasons": [int(s) for s in sorted(df.season.unique())],
            "odds_coverage": coverage,
            "dc_rho": config.DC_RHO,
            "eval": {}, "models": [], "sigma": {}, "shrink": {}}
    backend = "xgboost"

    for kind in ("total", "sup"):
        for market in (False, True):
            name = f"{kind}_{'market' if market else 'nomarket'}"
            params = search(df, kind, market) if args.search else None
            res = walk_forward(df, kind, market, params)
            meta["eval"][name] = res
            if res.get("skipped"):
                log.warning("%s: not enough data yet (%s rows) - skipping", name, res.get("n"))
                continue
            log.info("%s: mae %.3f vs market %.3f, cover %s, shrink %s", name,
                     res["mae_model"], res.get("mae_market_baseline", float("nan")),
                     res.get("cover_rate"), res.get("shrink"))
            feats = BASE_FEATURES + (MARKET_FEATURES if market else [])
            d = df.dropna(subset=[TARGETS[kind]])
            if market:
                d = d[d[MARKET_COL[kind]].notna()]
            model, backend = _fit(d, feats, TARGETS[kind], params)
            _save(model, backend, name)
            meta["models"].append(name)
            meta["sigma"][name] = res["sigma"]
            meta["shrink"][name] = res.get("shrink", config.DEFAULT_SHRINK)
            if market and not res.get("beats_market", False):
                log.warning("%s does NOT beat the market number (%.3f vs %.3f) - "
                            "treat its picks as unproven", name,
                            res["mae_model"], res["mae_market_baseline"])
            if hasattr(model, "feature_importances_"):
                meta.setdefault("top_features", {})[name] = dict(
                    sorted(zip(feats, map(float, model.feature_importances_)),
                           key=lambda kv: -kv[1])[:12])

    meta["probability"] = probability_report(
        df, meta["eval"].get("sup_market", {}), meta["eval"].get("total_market", {}))
    if not meta["probability"].get("skipped"):
        pr = meta["probability"]
        log.info("1X2 log loss: model %.4f vs market %.4f (edge %+.4f) | draw predicted "
                 "%.1f%% actual %.1f%%", pr["log_loss_model"], pr["log_loss_market"],
                 pr["log_loss_edge"], pr["draw"]["model_pct"], pr["draw"]["actual_pct"])

    meta["backend"] = backend
    config.ensure_dirs()
    (config.MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("saved %d models to %s", len(meta["models"]), config.MODEL_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
