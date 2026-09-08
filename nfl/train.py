"""Train the NFL totals and spread models.

    python -m nfl.train                # fetch data, train, evaluate, save
    python -m nfl.train --no-fetch     # use the local cache (offline / CI test)
    python -m nfl.train --search       # small randomised hyper-parameter search

Four models, same as the college pipeline:

    total_nomarket / margin_nomarket   - features only (used when no line is available yet)
    total_market   / margin_market     - features + the line (used when we have odds)

The output that decides whether any of this is worth betting is in data/nfl/models/meta.json:

    mae_model            model error
    mae_rating_baseline  the raw Elo/efficiency expectation, no ML
    mae_market_baseline  the closing line itself  <- if you don't beat this, you have no edge
    ats_rate             how often the model's side covered, out of sample

**Set your expectations before you read it.** NFL closing lines are the most efficient market
in sports. The realistic outcome is `beats_market: false` and a fitted `shrink` near zero,
which is the model correctly reporting that the line already knows everything it knows. That
is a finding, not a failure - and it is a far more useful one than a college-sized edge that
turns out to be noise.

Two things are sized differently from the college version because 272 games a season is not
800:

* the walk-forward pools **six** seasons rather than four, so the shrink weight and the
  ats buckets rest on a few thousand games rather than a few hundred;
* every segment minimum is stated in games, and segments that cannot clear it are dropped
  rather than reported with a number nobody should read.
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
from .sources import nflverse

log = logging.getLogger("nfl.train")

# Shallower and more regularised than the college defaults. There are roughly a fifth as many
# games per season here, so the same tree depth on the same number of features is a much
# better opportunity to memorise the training set.
DEFAULTS = dict(n_estimators=600, max_depth=3, learning_rate=0.02, subsample=0.8,
                colsample_bytree=0.7, min_child_weight=15, reg_lambda=5.0, random_state=42)

TARGETS = {"total": "total_points", "margin": "home_margin"}

# Hard ceiling on how far we'll move off the closing line. Lower than the college cap: the
# NFL close is sharper, so a fit that wants to move a long way off it is far likelier to be
# overfitting than to be insight.
SHRINK_CAP = float(os.environ.get("DEGEN_SHRINK_CAP", "0.45"))

# Pooled-sample minimums. Stated once, used everywhere, so the thresholds are arguable in
# one place rather than scattered through the file.
MIN_TRAIN_GAMES = 400
MIN_TEST_GAMES = 100
MIN_SHRINK_GAMES = 500     # below this we refuse to move off config.DEFAULT_SHRINK
MIN_BUCKET = 50
MIN_SEGMENT = 120
WALK_FORWARD_SEASONS = int(os.environ.get("DEGEN_WALK_SEASONS", "6"))


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


def assemble(fetch: bool = True) -> pd.DataFrame:
    if fetch:
        games = nflverse.update_games()
        nflverse.update_epa()
        nflverse.update_continuity()
    else:
        games = nflverse.load_games()
    if games.empty:
        raise SystemExit("no games cached - run once with network access")
    train_rows, _, _ = build(games, lines=nflverse.load_lines(), epa=nflverse.load_epa(),
                             continuity=nflverse.load_continuity())
    log.info("%d completed games, seasons %s, %d with a closing line",
             len(train_rows), sorted(train_rows["season"].unique()),
             int(train_rows["total_line"].notna().sum()))
    return train_rows


def complete_seasons(df: pd.DataFrame) -> list[int]:
    """Seasons that have finished. The in-progress season must never be the holdout: a
    handful of unrepresentative early games calibrates to nonsense, and with only 272 games
    in a full NFL season a partial one is worse still."""
    today = config.today_et()
    return sorted(int(s) for s in df["season"].unique() if config.season_end(int(s)) < today)


def walk_forward(df: pd.DataFrame, kind: str, market: bool, params=None,
                 n_seasons: int = WALK_FORWARD_SEASONS) -> dict:
    """Train on everything before season S, predict S, for each of the last n_seasons
    complete seasons; pool the out-of-sample predictions."""
    target = TARGETS[kind]
    feats = BASE_FEATURES + (MARKET_FEATURES if market else [])
    line_col = "total_line" if kind == "total" else "spread_home"
    rating_col = "exp_total" if kind == "total" else "exp_margin"

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
        line = (te[line_col] if kind == "total" else -te[line_col]).values \
            if market else np.full(len(te), np.nan)
        chunk = pd.DataFrame({"season": s, "game_id": te["game_id"].values,
                              "pred": pred, "actual": te[target].values,
                              "rating": te[rating_col].values, "line": line})
        chunks.append(chunk)
        per_season.append({"season": int(s), "n": int(len(te)),
                           "mae_model": round(float(mean_absolute_error(chunk.actual, chunk.pred)), 2),
                           "ats_rate": _ats(chunk)[0] if market else None})
    if not chunks:
        return {"skipped": True, "reason": "no season had enough data", "n": int(len(d))}

    pool = pd.concat(chunks, ignore_index=True)
    resid = pool["actual"] - pool["pred"]
    out = {
        "test_seasons": [int(s) for s in pool["season"].unique()],
        "n_test_total": int(len(pool)),
        "mae_model": round(float(mean_absolute_error(pool.actual, pool.pred)), 2),
        "mae_rating_baseline": round(float(mean_absolute_error(pool.actual, pool.rating)), 2),
        "sigma": round(float(resid.std()), 2),
        "bias": round(float(resid.mean()), 2),
        "per_season": per_season,
    }
    if market:
        out["mae_market_baseline"] = round(float(mean_absolute_error(pool.actual, pool.line)), 2)
        rate, n_dec = _ats(pool)
        out["ats_rate"] = rate
        out["ats_n"] = n_dec
        out["ats_stderr"] = round(float(100 * (0.25 / max(n_dec, 1)) ** 0.5), 2)
        out["beats_market"] = bool(out["mae_model"] < out["mae_market_baseline"])
        shrink = _best_shrink(pool.pred.values, pool.line.values, pool.actual.values)
        out["shrink_raw"] = round(float(shrink), 2)
        if n_dec < MIN_SHRINK_GAMES:
            log.warning("%s/mkt: only %d decided games - clamping shrink to the default",
                        kind, n_dec)
            shrink = min(shrink, config.DEFAULT_SHRINK)
        out["shrink"] = round(float(min(shrink, SHRINK_CAP)), 2)
        out["ats_by_disagreement"] = _ats_by_disagreement(pool)
        out["break_even_pct"] = round(config.BREAK_EVEN, 2)
        out["venue"] = config.VENUE
        out["market_softness"] = _softness(pool, d)
        # Correct across everything reported for this model at once. A reader scans the
        # softness segments and the disagreement buckets together looking for something
        # bettable, so they are one family of comparisons, not two.
        out["significance"] = _apply_multiple_comparisons(
            out["ats_by_disagreement"], *out["market_softness"].values())
    return out


def _decided(pool: pd.DataFrame) -> pd.DataFrame:
    p = pool.dropna(subset=["line"]).copy()
    p = p[p["actual"] != p["line"]].copy()
    p["right"] = np.where(p["pred"] > p["line"], p["actual"] > p["line"], p["actual"] < p["line"])
    p["disagree"] = (p["pred"] - p["line"]).abs()
    return p


def _roi(rate: float, price: float = 0.50) -> float:
    """Return per unit risked at the configured venue."""
    coef = config.FEE_COEF.get(config.VENUE, 0.07)
    if coef is None:
        return rate * (100 / 110) - (1 - rate)
    fee = coef * price * (1 - price)
    cost = price + fee
    return (rate * (1 - cost) - (1 - rate) * cost) / cost


MIN_SEASON_ROWS = 20        # below this a season's cover rate is not worth printing


def _seg(b: pd.DataFrame, label: str, min_n: int = MIN_SEGMENT) -> dict | None:
    """One segment's record, with the two things needed to judge whether it is real.

    `vs_break_even_se` on its own is what makes these tables dangerous: report twenty of them
    and one will clear two standard errors by chance roughly half the time. So each segment
    also carries how it behaved season by season - a real edge shows up in most of them, and
    a fluke is two bad years and four ordinary ones - and `_apply_multiple_comparisons` later
    stamps every segment with whether it survives correction for how many were looked at.
    """
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


def _softness(pool: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Where, if anywhere, is the NFL market soft?

    The college version cuts on things that proxy for how carefully a game is priced: how many
    books posted, conference tier, whether the line moved. Two of those do not exist here -
    nflverse publishes one closing number with no opening line and no book count - so the
    segments are re-cut on the axes that are actually NFL-specific:

    * **TV window.** Nine simultaneous 1pm kickoffs split the market's attention; Sunday Night
      Football does not. This is the closest NFL analogue to college's P4/G5 split.
    * **Rest.** Short weeks and byes are extreme, scheduled and known months ahead, which makes
      them the most-modelled situations in the sport - and therefore the least likely to be
      soft, which is worth confirming rather than assuming.
    * **Weather.** Wind is the largest single weather effect on totals and the hardest for a
      closing number to have priced days in advance.
    * **Divisional games**, **favourite size**, **week of season**, and **playoffs** - the last
      being a genuinely different market, with two weeks of buildup on one game.

    Every row reports `vs_break_even_se`. Treat anything under +2 as unproven, and remember you
    are looking at ~20 segments, so the best one landing at +2 is roughly what chance produces.
    """
    p = _decided(pool)
    if p.empty:
        return {}
    # NB: no "season" here. The pool already carries it, and merging it again from the
    # feature frame collides into season_x/season_y - which silently emptied the per-season
    # stability that these segments exist to be judged on.
    cols = ["game_id", "week", "week_frac", "playoff_round", "div_game", "is_primetime",
            "is_indoor", "wind", "total_line", "spread_home", "rest_diff", "h_short_week",
            "a_short_week", "h_bye", "a_bye", "neutral_site", "season_type"]
    have = [c for c in cols if c in meta.columns]
    p = p.merge(meta[have].astype({"game_id": str}), on="game_id", how="left")
    out: dict[str, list] = {}

    def add(key, rows):
        rows = [r for r in rows if r]
        if rows:
            out[key] = rows

    if "is_primetime" in p:
        add("by_tv_window", [
            _seg(p[p.is_primetime == 0], "Sunday afternoon (split attention)"),
            _seg(p[p.is_primetime == 1], "standalone national game"),
        ])

    if "playoff_round" in p:
        add("by_stage", [
            _seg(p[p.playoff_round == 0], "regular season"),
            _seg(p[p.playoff_round > 0], "playoffs", min_n=60),
        ])

    add("by_week", [
        _seg(p[p.week <= 4], "weeks 1-4"),
        _seg(p[(p.week >= 5) & (p.week <= 9)], "weeks 5-9"),
        _seg(p[(p.week >= 10) & (p.week <= 14)], "weeks 10-14"),
        _seg(p[p.week >= 15], "weeks 15+"),
    ])

    if "div_game" in p:
        add("by_divisional", [
            _seg(p[p.div_game == 1], "divisional"),
            _seg(p[p.div_game == 0], "non-divisional"),
        ])

    if "wind" in p and "is_indoor" in p:
        add("by_conditions", [
            _seg(p[p.is_indoor == 1], "indoors"),
            _seg(p[(p.is_indoor == 0) & (p.wind < 10)], "outdoors, calm (<10mph)"),
            _seg(p[(p.is_indoor == 0) & (p.wind >= 15)], "outdoors, windy (15mph+)"),
        ])

    if "rest_diff" in p:
        rd = p.rest_diff
        add("by_rest", [
            _seg(p[rd.abs() <= 1], "even rest"),
            _seg(p[rd >= 3], "home better rested"),
            _seg(p[rd <= -3], "away better rested"),
            _seg(p[(p.get("h_short_week", 0) == 1) | (p.get("a_short_week", 0) == 1)],
                 "a side on a short week"),
        ])

    if "spread_home" in p:
        fav = p.spread_home.abs()
        add("by_spread_size", [
            _seg(p[fav <= 3], "pick'em (<=3)"),
            _seg(p[(fav > 3) & (fav <= 7)], "3.5-7"),
            _seg(p[(fav > 7) & (fav <= 10)], "7.5-10"),
            _seg(p[fav > 10], "10+"),
        ])

    # The interesting cross: our biggest disagreements inside the least-watched window.
    if "is_primetime" in p:
        soft = p.is_primetime == 0
        add("soft_and_loud", [
            _seg(p[soft & (p.disagree >= 3)], "afternoon game & disagree 3+", min_n=100),
            _seg(p[~soft & (p.disagree >= 3)], "national game & disagree 3+", min_n=60),
        ])
    return out


FAMILY_ALPHA = 0.05


def _apply_multiple_comparisons(*groups) -> dict:
    """Stamp every reported segment with whether it survives the number of looks taken.

    These tables exist to be scanned for something to bet, which is exactly the situation
    where an uncorrected z-score misleads. Across k independent segments the chance that at
    least one clears |z| >= 2 is 1 - 0.954**k: about 60% at k=20. The Sidak correction turns
    the family-wise error rate back into the 5% a reader assumes they are getting.

    Every segment and disagreement bucket for a model counts as one look, because they are
    read together by someone looking for an edge. Anything that fails the corrected bar is
    marked `significant: false` - not deleted, because a suggestive segment is still worth
    watching, just not worth sizing a bet on.
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
    survivors = [r["segment"] if "segment" in r else r.get("disagreement") for r in segs
                 if r["significant"]]
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
    """How far from the closing line toward the model should we move? Minimises pooled MAE."""
    best, best_mae = 0.0, np.inf
    for w in np.arange(0, 1.01, 0.05):
        mae = mean_absolute_error(actual, line + w * (pred - line))
        if mae < best_mae:
            best, best_mae = float(w), mae
    return best


def _ats_by_disagreement(pool: pd.DataFrame) -> list[dict]:
    """Does the model get *better* when it disagrees with the line the most?

    Buckets pooled out-of-sample games by |model - line| (the RAW model output, before any
    shrink) and reports cover rate and ROI at the configured venue. If the high-disagreement
    buckets clear break-even by more than about two standard errors on a real sample, there is
    an edge in a subset. If they don't, larger disagreements are just larger errors - which is
    the more common finding, and against an NFL closing line it is the expected one.
    """
    decided = _decided(pool)
    if decided.empty:
        return []
    out = []
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 999)]:
        b = decided[(decided["disagree"] >= lo) & (decided["disagree"] < hi)]
        # Same reporting as a softness segment - per-season stability included - because a
        # bucket is read the same way and is just as capable of being one lucky slice.
        row = _seg(b, f"{lo}-{hi if hi < 999 else '+'}", min_n=MIN_BUCKET)
        if row:
            row["disagreement"] = row.pop("segment")
            out.append(row)
    return out


def _ats(chunk: pd.DataFrame) -> tuple[float | None, int]:
    decided = chunk["actual"] != chunk["line"]
    if not decided.any():
        return None, 0
    c = chunk[decided]
    right = np.where(c["pred"] > c["line"], c["actual"] > c["line"], c["actual"] < c["line"])
    return round(float(100 * right.mean()), 1), int(len(c))


def evaluate(df, kind, market, params=None):
    return walk_forward(df, kind, market, params)


def search(df, kind, market, n_iter=15):
    rng = np.random.default_rng(0)
    space = {"n_estimators": [400, 600, 1000], "max_depth": [2, 3, 4],
             "learning_rate": [0.01, 0.02, 0.04], "subsample": [0.7, 0.8, 1.0],
             "colsample_bytree": [0.6, 0.7, 0.9], "min_child_weight": [10, 15, 25]}
    best, best_mae = None, np.inf
    for _ in range(n_iter):
        p = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        r = evaluate(df, kind, market, p)
        mae = r.get("mae_model", np.inf)
        if mae < best_mae:
            best, best_mae = p, mae
    log.info("%s/%s best params %s (MAE %.2f)", kind, "mkt" if market else "base", best, best_mae)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--search", action="store_true")
    args = ap.parse_args(argv)

    df = assemble(fetch=not args.no_fetch)
    meta = {"trained_at": datetime.utcnow().isoformat(timespec="seconds"),
            "sport": "nfl",
            "base_features": BASE_FEATURES, "market_features": MARKET_FEATURES,
            "n_rows": int(len(df)), "seasons": [int(s) for s in sorted(df.season.unique())],
            "eval": {}, "models": [], "sigma": {}, "shrink": {}}
    backend = "xgboost"

    for kind in ("total", "margin"):
        for market in (False, True):
            name = f"{kind}_{'market' if market else 'nomarket'}"
            params = search(df, kind, market) if args.search else None
            res = walk_forward(df, kind, market, params)
            meta["eval"][name] = res
            if res.get("skipped"):
                log.warning("%s: not enough data yet (%s rows) - skipping", name, res.get("n"))
                continue
            log.info("%s: mae %.2f vs market %.2f, ats %s, shrink %s", name,
                     res["mae_model"], res.get("mae_market_baseline", float("nan")),
                     res.get("ats_rate"), res.get("shrink"))
            feats = BASE_FEATURES + (MARKET_FEATURES if market else [])
            d = df.dropna(subset=[TARGETS[kind]])
            if market:
                d = d[d["total_line" if kind == "total" else "spread_home"].notna()]
            model, backend = _fit(d, feats, TARGETS[kind], params)
            _save(model, backend, name)
            meta["models"].append(name)
            meta["sigma"][name] = res["sigma"]
            meta["shrink"][name] = res.get("shrink", config.DEFAULT_SHRINK)
            if market and not res.get("beats_market", False):
                log.warning("%s does NOT beat the closing line (%.2f vs %.2f) - "
                            "treat its picks as unproven", name,
                            res["mae_model"], res["mae_market_baseline"])
            if hasattr(model, "feature_importances_"):
                meta.setdefault("top_features", {})[name] = dict(
                    sorted(zip(feats, map(float, model.feature_importances_)),
                           key=lambda kv: -kv[1])[:12])

    meta["backend"] = backend
    config.ensure_dirs()
    (config.MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("saved %d models to %s", len(meta["models"]), config.MODEL_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
