"""Train the college-football totals and spread models.

    python -m cfb.train                # fetch data, train, evaluate, save
    python -m cfb.train --no-fetch     # use the local cache (offline / CI test)
    python -m cfb.train --search       # small randomised hyper-parameter search

Four models are fitted:

    total_nomarket / margin_nomarket   - features only (used when no line is available yet)
    total_market   / margin_market     - features + the line (used when we have odds)

Evaluation is a season holdout (train on seasons <= N-1, test on N) reported against three
baselines. The output you care about is in data/models/meta.json:

    mae_model            model error
    mae_rating_baseline  the raw Elo/efficiency expectation, no ML
    mae_market_baseline  the closing line itself  <- if you don't beat this, you have no edge
    ats_rate             how often the model's side covered, on the holdout season

Also calibrated here:
    sigma   - residual std, used to turn a point prediction into a probability
    shrink  - how far to move from the market toward the model (fitted, not guessed)
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
from .sources import cfbd

log = logging.getLogger("cfb.train")

DEFAULTS = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=8, reg_lambda=3.0, random_state=42)

TARGETS = {"total": "total_points", "margin": "home_margin"}

# Hard ceiling on how far we'll move off the closing line, no matter what the fit says.
# The market is the single best predictor available; fully overriding it is almost always
# overfitting, not insight.
SHRINK_CAP = float(os.environ.get("DEGEN_SHRINK_CAP", "0.6"))


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
        games = cfbd.update_games()
        cfbd.update_lines()
        cfbd.update_sp()
        cfbd.update_returning()
    else:
        games = cfbd.load_games()
    if games.empty:
        raise SystemExit("no games cached - run once with CFBD_API_KEY set")
    train_rows, _, _ = build(games, lines=cfbd.load_lines(), sp=cfbd.load_sp(),
                             returning=cfbd.load_returning())
    log.info("%d completed games, seasons %s, %d with a line",
             len(train_rows), sorted(train_rows["season"].unique()),
             int(train_rows["total_line"].notna().sum()))
    return train_rows


def complete_seasons(df: pd.DataFrame) -> list[int]:
    """Seasons that have finished. The in-progress season must never be the holdout: it is a
    handful of unrepresentative early games and calibrating on it produces garbage."""
    today = config.today_et()
    return sorted(int(s) for s in df["season"].unique() if config.season_end(int(s)) < today)


def walk_forward(df: pd.DataFrame, kind: str, market: bool, params=None,
                 n_seasons: int = 4) -> dict:
    """Train on everything before season S, predict S, for each of the last n_seasons
    complete seasons; pool the out-of-sample predictions.

    One season is not enough to calibrate on - a good or bad year swings MAE and, worse, the
    shrink weight. Pooling several thousand out-of-sample games makes both stable.
    """
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
        if len(tr) < 500 or len(te) < 100:
            continue
        model, _ = _fit(tr, feats, target, params)
        pred = model.predict(te[feats])
        line = (te[line_col] if kind == "total" else -te[line_col]).values if market else np.full(len(te), np.nan)
        chunk = pd.DataFrame({"season": s, "pred": pred, "actual": te[target].values,
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
        # 1 s.e. on a coin flip, so you can see whether ats_rate means anything
        out["ats_stderr"] = round(float(100 * (0.25 / max(n_dec, 1)) ** 0.5), 2)
        out["beats_market"] = bool(out["mae_model"] < out["mae_market_baseline"])
        shrink = _best_shrink(pool.pred.values, pool.line.values, pool.actual.values)
        # Never let a thin sample talk us into abandoning the line.
        if n_dec < 500:
            log.warning("%s/%s: only %d decided games - clamping shrink to the default",
                        kind, "mkt", n_dec)
            shrink = min(shrink, config.DEFAULT_SHRINK)
        out["shrink"] = round(float(min(shrink, SHRINK_CAP)), 2)
        out["shrink_raw"] = round(float(_best_shrink(pool.pred.values, pool.line.values,
                                                     pool.actual.values)), 2)
    return out


def _best_shrink(pred, line, actual) -> float:
    """How far from the closing line toward the model should we move? Minimises pooled MAE."""
    best, best_mae = 0.0, np.inf
    for w in np.arange(0, 1.01, 0.05):
        mae = mean_absolute_error(actual, line + w * (pred - line))
        if mae < best_mae:
            best, best_mae = float(w), mae
    return best


def _ats(chunk: pd.DataFrame) -> tuple[float | None, int]:
    decided = chunk["actual"] != chunk["line"]
    if not decided.any():
        return None, 0
    c = chunk[decided]
    right = np.where(c["pred"] > c["line"], c["actual"] > c["line"], c["actual"] < c["line"])
    return round(float(100 * right.mean()), 1), int(len(c))


# Backwards-compatible alias; the search routine scores with the same walk-forward.
def evaluate(df, kind, market, params=None):
    return walk_forward(df, kind, market, params)


def search(df, kind, market, n_iter=15):
    rng = np.random.default_rng(0)
    space = {"n_estimators": [300, 600, 1000], "max_depth": [3, 4, 5, 6],
             "learning_rate": [0.01, 0.02, 0.05], "subsample": [0.7, 0.8, 1.0],
             "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [5, 10, 20]}
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
            log.info("%s: %s", name, res)
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
                           key=lambda kv: -kv[1])[:10])

    meta["backend"] = backend
    config.ensure_dirs()
    (config.MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("saved %d models to %s", len(meta["models"]), config.MODEL_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
