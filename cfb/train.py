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


def evaluate(df: pd.DataFrame, kind: str, market: bool, params=None) -> dict:
    target = TARGETS[kind]
    feats = BASE_FEATURES + (MARKET_FEATURES if market else [])
    d = df.dropna(subset=[target])
    if market:
        col = "total_line" if kind == "total" else "spread_home"
        d = d[d[col].notna()]
    seasons = sorted(d["season"].unique())
    if len(seasons) < 2 or len(d) < 200:
        return {"skipped": True, "n": int(len(d))}
    test = seasons[-1]
    tr, te = d[d.season < test], d[d.season == test]
    if len(tr) < 100 or len(te) < 50:
        return {"skipped": True, "n": int(len(d))}

    model, _ = _fit(tr, feats, target, params)
    pred = model.predict(te[feats])
    resid = te[target].values - pred
    rating_col = "exp_total" if kind == "total" else "exp_margin"
    out = {
        "test_season": int(test), "n_train": int(len(tr)), "n_test": int(len(te)),
        "mae_model": round(float(mean_absolute_error(te[target], pred)), 2),
        "mae_rating_baseline": round(float(mean_absolute_error(te[target], te[rating_col])), 2),
        "sigma": round(float(np.std(resid)), 2),
        "bias": round(float(np.mean(resid)), 2),
    }
    if market:
        line = te["total_line"] if kind == "total" else -te["spread_home"]
        out["mae_market_baseline"] = round(float(mean_absolute_error(te[target], line)), 2)
        # how often would we have been right picking the model's side?
        edge = pred - line.values
        side_right = np.where(edge > 0, te[target].values > line.values, te[target].values < line.values)
        decided = te[target].values != line.values
        out["ats_rate"] = round(float(100 * side_right[decided].mean()), 1) if decided.any() else None
        out["ats_n"] = int(decided.sum())
        out["shrink"] = round(float(_best_shrink(pred, line.values, te[target].values)), 2)
    return out


def _best_shrink(pred, line, actual) -> float:
    """How far toward the model should we move from the line? Minimises MAE on the holdout."""
    best, best_mae = 0.0, np.inf
    for w in np.arange(0, 1.01, 0.05):
        blend = line + w * (pred - line)
        mae = mean_absolute_error(actual, blend)
        if mae < best_mae:
            best, best_mae = w, mae
    return best


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
            res = evaluate(df, kind, market, params)
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
