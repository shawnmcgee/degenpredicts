"""The model bake-off: try every candidate on seasons it never saw, keep the best per market.

    python -m ncaab.train                 # refresh ESPN finals, run the bake-off, write meta.json
    python -m ncaab.train --no-fetch      # use the committed games cache (offline / CI)

**What is compared.** Every candidate in :mod:`ncaab.models` - ratings only, ridge, Huber,
two flavours of gradient boosting, a random forest, a neural net, a blend and a stack - for
each of four jobs:

    margin_market / total_market       the published number: the line plus the model's view of
                                       how wrong the line is
    margin_nomarket / total_nomarket   used only when a game has no line on one market yet

**How it is scored: walk-forward.** For each of the last ``DEGEN_NCAAB_WALK_SEASONS`` complete
seasons S, every candidate is fitted on all seasons before S and predicts S. Nothing is ever
scored on a season it was fitted on - including the stack, whose weights for season S come only
from the seasons before S. The ratings feeding every model are leak-free by construction.

**How the winner is chosen.** By mean absolute error against the final score, pooled over every
scored game. Not by against-the-spread hit rate: over ~4,000 lined games a season the hit rate
has a standard error near 0.8 points of percentage, so most gaps between models on that measure
are noise, while MAE differences are paired game-by-game and far more stable. Both are reported.

The rule (``DEGEN_NCAAB_SELECT``, default ``1se``) is the standard one-standard-error rule: find
the lowest MAE, then take the SIMPLEST candidate whose paired difference from it is under one
standard error. A neural net that beats ridge by 0.01 points on 20,000 games has not beaten
ridge; it has been lucky in a way ridge would be lucky next season. ``lowest`` takes the raw
leader instead.

**What is saved** - ``data/ncaab/models/meta.json``: the full leaderboard per job, the chosen
model, its fitted shrink toward the line and its error spread, the stack's weights, and the
hit rate by size of disagreement. No fitted model is committed: ``predict.py`` refits the chosen
specification on every game through last night each run, which keeps multi-megabyte forests
out of the repo and the models current on the season in progress.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, models
from .features import BASE_FEATURES, MARKET_FEATURES, TARGETS, build, d1_rows, market_line
from .sources import espn

log = logging.getLogger("ncaab.train")

JOBS = [("margin", True), ("total", True), ("margin", False), ("total", False)]
MIN_TRAIN = 1500
MIN_TEST = 300
GAP_BUCKETS = {"margin": [(0, 1), (1, 2), (2, 3), (3, 5), (5, 99)],
               "total": [(0, 1.5), (1.5, 3), (3, 4.5), (4.5, 7), (7, 99)]}


def job_name(kind: str, market: bool) -> str:
    return f"{kind}_{'market' if market else 'nomarket'}"


def features_for(market: bool) -> list[str]:
    return BASE_FEATURES + (MARKET_FEATURES if market else [])


def candidates(market: bool) -> list[str]:
    names = ([] if market else ["ratings"]) + [b for b in models.BASES if models.available(b)]
    names = [n for n in names if n not in config.SKIP_MODELS]
    combo = [c for c in ("blend", "stack") if c not in config.SKIP_MODELS]
    return names + combo


def assemble(fetch: bool = True) -> pd.DataFrame:
    games = espn.update() if fetch else espn.load()
    if games.empty:
        raise SystemExit("no games in data/ncaab/games.csv - run once with network access")
    train, _, _ = build(games)
    D = d1_rows(train.dropna(subset=list(TARGETS.values())))
    log.info("%d D1-vs-D1 training rows, seasons %s, %d with a closing spread, %d with a total",
             len(D), sorted(int(s) for s in D["season"].unique()),
             int(D["spread_home"].notna().sum()), int(D["total_line"].notna().sum()))
    return D


def test_seasons(D: pd.DataFrame, n: int | None = None) -> list[int]:
    """The last ``n`` complete seasons that have at least one season before them to train on."""
    n = config.WALK_SEASONS if n is None else n
    current = config.season_of(config.today_et())
    seasons = sorted(int(s) for s in D["season"].unique() if int(s) < current)
    return seasons[1:][-n:] if len(seasons) > 1 else []


# ---------------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------------
def walk_forward(D: pd.DataFrame, kind: str, market: bool, seasons: list[int]) -> tuple[pd.DataFrame, dict]:
    """Out-of-sample predictions from every candidate, in the target's own units.

    Returns the predictions (one row per scored game, a ``pred_<name>`` column per candidate)
    and the stack weights used for each season."""
    target, feats = TARGETS[kind], features_for(market)
    pool = D[market_line(D, kind).notna()] if market else D
    names = candidates(market)
    bases = [n for n in names if n in models.BASES]
    frames, used = [], {}
    for S in seasons:
        tr, te = pool[pool["season"] < S], pool[pool["season"] == S]
        if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
            log.info("%s %d: skipped (%d train, %d test rows)", job_name(kind, market), S,
                     len(tr), len(te))
            continue
        line_tr, line_te = market_line(tr, kind), market_line(te, kind)
        y_tr = tr[target].to_numpy(float) - (line_tr.to_numpy(float) if market else 0.0)
        offset = line_te.to_numpy(float) if market else 0.0
        out = pd.DataFrame({"game_id": te["game_id"].values, "season": S,
                            "actual": te[target].to_numpy(float),
                            "line": line_te.to_numpy(float)})
        for b in bases:
            t0 = time.time()
            out[f"pred_{b}"] = models.fit_predict(b, tr[feats], y_tr, te[feats], market) + offset
            log.info("%s %d %-6s MAE %.3f  (%.0fs)", job_name(kind, market), S, b,
                     np.mean(np.abs(out[f"pred_{b}"] - out["actual"])), time.time() - t0)
        if "ratings" in names:
            out["pred_ratings"] = te[f"exp_{kind}"].to_numpy(float)
        members = [m for m in models.blend_members() if f"pred_{m}" in out]
        if "blend" in names and members:
            out["pred_blend"] = out[[f"pred_{m}" for m in members]].mean(axis=1)
        if "stack" in names and bases:
            w = _stack_weights(frames, bases, market) if frames else None
            if w is None:   # no earlier out-of-sample season yet: the stack starts as the blend
                out["pred_stack"] = out["pred_blend"] if "pred_blend" in out else \
                    out[[f"pred_{b}" for b in bases]].mean(axis=1)
                used[S] = "blend"
            else:
                P = out[[f"pred_{b}" for b in bases]].to_numpy(float)
                if market:          # the weights were fitted on line errors, so apply them there
                    P = P - offset[:, None]
                out["pred_stack"] = P @ w + offset
                used[S] = {b: round(float(x), 3) for b, x in zip(bases, w)}
        frames.append(out)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), used


def _stack_weights(frames: list[pd.DataFrame], bases: list[str], market: bool) -> np.ndarray | None:
    prev = pd.concat(frames, ignore_index=True)
    cols = [f"pred_{b}" for b in bases]
    if not all(c in prev for c in cols):
        return None
    P = prev[cols].to_numpy(float)
    y = prev["actual"].to_numpy(float)
    if market:
        line = prev["line"].to_numpy(float)
        P, y = P - line[:, None], y - line
    return models.nnls_weights(P, y)


# ---------------------------------------------------------------------------------
# the leaderboard
# ---------------------------------------------------------------------------------
def _side(pred, line, actual) -> tuple[float | None, int, float | None]:
    """How often the model's side of the line won, pushes and no-opinion games left out."""
    ok = np.isfinite(line) & np.isfinite(pred) & (actual != line) & (pred != line)
    if ok.sum() == 0:
        return None, 0, None
    right = np.sign(pred[ok] - line[ok]) == np.sign(actual[ok] - line[ok])
    p, n = float(right.mean()), int(ok.sum())
    return round(100 * p, 2), n, round(100 * np.sqrt(p * (1 - p) / n), 2)


def leaderboard(oos: pd.DataFrame, market: bool) -> list[dict]:
    names = [c[5:] for c in oos.columns if c.startswith("pred_")]
    y = oos["actual"].to_numpy(float)
    line = oos["line"].to_numpy(float)
    err = {n: np.abs(oos[f"pred_{n}"].to_numpy(float) - y) for n in names}
    best = min(names, key=lambda n: err[n].mean())
    rows = []
    for n in names:
        p = oos[f"pred_{n}"].to_numpy(float)
        d = err[n] - err[best]
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        hit, hit_n, hit_se = _side(p, line, y)
        rows.append({
            "model": n, "label": models.LABELS.get(n, n), "n": int(len(y)),
            "mae": round(float(err[n].mean()), 3),
            "rmse": round(float(np.sqrt(np.mean((p - y) ** 2))), 3),
            "bias": round(float(np.mean(p - y)), 3),
            "hit_rate": hit, "hit_n": hit_n, "hit_se": hit_se,
            "vs_best": round(float(d.mean()), 3), "vs_best_se": round(se, 4),
            "z": round(float(d.mean() / se), 2) if se > 0 else 0.0,
            "per_season": {str(int(s)): round(float(np.mean(err[n][oos["season"].values == s])), 3)
                           for s in sorted(oos["season"].unique())},
        })
    rows.sort(key=lambda r: r["mae"])
    if market and np.isfinite(line).all():
        rows.append({"model": "line", "label": models.LABELS["line"], "n": int(len(y)),
                     "mae": round(float(np.mean(np.abs(line - y))), 3),
                     "rmse": round(float(np.sqrt(np.mean((line - y) ** 2))), 3),
                     "bias": round(float(np.mean(line - y)), 3), "reference": True})
    return rows


def select(board: list[dict], rule: str | None = None) -> tuple[str, str]:
    """(chosen, best). ``1se``: the simplest model within one standard error of the best."""
    rule = (rule or config.SELECT).lower()
    cands = [r for r in board if not r.get("reference")]
    best = min(cands, key=lambda r: r["mae"])["model"]
    if rule == "lowest":
        return best, best
    by = {r["model"]: r for r in cands}
    for name in models.SIMPLICITY:
        if name in by and by[name]["z"] < 1.0:
            return name, best
    return best, best


def best_shrink(pred, line, actual) -> float:
    """How far from the line toward the model the published number should sit, by MAE."""
    best, best_mae = 1.0, np.inf
    for w in np.arange(0.0, 1.0001, 0.05):
        mae = float(np.mean(np.abs(line + w * (pred - line) - actual)))
        if mae < best_mae - 1e-9:
            best, best_mae = float(round(w, 2)), mae
    return best


def by_gap(pred, line, actual, edges) -> list[dict]:
    """The model's side record by how far it disagreed with the line - the evidence for the
    board's 'biggest gaps first' sort, or against it."""
    out = []
    gap = np.abs(pred - line)
    for lo, hi in edges:
        m = (gap >= lo) & (gap < hi)
        hit, n, se = _side(pred[m], line[m], actual[m])
        if n:
            out.append({"gap": f"{lo:g}-{hi:g}" if hi < 99 else f"{lo:g}+", "n": n,
                        "hit_rate": hit, "se": se})
    return out


def run_job(D: pd.DataFrame, kind: str, market: bool, seasons: list[int]) -> dict:
    name = job_name(kind, market)
    oos, used = walk_forward(D, kind, market, seasons)
    if oos.empty:
        log.warning("%s: not enough history for a walk-forward - skipped", name)
        return {"skipped": True, "reason": "not enough seasons with data"}
    board = leaderboard(oos, market)
    chosen, best = select(board)
    y = oos["actual"].to_numpy(float)
    line = oos["line"].to_numpy(float)
    pred = oos[f"pred_{chosen}"].to_numpy(float)
    res = {"chosen": chosen, "best": best, "rule": config.SELECT,
           "test_seasons": sorted(int(s) for s in oos["season"].unique()), "n": int(len(oos)),
           "leaderboard": board, "stack_by_season": {str(k): v for k, v in used.items()}}
    bases = [b for b in models.BASES if f"pred_{b}" in oos]
    if bases and "stack" in candidates(market):
        w = _stack_weights([oos], bases, market)
        res["stack_weights"] = {b: round(float(x), 4) for b, x in zip(bases, w)}
    if market:
        w = best_shrink(pred, line, y)
        pub = line + w * (pred - line)
        hit, hit_n, hit_se = _side(pred, line, y)
        res.update({"shrink": w, "sigma": round(float(np.std(y - pub)), 3),
                    "line_mae": round(float(np.mean(np.abs(line - y))), 3),
                    "hit_rate": hit, "hit_n": hit_n, "hit_se": hit_se,
                    "beats_line": bool(np.mean(np.abs(pub - y)) < np.mean(np.abs(line - y))),
                    "by_gap": by_gap(pred, line, y, GAP_BUCKETS[kind])})
    else:
        hit, hit_n, hit_se = _side(pred, line, y)
        res.update({"sigma": round(float(np.std(y - pred)), 3), "hit_rate": hit,
                    "hit_n": hit_n, "hit_se": hit_se})
    top = board[0]
    log.info("%s: chose %s (best %s, MAE %.3f; %s MAE %.3f)", name, chosen, best, top["mae"],
             chosen, next(r["mae"] for r in board if r["model"] == chosen))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args(argv)

    D = assemble(fetch=not args.no_fetch)
    seasons = test_seasons(D)
    log.info("walk-forward over %s; candidates: %s", seasons, ", ".join(candidates(True)))
    meta = {"sport": "ncaab", "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seasons": sorted(int(s) for s in D["season"].unique()), "n_rows": int(len(D)),
            "features": {"base": BASE_FEATURES, "market": MARKET_FEATURES},
            "select_rule": config.SELECT, "jobs": {}}
    for kind, market in JOBS:
        meta["jobs"][job_name(kind, market)] = run_job(D, kind, market, seasons)
    config.ensure_dirs()
    (config.MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("wrote %s", config.MODEL_DIR / "meta.json")
    return meta


def load_meta() -> dict:
    path = config.MODEL_DIR / "meta.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
