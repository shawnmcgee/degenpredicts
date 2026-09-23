"""Train the NHL models and write the report that says whether to bet any of it.

    python -m nhl.train                # refresh data, fit, evaluate, save
    python -m nhl.train --no-fetch     # use the committed cache (offline / CI)

Three things are fitted, in order:

1. **The scoreline model** (:mod:`nhl.scoreline`) - pull times, the extra-attacker and
   empty-net rates, the late-tie damping and the overtime split - by maximum likelihood on the
   final scores of the last four complete seasons. Refitted every run because goalie-pull
   behaviour keeps changing.
2. **Two Poisson models** for goals scored on a goalie, one side at a time
   (:mod:`nhl.glm`): ``nomarket`` for games without a price, and ``market``, which takes the
   market's implied goals as its offset and learns only where the market errs.
3. **Two shrink weights**, fitted on the pooled walk-forward: how far the published home/away
   SPLIT moves from the market toward the model, and how far the TOTAL does. They are separate
   because the backtest says they deserve different amounts of trust.

**Goalies.** Every model is fitted on the starters games actually had - a confirmed starter is
exactly that - and every goalie carries his own rating. The evaluation predicts each test game
twice: with the book's own guess about who starts (what the board knows before the day's news,
and the view every ROI below is measured on) and with the actual starters, to measure what the
news is worth (``eval.goalies``).

Everything is evaluated walk-forward - train on every season before S, predict S - over the
last six complete seasons. The numbers that decide whether to bet, in data/nhl/models/meta.json:

    eval.moneyline.log_loss_edge   the proper scoring rule against the de-vigged market
    eval.spread.roi_by_ev          puck-line ROI at the posted price, by expected value
    eval.total.roi_by_ev           the same for totals
    eval.*.significance            what |z| a bucket needs given how many were looked at

**What the research behind this found, before any of it was committed** (2012-2025):

* against CLOSING moneylines the market-aware model essentially ties the market: log loss
  -0.0003 better with the shrunk split, ROI +1.5% +/- 1.6% on 5,182 bets - not an edge;
* against NOON prices (2023-26, the market the morning board bets into) every positive-EV
  puck-line side returned +4.6% +/- 2.9% on 932 bets with NOTHING fitted on those seasons
  (see :func:`strict_holdout`), positive in all three - the strongest result on this site, and
  still well short of what a significance test would sign off on;
* on totals it found nothing, against anything. The published total sits close to the market.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .features import BASE_FEATURES, MARKET_FEATURES, build, sides, unstack
from .glm import PoissonGLM, recency_weights
from .odds_math import ev as expected_value
from .scoreline import DEFAULT_THETA, Table, calibration, fit_theta, grids
from .sources import history, nhle, odds

log = logging.getLogger("nhl.train")

THETA_SEASONS = 4
WALK_FORWARD_SEASONS = config._env_int("DEGEN_WALK_SEASONS", 6)
HALF_LIFE = 4.0          # seasons, for the recency weights
L2 = 1.0
MIN_TRAIN_GAMES = 2000
MIN_TEST_GAMES = 400
MIN_SHRINK_GAMES = 1000  # fewer than this and the shrink stays at its default
MIN_BUCKET = 60
MIN_SEGMENT = 150
MIN_SEASON_ROWS = 30
EV_BUCKETS = [(-1.0, 0.0), (0.0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 1.0)]
FAMILY_ALPHA = 0.05


# ---------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------
def _fit_pair(tr: pd.DataFrame, season: int) -> dict:
    w = recency_weights(tr["season"].values, season, HALF_LIFE)
    out = {"nomarket": PoissonGLM(BASE_FEATURES, l2=L2).fit(tr, weights=w)}
    mk = tr["log_mkt"].notna().values
    if mk.sum() >= 2 * MIN_TRAIN_GAMES:
        out["market"] = PoissonGLM(MARKET_FEATURES, offset="log_mkt", l2=L2).fit(
            tr[mk], weights=w[mk])
    return out


def blend(m_lh, m_la, lh, la, split: float, total: float):
    """Move from the market's goals toward the model's, separately for the total and the split.

    Where the market is missing the model's own numbers are returned unchanged.
    """
    m_lh, m_la, lh, la = (np.asarray(x, float) for x in (m_lh, m_la, lh, la))
    tm, tmod = m_lh + m_la, lh + la
    sm, smod = m_lh / tm, lh / tmod
    t = tm + total * (tmod - tm)
    s = sm + split * (smod - sm)
    has = np.isfinite(m_lh) & np.isfinite(m_la)
    return np.where(has, t * s, lh), np.where(has, t * (1 - s), la)


def save_models(models: dict) -> None:
    config.ensure_dirs()
    for name, m in models.items():
        m.save(config.MODEL_DIR / f"{name}.json")


def load_models() -> tuple[dict, dict]:
    meta = json.loads((config.MODEL_DIR / "meta.json").read_text())
    models = {n: PoissonGLM.load(config.MODEL_DIR / f"{n}.json") for n in meta["models"]}
    return models, meta


# ---------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------
def assemble(fetch: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = nhle.update_games() if fetch else nhle.load_games()
    if games.empty:
        raise SystemExit("no games cached - run `python -m nhl.sources.nhle` with network access")
    if fetch:
        odds.consolidate(games)
    return games, history.load_lines()


def complete_seasons(seasons) -> list[int]:
    """Seasons that have finished. The in-progress season must never be the holdout."""
    today = config.today_et()
    return sorted(int(s) for s in set(seasons) if config.season_end(int(s)) < today)


def fit_scoreline(games: pd.DataFrame, before: int | None = None) -> tuple[dict, list[dict]]:
    """Refit the late-game parameters on the last THETA_SEASONS complete regular seasons.

    ``before`` restricts the fit to seasons before it, for the strict holdout.
    """
    rows, _, _ = build(games, first_train_season=config.LOAD_FROM_SEASON + 1, starters="actual")
    seasons = [s for s in complete_seasons(rows["season"].unique()) if before is None or s < before]
    seasons = seasons[-THETA_SEASONS:]
    d = rows[rows["season"].isin(seasons) & (rows["game_type"] == "R")]
    if len(d) < 2000:
        log.warning("only %d games to fit the scoreline on - keeping the defaults", len(d))
        return dict(DEFAULT_THETA), []
    th = fit_theta(d["lam_h"].values, d["lam_a"].values, d["home_goals"].values,
                   d["away_goals"].values)
    th["seasons"] = seasons
    fit_view = {**th, "c_h": th["c_home_fit"], "c_a": th["c_away_fit"]}
    F, R = grids(d["lam_h"].values, d["lam_a"].values, fit_view)
    cal = calibration(F, R, d["home_goals"].values, d["away_goals"].values,
                      d["decided_in"].values)
    log.info("scoreline: fitted on %s (%d games), nll %.4f", seasons, len(d), th["nll"])
    return th, cal


# ---------------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------------
def _predict(models: dict, te: pd.DataFrame, gid, suffix: str = "") -> dict:
    lh, la = unstack(te, models["nomarket"].predict(te), gid)
    out = {f"lh_nomarket{suffix}": lh, f"la_nomarket{suffix}": la}
    if "market" in models:
        mk = te["log_mkt"].notna().values
        pred = np.full(len(te), np.nan)
        if mk.any():
            pred[mk] = models["market"].predict(te[mk])
        out[f"lh_market{suffix}"], out[f"la_market{suffix}"] = unstack(te, pred, gid)
    return out


def walk_forward(D: pd.DataFrame, S: pd.DataFrame, n_seasons: int = WALK_FORWARD_SEASONS,
                 S_eval: pd.DataFrame | None = None, S_known: pd.DataFrame | None = None
                 ) -> pd.DataFrame:
    """Out-of-sample goals for every game in the last n complete seasons, from both models.

    The models are fitted on ``S``. Each test season is predicted from ``S_eval`` (default
    ``S``) - the book's own guess about who starts, which is what the board knows before the
    day's news - and, when given, from ``S_known``, the same games with the starters they
    actually had, into ``*_known`` columns: the difference is what the news is worth.
    """
    S_eval = S if S_eval is None else S_eval
    comp = complete_seasons(D["season"].unique())
    tests = [s for s in comp[-n_seasons:] if s > comp[0]] if comp else []
    chunks = []
    for s in tests:
        tr, te = S[S["season"] < s], S_eval[S_eval["season"] == s]
        if len(tr) < 2 * MIN_TRAIN_GAMES or len(te) < 2 * MIN_TEST_GAMES:
            continue
        models = _fit_pair(tr, s)
        gid = D.loc[D["season"] == s, "game_id"].values
        chunk = {"game_id": gid, "season": s, **_predict(models, te, gid)}
        if S_known is not None:
            chunk.update(_predict(models, S_known[S_known["season"] == s], gid, "_known"))
        chunks.append(pd.DataFrame(chunk))
        log.info("walk-forward %d: %d games", s, len(gid))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).merge(D, on=["game_id", "season"], how="left")


def _logloss(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _markets(pool: pd.DataFrame, lh, la, table: Table) -> pd.DataFrame:
    """Model probabilities for every market line in the pool."""
    out = pd.DataFrame(index=pool.index)
    out["p_home"] = table.p_home(lh, la)
    pl = pool["pl_home"].fillna(-1.5).values
    out["p_pl_home"], out["pl_push"] = table.cover(lh, la, pl)
    tl = pool["total_line"].fillna(6.0).values
    po, pp = table.over(lh, la, tl)
    out["p_over"], out["tot_push"] = po, pp
    out["e_total"] = table.e_total(lh, la)
    return out


def fit_shrinks(pool: pd.DataFrame, table: Table) -> dict:
    """Split shrink by moneyline log loss; total shrink by over/under log loss on PRICED totals."""
    has = pool["lh_market"].notna() & pool["m_lh"].notna()
    p = pool[has]
    mar = (p["home_goals"] - p["away_goals"]).values
    tot = (p["home_goals"] + p["away_goals"]).values
    out = {"n_split": int(len(p))}
    grid = np.round(np.arange(0, config.SHRINK_CAP + 1e-9, 0.05), 2)
    scores = []
    for b in grid:
        lh, la = blend(p["m_lh"], p["m_la"], p["lh_market"], p["la_market"], b, 0.0)
        scores.append(float(_logloss(table.p_home(lh, la), mar > 0).mean()))
    b_raw = float(grid[int(np.argmin(scores))])
    out["split_raw"] = b_raw
    out["split"] = b_raw if len(p) >= MIN_SHRINK_GAMES else config.DEFAULT_SPLIT_SHRINK
    priced = (~p["p_over_assumed"].astype(bool)) & p["total_line"].notna() & \
        (tot != p["total_line"].values)
    q = p[priced.values]
    out["n_total"] = int(len(q))
    if len(q):
        tq = (q["home_goals"] + q["away_goals"]).values
        scores = []
        for a in grid:
            lh, la = blend(q["m_lh"], q["m_la"], q["lh_market"], q["la_market"], out["split"], a)
            o, pu = table.over(lh, la, q["total_line"].values)
            scores.append(float(_logloss(o / np.maximum(1 - pu, 1e-9), tq > q["total_line"].values).mean()))
        a_raw = float(grid[int(np.argmin(scores))])
    else:
        a_raw = config.DEFAULT_TOTAL_SHRINK
    out["total_raw"] = a_raw
    out["total"] = a_raw if len(q) >= MIN_SHRINK_GAMES else config.DEFAULT_TOTAL_SHRINK
    return out


def _seg(b: pd.DataFrame, label: str, min_n: int = MIN_SEGMENT) -> dict | None:
    """One slice of priced bets: ROI with its standard error and per-season stability."""
    if len(b) < min_n:
        return None
    r = b["profit"].to_numpy(float)
    roi, se = float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan
    out = {"segment": label, "n": int(len(b)), "hit_pct": round(100 * float(b["won"].mean()), 1),
           "roi_pct": round(100 * roi, 2), "stderr": round(100 * se, 2) if se == se else None,
           "vs_zero_se": round(roi / se, 2) if se and se == se else None}
    by = b.groupby("season")["profit"].agg(["size", "mean"])
    by = by[by["size"] >= MIN_SEASON_ROWS]
    if len(by):
        out["season_roi_pct"] = {int(s): round(100 * float(m), 1) for s, m in by["mean"].items()}
        out["seasons_measured"] = int(len(by))
        out["seasons_positive"] = int((by["mean"] > 0).sum())
    return out


def _apply_multiple_comparisons(*groups) -> dict:
    """Stamp every reported slice with whether it survives the number of looks taken.

    These tables exist to be scanned for something to bet, which is exactly when an uncorrected
    z-score misleads: across k slices the chance one clears |z| >= 2 by luck is 1 - 0.954**k,
    about 60% at k=20. Sidak turns the family-wise error rate back into the 5% a reader assumes.
    """
    segs = [r for g in groups for r in (g or []) if r]
    if not segs:
        return {}
    from scipy.stats import norm
    k = len(segs)
    crit = float(norm.isf((1 - (1 - FAMILY_ALPHA) ** (1 / k)) / 2))
    for r in segs:
        z = r.get("vs_zero_se")
        r["significant"] = bool(z is not None and abs(z) >= crit)
    survivors = [r.get("segment") or r.get("ev") for r in segs if r["significant"]]
    return {"comparisons": k, "family_alpha": FAMILY_ALPHA, "z_required": round(crit, 2),
            "significant_segments": survivors,
            "verdict": ("no segment survives correction for the number of looks taken"
                        if not survivors else f"{len(survivors)} of {k} segments survive correction")}


def _bets(pool: pd.DataFrame, probs: pd.DataFrame, market: str) -> pd.DataFrame:
    """The bet the board would have made on every PRICED game: the side with the larger EV."""
    mar = (pool["home_goals"] - pool["away_goals"]).values
    tot = (pool["home_goals"] + pool["away_goals"]).values
    if market == "spread":
        a_dec, b_dec = pool["pl_home_dec"].values, pool["pl_away_dec"].values
        p_a, push = probs["p_pl_home"].values, probs["pl_push"].values
        adj = mar + pool["pl_home"].values
        won_a, pushed = adj > 0, adj == 0
        mkt_a = pool["p_pl_home"].values
    else:
        a_dec, b_dec = pool["over_dec"].values, pool["under_dec"].values
        p_a, push = probs["p_over"].values, probs["tot_push"].values
        won_a, pushed = tot > pool["total_line"].values, tot == pool["total_line"].values
        mkt_a = pool["p_over"].values
    p_b = 1 - p_a - push
    ev_a, ev_b = expected_value(p_a, push, a_dec), expected_value(p_b, push, b_dec)
    take_a = ev_a >= ev_b
    evv = np.where(take_a, ev_a, ev_b)
    dec = np.where(take_a, a_dec, b_dec)
    won = np.where(take_a, won_a, ~won_a & ~pushed)
    profit = np.where(pushed, 0.0, np.where(won, dec - 1, -1.0))
    priced = np.isfinite(evv) & np.isfinite(dec)
    b = pd.DataFrame({"season": pool["season"].values, "ev": evv, "won": won, "profit": profit,
                      "pushed": pushed, "edge_pp": 100 * (np.where(take_a, p_a, p_b)
                                                          - np.where(take_a, mkt_a, 1 - mkt_a))})
    for c in ("h_b2b", "a_b2b", "h_games", "a_games", "playoff", "p_home", "g_h_conf", "g_a_conf"):
        if c in pool:
            b[c] = pool[c].values
    return b[priced & ~pushed]


def _report(pool: pd.DataFrame, probs: pd.DataFrame, market: str) -> dict:
    """ROI by EV, softness segments and significance for one market."""
    b = _bets(pool, probs, market)
    if b.empty:
        return {"skipped": True, "reason": "no priced games in the walk-forward"}
    buckets = []
    for lo, hi in EV_BUCKETS:
        row = _seg(b[(b["ev"] >= lo) & (b["ev"] < hi)],
                   f"{lo:+.2f} to {hi:+.2f}" if hi < 1 else f"{lo:+.2f}+", min_n=MIN_BUCKET)
        if row:
            row["ev"] = row.pop("segment")
            # A negative-EV bucket is the board declining the bet. It loses the vig, as it
            # should, and it is shown for that reason - but it is not a candidate bet, so it
            # is kept out of the family the significance correction is taken over.
            row["candidate"] = lo >= 0
            buckets.append(row)
    threshold = config.SPREAD_EV_MIN if market == "spread" else config.TOTAL_EV_MIN
    rules = [r for r in (_seg(b[b["ev"] >= 0], "every positive-EV side", min_n=MIN_BUCKET),
                         _seg(b[b["ev"] >= threshold], f"EV >= {threshold:.2f} (the staking rule)",
                              min_n=MIN_BUCKET)) if r]
    play = b[b["ev"] >= 0]
    soft = {}

    def add(key, rows):
        rows = [r for r in rows if r]
        if rows:
            soft[key] = rows

    if "h_b2b" in play:
        b2b = (play["h_b2b"] == 1) | (play["a_b2b"] == 1)
        add("by_back_to_back", [_seg(play[b2b], "a side on a back-to-back"),
                                _seg(play[~b2b], "no back-to-back")])
    if "h_games" in play:
        early = play[["h_games", "a_games"]].min(axis=1) < 15
        add("by_stage", [_seg(play[early], "first 15 games"), _seg(play[~early], "after 15 games")])
    if "p_home" in play:
        fav = (play["p_home"] - 0.5).abs()
        add("by_favourite", [_seg(play[fav < 0.1], "near even (<60%)"),
                             _seg(play[fav >= 0.1], "clear favourite (60%+)")])
    if "g_h_conf" in play:
        unsure = play[["g_h_conf", "g_a_conf"]].min(axis=1) < 0.6
        add("by_goalie_certainty", [_seg(play[unsure], "a starter in doubt"),
                                    _seg(play[~unsure], "both starters likely")])
    sig = _apply_multiple_comparisons([r for r in buckets if r["candidate"]], rules, *soft.values())
    for r in buckets:
        r.setdefault("significant", False)       # a declined bet is never a finding
    if not sig:
        sig = {"comparisons": 0, "verdict": "no positive-EV side in the backtest - nothing to test"}
    return {"n_priced": int(len(b)), "roi_by_ev": buckets, "rules": rules,
            "market_softness": soft, "significance": sig}


def top_pick_rate(DE: pd.DataFrame, games: pd.DataFrame, seasons) -> float | None:
    """How often the book's own guess from recent starts named the goalie who started."""
    ids = games[["game_id", "home_goalie_id", "away_goalie_id"]].copy()
    ids["game_id"] = ids["game_id"].astype(str)
    d = DE[DE["season"].isin(list(seasons))].merge(ids, on="game_id", how="left")
    d = d.dropna(subset=["home_goalie_id", "away_goalie_id", "g_h_id", "g_a_id"])
    if d.empty:
        return None
    hits = np.concatenate([(d["g_h_id"].astype(float) == d["home_goalie_id"].astype(float)).values,
                           (d["g_a_id"].astype(float) == d["away_goalie_id"].astype(float)).values])
    return round(100 * float(hits.mean()), 1)


def goalie_value(pool: pd.DataFrame, table: Table, shrink: dict) -> dict:
    """What knowing the starter is worth: the same models, the same games, predicted with the
    book's guess and with the actual starters. Positive means the actual starter helped."""
    if "lh_nomarket_known" not in pool:
        return {}
    mar = (pool["home_goals"] - pool["away_goals"]).values
    won = mar > 0
    out = {"n_games": int(len(pool))}
    nm = pool["lh_nomarket"].notna().values & pool["lh_nomarket_known"].notna().values
    guess = _logloss(table.p_home(pool["lh_nomarket"].values[nm], pool["la_nomarket"].values[nm]), won[nm])
    known = _logloss(table.p_home(pool["lh_nomarket_known"].values[nm],
                                  pool["la_nomarket_known"].values[nm]), won[nm])
    out["nomarket_ml_gain"] = round(float(guess.mean() - known.mean()), 5)
    if "lh_market_known" in pool:
        closing = pool["source"].astype(str).str.contains("close").values
        mk = (pool["lh_market"].notna() & pool["lh_market_known"].notna() & pool["m_lh"].notna()
              & pool["p_home"].notna()).values
        views = {}
        for view in ("", "_known"):
            lh, la = blend(pool["m_lh"], pool["m_la"], pool[f"lh_market{view}"],
                           pool[f"la_market{view}"], shrink["split"], shrink["total"])
            views[view] = _logloss(table.p_home(lh, la), won)
        for label, sel in (("close", mk & closing), ("noon", mk & ~closing)):
            if sel.sum() >= MIN_TEST_GAMES:
                out[f"market_ml_gain_{label}"] = round(
                    float(views[""][sel].mean() - views["_known"][sel].mean()), 5)
                out[f"n_{label}"] = int(sel.sum())
    return out


def evaluate(pool: pd.DataFrame, table: Table, shrink: dict) -> dict:
    mk = pool["lh_market"].notna() & pool["m_lh"].notna()
    lh, la = blend(pool["m_lh"], pool["m_la"],
                   pool["lh_market"].fillna(pool["lh_nomarket"]),
                   pool["la_market"].fillna(pool["la_nomarket"]),
                   shrink["split"], shrink["total"])
    probs = _markets(pool, lh, la, table)
    mprobs = _markets(pool, pool["m_lh"].fillna(3.0).values, pool["m_la"].fillna(3.0).values, table)
    mar = (pool["home_goals"] - pool["away_goals"]).values
    tot = (pool["home_goals"] + pool["away_goals"]).values
    seasons = sorted(int(s) for s in pool["season"].unique())
    out = {"test_seasons": seasons, "n_test_total": int(len(pool))}

    # ---- moneyline: the cleanest test of who-beats-whom -------------------------------
    m = mk.values & pool["p_home"].notna().values
    closing = pool["source"].astype(str).str.contains("close").values
    ml = {"n": int(m.sum()),
          "log_loss_model": round(float(_logloss(probs["p_home"].values[m], mar[m] > 0).mean()), 5),
          "log_loss_market": round(float(_logloss(pool["p_home"].values[m], mar[m] > 0).mean()), 5)}
    ml["log_loss_edge"] = round(ml["log_loss_market"] - ml["log_loss_model"], 5)
    ml["beats_market"] = bool(ml["log_loss_edge"] > 0)
    for label, sel in (("closing", m & closing), ("pregame", m & ~closing)):
        if sel.sum() >= MIN_TEST_GAMES:
            ml[f"vs_{label}"] = {
                "n": int(sel.sum()),
                "log_loss_edge": round(float(_logloss(pool["p_home"].values[sel], mar[sel] > 0).mean()
                                             - _logloss(probs["p_home"].values[sel], mar[sel] > 0).mean()), 5)}
    nm = pool["lh_nomarket"].notna().values
    p_nm = table.p_home(pool["lh_nomarket"].values[nm], pool["la_nomarket"].values[nm])
    ml["log_loss_nomarket_model"] = round(float(_logloss(p_nm, mar[nm] > 0).mean()), 5)
    out["moneyline"] = ml

    # ---- puck line --------------------------------------------------------------------
    pl = pool["pl_home"].isin([-1.5, 1.5]).values & mk.values
    cover = mar + pool["pl_home"].values > 0
    sp = {"n": int(pl.sum())}
    if pl.any():
        # A bare puck-line cover rate means nothing - a +1.5 dog covers about 70% of the time
        # and a -1.5 favourite about 35% - so it is reported beside the rate the MARKET's own
        # number expected for the same sides. The gap is the model's contribution.
        took_home = probs["p_pl_home"].values > mprobs["p_pl_home"].values
        right = np.where(took_home, cover, ~cover)[pl]
        expect = np.where(took_home, mprobs["p_pl_home"].values, 1 - mprobs["p_pl_home"].values)[pl]
        sp.update(cover_rate=round(100 * float(right.mean()), 1),
                  market_expected_rate=round(100 * float(expect.mean()), 1), cover_n=int(pl.sum()),
                  cover_stderr=round(100 * float(np.sqrt((expect * (1 - expect)).mean() / pl.sum())), 2))
        pr = pl & pool["p_pl_home"].notna().values
        if pr.sum() >= MIN_TEST_GAMES:
            sp["log_loss_model"] = round(float(_logloss(probs["p_pl_home"].values[pr], cover[pr]).mean()), 5)
            sp["log_loss_market"] = round(float(_logloss(pool["p_pl_home"].values[pr], cover[pr]).mean()), 5)
            sp["log_loss_market_via_scoreline"] = round(
                float(_logloss(mprobs["p_pl_home"].values[pr], cover[pr]).mean()), 5)
            sp["beats_market"] = bool(sp["log_loss_model"] < sp["log_loss_market"])
    sp.update(_report(pool[pl], probs[pl], "spread"))
    out["spread"] = sp

    # ---- totals -----------------------------------------------------------------------
    tl = pool["total_line"].notna().values & mk.values & (tot != pool["total_line"].values)
    over = tot > pool["total_line"].values
    to = {"n": int(tl.sum())}
    if tl.any():
        took_over = probs["p_over"].values > mprobs["p_over"].values
        right = np.where(took_over, over, ~over)[tl]
        mo = mprobs["p_over"].values / np.maximum(1 - mprobs["tot_push"].values, 1e-9)
        expect = np.where(took_over, mo, 1 - mo)[tl]
        to.update(cover_rate=round(100 * float(right.mean()), 1),
                  market_expected_rate=round(100 * float(expect.mean()), 1), cover_n=int(tl.sum()),
                  cover_stderr=round(100 * float(np.sqrt(0.25 / tl.sum())), 2))
        pr = tl & ~pool["p_over_assumed"].astype(bool).values
        if pr.sum() >= MIN_TEST_GAMES:
            p_ex = probs["p_over"].values / np.maximum(1 - probs["tot_push"].values, 1e-9)
            to["log_loss_model"] = round(float(_logloss(p_ex[pr], over[pr]).mean()), 5)
            to["log_loss_market"] = round(float(_logloss(pool["p_over"].values[pr], over[pr]).mean()), 5)
            to["beats_market"] = bool(to["log_loss_model"] < to["log_loss_market"])
    to["mae_model"] = round(float(np.abs(probs["e_total"].values[mk.values] - tot[mk.values]).mean()), 3)
    to["mae_market"] = round(float(np.abs(mprobs["e_total"].values[mk.values] - tot[mk.values]).mean()), 3)
    to.update(_report(pool[pool["total_line"].notna().values & mk.values],
                      probs[pool["total_line"].notna().values & mk.values], "total"))
    out["total"] = to

    per = []
    for s in seasons:
        sel = (pool["season"].values == s) & m
        if sel.sum():
            per.append({"season": s, "n": int(sel.sum()),
                        "ml_log_loss_edge": round(float(
                            _logloss(pool["p_home"].values[sel], mar[sel] > 0).mean()
                            - _logloss(probs["p_home"].values[sel], mar[sel] > 0).mean()), 5)})
    out["per_season"] = per
    return out


def strict_holdout(games: pd.DataFrame, lines: pd.DataFrame) -> dict:
    """The number to trust: nothing fitted on the seasons it scores.

    The walk-forward above keeps the Poisson models out of sample, but the scoreline model and
    the shrinks are fitted on the latest seasons, which overlap the priced seasons being scored -
    and the late-game shape turns out to matter a great deal to the puck line. So this re-runs
    the evaluation with the scoreline fitted only on seasons BEFORE the first priced one and the
    shrinks fitted only on walk-forward seasons before it too. The first time this was run it
    cut the puck line's positive-EV ROI from +16% to +4.6% +/- 2.9% (932 bets, 2023-26) - still
    positive in every season, but no longer anything a significance test would sign off on.
    """
    priced = lines.dropna(subset=["pl_home_dec"]).groupby("season").size()
    comp = set(complete_seasons(priced.index))
    tests = sorted(int(s) for s, n in priced.items() if n >= MIN_TEST_GAMES and s in comp)
    if not tests:
        return {"skipped": True, "reason": "no complete season with puck-line prices"}
    first = tests[0]
    theta, _ = fit_scoreline(games, before=first)
    table = Table(theta)
    D, _, _ = build(games, lines=lines, table=table, starters="actual")
    DE, _, _ = build(games, lines=lines, table=table, starters="expected")
    pool = walk_forward(DE, sides(D), S_eval=sides(DE))
    pre, post = pool[pool["season"] < first], pool[pool["season"] >= first]
    if pre.empty or post.empty:
        return {"skipped": True, "reason": "not enough seasons before the first priced one"}
    shrink = fit_shrinks(pre.reset_index(drop=True), table)
    ev = evaluate(post.reset_index(drop=True), table, shrink)
    return {"test_seasons": sorted(int(x) for x in post["season"].unique()),
            "theta_seasons": theta.get("seasons"), "shrink_seasons":
                sorted(int(x) for x in pre["season"].unique()),
            "shrink": shrink, "moneyline": ev["moneyline"],
            "spread": {k: ev["spread"].get(k) for k in ("n", "log_loss_model", "log_loss_market",
                                                         "roi_by_ev", "rules", "significance")},
            "total": {k: ev["total"].get(k) for k in ("n", "log_loss_model", "log_loss_market",
                                                        "roi_by_ev", "rules", "significance")}}


# ---------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args(argv)

    games, lines = assemble(fetch=not args.no_fetch)
    theta, cal = fit_scoreline(games)
    table = Table(theta)
    # fitted on the starters games actually had; evaluated on the book's guess about them
    D, _, _ = build(games, lines=lines, table=table, starters="actual")
    DE, _, _ = build(games, lines=lines, table=table, starters="expected")
    S, SE = sides(D), sides(DE)
    log.info("%d training games, seasons %d-%d, %d with a market number", len(D),
             int(D["season"].min()), int(D["season"].max()), int(D["m_lh"].notna().sum()))

    pool = walk_forward(DE, S, S_eval=SE, S_known=S)
    if pool.empty:
        raise SystemExit("not enough complete seasons to evaluate")
    shrink = fit_shrinks(pool, table)
    ev = evaluate(pool, table, shrink)
    ev["goalies"] = {"top_pick_started_pct": top_pick_rate(DE, games, ev["test_seasons"]),
                     **goalie_value(pool, table, shrink)}

    last = int(D["season"].max()) + 1
    models = _fit_pair(S, last)
    save_models(models)
    meta = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sport": "nhl", "models": sorted(models),
            "base_features": BASE_FEATURES, "market_features": MARKET_FEATURES,
            "n_rows": int(len(D)), "seasons": sorted(int(s) for s in D["season"].unique()),
            "half_life_seasons": HALF_LIFE, "theta": theta, "scoreline_calibration": cal,
            "shrink": shrink, "effects": {n: m.effects() for n, m in models.items()},
            "lines": lines["source"].value_counts().to_dict() if len(lines) else {},
            "eval": ev, "eval_strict": strict_holdout(games, lines)}
    config.ensure_dirs()
    (config.MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    ml, sp, to = ev["moneyline"], ev["spread"], ev["total"]
    log.info("moneyline log loss %.5f vs market %.5f (edge %+.5f) | shrink split %.2f total %.2f",
             ml["log_loss_model"], ml["log_loss_market"], ml["log_loss_edge"],
             shrink["split"], shrink["total"])
    log.info("goalies: %s", ev["goalies"])
    for name, blk in (("puck line", sp), ("totals", to)):
        for b in blk.get("roi_by_ev", []) + blk.get("rules", []):
            log.info("%s %s: n=%d ROI %+.2f%% (se %s)", name, b.get("ev") or b.get("segment"),
                     b["n"], b["roi_pct"], b["stderr"])
        log.info("%s: %s", name, (blk.get("significance") or {}).get("verdict"))
    st = meta["eval_strict"]
    for r in (st.get("spread") or {}).get("rules") or []:
        log.info("STRICT holdout %s puck line %s: n=%d ROI %+.2f%% (se %s)", st["test_seasons"],
                 r["segment"], r["n"], r["roi_pct"], r["stderr"])
    log.info("saved %s to %s", sorted(models), config.MODEL_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
