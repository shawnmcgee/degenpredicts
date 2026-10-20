"""Train the NBA props models, and the walk-forward backtest that says how far to trust them.

    python -m nba.props.model              # build, walk-forward, fit, save
    python -m nba.props.model --market threes

Four models, one per Kalshi series - points, rebounds, assists and threes - each a
gradient-boosted regression with a Poisson loss for the player's expected stat line. The log
link is the point: a stat line is minutes times a rate, and a log link makes the two multiply
rather than add. They are scikit-learn's histogram boosters, as the NFL props models are: a
missing value (a player new to the league, a season with no closing line) is its own branch,
and the models are fitted inside the props job, so a pickle never crosses scikit-learn versions.

**A mean is not a price.** Points are skewed - a scorer projected for 24 clears 23.5 a little
under half the time - and threes are a small count with a lot of mass at zero. So each market's
out-of-sample predictions are kept beside what happened, and the chance of clearing any rung is
read off past player-games the model projected about the same, per position group: the NFL
props board's distribution table, unchanged. It is also what corrects the model's own tails.

**What the backtest can and cannot say.** There is no free archive of past prop prices, so it
scores the model against outcomes, with a proxy line at the player's trailing average - the
crude opener a book starts from. That flatters the model; it is a floor on competence, not a
claim of edge. Whether there is an edge against Kalshi is what the logged ladders answer.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from .. import config
from .features import FEATURES, TARGETS, eligible

log = logging.getLogger("nba.props.model")

PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=80,
              l2_regularization=2.0, early_stopping=False, random_state=42)
BANDWIDTH = 0.08          # log-scale kernel width when reading a distribution off past games
MIN_ESS = 200             # effective sample size before a position group gets its own table
POOL_CAP = 40000          # out-of-sample pairs kept per market in the saved table
MIN_TRAIN, MIN_TEST = 5000, 1000
BASELINE = {"points": "ew_pts", "rebounds": "ew_reb", "assists": "ew_ast", "threes": "ew_fg3m"}


def _model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="poisson", **PARAMS)


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(X), 0.0, None)


def rows_for(df: pd.DataFrame, market: str) -> pd.DataFrame:
    keep = ["n_eff", "ew_min", "ew_fg3a"]
    mask = np.fromiter((eligible(r, market) for r in df[keep].to_dict("records")), bool, len(df))
    return df[mask]


def complete_seasons(df: pd.DataFrame) -> list[int]:
    today = config.today_et()
    return sorted(int(s) for s in df["season"].unique() if config.season_end(int(s)) < today)


# ---------------------------------------------------------------------------------------
# Distributions read off past projections
# ---------------------------------------------------------------------------------------
class Dist:
    """P(stat >= k | projected mean, position group), from out-of-sample (projection, outcome)
    pairs. A Gaussian kernel in log(projection) picks the comparable past games."""

    def __init__(self, mu: np.ndarray, y: np.ndarray, group: np.ndarray):
        ok = np.isfinite(mu) & np.isfinite(y)
        self.lmu = np.log(np.maximum(mu[ok], 0.25))
        self.y = y[ok]
        self.group = group[ok]

    def _weights(self, mu: float, group: str | None) -> np.ndarray:
        w = np.exp(-0.5 * ((self.lmu - np.log(max(mu, 0.25))) / BANDWIDTH) ** 2)
        if group is not None:
            wg = w * (self.group == group)
            if wg.sum() > 0 and wg.sum() ** 2 / max((wg ** 2).sum(), 1e-12) >= MIN_ESS:
                return wg
        return w

    def p_at_least(self, mu: float, k: float, group: str | None = None) -> float:
        w = self._weights(mu, group)
        s = w.sum()
        return float((w * (self.y >= k)).sum() / s) if s > 0 else float("nan")

    def mean(self, mu: float, group: str | None = None) -> float:
        """What players projected like this actually averaged - the card's number."""
        w = self._weights(mu, group)
        return float((w * self.y).sum() / w.sum()) if w.sum() > 0 else float("nan")

    def quantile(self, mu: float, q: float, group: str | None = None) -> float:
        w = self._weights(mu, group)
        order = np.argsort(self.y)
        cw = np.cumsum(w[order])
        if cw[-1] <= 0:
            return float("nan")
        return float(self.y[order][np.searchsorted(cw, q * cw[-1])])

    def to_json(self) -> dict:
        idx = np.arange(len(self.y))
        if len(idx) > POOL_CAP:
            idx = np.random.default_rng(0).choice(idx, POOL_CAP, replace=False)
        y = self.y[idx]
        # box-score counts are whole numbers; writing them as such halves the file
        y = y.astype(int).tolist() if np.all(y == np.round(y)) else np.round(y, 1).tolist()
        return {"mu": np.round(np.exp(self.lmu[idx]), 2).tolist(), "y": y,
                "group": "".join(self.group[idx].astype(str)) if all(
                    len(str(g)) == 1 for g in self.group[idx]) else self.group[idx].tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "Dist":
        g = d["group"]
        return cls(np.asarray(d["mu"], float), np.asarray(d["y"], float),
                   np.asarray(list(g) if isinstance(g, str) else g, object))


# ---------------------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------------------
def walk_forward(df: pd.DataFrame, market: str, n_seasons: int | None = None) -> pd.DataFrame:
    """Out-of-sample projections for the last n complete seasons, each from a model trained
    only on the seasons before it."""
    n = n_seasons or config.PROPS_WALK_SEASONS
    d = rows_for(df, market)
    target = TARGETS[market]
    comp = complete_seasons(d)
    tests = [s for s in comp[-n:] if s > comp[0] + 1]
    out = []
    for s in tests:
        tr, te = d[d["season"] < s], d[d["season"] == s]
        if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
            continue
        m = _model().fit(tr[FEATURES], tr[target].to_numpy(float))
        out.append(pd.DataFrame({
            "season": s, "athlete_id": te["athlete_id"].values, "position": te["position"].values,
            "pred": _predict(m, te[FEATURES]), "base": te[BASELINE[market]].to_numpy(float),
            "y": te[target].to_numpy(float)}))
        log.info("%s walk-forward %d: %d rows", market, s, len(te))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def _brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def _calibration(p, y, edges=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001)) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 30:
            out.append({"bucket": f"{lo:.0%}-{min(hi, 1):.0%}", "n": int(m.sum()),
                        "predicted": round(100 * float(p[m].mean()), 1),
                        "actual": round(100 * float(y[m].mean()), 1)})
    return out


def evaluate(oos: pd.DataFrame, market: str) -> dict:
    """Score the walk-forward. Probabilities for season S are read off a table built only from
    the out-of-sample seasons before S, so the distribution is as out-of-sample as the mean."""
    if oos.empty:
        return {"skipped": True}
    seasons = sorted(int(s) for s in oos["season"].unique())
    rep = {"market": market, "seasons": seasons, "n": int(len(oos)),
           "mae_model": round(float(np.mean(np.abs(oos.pred - oos.y))), 3),
           "mae_baseline": round(float(np.mean(np.abs(oos.base - oos.y))), 3),
           "mean_outcome": round(float(oos.y.mean()), 2), "mean_pred": round(float(oos.pred.mean()), 2)}
    # Proxy line at the player's trailing average (to the half point), both sides' chances read
    # off tables built only from earlier out-of-sample seasons.
    probs, bprobs, ys, ss = [], [], [], []
    for s in seasons[1:]:
        prior, cur = oos[oos["season"] < s], oos[oos["season"] == s]
        dm = Dist(prior.pred.to_numpy(float), prior.y.to_numpy(float), prior.position.to_numpy(object))
        db = Dist(prior.base.to_numpy(float), prior.y.to_numpy(float), prior.position.to_numpy(object))
        sample = cur.sample(min(len(cur), 3000), random_state=1)
        for r in sample.itertuples():
            line = np.floor(r.base) + 0.5
            if line < 0.5:
                continue
            probs.append(dm.p_at_least(r.pred, line + 0.5, r.position))
            bprobs.append(db.p_at_least(r.base, line + 0.5, r.position))
            ys.append(float(r.y > line))
            ss.append(int(s))
    probs, bprobs, ys = np.array(probs), np.array(bprobs), np.array(ys)
    if len(ys):
        right = lambda p: np.where(p > 0.5, ys == 1, ys == 0)
        conf = np.abs(probs - 0.5) >= 0.10
        rep["proxy_line"] = {
            "what": ("a line at the player's trailing average, the crude opener a book starts "
                     "from; books use far more, so these rates are a floor on competence, not a "
                     "measure of edge"),
            "seasons": sorted(set(ss)), "n": int(len(ys)),
            "over_rate": round(100 * float(ys.mean()), 1),
            "brier_model": round(_brier(probs, ys), 4),
            "brier_trailing_avg": round(_brier(bprobs, ys), 4),
            "calibration": _calibration(probs, ys),
            "always_under_hit_rate": round(100 * float(1 - ys.mean()), 1),
            "trailing_avg_side_hit_rate": round(100 * float(right(bprobs).mean()), 1),
            "side_hit_rate_all": round(100 * float(right(probs).mean()), 1),
            "side_hit_rate_confident": round(100 * float(right(probs)[conf].mean()), 1) if conf.any() else None,
            "n_confident": int(conf.sum()),
        }
    rep["per_season"] = [{"season": int(s), "n": int(len(g)),
                          "mae_model": round(float(np.mean(np.abs(g.pred - g.y))), 3),
                          "mae_baseline": round(float(np.mean(np.abs(g.base - g.y))), 3)}
                         for s, g in oos.groupby("season")]
    return rep


# ---------------------------------------------------------------------------------------
# Fit, save, load
# ---------------------------------------------------------------------------------------
def assemble() -> pd.DataFrame:
    """Every training row, from the committed player files and the game board's own replay."""
    from .. import features as game_features
    from ..sources import history, hoopr
    from .features import build, game_context
    games = hoopr.load_games()
    first_load = config.PROPS_FIRST_SEASON - 2
    players = hoopr.load_players(seasons=range(first_load, config.season_of(config.today_et()) + 1))
    # the game board's replay supplies each game's pre-game expected margin and total
    g_rows, _, _, _ = game_features.build(games, hoopr.load_players(), history.load_lines(),
                                          first_train_season=first_load)
    rows, _, _ = build(players, games[games["season"] >= first_load], game_context(g_rows),
                       hoopr.load_positions())
    log.info("%d player-game rows, seasons %s", len(rows), sorted(rows["season"].unique()))
    return rows


def train(rows: pd.DataFrame, markets=None) -> dict:
    config.PROPS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    markets = markets or list(TARGETS)
    import sklearn
    meta_path = config.PROPS_MODEL_DIR / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() and set(markets) != set(TARGETS) \
        else {"markets": {}}
    meta.update({"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "sklearn": sklearn.__version__, "features": FEATURES})
    for market in markets:
        oos = walk_forward(rows, market)
        rep = evaluate(oos, market)
        d = rows_for(rows, market)
        m = _model().fit(d[FEATURES], d[TARGETS[market]].to_numpy(float))
        # compressed: the models are refitted weekly and committed, so every byte is history
        joblib.dump(m, config.PROPS_MODEL_DIR / f"{market}.joblib", compress=3)
        if len(oos):
            dist = Dist(oos.pred.to_numpy(float), oos.y.to_numpy(float), oos.position.to_numpy(object))
            (config.PROPS_MODEL_DIR / f"{market}_dist.json").write_text(
                json.dumps(dist.to_json(), separators=(",", ":")))
        rep["n_train"] = int(len(d))
        meta["markets"][market] = rep
        log.info("%s: MAE %s vs trailing avg %s | Brier at the proxy line %s vs %s", market,
                 rep.get("mae_model"), rep.get("mae_baseline"),
                 (rep.get("proxy_line") or {}).get("brier_model"),
                 (rep.get("proxy_line") or {}).get("brier_trailing_avg"))
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load():
    meta = json.loads((config.PROPS_MODEL_DIR / "meta.json").read_text())
    models, dists = {}, {}
    for market in meta["markets"]:
        models[market] = joblib.load(config.PROPS_MODEL_DIR / f"{market}.joblib")
        p = config.PROPS_MODEL_DIR / f"{market}_dist.json"
        if p.exists():
            dists[market] = Dist.from_json(json.loads(p.read_text()))
    return models, dists, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", action="append")
    a = ap.parse_args(argv)
    train(assemble(), a.market)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
