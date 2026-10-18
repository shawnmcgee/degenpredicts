"""Train the props models and the walk-forward backtest that decides how far to trust them.

    python -m nfl.props.model              # fetch, build, walk-forward, fit, save
    python -m nfl.props.model --no-fetch   # use the committed player files

Five models, one per Kalshi series: passing, rushing and receiving yards and receptions are
gradient-boosted regressions with a Poisson loss for the player's expected stat line - a log
link, so opportunity and efficiency multiply rather than add, which is how a stat line is
actually produced - and anytime touchdown is a gradient-boosted classifier.

They are scikit-learn's histogram boosters rather than XGBoost, for two reasons: they treat a
missing value (a rookie's empty snap history) as its own branch without being told, and the
props models are fitted inside the props workflow rather than committed from elsewhere, so a
pickle never crosses scikit-learn versions.

**A mean is not a price.** A prop is a bet on one side of a line, so what is needed is the
whole distribution - and yardage is badly skewed: a receiver projected for 60 yards clears
59.5 well under half the time, because his median is below his mean. So each model's
out-of-sample predictions are kept beside what actually happened, and the probability of
clearing any line is read off the outcomes of past player-games the model projected about the
same - the way ``nfl/keynumbers.py`` reads a spread off past games that closed near it. The
same table gives the median, the zero mass, and the shape. It is kept per position group,
because a quarterback projected for 40 rushing yards is not a back projected for 40.

**What the backtest can and cannot say.** There is no free archive of past prop lines, so it
scores the model against outcomes, not against the market:

* error against a trailing-average baseline (the player's own recent per-game numbers);
* calibration - when it says 60%, does it happen 60% of the time;
* a *proxy line* at the player's trailing average, the crude opener a book starts from, and
  how often the model's side of it won. Books set their lines with far more than a trailing
  average, so this flatters the model; it is a floor on competence, not a claim of edge.

Whether there is an edge against Kalshi's real prices is a question the logged ladders answer,
one Sunday at a time.
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
from ..sources import nflverse
from . import data
from .features import FEATURES, TARGETS, build, eligible

log = logging.getLogger("nfl.props.model")

PARAMS = dict(max_iter=400, learning_rate=0.04, max_leaf_nodes=24, min_samples_leaf=60,
              l2_regularization=2.0, early_stopping=False, random_state=42)
BANDWIDTH = 0.10          # log-scale kernel width when reading a distribution off past games
MIN_ESS = 150             # effective sample size before a position group gets its own table
POOL_CAP = 40000          # out-of-sample pairs kept per market in the saved table


def _is_binary(market: str) -> bool:
    return market == "anytime_td"


def _model(market: str):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    if _is_binary(market):
        return HistGradientBoostingClassifier(**PARAMS)
    return HistGradientBoostingRegressor(loss="poisson", **PARAMS)


def _fit(model, X: pd.DataFrame, y: pd.Series, market: str):
    # The Poisson loss needs a non-negative target. Only rushing yards can go below zero (a
    # 2-yard loss on a lone carry), and a projection of the mean barely notices; the outcome
    # distribution the probabilities are read from keeps the real, signed numbers.
    model.fit(X, y if _is_binary(market) else np.clip(y.to_numpy(float), 0.0, None))
    return model


def _predict(model, X: pd.DataFrame, market: str) -> np.ndarray:
    if _is_binary(market):
        return model.predict_proba(X)[:, 1]
    return np.clip(model.predict(X), 0.0, None)


def rows_for(df: pd.DataFrame, market: str) -> pd.DataFrame:
    mask = [eligible(r, market) for r in df[["position", "n_eff", "is_starter_qb", "ew_carries",
                                              "ew_targets", "ew_snap_pct"]].to_dict("records")]
    out = df[np.array(mask, dtype=bool)]
    return out[out["season_type"].astype(str).str.upper() == "REG"] if "season_type" in out else out


def baseline(df: pd.DataFrame, market: str) -> np.ndarray:
    """The player's own trailing per-game number - what a model has to beat to be worth having."""
    col = {"pass_yds": "ew_pass_yds", "rush_yds": "ew_rush_yds", "rec_yds": "ew_rec_yds",
           "receptions": "ew_receptions"}.get(market)
    if col:
        return df[col].to_numpy(float)
    return np.clip(1 - np.exp(-df["ew_tds"].to_numpy(float)), 0.01, 0.99)


# ---------------------------------------------------------------------------------------
# Distributions read off past projections
# ---------------------------------------------------------------------------------------
class Dist:
    """P(stat >= k | projected mean, position group), from out-of-sample (projection, outcome)
    pairs. A Gaussian kernel in log(projection) picks the comparable past games."""

    def __init__(self, mu: np.ndarray, y: np.ndarray, group: np.ndarray):
        ok = np.isfinite(mu) & np.isfinite(y)
        self.lmu = np.log(np.maximum(mu[ok], 0.5))
        self.y = y[ok]
        self.group = group[ok]

    def _weights(self, mu: float, group: str | None) -> np.ndarray:
        w = np.exp(-0.5 * ((self.lmu - np.log(max(mu, 0.5))) / BANDWIDTH) ** 2)
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
        """What players projected like this actually averaged. Differs from `mu` where the
        model's projections run hot or cold - at the high end they run hot, and this is the
        number the card shows."""
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
        return {"mu": np.round(np.exp(self.lmu[idx]), 2).tolist(),
                "y": np.round(self.y[idx], 1).tolist(), "group": self.group[idx].tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "Dist":
        return cls(np.asarray(d["mu"], float), np.asarray(d["y"], float),
                   np.asarray(d["group"], object))


# ---------------------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------------------
def complete_seasons(df: pd.DataFrame) -> list[int]:
    today = config.today_et()
    return sorted(int(s) for s in df["season"].unique() if config.season_end(int(s)) < today)


def walk_forward(df: pd.DataFrame, market: str, n_seasons: int | None = None) -> pd.DataFrame:
    """Out-of-sample projections for the last n complete seasons, each from a model trained
    only on the seasons before it."""
    n = n_seasons or (config.PROPS_WALK_SEASONS + 2)
    d = rows_for(df, market)
    target = TARGETS[market]
    comp = complete_seasons(d)
    tests = [s for s in comp[-n:] if s > comp[0] + 1]
    out = []
    for s in tests:
        tr, te = d[d["season"] < s], d[d["season"] == s]
        if len(tr) < 2000 or len(te) < 200:
            continue
        m = _fit(_model(market), tr[FEATURES], tr[target], market)
        out.append(pd.DataFrame({
            "season": s, "player_id": te["player_id"].values, "position": te["position"].values,
            "pred": _predict(m, te[FEATURES], market), "base": baseline(te, market),
            "y": te[target].to_numpy(float)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def platt_fit(p, y) -> tuple[float, float]:
    """logit(P) = a + b * logit(p). b below 1 means the raw probabilities were too extreme."""
    from sklearn.linear_model import LogisticRegression
    ok = np.isfinite(p)
    if ok.sum() < 200 or len(set(np.asarray(y)[ok])) < 2:
        return 0.0, 1.0
    lr = LogisticRegression(C=1e6).fit(_logit(p[ok]).reshape(-1, 1), np.asarray(y)[ok])
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def platt(p, a: float, b: float):
    return 1 / (1 + np.exp(-(a + b * _logit(p))))


def _brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def _logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


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
    seasons = sorted(oos["season"].unique())
    rep = {"market": market, "seasons": [int(s) for s in seasons], "n": int(len(oos))}
    if _is_binary(market):
        ev = oos[oos["season"] >= seasons[0]]
        rep.update({
            "log_loss_model": round(_logloss(ev.pred, ev.y), 4),
            "log_loss_baseline": round(_logloss(ev.base, ev.y), 4),
            "log_loss_base_rate": round(_logloss(np.full(len(ev), ev.y.mean()), ev.y), 4),
            "brier_model": round(_brier(ev.pred, ev.y), 4),
            "brier_baseline": round(_brier(ev.base, ev.y), 4),
            "td_rate": round(100 * float(ev.y.mean()), 1),
            "calibration": _calibration(ev.pred, ev.y),
            "per_season": [{"season": int(s), "n": int(len(g)),
                            "log_loss_model": round(_logloss(g.pred, g.y), 4),
                            "log_loss_baseline": round(_logloss(g.base, g.y), 4)}
                           for s, g in ev.groupby("season")],
        })
        return rep

    rep["mae_model"] = round(float(np.mean(np.abs(oos.pred - oos.y))), 2)
    rep["mae_baseline"] = round(float(np.mean(np.abs(oos.base - oos.y))), 2)
    rep["mean_outcome"] = round(float(oos.y.mean()), 2)
    rep["mean_pred"] = round(float(oos.pred.mean()), 2)
    # Proxy-line test: the line sits at the player's trailing average (half-point), the crude
    # opener a book starts from. P(over) from the model's distribution against the same
    # machinery driven by the trailing average, both built only from earlier seasons.
    probs, bprobs, ys, ss = [], [], [], []
    for s in seasons[2:]:
        prior = oos[oos["season"] < s]
        cur = oos[oos["season"] == s]
        dm = Dist(prior.pred.to_numpy(float), prior.y.to_numpy(float), prior.position.to_numpy(object))
        db = Dist(prior.base.to_numpy(float), prior.y.to_numpy(float), prior.position.to_numpy(object))
        sample = cur.sample(min(len(cur), 2500), random_state=1)
        for r in sample.itertuples():
            line = np.floor(r.base) + 0.5
            if line < 0.5:
                continue
            probs.append(dm.p_at_least(r.pred, line + 0.5, r.position))
            bprobs.append(db.p_at_least(r.base, line + 0.5, r.position))
            ys.append(float(r.y > line))
            ss.append(int(s))
    probs, bprobs, ys, ss = np.array(probs), np.array(bprobs), np.array(ys), np.array(ss)
    if len(ys):
        # The raw distribution is calibrated against a FIXED threshold, but a line is not fixed:
        # it is set where the player's own history puts it, and a projection far from that line
        # is far from it partly by noise. So the raw probabilities run too extreme on both ends.
        # Platt scaling, fitted only on seasons before the one it is scored on, pulls them in.
        cal = np.full(len(ys), np.nan)
        ev_seasons = sorted(set(ss.tolist()))
        for s in ev_seasons[1:]:
            a, b = platt_fit(probs[ss < s], ys[ss < s])
            cal[ss == s] = platt(probs[ss == s], a, b)
        m = ~np.isnan(cal)

        def side_hits(p, mask):
            right = np.where(p > 0.5, ys == 1, ys == 0)
            return round(100 * float(right[mask].mean()), 1) if mask.any() else None

        conf = m & (np.abs(np.where(m, cal, 0.5) - 0.5) >= 0.10)
        rep["proxy_line"] = {
            "what": ("a line at the player's trailing average, the crude opener a book starts "
                     "from. Books use far more than that, so these rates flatter the model; "
                     "they are a floor on competence, not a measure of edge"),
            "n_raw": int(len(ys)),
            "brier_raw": round(_brier(probs, ys), 4),
            "calibration_raw": _calibration(probs, ys),
            "seasons": [s for s in ev_seasons[1:]], "n": int(m.sum()),
            "over_rate": round(100 * float(ys[m].mean()), 1),
            "brier_model": round(_brier(cal[m], ys[m]), 4),
            "brier_trailing_avg": round(_brier(bprobs[m], ys[m]), 4),
            "calibration": _calibration(cal[m], ys[m]),
            "always_under_hit_rate": round(100 * float(1 - ys[m].mean()), 1),
            "trailing_avg_side_hit_rate": side_hits(bprobs, m),
            "side_hit_rate_all": side_hits(cal, m),
            "side_hit_rate_confident": side_hits(cal, conf),
            "n_confident": int(conf.sum()),
            "stderr_confident": round(100 * float(np.sqrt(0.25 / max(conf.sum(), 1))), 2),
        }
        a, b = platt_fit(probs, ys)
        rep["platt"] = {"a": round(a, 4), "b": round(b, 4)}
    rep["per_season"] = [{"season": int(s), "n": int(len(g)),
                          "mae_model": round(float(np.mean(np.abs(g.pred - g.y))), 2),
                          "mae_baseline": round(float(np.mean(np.abs(g.base - g.y))), 2)}
                         for s, g in oos.groupby("season")]
    return rep


# ---------------------------------------------------------------------------------------
# Fit, save, load
# ---------------------------------------------------------------------------------------
def assemble(fetch: bool = True) -> pd.DataFrame:
    if fetch:
        data.update()
        games = nflverse.update_games()
    else:
        games = nflverse.load_games()
    players = data.load()
    rows, _, _ = build(players, games, nflverse.load_lines())
    log.info("%d player-game rows, seasons %s", len(rows), sorted(rows["season"].unique()))
    return rows


def train(rows: pd.DataFrame, markets=None) -> dict:
    config.PROPS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    markets = markets or list(TARGETS)
    import sklearn
    meta = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn": sklearn.__version__, "features": FEATURES, "markets": {}}
    for market in markets:
        oos = walk_forward(rows, market)
        rep = evaluate(oos, market)
        d = rows_for(rows, market)
        m = _fit(_model(market), d[FEATURES], d[TARGETS[market]], market)
        joblib.dump(m, config.PROPS_MODEL_DIR / f"{market}.joblib")
        if not _is_binary(market) and len(oos):
            dist = Dist(oos.pred.to_numpy(float), oos.y.to_numpy(float),
                        oos.position.to_numpy(object))
            (config.PROPS_MODEL_DIR / f"{market}_dist.json").write_text(
                json.dumps(dist.to_json(), separators=(",", ":")))
        if _is_binary(market) and len(oos):
            # isotonic-style recalibration table from the walk-forward: predicted bucket ->
            # actual rate, applied on the board so a 30% means 30%
            rep["recalibration"] = _calibration(oos.pred, oos.y, edges=tuple(np.linspace(0, 1, 21)))
        rep["n_train"] = int(len(d))
        meta["markets"][market] = rep
        log.info("%s: %s", market, {k: v for k, v in rep.items()
                                    if k in ("mae_model", "mae_baseline", "log_loss_model",
                                             "log_loss_baseline", "n")})
    (config.PROPS_MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
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


def recalibrate(p: float, table: list[dict]) -> float:
    """Map a raw TD probability through the walk-forward's predicted->actual table."""
    if not table or p != p:
        return p
    xs = [t["predicted"] / 100 for t in table]
    ys = [t["actual"] / 100 for t in table]
    return float(np.interp(p, xs, ys))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--market", action="append")
    a = ap.parse_args(argv)
    rows = assemble(fetch=not a.no_fetch)
    train(rows, a.market)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
