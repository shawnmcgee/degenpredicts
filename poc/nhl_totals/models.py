"""Feature engineering and the model families compared in the backtest.

Every model outputs P(over) for a game, pushes excluded. They differ in what they assume:

  base     logistic regression on the market's structure only (the line, the moneyline favourite).
           It knows nothing about the teams, and it is the bar every other model must clear: in
           the unpriced years, "beating 50%" can come from nothing more than which side of a 5.5
           tends to carry the juice, and this model captures exactly that.
  logit    L2 logistic regression on every engineered feature, standardised
  gbm      LightGBM, kept shallow and slow (8 leaves, 300+ games per leaf) - with ~1,200 games a
           season there is little for trees to find, and these settings stop them inventing it
  pois     Poisson regression on total goals anchored to the line (offset log line), read out as
           P(over) through a dispersion-matched count distribution
  ens      the average of logit, gbm and pois
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

LINES = [5.0, 5.5, 6.0, 6.5]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw state columns into features that speak in goals and in ratios to the line."""
    x = pd.DataFrame(index=df.index)
    line = df["total_line"]
    x["line"] = line
    for L in LINES:
        x[f"line_{L}"] = (line == L).astype(float)
    x["fav"] = (df["p_home_mkt"] - 0.5).abs().fillna(0.0)
    x["fav_x_line"] = x["fav"] * (line - 5.75)
    x["home_fav"] = (df["p_home_mkt"].fillna(0.5) - 0.5)

    def both(goalie):  # expected non-empty-net goals, both sides, through tonight's goalies
        out = {}
        for tag in ("f", "s"):
            out[tag] = df[f"{tag}_xgf_h"] * df[f"{goalie}_a"] + df[f"{tag}_xgf_a"] * df[f"{goalie}_h"]
        return out

    lg2 = 2 * df["lg_xgf"]
    for name, goalie in (("act", "gk"), ("proj", "gkp")):
        e = both(goalie)
        x[f"xg_f_{name}"] = np.log(e["f"] / lg2)
        x[f"xg_s_{name}"] = np.log(e["s"] / lg2)
        x[f"gk_{name}"] = np.log(df[f"{goalie}_h"]) + np.log(df[f"{goalie}_a"])
        # the model's expected final total, in goals, against the line
        x[f"exp_tot_{name}"] = (e["s"] + df["s_en_f_tot"]) * (df["lg_tot"] / (lg2 + 2 * df["lg_en_f"]))
        x[f"gap_{name}"] = x[f"exp_tot_{name}"] - line
    x["g_s"] = np.log(df["s_gf_tot"] / (2 * df["lg_gf"]))
    x["g_f"] = np.log(df["f_gf_tot"] / (2 * df["lg_gf"]))
    x["cf_s"] = np.log(df["s_cf_tot"] / (2 * df["lg_cf"]))
    x["sf_s"] = np.log(df["s_sf_tot"] / (2 * df["lg_sf"]))
    x["pp_s"] = np.log(df["s_xgf_pp_tot"] / (2 * df["lg_xgf_pp"]))
    x["pen_s"] = np.log(df["s_pen_d_tot"] / (2 * df["lg_pen_d"]))
    x["en_s"] = df["s_en_f_tot"] - 2 * df["lg_en_f"]
    x["xg5_s"] = np.log(df["s_xgf5_tot"] / (2 * df["lg_xgf5"]))
    x["fin_s"] = df["s_fin_h"] + df["s_fin_a"] - 2
    x["sv_s"] = df["s_sv_h"] + df["s_sv_a"] - 2
    x["fin_f"] = df["f_fin_h"] + df["f_fin_a"] - 2
    x["lg_vs_line"] = df["lg_tot"] - line
    x["lg_tot"] = df["lg_tot"]
    # lineups (tonight's dressed skaters against the usual group, in goals per game)
    for k in ("dixg", "dpts", "doixgf", "doixga"):
        x[f"lu_{k}"] = df[f"lu_{k}_h"].fillna(0) + df[f"lu_{k}_a"].fillna(0)
    x["lu_miss"] = df["lu_miss_h"] + df["lu_miss_a"]
    x["lu_new"] = df["lu_new_h"] + df["lu_new_a"]
    x["gk_backup"] = df["gk_backup_h"] + df["gk_backup_a"]
    x["gk_young"] = (df["gk_n_h"] < 40).astype(float) + (df["gk_n_a"] < 40).astype(float)
    x["gk_rest_min"] = np.minimum(df["gk_rest_h"], df["gk_rest_a"])
    # referees
    x["ref_pen"] = np.log(df["ref_pen"])
    x["ref_goals"] = np.log(df["ref_goals"])
    x["ref_ou"] = df["ref_ou"]
    # schedule
    x["b2b"] = df["b2b_h"] + df["b2b_a"]
    x["b2b_a"] = df["b2b_a"]
    x["rest_diff"] = df["rest_h"] - df["rest_a"]
    x["km"] = np.log1p(df["km_h"] + df["km_a"])
    x["tz"] = df["tz_h"].abs() + df["tz_a"].abs()
    x["three4"] = df["3in4_h"] + df["3in4_a"]
    # market memory: how teams and the season have run against the line
    x["team_ou"] = df["ou_h"] + df["ou_a"]
    x["early"] = (np.minimum(df["gp_h"], df["gp_a"]) < 10).astype(float)
    x["gp"] = np.minimum(df["gp_h"], df["gp_a"]).clip(upper=82) / 82
    x["playoff"] = (df["game_type"] == "P").astype(float)
    x["month"] = pd.to_datetime(df["date"]).dt.month
    return x


MARKET = ["line", "line_5.0", "line_5.5", "line_6.0", "line_6.5", "fav", "fav_x_line", "home_fav"]
COMMON = ["g_s", "g_f", "cf_s", "sf_s", "pp_s", "pen_s", "en_s", "xg5_s", "fin_s", "sv_s", "fin_f",
          "lg_vs_line", "ref_pen", "ref_goals", "ref_ou", "b2b", "b2b_a", "rest_diff", "km", "tz",
          "three4", "team_ou", "early", "gp", "playoff"]
# what a closing line knows: tonight's real starters and lineups
CLOSE = MARKET + COMMON + ["xg_f_act", "xg_s_act", "gk_act", "gap_act", "lu_dixg", "lu_dpts",
                           "lu_doixgf", "lu_doixga", "lu_miss", "lu_new", "gk_backup", "gk_young",
                           "gk_rest_min"]
# what is known at noon: the usual starter (back-to-back rule) and the usual lineup
NOON = MARKET + COMMON + ["xg_f_proj", "xg_s_proj", "gk_proj", "gap_proj"]


def _weights(seasons: pd.Series, half_life: float = 4.0) -> np.ndarray:
    return 0.5 ** ((seasons.max() - seasons) / half_life)


class Base:
    def __init__(self):
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))

    def fit(self, X, y, w):
        self.m.fit(X[MARKET], y, logisticregression__sample_weight=w)
        return self

    def predict(self, X):
        return self.m.predict_proba(X[MARKET])[:, 1]


class Logit:
    def __init__(self, cols, C=0.05):
        self.cols, self.m = cols, make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000))

    def fit(self, X, y, w):
        self.m.fit(X[self.cols].fillna(0), y, logisticregression__sample_weight=w)
        return self

    def predict(self, X):
        return self.m.predict_proba(X[self.cols].fillna(0))[:, 1]


class GBM:
    def __init__(self, cols, n=400):
        self.cols = cols
        self.m = lgb.LGBMClassifier(n_estimators=n, learning_rate=0.01, num_leaves=8,
                                    min_child_samples=300, subsample=0.7, subsample_freq=1,
                                    colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)

    def fit(self, X, y, w):
        self.m.fit(X[self.cols], y, sample_weight=w)
        return self

    def predict(self, X):
        return self.m.predict_proba(X[self.cols])[:, 1]


class Pois:
    """E[total] = line * exp(b.x); P(over) from a negative binomial with the fitted dispersion."""

    def __init__(self, cols, alpha=1.0):
        self.cols = [c for c in cols if c not in ("line",)]
        self.sc = StandardScaler()
        self.m = PoissonRegressor(alpha=alpha / 1000, max_iter=3000)

    def fit(self, X, total, w, line):
        Z = self.sc.fit_transform(X[self.cols].fillna(0))
        # offset via the target: model total/line, weighted by line (equivalent Poisson score)
        self.m.fit(Z, total / line, sample_weight=w * line)
        mu = self.m.predict(Z) * line
        # method-of-moments dispersion for a negative binomial: var = mu + mu^2 / k
        excess = np.average((total - mu) ** 2 - mu, weights=w)
        self.k = np.inf if excess <= 0 else np.average(mu ** 2, weights=w) / excess
        return self

    def mean(self, X):
        return self.m.predict(self.sc.transform(X[self.cols].fillna(0))) * X["line"].values

    def predict(self, X):
        mu, line = self.mean(X), X["line"].values
        fl = np.floor(line).astype(int)                          # P(T <= floor(line)) below
        if np.isinf(self.k):
            cdf = stats.poisson.cdf(fl, mu)
            pmf = stats.poisson.pmf(fl, mu)
        else:
            n, p = self.k, self.k / (self.k + mu)
            cdf = stats.nbinom.cdf(fl, n, p)
            pmf = stats.nbinom.pmf(fl, n, p)
        whole = np.isclose(line, fl)
        p_over = 1 - cdf
        p_under = np.where(whole, cdf - pmf, cdf)
        return p_over / (p_over + p_under)                       # pushes excluded


# --- market-anchored models -------------------------------------------------------------------
#
# The unpriced years carry a closing line but not its juice, and the juice is where the market
# hides its view at a 5.5. So the juice is modelled: a ridge regression of the market's fair
# logit(P(over)) on the features, fitted on the priced games' PRICES alone (never their results).
# Its prediction stands in for the missing price in the unpriced years. What a residual model then
# learns there is what the features know beyond the price a book would have posted - which is the
# only thing worth betting.

RESID = ["gap_proj", "xg_s_proj", "xg_f_proj", "gk_proj", "g_s", "g_f", "sv_s", "fin_s", "fin_f",
         "cf_s", "pp_s", "pen_s", "en_s", "b2b", "b2b_a", "three4", "rest_diff", "km", "tz", "early",
         "lg_vs_line", "ref_pen", "ref_goals", "ref_ou", "team_ou", "playoff"]
JUICE = MARKET + RESID


class Juice:
    def __init__(self, alpha=10.0):
        from sklearn.linear_model import Ridge
        self.m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, X):
        pr = X[X["p_over_mkt"].notna()]
        self.m.fit(pr[JUICE].fillna(0), logit(pr["p_over_mkt"].values))
        return self

    def logit_price(self, X):
        """Real price where there is one, the modelled price where there is not."""
        est = self.m.predict(X[JUICE].fillna(0))
        return np.where(X["p_over_mkt"].notna(), logit(X["p_over_mkt"].fillna(0.5).values), est)


class Resid:
    """P(over) = sigmoid(logit price + b.x): the price, moved only by what the features add."""

    def __init__(self, cols=RESID, C=0.02, kind="logit"):
        self.cols, self.C, self.kind = cols, C, kind

    def fit(self, X, y, w, offset):
        if self.kind == "logit":
            # sklearn has no offset: fold it in as a feature with a fixed unit weight
            self.sc = StandardScaler().fit(X[self.cols].fillna(0))
            Z = self.sc.transform(X[self.cols].fillna(0))
            self.b = _offset_logit(Z, y, w, offset, lam=1.0 / self.C)
        else:
            self.m = lgb.LGBMRegressor(objective="binary", n_estimators=300, learning_rate=0.01,
                                       num_leaves=4, min_child_samples=400, subsample=0.7,
                                       subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, verbose=-1)
            self.m.fit(X[self.cols], y, sample_weight=w, init_score=offset)
        return self

    def predict(self, X, offset):
        if self.kind == "logit":
            z = offset + self.sc.transform(X[self.cols].fillna(0)) @ self.b
        else:
            z = offset + self.m.predict(X[self.cols], raw_score=True)
        return 1 / (1 + np.exp(-z))


def _offset_logit(Z, y, w, offset, lam):
    """Ridge logistic regression with a fixed offset and no intercept (the price is the intercept)."""
    from scipy.optimize import minimize

    def f(b):
        z = offset + Z @ b
        p = 1 / (1 + np.exp(-z))
        ll = -np.sum(w * (y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))))
        g = -Z.T @ (w * (y - p))
        return ll + 0.5 * lam * b @ b, g + lam * b

    return minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols, models=("base", "logit", "gbm", "pois", "ens")):
    """Fit every model on `train` (rows with a decided over/under) and score `test`."""
    tr = train[train["y"].notna()]
    w = _weights(tr["season"])
    out = {}
    if "base" in models:
        out["base"] = Base().fit(tr, tr["y"], w).predict(test)
    if "logit" in models or "ens" in models:
        out["logit"] = Logit(cols).fit(tr, tr["y"], w).predict(test)
    if "gbm" in models or "ens" in models:
        out["gbm"] = GBM(cols).fit(tr, tr["y"], w).predict(test)
    if "pois" in models or "ens" in models:
        allr = train
        out["pois"] = Pois(cols).fit(allr, allr["total"].values, _weights(allr["season"]), allr["line"].values).predict(test)
    if "ens" in models:
        out["ens"] = (out["logit"] + out["gbm"] + out["pois"]) / 3
    return out
