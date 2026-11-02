"""The candidate models the bake-off chooses between, behind one interface.

Nine candidates, in order of simplicity (the order matters - see ``SIMPLICITY``):

| name | what it is |
|---|---|
| ``ratings`` | no machine learning: the rating engine's own expected margin / total |
| ``ridge`` | linear regression with an L2 penalty chosen by cross-validation |
| ``huber`` | linear, but with a loss that stops a 40-point blowout from dragging every coefficient |
| ``hgb`` | scikit-learn's histogram gradient-boosted trees |
| ``xgb`` | XGBoost gradient-boosted trees (skipped if xgboost is not installed) |
| ``rf`` | random forest |
| ``mlp`` | a small neural network (two hidden layers, early stopping) |
| ``blend`` | the plain average of ridge, hgb and xgb (rf stands in for xgb if it is missing) |
| ``stack`` | a non-negative weighted mix of every base model, the weights fitted only on seasons the models had not seen when they made those predictions |

**Two variants of each.** *No-market* models see ratings and context only. *Market* models also
see the line, and predict the line's ERROR (final minus line) instead of the final itself. That
difference decides whether tree models can compete at all: asked for the total, a tree has to
rebuild the line out of a staircase of splits and gives back a point of accuracy doing it; asked
for the error, it starts from the line and only has to learn when the line is wrong. The linear
models are indifferent - the line is a feature either way - but the residual form keeps every
candidate on the same footing.

Nothing here touches the clock or the disk. Hyper-parameters are fixed and conservative on
purpose: the bake-off compares model FAMILIES on seasons none of them saw, and tuning each one on
those same seasons would make the comparison measure the tuning instead.
"""
from __future__ import annotations

import importlib.util
import logging
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SIMPLICITY = ["ratings", "ridge", "huber", "hgb", "xgb", "rf", "mlp", "blend", "stack"]
BASES = ["ridge", "huber", "hgb", "xgb", "rf", "mlp"]
LABELS = {"ratings": "Ratings only (no ML)", "ridge": "Ridge regression",
          "huber": "Huber regression", "hgb": "Gradient boosting (sklearn)",
          "xgb": "XGBoost", "rf": "Random forest", "mlp": "Neural net (MLP)",
          "blend": "Blend: ridge + boosting", "stack": "Stack: weighted mix of all",
          "line": "The line itself"}


def available(name: str) -> bool:
    if name == "xgb":
        return importlib.util.find_spec("xgboost") is not None
    return name in SIMPLICITY


def blend_members() -> list[str]:
    return ["ridge", "hgb", "xgb" if available("xgb") else "rf"]


def make(name: str, market: bool):
    """A fresh, unfitted estimator. Market variants are regularised harder: their target is the
    line's error, which is mostly noise, and a flexible model will happily fit the noise."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "ridge":
        from sklearn.linear_model import RidgeCV
        return make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-1, 4.5, 12)))
    if name == "huber":
        from sklearn.linear_model import HuberRegressor
        return make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(),
                             HuberRegressor(epsilon=1.35, alpha=1e-2 if market else 1e-4,
                                            max_iter=2000))
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        if market:
            return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03,
                                                 max_leaf_nodes=7, min_samples_leaf=200,
                                                 l2_regularization=5.0, random_state=0)
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04,
                                             max_leaf_nodes=15, min_samples_leaf=60,
                                             l2_regularization=1.0, random_state=0)
    if name == "xgb":
        from xgboost import XGBRegressor
        if market:
            return XGBRegressor(n_estimators=250, max_depth=3, learning_rate=0.03,
                                subsample=0.8, colsample_bytree=0.8, min_child_weight=100,
                                reg_lambda=10.0, tree_method="hist", n_jobs=-1, random_state=42)
        return XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
                            tree_method="hist", n_jobs=-1, random_state=42)
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestRegressor(n_estimators=150,
                                                   min_samples_leaf=150 if market else 40,
                                                   max_features=0.4, n_jobs=-1, random_state=0))
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor
        net = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0 if market else 1e-2,
                           learning_rate_init=1e-3, early_stopping=True,
                           validation_fraction=0.15, n_iter_no_change=10, max_iter=300,
                           random_state=0)
        return TransformedTargetRegressor(
            regressor=make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), net),
            transformer=StandardScaler())
    raise KeyError(f"no such base model: {name}")


def _fit(est, X: pd.DataFrame, y: np.ndarray):
    with warnings.catch_warnings():
        # Huber and the MLP warn when they stop on max_iter / early stopping; both are expected
        warnings.simplefilter("ignore")
        est.fit(X, y)
    return est


def fit_predict(name: str, Xtr: pd.DataFrame, ytr, Xte: pd.DataFrame, market: bool) -> np.ndarray:
    est = _fit(make(name, market), Xtr, np.asarray(ytr, float))
    return np.asarray(est.predict(Xte), float)


def nnls_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least-squares weights for combining columns of predictions ``P``.

    Non-negative because a negative weight on a model means betting against it, which on a few
    thousand games is almost always the fit chasing noise. Not forced to sum to one: for the
    market variants the predictions are line errors, and a mix that sums to 0.6 is the stack
    saying "trust these models a bit less than they trust themselves".
    """
    from scipy.optimize import nnls
    ok = np.isfinite(P).all(axis=1) & np.isfinite(y)
    if ok.sum() < 50:
        return np.full(P.shape[1], 1.0 / P.shape[1])
    w, _ = nnls(P[ok], y[ok])
    if not np.isfinite(w).all() or w.sum() == 0:
        return np.full(P.shape[1], 1.0 / P.shape[1])
    return w


class Fitted:
    """A chosen model fitted for production: one of the bases, the ratings, a blend, or a
    stack. ``predict`` returns the target (for market variants: the line's error)."""

    def __init__(self, name: str, market: bool, exp_col: str, weights: dict | None = None):
        self.name, self.market, self.exp_col = name, market, exp_col
        self.weights = dict(weights or {})
        self.parts: dict = {}

    def members(self) -> list[str]:
        if self.name == "blend":
            return blend_members()
        if self.name == "stack":
            return [m for m in self.weights if available(m)]
        if self.name == "ratings":
            return []
        return [self.name]

    def fit(self, X: pd.DataFrame, y) -> "Fitted":
        for m in self.members():
            self.parts[m] = _fit(make(m, self.market), X, np.asarray(y, float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.name == "ratings":
            return np.zeros(len(X)) if self.market else X[self.exp_col].to_numpy(float)
        preds = {m: np.asarray(est.predict(X), float) for m, est in self.parts.items()}
        if self.name == "blend":
            return np.mean(list(preds.values()), axis=0)
        if self.name == "stack":
            w = {m: self.weights[m] for m in preds}
            return sum(w[m] * preds[m] for m in preds)
        return preds[self.name]
