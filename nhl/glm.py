"""A Poisson regression with an offset, small enough to read and saved as plain JSON.

Why not XGBoost like the other sports: it lost. Walk-forward over 2012-2025 the gradient-boosted
models were worse than a Poisson GLM on every market - moneyline log loss 0.6694 against 0.6681,
and on totals and the puck line they were the worst of the models tried. There are roughly
2,600 side-rows a season, each a count with a mean near 2.8 and a standard deviation near 1.7;
trees have nothing to find in that except noise. A log-linear model on well-built ratings is the
right shape for a multiplicative scoring process, and its coefficients can be read.

**The market-aware model is an offset model.** Rather than feeding the market's number in as one
feature among many, the market's implied goals are the model's baseline:

    log E[goals] = log(market goals) + b0 + b . x

so the coefficients describe only where the market goes wrong - a back-to-back it under-weights,
a rating gap it has not caught up with. With every coefficient at zero the model IS the market,
which is the correct default for a market this efficient and the reason this form beat the
alternatives on the moneyline, the puck line and total-goals likelihood alike.

Rows are weighted toward recent seasons (half-life four seasons), because the sport keeps moving:
shot counting, empty-net strategy and scoring have all shifted within the training window.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize


class PoissonGLM:
    def __init__(self, features: list[str], offset: str | None = None, l2: float = 1.0):
        self.features = list(features)
        self.offset = offset
        self.l2 = float(l2)
        self.mu = self.sd = self.coef = None
        self.intercept = 0.0

    def _z(self, X):
        return (X - self.mu) / self.sd

    def fit(self, df, y="y", weights=None) -> "PoissonGLM":
        X = df[self.features].to_numpy(float)
        yy = df[y].to_numpy(float)
        off = df[self.offset].to_numpy(float) if self.offset else np.zeros(len(yy))
        ok = np.isfinite(X).all(1) & np.isfinite(yy) & np.isfinite(off)
        X, yy, off = X[ok], yy[ok], off[ok]
        w = np.ones(len(yy)) if weights is None else np.asarray(weights, float)[ok]
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        Z = self._z(X)
        W = w.sum()
        # the intercept starts at the log of the mean residual rate, so an offset model begins
        # where the offset leaves it rather than at exp(0) = 1 goal
        b0 = float(np.log(max((w * yy).sum(), 1e-9) / max((w * np.exp(off)).sum(), 1e-9)))

        def f(b):
            eta = off + b[0] + Z @ b[1:]
            lam = np.exp(np.clip(eta, -20, 5))
            val = (w * (lam - yy * eta)).sum() / W + self.l2 * (b[1:] ** 2).sum() / len(yy)
            r = w * (lam - yy) / W
            g = np.concatenate([[r.sum()], Z.T @ r + 2 * self.l2 * b[1:] / len(yy)])
            return val, g

        res = minimize(f, np.concatenate([[b0], np.zeros(Z.shape[1])]), jac=True,
                       method="L-BFGS-B")
        self.intercept, self.coef = float(res.x[0]), res.x[1:]
        return self

    def predict(self, df) -> np.ndarray:
        X = df[self.features].to_numpy(float)
        off = df[self.offset].to_numpy(float) if self.offset else np.zeros(len(X))
        return np.exp(np.clip(off + self.intercept + self._z(X) @ self.coef, -20, 5))

    def to_dict(self) -> dict:
        return {"features": self.features, "offset": self.offset, "l2": self.l2,
                "intercept": self.intercept, "coef": [float(c) for c in self.coef],
                "mu": [float(m) for m in self.mu], "sd": [float(s) for s in self.sd]}

    @classmethod
    def from_dict(cls, d: dict) -> "PoissonGLM":
        m = cls(d["features"], d.get("offset"), d.get("l2", 1.0))
        m.intercept = float(d["intercept"])
        m.coef = np.asarray(d["coef"], float)
        m.mu = np.asarray(d["mu"], float)
        m.sd = np.asarray(d["sd"], float)
        return m

    def save(self, path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "PoissonGLM":
        return cls.from_dict(json.loads(path.read_text()))

    def effects(self) -> dict:
        """Per-feature effect of one standard deviation, as a percentage on goals."""
        return {f: round(100 * (float(np.exp(c)) - 1), 2) for f, c in zip(self.features, self.coef)}


def recency_weights(seasons, target_season: int, half_life: float = 4.0) -> np.ndarray:
    s = np.asarray(seasons, float)
    return 0.5 ** ((target_season - 1 - s) / half_life)
