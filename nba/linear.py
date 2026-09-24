"""A ridge regression small enough to read, saved as plain JSON coefficients.

Why not XGBoost like the NFL and college football: it lost, and the way it lost is worth
knowing. On the first feature set gradient-boosted trees looked clearly better on margins
(10.50 points of error against ridge's 10.64, walk-forward over 2017-2026). They were reading a
leak. That version marked a player "available" if he logged minutes, and garbage time hands
minutes to the end of the bench only in blowouts - so the trees had learned to see a blowout
coming from who got into the game. With availability rebuilt from pre-game information only
(see :mod:`nba.players`) the trees fell to 10.66 and ridge held at 10.60. A linear model on
well-built ratings is also the right shape here: basketball scoring is close to additive in the
things that move it, there are only ~1,300 games a season, and the coefficients can be read.

Features are standardised with the training means and deviations, which are saved alongside
the coefficients, so a saved model reproduces its predictions exactly. Rows can be weighted -
the models weight recent seasons more, because the sport keeps moving: pace, three-point
volume, home-court advantage and scoring have all shifted inside the training window.
"""
from __future__ import annotations

import json

import numpy as np


class Ridge:
    def __init__(self, features: list[str], alpha: float = 10.0):
        self.features = list(features)
        self.alpha = float(alpha)
        self.mu = self.sd = self.coef = None
        self.intercept = 0.0

    def _X(self, df) -> np.ndarray:
        X = df[self.features].to_numpy(float)
        return np.where(np.isfinite(X), X, 0.0)

    def fit(self, df, y, weights=None) -> "Ridge":
        X = self._X(df)
        yy = np.asarray(df[y] if isinstance(y, str) else y, float)
        ok = np.isfinite(yy)
        X, yy = X[ok], yy[ok]
        w = np.ones(len(yy)) if weights is None else np.asarray(weights, float)[ok]
        w = w / w.mean()
        self.mu = (w[:, None] * X).sum(0) / w.sum()
        self.sd = np.sqrt((w[:, None] * (X - self.mu) ** 2).sum(0) / w.sum()) + 1e-9
        Z = (X - self.mu) / self.sd
        ym = float((w * yy).sum() / w.sum())
        A = Z.T @ (w[:, None] * Z) + self.alpha * np.eye(Z.shape[1])
        self.coef = np.linalg.solve(A, Z.T @ (w * (yy - ym)))
        self.intercept = ym
        return self

    def predict(self, df) -> np.ndarray:
        Z = (self._X(df) - self.mu) / self.sd
        return self.intercept + Z @ self.coef

    def to_dict(self) -> dict:
        return {"features": self.features, "alpha": self.alpha, "intercept": self.intercept,
                "coef": [float(c) for c in self.coef], "mu": [float(m) for m in self.mu],
                "sd": [float(s) for s in self.sd]}

    @classmethod
    def from_dict(cls, d: dict) -> "Ridge":
        m = cls(d["features"], d.get("alpha", 10.0))
        m.intercept = float(d["intercept"])
        m.coef = np.asarray(d["coef"], float)
        m.mu = np.asarray(d["mu"], float)
        m.sd = np.asarray(d["sd"], float)
        return m

    def save(self, path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "Ridge":
        return cls.from_dict(json.loads(path.read_text()))

    def raw(self, feature: str) -> float:
        """Points per unit of `feature` on its own scale, not per standard deviation."""
        i = self.features.index(feature)
        return float(self.coef[i] / self.sd[i])

    def effects(self) -> dict:
        """Points per standard deviation of each feature, largest first."""
        pairs = sorted(zip(self.features, self.coef), key=lambda kv: -abs(kv[1]))
        return {f: round(float(c), 3) for f, c in pairs}


def recency_weights(seasons, target_season: int, half_life: float = 4.0) -> np.ndarray:
    s = np.asarray(seasons, float)
    return 0.5 ** ((target_season - 1 - s) / half_life)
