"""Key-number-aware probabilities for NFL spreads and totals.

A normal curve treats every margin alike. NFL games do not: since 2015, 14.8% have ended on
exactly 3 points and 8.7% on exactly 7, and a 3-point favourite wins by exactly 3 about one game
in ten. That mass is the whole difference between +2.5 and +3.5 - roughly ten points of cover
rate - and a bell curve smears it across the neighbouring half-points, so it misprices exactly the
numbers worth shopping for.

So the distribution is read from history instead. For an expected margin of m points it is the
final margins of past games whose closing spread was near m, oriented to the favourite and
weighted by how near; totals work the same way off the closing total. A little of a normal curve
is mixed in so a margin nobody has seen at that spread is unlikely rather than impossible.

The table supplies the SHAPE - where the spikes are, how often a number pushes - and never the
direction. Each distribution is tilted so its market number is its midpoint: history leans
slightly under (48.9% overs since 2015, anywhere from 44.5% to 53.9% by season), and left in, that
lean would quietly make every under on the board look like a bet. Which side to take is the
model's call, through the expectation it hands in.

2015 onward only: the extra point moved back that season, and margins of 7 and 8 have not been
as common since.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

FROM_SEASON = 2015
MARGINS = np.arange(-70, 71)
TOTALS = np.arange(0, 131)
SPREAD_GRID = np.arange(0.0, 24.01, 0.5)       # the favourite's expected margin
TOTAL_GRID = np.arange(30.0, 62.01, 0.5)
BANDWIDTH = {"margin": 1.0, "total": 1.5}       # points either side that count as "near"
MIN_EFFECTIVE = 150                             # widen the window until this many games inform it
NORMAL_MIX = 0.05
SIGMA = {"margin": 13.3, "total": 13.3}


def payout(price) -> float:
    """Profit per unit staked at an American price; -110 when there is none."""
    if price is None or price != price:
        return 100 / 110
    p = float(price)
    return p / 100 if p > 0 else 100 / -p


def _pmf_table(x: np.ndarray, y: np.ndarray, grid: np.ndarray, support: np.ndarray,
               h0: float, sigma: float) -> np.ndarray:
    """One PMF over ``support`` per grid point: outcomes ``y`` of games whose expectation ``x``
    sat near it, kernel-weighted, the window widened until enough games count."""
    onehot = (y[:, None] == support[None, :]).astype(float)
    table = np.zeros((len(grid), len(support)))
    for i, g in enumerate(grid):
        h = h0
        while True:
            w = np.exp(-0.5 * ((x - g) / h) ** 2)
            n_eff = w.sum() ** 2 / max((w ** 2).sum(), 1e-12)
            if n_eff >= MIN_EFFECTIVE or h > 8:
                break
            h *= 1.5
        emp = w @ onehot / max(w.sum(), 1e-12)
        edges = np.append(support - 0.5, support[-1] + 0.5)
        smooth = np.diff(norm.cdf(edges, loc=g, scale=sigma))
        table[i] = _center((1 - NORMAL_MIX) * emp + NORMAL_MIX * smooth / smooth.sum(),
                           support, g)
    return table


def _center(pmf: np.ndarray, support: np.ndarray, mid: float) -> np.ndarray:
    """Tilt ``pmf`` (exponentially, so relative spikes survive) until ``mid`` is its midpoint:
    as likely to finish above it as below."""
    def gap(theta):
        w = pmf * np.exp(theta * (support - mid))
        w = w / w.sum()
        return w[support > mid].sum() - w[support < mid].sum(), w
    lo, hi = -1.0, 1.0
    for _ in range(60):
        th = (lo + hi) / 2
        g, w = gap(th)
        if g > 0:
            hi = th
        else:
            lo = th
    return gap((lo + hi) / 2)[1]


def _interp(table: np.ndarray, grid: np.ndarray, v: float) -> np.ndarray:
    v = float(np.clip(v, grid[0], grid[-1]))
    j = int(np.clip(np.searchsorted(grid, v) - 1, 0, len(grid) - 2))
    a = (v - grid[j]) / (grid[j + 1] - grid[j])
    return (1 - a) * table[j] + a * table[j + 1]


class KeyNumbers:
    """P(final margin) and P(final total) given an expected margin or total."""

    def __init__(self, games: pd.DataFrame, lines: pd.DataFrame, from_season: int = FROM_SEASON):
        g = games.copy()
        g["game_id"] = g["game_id"].astype(str)
        ln = lines[["game_id", "spread_home", "total_line"]].copy()
        ln["game_id"] = ln["game_id"].astype(str)
        g = g.drop(columns=[c for c in ("spread_home", "total_line") if c in g])
        d = g.merge(ln, on="game_id", how="inner")
        done = d["completed"].astype(bool) & d["home_points"].notna() & d["away_points"].notna()
        d = d[done & (d["season"] >= from_season)]
        margin = (d["home_points"] - d["away_points"]).round().astype(int).values
        total = (d["home_points"] + d["away_points"]).round().astype(int).values
        sp = d["spread_home"].values.astype(float)
        ok = np.isfinite(sp)
        fav = -sp[ok]                                 # the market's expected home margin
        fav_margin = np.where(fav >= 0, margin[ok], -margin[ok])
        self.n_margin = int(ok.sum())
        self.margin_table = _pmf_table(np.abs(fav), fav_margin, SPREAD_GRID, MARGINS,
                                       BANDWIDTH["margin"], SIGMA["margin"])
        tl = d["total_line"].values.astype(float)
        okt = np.isfinite(tl)
        self.n_total = int(okt.sum())
        self.total_table = _pmf_table(tl[okt], total[okt], TOTAL_GRID, TOTALS,
                                      BANDWIDTH["total"], SIGMA["total"])

    @property
    def usable(self) -> bool:
        return self.n_margin >= MIN_EFFECTIVE and self.n_total >= MIN_EFFECTIVE

    def margin_pmf(self, mu: float) -> np.ndarray:
        """P(home margin = k) for k in MARGINS, for an expected home margin ``mu``."""
        fav = _interp(self.margin_table, SPREAD_GRID, abs(mu))
        return fav if mu >= 0 else fav[::-1]           # MARGINS is symmetric about 0

    def spread(self, mu: float, line: float, home: bool) -> tuple[float, float, float]:
        """(P win, P push, P loss) for one side at its own spread - ``line`` is the points that
        side gets, so a home -3 is ``line=-3`` and an away +3.5 is ``line=3.5``."""
        pmf = self.margin_pmf(mu)
        side = MARGINS if home else -MARGINS          # the side's own margin
        adj = side + line
        return float(pmf[adj > 0].sum()), float(pmf[adj == 0].sum()), float(pmf[adj < 0].sum())

    def total(self, expected: float, line: float, over: bool) -> tuple[float, float, float]:
        """(P win, P push, P loss) for the over or the under at ``line``."""
        pmf = _interp(self.total_table, TOTAL_GRID, expected)
        win = TOTALS > line if over else TOTALS < line
        lose = TOTALS < line if over else TOTALS > line
        return float(pmf[win].sum()), float(pmf[TOTALS == line].sum()), float(pmf[lose].sum())

    @staticmethod
    def ev(probs: tuple[float, float, float], price) -> float:
        """Expected profit per unit at an American price; a push returns the stake."""
        p_win, _, p_loss = probs
        return p_win * payout(price) - p_loss

    def worst_spread(self, mu: float, home: bool, price=-110, span: float = 14.0) -> float:
        """The fewest points this side can take and still have positive EV at ``price``:
        the "bet at X or better" number. Searched within ``span`` of the side's fair number, so
        a 17-point underdog is priced as readily as a pick'em; NaN if nothing there is."""
        fair = round((-mu if home else mu) * 2) / 2
        lines = np.arange(fair - span, fair + span + 0.01, 0.5)
        ok = [x for x in lines if self.ev(self.spread(mu, x, home), price) > 0]
        return float(min(ok)) if ok else np.nan

    def worst_total(self, expected: float, over: bool, price=-110, span: float = 14.0) -> float:
        """The highest total an over (lowest an under) is still worth at ``price``."""
        lines = np.arange(round(expected * 2) / 2 - span, round(expected * 2) / 2 + span + 0.01, 0.5)
        ok = [x for x in lines if self.ev(self.total(expected, x, over), price) > 0]
        if not ok:
            return np.nan
        return float(max(ok)) if over else float(min(ok))
