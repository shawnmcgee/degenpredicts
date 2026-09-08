"""The scoreline distribution: Dixon-Coles bivariate Poisson over a grid of results.

This module is the reason the EPL pipeline is not just the NFL one with the word "points"
replaced by "goals". The other two sports model a margin as a Gaussian around a point
prediction and read probabilities off the normal CDF. That works when a margin is a
continuous-ish quantity with a standard deviation of thirteen. It does not work here:

* **The scale is tiny and discrete.** A Premier League match produces about 2.8 goals. The
  entire distribution lives on the integers 0-5, so a normal approximation is not
  approximating anything - the difference between a 0.4-goal and a 0.6-goal supremacy is
  several percentage points of win probability, and a continuous model smears it.
* **The draw is a real outcome**, about 24% of matches, and a Gaussian margin model has no way
  to express it. ``P(margin == 0)`` is zero under a continuous density; you would have to
  bolt on a fudge, and the fudge would be doing the most important work in the model.
* **Low scores are correlated.** Independent Poisson marginals systematically understate 0-0
  and 1-1 and overstate 1-0 and 0-1. Those four scorelines are roughly a fifth of all matches
  and they are exactly the ones that decide whether a match is drawn.

So instead of a point prediction plus a sigma, the models predict two quantities - **supremacy**
(home goals minus away goals) and **total goals** - and this module turns that pair into a full
joint distribution over scorelines. Every market is then read off the same grid, which means
the 1X2 price, the Asian handicap and the over/under can never contradict each other. That
internal consistency is not a nicety: it is what makes it safe to compare our number against
three different markets on the same match.

    lambda_home = (total + supremacy) / 2
    lambda_away = (total - supremacy) / 2
    P(i, j)     = Poisson(i; lh) * Poisson(j; la) * tau(i, j)     # Dixon-Coles correction

``tau`` is the Dixon-Coles low-score adjustment, which lifts 0-0 and 1-1 and trims 1-0 and 0-1
by a single fitted parameter ``rho``. Dixon and Coles fitted -0.13 on English football of the
early nineties; the modern, higher-scoring league fits nearer -0.04, which is ``config.DC_RHO``.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from . import config

# Expected goals are floored rather than allowed to reach zero. A model that has been handed a
# supremacy larger than the total - which a tree can do on an extreme extrapolation - would
# otherwise produce a negative lambda, and scipy answers that with silent NaNs across the whole
# grid rather than an error. 0.05 goals is far below any real team-match and keeps the grid finite.
MIN_LAMBDA = 0.05


def lambdas(supremacy: float, total: float) -> tuple[float, float]:
    """Split a (supremacy, total) prediction into the two sides' expected goals."""
    if supremacy != supremacy or total != total:
        return float("nan"), float("nan")
    total = max(float(total), 2 * MIN_LAMBDA)
    lh = (total + float(supremacy)) / 2.0
    la = (total - float(supremacy)) / 2.0
    return max(lh, MIN_LAMBDA), max(la, MIN_LAMBDA)


def _tau(lh: float, la: float, rho: float, n: int) -> np.ndarray:
    """The Dixon-Coles correction, as a multiplicative mask over the scoreline grid.

    Only the four lowest scorelines are touched; everything else is 1. The correction has to
    keep every cell positive, so rho is clamped to the range in which it does - an unclamped
    rho with two low-scoring sides can drive tau(0,0) negative and produce negative
    "probabilities" that still sum to one after normalisation, which is the kind of bug that
    survives every sanity check except a direct one.
    """
    t = np.ones((n, n))
    if not rho:
        return t
    lo = -1.0 / max(lh * la, 1e-9)                       # keeps tau(0,0) > 0
    hi = min(1.0 / max(lh, 1e-9), 1.0 / max(la, 1e-9))   # keeps tau(0,1), tau(1,0) > 0
    r = float(np.clip(rho, max(lo, -1.0), min(hi, 1.0)))
    t[0, 0] = 1.0 - lh * la * r
    t[0, 1] = 1.0 + lh * r
    t[1, 0] = 1.0 + la * r
    t[1, 1] = 1.0 - r
    return np.maximum(t, 1e-12)


def grid(supremacy: float, total: float, rho: float | None = None,
         max_goals: int | None = None) -> np.ndarray:
    """Joint P(home goals = i, away goals = j) as an (n+1) x (n+1) array.

    Truncated at ``max_goals`` per side and renormalised, so the tail beyond the grid is
    redistributed rather than silently lost. At 10 goals a side the truncated mass is under
    one part in ten thousand.
    """
    n = (max_goals if max_goals is not None else config.MAX_GOALS) + 1
    lh, la = lambdas(supremacy, total)
    if lh != lh or la != la:
        return np.full((n, n), np.nan)
    k = np.arange(n)
    m = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
    m = m * _tau(lh, la, config.DC_RHO if rho is None else rho, n)
    s = m.sum()
    return m / s if s > 0 else m


# ---------------------------------------------------------------------------------
# Reading markets off the grid
# ---------------------------------------------------------------------------------
def match_odds(m: np.ndarray) -> tuple[float, float, float]:
    """(P home win, P draw, P away win) - the 1X2 market."""
    if m.size == 0 or not np.isfinite(m).all():
        return float("nan"), float("nan"), float("nan")
    n = m.shape[0]
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return float(m[i > j].sum()), float(np.trace(m)), float(m[i < j].sum())


def over_under(m: np.ndarray, line: float) -> tuple[float, float, float]:
    """(P over, P push, P under) for a total-goals line.

    The push leg is not decoration. Football over/under markets are quoted at whole numbers as
    often as at halves - "over 3 goals" is a standard price - and a whole-number line pushes on
    an exact hit, which is ~15% of matches at the 3-goal line. Folding that into either side
    would misprice the bet by more than any edge the model could plausibly have.
    """
    if m.size == 0 or not np.isfinite(m).all() or line != line:
        return float("nan"), float("nan"), float("nan")
    n = m.shape[0]
    tot = np.arange(n)[:, None] + np.arange(n)[None, :]
    over = float(m[tot > line].sum())
    push = float(m[tot == line].sum())
    return over, push, float(m[tot < line].sum())


def _ah_half(m: np.ndarray, handicap: float) -> tuple[float, float, float]:
    """Asian handicap for a line that is a whole or half number, from the HOME side.

    ``handicap`` is added to the home team's goals: -1.0 means the home side gives a goal.
    """
    n = m.shape[0]
    adj = (np.arange(n)[:, None] - np.arange(n)[None, :]) + handicap
    return float(m[adj > 0].sum()), float(m[adj == 0].sum()), float(m[adj < 0].sum())


def asian_handicap(m: np.ndarray, handicap: float) -> tuple[float, float, float]:
    """(P win, P push, P loss) on the HOME side of an Asian handicap.

    Quarter lines (-0.25, -0.75, +1.25 ...) are the reason this is not a one-liner. They are
    not a single bet: the stake is split across the two adjacent half/whole lines, so -0.75 is
    half at -0.5 and half at -1.0. Half of that stake can win while the other half pushes,
    which no single-outcome formula can express. Treating -0.75 as if it were -0.5 or -1.0 -
    or rounding it away - misprices a line that is quoted on most Premier League matches.

    The two halves are averaged here, which gives the correct *expected* result and correct EV.
    """
    if m.size == 0 or not np.isfinite(m).all() or handicap != handicap:
        return float("nan"), float("nan"), float("nan")
    h = float(handicap)
    q = round(h * 4) / 4
    if abs(q * 2 - round(q * 2)) < 1e-9:          # whole or half line: a single bet
        return _ah_half(m, q)
    lo, hi = q - 0.25, q + 0.25                   # quarter line: split across the neighbours
    a, b = _ah_half(m, lo), _ah_half(m, hi)
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


def both_teams_score(m: np.ndarray) -> float:
    if m.size == 0 or not np.isfinite(m).all():
        return float("nan")
    return float(m[1:, 1:].sum())


def most_likely_score(m: np.ndarray) -> tuple[int, int, float]:
    if m.size == 0 or not np.isfinite(m).all():
        return 0, 0, float("nan")
    i, j = np.unravel_index(int(np.argmax(m)), m.shape)
    return int(i), int(j), float(m[i, j])


def expected(m: np.ndarray) -> tuple[float, float]:
    """(expected supremacy, expected total) implied by the grid.

    Not the same as the inputs once the Dixon-Coles correction and truncation are applied, so
    this is what the site should quote when it says "our number".
    """
    if m.size == 0 or not np.isfinite(m).all():
        return float("nan"), float("nan")
    n = m.shape[0]
    k = np.arange(n)
    eh = float((m.sum(axis=1) * k).sum())
    ea = float((m.sum(axis=0) * k).sum())
    return eh - ea, eh + ea


def fair_handicap(m: np.ndarray) -> float:
    """The handicap at which the home side is a 50/50 - the market's "true" Asian line.

    Solved on the quarter-goal ladder that books actually quote rather than continuously,
    because a fair line of -0.63 is not a thing anyone can bet.
    """
    if m.size == 0 or not np.isfinite(m).all():
        return float("nan")
    best, gap = float("nan"), np.inf
    for h in np.arange(-6.0, 6.01, 0.25):
        w, p, l = asian_handicap(m, float(h))
        if w != w or (w + l) <= 0:
            continue
        d = abs(w / (w + l) - 0.5)
        if d < gap:
            best, gap = float(h), d
    return best
