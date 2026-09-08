"""Decimal odds, and dividing the vig out of a three-way market.

Two things here have no counterpart in the other pipelines.

**Odds are decimal.** Every source for this sport - football-data.co.uk, the European books,
the exchanges - quotes decimal. A price of 2.50 returns 2.50 per unit staked including the
stake, so the profit multiple this project's EV maths wants is ``price - 1``. American odds
never appear, so no conversion layer exists to get backwards.

**The market is three-way, and that changes how the vig comes out.** With two outcomes at
1.95/1.95, the implied probabilities sum to 1.026 and dividing each by the sum is exactly
right. With three outcomes it is not, because the overround is not spread evenly: bookmakers
load proportionally more margin onto the longshot. This is the favourite-longshot bias, it is
one of the most replicated findings in the sports-betting literature, and it is large enough
in football to matter - a 15.0 away price at a typical book implies about 6.7% under
proportional de-vigging when the true probability is nearer 5.5%.

Proportional de-vigging therefore hands the model a systematically *inflated* probability on
every longshot, which is exactly where a "value" bet looks most attractive and where the model
is most likely to be talking itself into one. Shin's method is the standard correction: it
models the bookmaker as pricing against a proportion ``z`` of insider money and backs out the
probabilities that assumption implies. It shrinks longshots and lifts favourites, in the
direction the empirical bias goes.

All three methods are here because the right answer is "check whether it matters" rather than
"trust the literature": ``method="proportional"`` reproduces the naive figure, and the
difference between the two is reported in the training metadata.
"""
from __future__ import annotations

import numpy as np

# Bookmaker margins outside this range mean the quote is broken - a stale price, a suspended
# market, or a column read from the wrong place in a CSV whose layout changed. A market that
# sums to meaningfully less than 1 is an arbitrage, which does not happen at a real book and
# always means a data error; one over 1.5 is not a football market.
#
# The floor sits a hair below 1 rather than at it. An exactly-fair set of prices - which is what
# a round-trip through the scoreline model produces, and what a zero-commission exchange
# approaches - sums to 1.0 in exact arithmetic and to 0.9999999 in floating point. Rejecting
# that as "broken" would silently return NaN for the one input that is definitionally correct.
# The 0.1% slack is far too small to admit a real mispriced book.
MIN_OVERROUND = 0.999
MAX_OVERROUND = 1.5


def decimal_prob(price) -> float:
    """Raw implied probability from a decimal price, before any vig removal."""
    try:
        p = float(price)
    except (TypeError, ValueError):
        return float("nan")
    return 1.0 / p if p > 1.0 else float("nan")


def payout(price) -> float:
    """Profit multiple per unit staked: what EV and Kelly want. 2.50 -> 1.50."""
    try:
        p = float(price)
    except (TypeError, ValueError):
        return float("nan")
    return p - 1.0 if p > 1.0 else float("nan")


def overround(*prices) -> float:
    """Sum of implied probabilities. 1.05 means a 5% book margin."""
    ps = [decimal_prob(p) for p in prices]
    if any(p != p for p in ps) or not ps:
        return float("nan")
    return float(sum(ps))


def _proportional(pi: np.ndarray) -> np.ndarray:
    return pi / pi.sum()


def _shin_z(pi: np.ndarray, tol: float = 1e-10, iters: int = 100) -> float:
    """Solve for Shin's insider-trading proportion z, by bisection on [0, 0.35].

    ``sum_i p_i(z) = 1`` is monotonic in z over this range, so bisection is both sufficient and
    far more robust than the fixed-point iteration usually quoted for this - that one diverges
    on the lopsided books football produces, where a 1.05 favourite sits next to a 34.0
    longshot.
    """
    phi = float(pi.sum())
    if phi <= 1.0:
        return 0.0

    def total(z: float) -> float:
        if z >= 1.0:
            return np.inf
        return float((np.sqrt(z * z + 4 * (1 - z) * pi * pi / phi) - z).sum() / (2 * (1 - z)))

    lo, hi = 0.0, 0.35
    if total(hi) > 1.0:          # margin too big for the model; fall back to proportional
        return float("nan")
    for _ in range(iters):
        mid = (lo + hi) / 2
        if total(mid) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def _shin(pi: np.ndarray) -> np.ndarray:
    z = _shin_z(pi)
    if z != z:
        return _proportional(pi)
    phi = float(pi.sum())
    p = (np.sqrt(z * z + 4 * (1 - z) * pi * pi / phi) - z) / (2 * (1 - z))
    s = p.sum()
    return p / s if s > 0 else _proportional(pi)


def _power(pi: np.ndarray, tol: float = 1e-10, iters: int = 200) -> np.ndarray:
    """Odds-ratio / power method: find k such that sum(pi_i ** k) == 1.

    A second view on the same correction, with a different functional form. Reported alongside
    Shin so a reader can see whether the choice of method is doing any of the work - if the two
    disagree by more than a point on a real price, neither should be trusted much.
    """
    lo, hi = 0.5, 3.0
    for _ in range(iters):
        k = (lo + hi) / 2
        s = float((pi ** k).sum())
        if s > 1.0:
            lo = k
        else:
            hi = k
        if hi - lo < tol:
            break
    k = (lo + hi) / 2
    p = pi ** k
    return p / p.sum()


METHODS = {"proportional": _proportional, "shin": _shin, "power": _power}


def devig(prices, method: str = "shin") -> np.ndarray:
    """Vig-free probabilities from a list of decimal prices covering every outcome.

    Works for any number of outcomes, so the same function serves the three-way 1X2 market and
    the two-way over/under and Asian handicap markets. Returns NaNs rather than guessing when
    a price is missing or the book does not look like a book.
    """
    pi = np.array([decimal_prob(p) for p in prices], dtype=float)
    if pi.size == 0 or np.isnan(pi).any():
        return np.full(max(pi.size, 1), np.nan)
    phi = pi.sum()
    if not (MIN_OVERROUND <= phi <= MAX_OVERROUND):
        return np.full(pi.size, np.nan)
    fn = METHODS.get(method, _shin)
    return fn(pi)


def devig_three(home, draw, away, method: str = "shin") -> tuple[float, float, float]:
    """(P home, P draw, P away) with the vig removed from a 1X2 market."""
    p = devig([home, draw, away], method=method)
    return float(p[0]), float(p[1]), float(p[2])


def devig_two(a, b, method: str = "shin") -> tuple[float, float]:
    """Two-way market - over/under, or the two sides of an Asian handicap."""
    p = devig([a, b], method=method)
    return float(p[0]), float(p[1])


def supremacy_from_prices(home, draw, away, method: str = "shin") -> float:
    """The market's implied goal supremacy, backed out of a 1X2 price.

    Inverts :func:`epl.poisson.match_odds` on a fixed league total, so a 1X2 price can be
    compared against the model on the model's own scale. Used when a match has a 1X2 price but
    no Asian handicap - common in the older seasons, where football-data carries 1X2 back to
    2000-01 but handicaps only from 2006-07.
    """
    from .poisson import grid, match_odds
    from .ratings import LEAGUE_GPG

    ph, pd_, pa = devig_three(home, draw, away, method=method)
    if ph != ph or pa != pa:
        return float("nan")
    lo, hi = -4.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        h, _d, a = match_odds(grid(mid, 2 * LEAGUE_GPG))
        if h != h:
            return float("nan")
        # The home/away ratio is monotonic in supremacy and, unlike the raw home probability,
        # does not move when the total does - so this inversion is stable even though the true
        # total for the match is unknown at this point.
        if (h - a) < (ph - pa):
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3)


def total_from_prices(over, under, line: float, method: str = "shin") -> float:
    """The market's implied total goals, backed out of an over/under price."""
    from .poisson import grid, over_under

    po, _pu = devig_two(over, under, method=method)
    if po != po or line != line:
        return float("nan")
    lo, hi = 0.5, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        o, _p, _u = over_under(grid(0.0, mid), float(line))
        if o != o:
            return float("nan")
        if o < po:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 3)
