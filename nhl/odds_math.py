"""Prices: American at the books, decimal everywhere inside this package.

The one rule this module exists to enforce is that **prices are aggregated in decimal, never in
American odds.** American odds are discontinuous at +/-100: there is no price between -100 and
+100, so an average or median across books that straddle even money is meaningless. The median
of -115 and +105 is -5, which reads as a 20x payout, and the first backtest of this pipeline did
exactly that and reported a +2,000% ROI on totals. Everything below takes or returns decimal
unless its name says otherwise.
"""
from __future__ import annotations

import numpy as np


def american_to_decimal(a):
    """-150 -> 1.667, +130 -> 2.30. Works on scalars and arrays; |a| < 100 is not a price."""
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(a >= 100, 1 + a / 100, np.where(a <= -100, 1 + 100 / np.abs(a), np.nan))
    return out if out.ndim else float(out)


def decimal_to_american(d):
    d = np.asarray(d, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(d >= 2, (d - 1) * 100, np.where(d > 1, -100 / (d - 1), np.nan))
    return out if out.ndim else float(out)


def implied(d):
    """Raw implied probability of a decimal price, vig included."""
    d = np.asarray(d, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(d > 1, 1 / d, np.nan)
    return out if out.ndim else float(out)


def devig_two(d_a, d_b):
    """Vig-free probability of side A in a two-way market, proportional method.

    Proportional is the standard for two-way markets. It does leave a little of the
    favourite-longshot bias in a lopsided puck line (+150/-180), which is worth knowing when
    reading an edge on the +1.5 dog: part of it may be the de-vig rather than the model.
    """
    pa, pb = implied(d_a), implied(d_b)
    s = np.asarray(pa, float) + np.asarray(pb, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((s > 0.98) & (s < 1.25), pa / s, np.nan)
    return out if np.ndim(out) else float(out)


def ev(p_win, p_push, dec):
    """Expected profit per unit staked, with a push returning the stake."""
    p_win, p_push, dec = (np.asarray(x, float) for x in (p_win, p_push, dec))
    p_loss = np.clip(1 - p_win - p_push, 0, 1)
    out = p_win * (dec - 1) - p_loss
    return out if out.ndim else float(out)


def kelly(p_win, p_push, dec, fraction: float) -> float:
    """Fractional Kelly on a decimal price, conditioned on the bet not pushing."""
    if any(x is None or x != x for x in (p_win, p_push, dec)) or dec <= 1:
        return 0.0
    live = 1 - p_push
    if live <= 0:
        return 0.0
    p = p_win / live
    b = dec - 1
    f = (p * (b + 1) - 1) / b
    return max(0.0, f) * fraction
