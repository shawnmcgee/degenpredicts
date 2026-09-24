"""Prices and probabilities: American at the books, decimal everywhere inside this package.

Two rules this module exists to enforce.

**Prices are aggregated in decimal, never in American odds.** American odds are discontinuous at
+/-100, so an average or median across books that straddle even money is meaningless: the median
of -115 and +105 is -5, which reads as a 20x payout. The NHL pipeline's first backtest did exactly
that and reported a +2,000% ROI; this module is where that mistake is kept out of the NBA.

**A final margin is an integer that is never zero, and a whole-number line pushes.** Reading a
cover probability off a continuous normal ignores both: it puts probability on ties that cannot
happen and none on the push that returns the stake on a 7-point spread when the favourite wins by
seven. So cover and over/under probabilities are read off a normal DISCRETISED onto the
integers - zero removed for margins - and every price, EV and grade carries a push leg.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


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

    Books with an impossible overround (under 0.98 or over 1.25) are refused rather than
    normalised: they are a parsing error or a stale quote, not a price.
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


# --- the discretised normal ---------------------------------------------------------------
_K = np.arange(-80, 81)          # home margins; every NBA final in the data sits well inside
_T = np.arange(120, 361)         # totals


def _mass(mu, sigma, support, drop_zero: bool):
    mu = np.atleast_1d(np.asarray(mu, float))[:, None]
    sigma = np.atleast_1d(np.asarray(sigma, float))[:, None]
    w = norm.pdf((support[None, :] - mu) / sigma)
    if drop_zero:
        w = np.where(support[None, :] == 0, 0.0, w)
    return w / w.sum(axis=1, keepdims=True)


def cover(mu, sigma, home_line):
    """(P home covers, P push, P away covers) for a home spread in book convention.

    ``home_line`` -6.5 means the home side gives 6.5: it covers when margin - 6.5 > 0. ``mu`` is
    the expected home margin. Vectorised over games.
    """
    m = _mass(mu, sigma, _K, drop_zero=True)
    adj = _K[None, :] + np.atleast_1d(np.asarray(home_line, float))[:, None]
    h = (m * (adj > 0)).sum(1)
    p = (m * (adj == 0)).sum(1)
    return h, p, 1 - h - p


def over_under(mu, sigma, line):
    """(P over, P push, P under) for a total line. Vectorised over games."""
    m = _mass(mu, sigma, _T, drop_zero=False)
    ln = np.atleast_1d(np.asarray(line, float))[:, None]
    o = (m * (_T[None, :] > ln)).sum(1)
    p = (m * (_T[None, :] == ln)).sum(1)
    return o, p, 1 - o - p


def win_prob(mu, sigma):
    """P(home wins) - there are no ties, so this is P(margin > 0) on the zero-free support."""
    h, _, _ = cover(mu, sigma, np.zeros(np.size(mu)))
    return h


def margin_from_prob(p, sigma):
    """The expected home margin a win probability implies - inverts :func:`win_prob`.

    Used to read the market's moneyline as a margin, so a moneyline-only game still has a
    market number on the same scale as the model's.
    """
    p = np.clip(np.asarray(p, float), 0.005, 0.995)
    mu = sigma * norm.ppf(p)
    for _ in range(3):                 # the discretisation shifts it slightly; three steps fix it
        mu = mu + sigma * (norm.ppf(p) - norm.ppf(np.clip(win_prob(mu, sigma), 0.005, 0.995)))
    return mu
