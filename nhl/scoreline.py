"""The final-score distribution, and the reason this is not the EPL pipeline with "goals" renamed.

The Premier League reads every market off a Dixon-Coles Poisson grid. Hockey breaks that model
in the last three minutes of every close game. A team trailing by one or two pulls its goalie
for an extra attacker; its own scoring rate roughly doubles, and the leader starts scoring into
an empty net. The effect on the margin is enormous and it is exactly where the puck line lives.
Regulation margins across 2021-2025, against independent Poisson at the same scoring rates:

    |margin|   actual   Poisson
       0       22.2%     16.6%    <- goes to overtime
       1       17.7%     30.3%    <- one-goal games get tied up or iced
       2       19.4%     23.2%
       3       23.9%     15.0%    <- two-goal games become three

A Poisson or Skellam model puts 30% of games on a one-goal margin when 18% actually land there,
so it misprices every -1.5 in the league. Empty-net goals have also more than doubled since
2007 (0.17 to 0.39 a game) as coaches pull earlier, so the parameters are fitted on recent
seasons at every retrain rather than fixed here.

So the grid is built in two phases:

* **Minutes 0 to 50:** independent Poisson at the base rates - goals scored against a goalie.
* **The last ten minutes:** a Markov chain over the score, stepped every 20 seconds. A side down
  by one or two goes into a CHASE - it scores at ``m6`` times its base rate and the leader picks
  up ``en`` extra goals a minute (into the empty net, and on the counter). Tied games tighten up
  (``tie``): the side that would lose still banks a point in overtime, and it shows.

The chase is deliberately one effect, not two. It blends the ordinary score effect (a trailing
team pushes for the last ten minutes) with the pulled goalie (the last two or three), because
box scores carry no goal times and the data cannot tell them apart: a fit that tried to
separate them landed on a trailing side scoring 2.5 times as fast as normal for ten minutes and
put 25% of games in overtime against a real 22%. So ``t1`` and ``t2`` are when the chase starts
for a side down one and down two - fitted at the edge of the window, which is the honest reading
that the effect builds for most of the last period - and ``w`` spreads that start over two
minutes. Fitted this way the grid reproduces 2022-2025 to the tenth of a point on everything the
puck line turns on: home -1.5 32.7% predicted and actual, away -1.5 27.3% against 27.2%,
one-goal margins 39.9% against 40.0%, overtime 22.0% against 22.3%.

Then overtime and the shootout: a tied regulation becomes a one-goal win, home with a
probability that leans toward the stronger side (``ot0``, ``ot1``), because 3-on-3 rewards
skill. The shootout winner is credited one goal - which is how the NHL publishes the final
score AND how books settle totals and puck lines, so the grid, the grade and the book agree.

Every market is then read off the same grid, so the moneyline, puck line and total can never
contradict each other.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

K = 15            # goals 0..14 per side; the mass past 14 is folded onto the edge
DT = 1 / 3        # minutes per step in the late phase

# Fitted on the 2022-2025 regular seasons (5,248 games) with the replayed ratings as base rates.
# Retraining refits these on the latest complete seasons; these are only the fallback.
DEFAULT_THETA = {"T": 10.0, "w": 1.0, "t1": 9.0, "t2": 9.0, "m6": 1.60, "en": 0.074,
                 "tie": 0.36, "c": 1.011, "ot0": 0.032, "ot1": 1.10}
# c_h and c_a are nuisance scales, fitted and then averaged into c (see fit_theta).
FIT_KEYS = ("t1", "t2", "m6", "en", "tie", "c_h", "c_a", "ot0", "ot1")
BOUNDS = {"t1": (0.3, 9.0), "t2": (0.3, 9.0), "m6": (0.5, 5.0), "en": (0.0, 1.5),
          "tie": (-1.0, 1.5), "c_h": (0.7, 1.4), "c_a": (0.7, 1.4), "ot0": (-1.0, 1.0),
          "ot1": (-3.0, 4.0)}


def grids(lh, la, th: dict | None = None, dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
    """(final, regulation) score grids, shape (n, K, K), for arrays of base rates.

    ``lh``/``la`` are expected goals scored against a goalie in regulation - the quantity the
    models predict - not final goals, which also contain empty-net and overtime goals that
    this function adds itself.
    """
    th = {**DEFAULT_THETA, **(th or {})}
    lh = np.atleast_1d(np.asarray(lh, float)) * th.get("c_h", th["c"])
    la = np.atleast_1d(np.asarray(la, float)) * th.get("c_a", th["c"])
    lh = np.clip(np.nan_to_num(lh, nan=3.0), 0.05, 12.0)
    la = np.clip(np.nan_to_num(la, nan=3.0), 0.05, 12.0)
    T = float(th["T"])
    k = np.arange(K)
    d = k[:, None] - k[None, :]
    frac = (60 - T) / 60
    P = poisson.pmf(k[None, :], (lh * frac)[:, None])[:, :, None] * \
        poisson.pmf(k[None, :], (la * frac)[:, None])[:, None, :]
    rh, ra = lh / 60.0, la / 60.0
    steps = int(round(T / dt))
    w = max(float(th["w"]), 1e-6)
    for s in range(steps):
        left = T - s * dt
        mh = np.ones((K, K))
        ma = np.ones((K, K))
        if th["tie"]:
            damp = np.exp(-th["tie"])
            mh = np.where(d == 0, damp, mh)
            ma = np.where(d == 0, damp, ma)
        # share of trailing sides that have pulled the goalie by now
        r1 = float(np.clip((th["t1"] + w - left) / (2 * w), 0.0, 1.0))
        r2 = float(np.clip((th["t2"] + w - left) / (2 * w), 0.0, 1.0))
        hp = np.where(d == -1, r1, np.where(d == -2, r2, 0.0))
        ap = np.where(d == 1, r1, np.where(d == 2, r2, 0.0))
        mh = np.where(hp > 0, (1 - hp) * mh + hp * th["m6"], mh)
        ma = np.where(ap > 0, (1 - ap) * ma + ap * th["m6"], ma)
        Rh = (rh[:, None, None] * mh[None] + (ap * th["en"])[None]) * dt
        Ra = (ra[:, None, None] * ma[None] + (hp * th["en"])[None]) * dt
        mv_h, mv_a = P * Rh, P * Ra
        P = P * (1 - Rh - Ra)
        P[:, 1:, :] += mv_h[:, :-1, :]
        P[:, -1, :] += mv_h[:, -1, :]
        P[:, :, 1:] += mv_a[:, :, :-1]
        P[:, :, -1] += mv_a[:, :, -1]
    p_ot = 1 / (1 + np.exp(-(th["ot0"] + th["ot1"] * np.log(lh / la))))
    F = P.copy()
    i = np.arange(K - 1)
    F[:, i, i] = 0.0
    F[:, i + 1, i] += P[:, i, i] * p_ot[:, None]
    F[:, i, i + 1] += P[:, i, i] * (1 - p_ot)[:, None]
    F[:, K - 1, K - 1] = 0.0
    F /= F.sum(axis=(1, 2), keepdims=True)
    return F, P


# ---------------------------------------------------------------------------------
# Reading markets off a grid (single game, 2-D)
# ---------------------------------------------------------------------------------
def _dt():
    k = np.arange(K)
    return k[:, None] - k[None, :], k[:, None] + k[None, :]


def moneyline(F: np.ndarray) -> float:
    """P(home wins), overtime and shootout included - hockey moneylines have no draw."""
    d, _ = _dt()
    return float(F[d > 0].sum())


def cover(F: np.ndarray, home_line: float) -> tuple[float, float, float]:
    """(P home covers, P push, P away covers) for a home puck line such as -1.5 or +1.5."""
    d, _ = _dt()
    adj = d + float(home_line)
    return float(F[adj > 0].sum()), float(F[adj == 0].sum()), float(F[adj < 0].sum())


def over_under(F: np.ndarray, line: float) -> tuple[float, float, float]:
    """(P over, P push, P under). Whole-number lines push, and 6.0 is the commonest line."""
    _, t = _dt()
    return float(F[t > line].sum()), float(F[t == line].sum()), float(F[t < line].sum())


def regulation_split(R: np.ndarray) -> tuple[float, float, float]:
    """(P home wins in regulation, P overtime, P away wins in regulation)."""
    d, _ = _dt()
    return float(R[d > 0].sum()), float(R[d == 0].sum()), float(R[d < 0].sum())


def expected(F: np.ndarray) -> tuple[float, float]:
    """(expected final margin, expected final total), empty-net and overtime goals included."""
    d, t = _dt()
    return float((F * d).sum()), float((F * t).sum())


# ---------------------------------------------------------------------------------
# A lookup table for pricing thousands of games at once, and for inverting the market
# ---------------------------------------------------------------------------------
class Table:
    """Margin and total distributions tabulated on (total rate, home share).

    Walk-forward evaluation prices tens of thousands of games and the market inversion needs
    Newton steps on top of that, so building a grid per game per step is far too slow. The
    distributions are smooth in both coordinates, and a linear blend of two probability mass
    functions is still one, so bilinear interpolation of the PMFs is exact enough and never
    produces a probability outside [0, 1].
    """
    MARGINS = np.arange(-(K - 1), K)
    TOTALS = np.arange(0, 2 * K - 1)

    def __init__(self, th: dict | None = None, tau=None, share=None):
        self.th = {**DEFAULT_THETA, **(th or {})}
        self.tau = np.arange(2.0, 10.01, 0.05) if tau is None else tau
        self.share = np.arange(0.25, 0.7501, 0.005) if share is None else share
        T, S = np.meshgrid(self.tau, self.share, indexing="ij")
        F, _ = grids((T * S).ravel(), (T * (1 - S)).ravel(), self.th)
        d, t = _dt()
        n = F.shape[0]
        mpmf = np.zeros((n, len(self.MARGINS)))
        tpmf = np.zeros((n, len(self.TOTALS)))
        for j, m in enumerate(self.MARGINS):
            mpmf[:, j] = F[:, d == m].sum(1)
        for j, v in enumerate(self.TOTALS):
            tpmf[:, j] = F[:, t == v].sum(1)
        shp = (len(self.tau), len(self.share))
        self.mpmf = mpmf.reshape(*shp, -1)
        self.tpmf = tpmf.reshape(*shp, -1)

    def _interp(self, arr, lh, la):
        lh = np.asarray(lh, float)
        la = np.asarray(la, float)
        # a game without a market number comes through as NaN and must come out as NaN - without
        # casting NaN to an index on the way, which is what floods the logs with warnings
        bad = ~(np.isfinite(lh) & np.isfinite(la))
        if bad.any():
            lh, la = np.where(bad, 3.0, lh), np.where(bad, 3.0, la)
        tau = np.clip(lh + la, self.tau[0], self.tau[-1])
        s = np.clip(lh / np.maximum(lh + la, 1e-9), self.share[0], self.share[-1])
        fi = (tau - self.tau[0]) / (self.tau[1] - self.tau[0])
        fj = (s - self.share[0]) / (self.share[1] - self.share[0])
        i0 = np.clip(np.floor(fi).astype(int), 0, len(self.tau) - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, len(self.share) - 2)
        a, b = (fi - i0)[:, None], (fj - j0)[:, None]
        out = ((1 - a) * (1 - b) * arr[i0, j0] + a * (1 - b) * arr[i0 + 1, j0]
               + (1 - a) * b * arr[i0, j0 + 1] + a * b * arr[i0 + 1, j0 + 1])
        if bad.any():
            out[bad] = np.nan
        return out

    def margin_pmf(self, lh, la):
        return self._interp(self.mpmf, lh, la)

    def total_pmf(self, lh, la):
        return self._interp(self.tpmf, lh, la)

    def p_home(self, lh, la):
        return self.margin_pmf(lh, la)[:, self.MARGINS > 0].sum(1)

    def cover(self, lh, la, home_line):
        """Vectorised (P home covers, P push) for per-game home lines."""
        pm = self.margin_pmf(lh, la)
        hl = np.asarray(home_line, float)[:, None]
        adj = self.MARGINS[None, :] + hl
        return (pm * (adj > 0)).sum(1), (pm * (adj == 0)).sum(1)

    def over(self, lh, la, line):
        """Vectorised (P over, P push) for per-game total lines."""
        pt = self.total_pmf(lh, la)
        ln = np.asarray(line, float)[:, None]
        return (pt * (self.TOTALS[None, :] > ln)).sum(1), (pt * (self.TOTALS[None, :] == ln)).sum(1)

    def e_total(self, lh, la):
        return self.total_pmf(lh, la) @ self.TOTALS

    def invert(self, p_home, line, p_over, iters: int = 30):
        """Market (de-vigged P home win, total line, de-vigged P over | no push) -> base rates.

        The moneyline pins down who is better and the over price at the posted line pins down
        how many goals, so two prices give exactly the two numbers the grid needs. That is the
        market's view in the model's own units, which is what makes "we differ by 0.2 goals"
        a meaningful sentence and what the market-aware model is trained against.
        """
        p_home = np.asarray(p_home, float)
        line = np.asarray(line, float)
        p_over = np.asarray(p_over, float)
        n = len(p_home)
        ok = np.isfinite(p_home) & np.isfinite(line) & np.isfinite(p_over)
        ln = np.where(ok, line, 6.0)
        tau, s = np.full(n, 5.6), np.full(n, 0.52)

        def f(tau_, s_):
            lh, la = tau_ * s_, tau_ * (1 - s_)
            o, pu = self.over(lh, la, ln)
            return self.p_home(lh, la), o / np.maximum(1 - pu, 1e-9)

        tgt1, tgt2 = np.where(ok, p_home, 0.5), np.where(ok, p_over, 0.5)
        for _ in range(iters):
            f1, f2 = f(tau, s)
            e = 1e-3
            g1t, g2t = f(tau + e, s)
            g1s, g2s = f(tau, s + e)
            a11, a21 = (g1t - f1) / e, (g2t - f2) / e
            a12, a22 = (g1s - f1) / e, (g2s - f2) / e
            det = a11 * a22 - a12 * a21
            det = np.where(np.abs(det) < 1e-9, 1e-9, det)
            r1, r2 = f1 - tgt1, f2 - tgt2
            tau = np.clip(tau - 0.8 * (a22 * r1 - a12 * r2) / det, self.tau[0], self.tau[-1])
            s = np.clip(s - 0.8 * (-a21 * r1 + a11 * r2) / det, self.share[0], self.share[-1])
        lh, la = tau * s, tau * (1 - s)
        return np.where(ok, lh, np.nan), np.where(ok, la, np.nan)


# ---------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------
def fit_theta(lh, la, home_goals, away_goals, x0: dict | None = None,
              maxiter: int = 200) -> dict:
    """Maximum likelihood on final scores, given each game's base rates.

    Rates are binned to 0.05 goals so each distinct pair is priced once; over a few thousand
    games that is ~800 grids per evaluation instead of thousands.

    The base rates used here come from the replayed ratings, whose home edge lags the league's
    (home ice has been shrinking). A single scale would make the shape parameters - overtime
    especially - absorb that lag. So home and away get a scale each while fitting, and only
    their average is kept: in production the rates come from the Poisson models, which learn
    the current home edge themselves, and a separate home scale would count it twice.

    The pull times are "effective" values. They also absorb the ordinary score effect - a
    trailing team pushes long before its goalie comes out - so they read earlier than the
    three minutes a stopwatch would show.
    """
    x0 = {**DEFAULT_THETA, **(x0 or {})}
    x0.setdefault("c_h", x0["c"])
    x0.setdefault("c_a", x0["c"])
    bh = np.round(np.asarray(lh, float) / 0.05) * 0.05
    ba = np.round(np.asarray(la, float) / 0.05) * 0.05
    pairs, inv = np.unique(np.column_stack([bh, ba]), axis=0, return_inverse=True)
    inv = inv.ravel()
    hg = np.clip(np.asarray(home_goals, int), 0, K - 1)
    ag = np.clip(np.asarray(away_goals, int), 0, K - 1)

    def nll(x):
        th = {**x0, **dict(zip(FIT_KEYS, x))}
        F, _ = grids(pairs[:, 0], pairs[:, 1], th)
        p = F[inv, hg, ag]
        return -float(np.log(np.clip(p, 1e-12, None)).mean())

    # Bounded quasi-Newton rather than Nelder-Mead: the surface is flat along the pull-time
    # directions, and a simplex wanders along a flat ridge until it runs out of iterations.
    start = np.array([float(np.clip(x0[k], *BOUNDS[k])) for k in FIT_KEYS])
    r = minimize(nll, start, method="L-BFGS-B", bounds=[BOUNDS[k] for k in FIT_KEYS],
                 options={"maxiter": maxiter, "eps": 1e-4})
    th = {**x0, **dict(zip(FIT_KEYS, (float(v) for v in r.x)))}
    th["c_home_fit"], th["c_away_fit"] = th.pop("c_h"), th.pop("c_a")
    th["c"] = (th["c_home_fit"] + th["c_away_fit"]) / 2
    th["nll"] = float(r.fun)
    th["n_games"] = int(len(hg))
    th["converged"] = bool(r.success)
    return th


def calibration(F: np.ndarray, R: np.ndarray, home_goals, away_goals,
                decided_in=None) -> list[dict]:
    """Predicted against actual for the events the markets turn on."""
    hg, ag = np.asarray(home_goals), np.asarray(away_goals)
    d, t = _dt()
    mar, tot = hg - ag, hg + ag
    went_ot = None if decided_in is None else np.isin(np.asarray(decided_in, str), ["OT", "SO"])
    rows = [("goes to overtime", R[:, d == 0].sum(1), went_ot),
            ("home wins", F[:, d > 0].sum(1), mar > 0),
            ("home by 2+ (home -1.5 covers)", F[:, d >= 2].sum(1), mar >= 2),
            ("away by 2+ (away -1.5 covers)", F[:, d <= -2].sum(1), mar <= -2),
            ("one-goal margin", F[:, np.abs(d) == 1].sum(1), np.abs(mar) == 1),
            ("total over 5.5", F[:, t > 5.5].sum(1), tot > 5.5),
            ("total exactly 6", F[:, t == 6].sum(1), tot == 6),
            ("total over 6.5", F[:, t > 6.5].sum(1), tot > 6.5)]
    out = []
    for name, pred, act in rows:
        r = {"event": name, "predicted_pct": round(100 * float(pred.mean()), 1)}
        if act is not None:
            r["actual_pct"] = round(100 * float(np.mean(act)), 1)
        out.append(r)
    return out


def dumps(th: dict) -> str:
    return json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in th.items()},
                      indent=2)
