"""Venue-agnostic exchange maths, and the fan-out that runs every enabled venue.

Two prediction-market venues quote the same binary event, so the pipeline reads both and
prices our probability against whichever ask is actually cheaper. What makes that a small
change rather than a rewrite is a coincidence in the fee schedules: both venues charge

    fee = coefficient * P * (1 - P)

per contract, differing only in the coefficient (Kalshi taker 0.07, Polymarket sports 0.05,
Polymarket maker 0.00). So the cost model, EV, ROI and Kelly sizing are shared and only the
coefficient is per-venue.

**The coefficient must travel with the quote.** `fee()` used to read the single global
`config.VENUE`, which was fine with one venue and is a silent mispricing with two: a
Polymarket ask charged at Kalshi's 0.07 overstates cost by ~0.5c at the money, which is
enough to flip a marginal pick either way. Every function here therefore takes `venue`, and
every board carries a ``venue`` column.

A note on the winner's curse. Choosing the best of N noisy estimates returns a positive number
even with no edge, which is why the ladder guards exist. Taking the cheaper of two venues'
asks for the *same* event is not that: the model probability is held fixed and only the price
varies, so a genuinely cheaper ask is genuinely better. But a stale or placeholder quote also
looks cheap, and now there are two books' worth of them, so the liquidity gate does more work
than before rather than less. Both venues fail closed: no confirmed ask and size means not
tradeable.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from .. import config

log = logging.getLogger(__name__)

# Which venues to read, in preference order for tie-breaking. A venue that is unreachable, or
# whose market shapes have changed, contributes nothing and is logged - it never takes the run
# down with it, because the exchange columns are a garnish on a pipeline that works without
# them.
DEFAULT_VENUES = ("kalshi", "polymarket")


def enabled() -> tuple[str, ...]:
    """Venues to read this run, from ``DEGEN_VENUES`` (comma-separated), else all of them.

    Read at call time, not import, so a workflow variable applies whenever it is set. An
    unset-but-present environment variable is blank rather than absent, which is why this
    filters empties instead of trusting `os.environ.get(..., default)`.
    """
    raw = os.environ.get("DEGEN_VENUES", "").strip()
    if not raw:
        return DEFAULT_VENUES
    want = tuple(v.strip().lower() for v in raw.split(",") if v.strip())
    unknown = [v for v in want if v not in DEFAULT_VENUES]
    if unknown:
        log.warning("ignoring unknown venue(s) %s - known venues are %s",
                    unknown, list(DEFAULT_VENUES))
    return tuple(v for v in want if v in DEFAULT_VENUES) or DEFAULT_VENUES


def _module(venue: str):
    if venue == "kalshi":
        from . import kalshi
        return kalshi
    if venue == "polymarket":
        from . import polymarket
        return polymarket
    raise KeyError(venue)


# ---------------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------------
def fee_key(venue: str) -> str:
    """The `config.FEE_COEF` key for a venue, honouring the configured taker/maker side.

    `config.VENUE` still selects taker vs maker (and `sportsbook`, which has no exchange
    quotes at all); the venue argument selects whose schedule to read.
    """
    side = "maker" if str(config.VENUE).endswith("_maker") else "taker"
    return f"{venue}_{side}"


def fee_coef(venue: str | None = None) -> float | None:
    """Fee coefficient for a venue. ``None`` means "no explicit fee" (a sportsbook)."""
    if venue is None:
        return config.FEE_COEF.get(config.VENUE, 0.07)
    key = fee_key(venue)
    if key in config.FEE_COEF:
        return config.FEE_COEF[key]
    # An unknown venue must not silently price at zero, which would make every quote look
    # like a bargain. Fall back to the most expensive schedule we know.
    log.warning("no fee schedule for %r - charging the most expensive known coefficient", key)
    return max(v for v in config.FEE_COEF.values() if v is not None)


def fee(price: float, venue: str | None = None, coef: float | None = None) -> float:
    """Per-contract fee at `price`. Both venues use ``coef * P * (1 - P)``.

    Schedules change - verify at kalshi.com/fee-schedule and docs.polymarket.com/trading/fees.
    """
    coef = fee_coef(venue) if coef is None else coef
    if coef is None:
        return 0.0
    return coef * price * (1 - price)


def contract_ev(p_win: float, ask: float, venue: str | None = None) -> tuple[float, float]:
    """Buying one YES contract at `ask` costs ask + fee and pays 1.00 if it hits.

    Returns (ev_per_contract, roi_fraction).
    """
    if p_win is None or p_win != p_win or ask != ask or ask <= 0 or ask >= 1:
        return float("nan"), float("nan")
    cost = ask + fee(ask, venue)
    return p_win - cost, (p_win - cost) / cost


def multiplier(ask, prob, venue: str | None = None):
    """What the contract pays per unit risked, and what it *should* pay.

    Buying YES costs ask + fee and returns 1.00, so the payout multiple is 1/(ask+fee). The
    fair multiple implied by our probability is 1/prob. The ratio between them is the ROI in a
    form that reads like odds instead of cents.
    """
    if ask is None or ask != ask or ask <= 0:
        return float("nan"), float("nan"), float("nan")
    cost = ask + fee(ask, venue)
    pays = 1.0 / cost if cost > 0 else float("nan")
    fair = 1.0 / prob if prob and prob == prob and prob > 0 else float("nan")
    edge = (pays / fair - 1.0) * 100 if pays == pays and fair == fair else float("nan")
    return round(pays, 3), round(fair, 3), round(edge, 1)


# ---------------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------------
def monotonicity_breaks(ladder: pd.DataFrame) -> list[dict]:
    """Rungs that contradict each other within one venue's ladder.

    P(X > 8.5) can never exceed P(X > 4.5). When the asks say otherwise, at least one quote is
    stale or a placeholder.

    Grouped by venue as well as event and team: two venues legitimately quote the same strike
    at different prices, and comparing across them would report every price difference as an
    incoherence. Cross-venue disagreement is a signal, not a break.
    """
    if ladder is None or ladder.empty:
        return []
    keys = ["event_ticker", ladder["team"].fillna("")]
    if "venue" in ladder.columns:
        keys.insert(0, "venue")
    out = []
    for key, grp in ladder.groupby(keys):
        venue, ev, team = (key if len(key) == 3 else (None, *key))
        g = grp.dropna(subset=["yes_ask"]).sort_values("strike")
        asks = g["yes_ask"].tolist()
        for i in range(len(asks) - 1):
            if asks[i + 1] > asks[i] + 0.01:      # higher strike should not cost more
                out.append({"venue": venue, "event_ticker": ev, "team": team or None,
                            "lower_strike": g["strike"].iloc[i], "lower_ask": asks[i],
                            "higher_strike": g["strike"].iloc[i + 1],
                            "higher_ask": asks[i + 1]})
    if out:
        log.info("%d monotonicity breaks across ladders", len(out))
    return out


# ---------------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------------
# Columns every venue must supply, so predict.py never has to know which one answered. These
# are the true minimum - exactly what `predict` reads - rather than everything a venue happens
# to return, so a venue is only dropped for missing something that would actually break it.
MONEYLINE_COLS = ["venue", "date", "home_team", "away_team", "team", "ticker",
                  "yes_ask", "quote_spread", "tradeable"]
# Ladders additionally need `yes_bid` (the "under" side is priced off 1 - bid) and
# `event_ticker` (the coherence check groups on it).
LADDER_COLS = ["venue", "date", "home_team", "away_team", "team", "strike", "event_ticker",
               "yes_bid", "yes_ask", "quote_spread", "tradeable"]


def _collect(call, cols: list[str], label: str, matcher=None, **kw) -> pd.DataFrame:
    frames = []
    for venue in enabled():
        try:
            df = call(_module(venue), matcher=matcher, **kw)
        except Exception as e:              # one venue's outage is not the run's problem
            log.warning("%s %s unavailable (%s) - continuing without it", venue, label, e)
            continue
        if df is None or df.empty:
            log.info("%s %s: nothing returned", venue, label)
            continue
        df = df.copy()
        df["venue"] = venue
        missing = [c for c in cols if c not in df.columns]
        if missing:
            # A venue that cannot fill the contract is dropped rather than half-merged, which
            # would leave predict.py reading NaN where it expects a price.
            log.warning("%s %s is missing %s - dropping it from this run", venue, label, missing)
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True, sort=False)
    log.info("%s across %d venue(s): %d rows, %d tradeable (%s)", label, out["venue"].nunique(),
             len(out), int(out["tradeable"].sum()),
             out.groupby("venue").size().to_dict())
    return out


def moneyline_board(matcher=None) -> pd.DataFrame:
    """One row per (venue, game, side) with that venue's price."""
    return _collect(lambda m, matcher: m.moneyline_board(matcher),
                    MONEYLINE_COLS, "moneyline", matcher)


def ladder_board(kind: str, matcher=None) -> pd.DataFrame:
    """Every strike of every open spread/total market, from every enabled venue."""
    return _collect(lambda m, matcher: m.ladder_board(kind, matcher),
                    LADDER_COLS, f"{kind} ladder", matcher)
