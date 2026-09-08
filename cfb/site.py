"""Render the static site into docs/ (GitHub Pages serves that folder).

    python -m cfb.site

No Flask, no server, no uploads. The workflow commits docs/ and Pages publishes it.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import config

log = logging.getLogger("cfb.site")
TEMPLATES = Path(__file__).resolve().parent / "templates"  # ships with the package


FEE_COEF = 0.07  # Kalshi taker: roundup(0.07 * C * P * (1-P))


def _fee(price):
    import math
    if price is None or price != price:
        return float("nan")
    return math.ceil(FEE_COEF * price * (1 - price) * 100) / 100


def _pays(ask):
    """What a winning contract returns per dollar risked, fee included.

    A Kalshi contract settles at $1. Buying at 58c with a 1c fee costs 59c and returns 100c,
    i.e. 1.69x. This is the number to compare against the model's fair multiplier.
    """
    if ask is None or ask != ask:
        return None
    cost = ask + _fee(ask)
    return round(1 / cost, 2) if 0 < cost < 1 else None


def _fair(prob):
    """The multiplier that would make the bet break even at our probability."""
    if prob is None or prob != prob or prob <= 0:
        return None
    return round(1 / prob, 2)


def _mult(ask, prob):
    """Turn an exchange ask into what a punter actually reads: a payout multiple.

    You pay `ask` plus the taker fee for a contract that settles at $1, so the return per
    dollar risked is 1/(ask+fee). "Fair" is 1/prob — what the multiple would have to be for the
    bet to break even at our estimated probability. Edge is the ratio, i.e. the ROI.
    """
    from .sources.kalshi import fee
    try:
        ask = float(ask); prob = float(prob)
    except (TypeError, ValueError):
        return None, None, None
    if not (0 < ask < 1) or not (0 < prob < 1):
        return None, None, None
    cost = ask + fee(ask)
    if cost <= 0 or cost >= 1:
        return None, None, None
    pays = 1.0 / cost
    fair = 1.0 / prob
    return round(pays, 2), round(fair, 2), round(100 * (prob / cost - 1), 1)


def _prob_cents(p):
    """Model win probability as exchange cents, so it sits next to a Kalshi quote."""
    try:
        return int(round(float(p) * 100))
    except (TypeError, ValueError):
        return None


def _recent_results(limit: int = 12) -> list[dict]:
    """Last graded games, newest first - gives the page a memory instead of resetting daily."""
    if not config.RESULTS.exists():
        return []
    df = pd.read_csv(config.RESULTS, dtype={"game_id": str})
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date", ascending=False).head(limit)
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def _kick_key(kickoff_utc, tip_et) -> str:
    """Sort key for the "Sort by kickoff" control: the ISO timestamp, or "" if unannounced.

    The page used to sort on the display label ("Wed Sep 09, 8:20 PM"). That sorts
    alphabetically, which means by WEEKDAY NAME - Fri, Mon, Sat, Sun, Thu, Tue, Wed - so
    Monday came first and Wednesday last. It also compared the hour as text, putting a
    10:00 PM game ahead of a 12:00 PM one on the same day, and interleaved dates across
    weeks. The ISO timestamp is the only field here that orders correctly.

    A game with no announced kickoff returns "", which the page sorts to the end rather than
    slotting it in at whatever placeholder time the feed stamped on it.
    """
    def blank(v):
        return (v is None or (isinstance(v, float) and v != v)
                or str(v).strip() in ("", "nan", "NaT", "None"))

    if blank(tip_et) or blank(kickoff_utc):
        return ""
    return str(kickoff_utc).strip()


def _board() -> tuple[list[dict], int | None]:
    """This week's games, freshest prediction per game, ready for the template."""
    if not config.PICKS.exists():
        return [], None
    df = pd.read_csv(config.PICKS, dtype={"game_id": str})
    if df.empty:
        return [], None
    df["date"] = pd.to_datetime(df["date"]).dt.date
    upcoming = df[df["date"] >= config.today_et()]
    df = upcoming if len(upcoming) else df[df["week"] == df["week"].max()]
    week = int(df["week"].mode().iloc[0]) if len(df) else None
    df = df.sort_values("prediction_date").drop_duplicates("game_id", keep="last")

    # Rank by the model's largest disagreement with the line, but a tradeable exchange edge
    # outranks it - that's the only number here tied to a price you could actually pay.
    df["_rank"] = df[["total_disagree", "margin_disagree"]].abs().max(axis=1).fillna(0)
    playable = pd.Series(False, index=df.index)
    for c in ("kt_pick", "ks_pick", "ml_pick"):
        if c in df:
            playable |= df[c].notna()
    df["has_play"] = playable
    df.loc[playable, "_rank"] = df.loc[playable, "_rank"] + 100
    df = df.sort_values("_rank", ascending=False)

    # Exchange quotes as payout multiples. `pays` is what the contract returns per unit risked
    # after the taker fee; `fair` is what our probability says it should return. Paying more
    # than fair is the edge, and the gap between them is the ROI.
    for pre, ask_c, prob_c in (("kt", "kt_ask", "kt_prob"), ("ks", "ks_ask", "ks_prob"),
                               ("ml", "ml_ask", "ml_ref_prob")):
        ref_ask, ref_prob = f"{pre}_ref_ask", f"{pre}_ref_prob"
        ask = df[ask_c] if ask_c in df else None
        if ref_ask in df:
            ask = df[ref_ask] if ask is None else ask.fillna(df[ref_ask])
        prob = df[prob_c] if prob_c in df else None
        if ref_prob in df:
            prob = df[ref_prob] if prob is None else prob.fillna(df[ref_prob])
        if ask is None or prob is None:
            continue
        trio = [_mult(a, p) for a, p in zip(ask, prob)]
        df[f"{pre}_pays"] = [t[0] for t in trio]
        df[f"{pre}_fair"] = [t[1] for t in trio]
        df[f"{pre}_edge"] = [t[2] for t in trio]
        df[f"{pre}_prob_pct"] = (prob * 100).round(0)

    # Day tabs, a TBD-safe kickoff label, and a searchable blob so filtering needs no backend.
    df["day"] = pd.to_datetime(df["date"]).dt.strftime("%a %b %-d")
    df["day_key"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["time_label"] = [("Time TBD" if (not t or str(t) in ("nan", "")) else str(t))
                        for t in df.get("tip_et", pd.Series([""] * len(df)))]
    _blank = pd.Series([""] * len(df), index=df.index)
    df["kick_sort"] = [_kick_key(k, t) for k, t in
                       zip(df["kickoff_utc"] if "kickoff_utc" in df else _blank,
                           df["tip_et"] if "tip_et" in df else _blank)]
    conf_h = df["home_conf"] if "home_conf" in df else pd.Series([""] * len(df), index=df.index)
    conf_a = df["away_conf"] if "away_conf" in df else pd.Series([""] * len(df), index=df.index)
    df["search"] = (df["home_team"].fillna("") + " " + df["away_team"].fillna("") + " "
                    + conf_h.fillna("") + " " + conf_a.fillna("")).str.lower()

    P4 = {"SEC", "Big Ten", "Big 12", "ACC"}
    df["is_g5"] = ~(conf_h.isin(P4) & conf_a.isin(P4))

    df = df.astype(object).where(pd.notna(df), None)
    return df.to_dict("records"), week


def _metrics() -> dict:
    if config.METRICS.exists():
        return json.loads(config.METRICS.read_text())
    return {}


def _days(picks: list[dict]) -> list[dict]:
    """Distinct kickoff days in order, for the day tabs."""
    seen: dict[str, str] = {}
    for p in picks:
        k = p.get("day_key")
        if k and k not in seen:
            seen[k] = p.get("day") or k
    return [{"key": k, "label": v} for k, v in sorted(seen.items())]


def build() -> None:
    config.ensure_dirs()
    env = Environment(loader=FileSystemLoader(TEMPLATES),
                      autoescape=select_autoescape(["html"]))
    env.filters["money"] = lambda v: ("+" if (v or 0) >= 0 else "") + f"{v or 0:.2f}"
    picks, week = _board()
    metrics = _metrics()
    html = env.get_template("index.html").render(
        title=config.SITE_TITLE, picks=picks, m=metrics, week=week,
        results=_recent_results(), venue=config.VENUE,
        support_url=config.SUPPORT_URL, support_label=config.SUPPORT_LABEL,
        days=_days(picks), updated=metrics.get("updated", ""),
        total_min=config.TOTAL_EDGE_MIN, spread_min=config.SPREAD_EDGE_MIN,
    )
    (config.DOCS / "index.html").write_text(html)
    (config.DOCS / ".nojekyll").touch()
    # expose the raw data so the page (or you) can fetch it
    for src in (config.PICKS, config.METRICS, config.RESULTS):
        if src.exists():
            shutil.copy(src, config.DOCS / src.name)
    log.info("built %s: week %s, %d games", config.DOCS / "index.html", week, len(picks))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    build()
