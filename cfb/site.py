"""Render the static site into docs/ (GitHub Pages serves that folder).

    python -m cfb.site

No Flask, no server, no uploads. The workflow commits docs/ and Pages publishes it.
"""
from __future__ import annotations

import json
import logging
import re
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


_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})\s*([AaPp])\.?[Mm]")


def _et_hour(kickoff_utc, tip_et) -> float | None:
    """Kickoff as a fractional ET hour, or None when it is genuinely unannounced.

    Two sources because they fail in different places. `kickoff_utc` carries an explicit UTC
    offset and is exact, but older pick rows predate the column. Those rows still carry a
    printed `tip_et` ("Sat Sep 12, 3:30 PM"), and the page already shows it - so a game with a
    visible time must not read as "TBD" in the slate filter just because the ISO field is
    missing. Parsing the label back is ugly and it is the honest fallback.
    """
    if kickoff_utc is not None and str(kickoff_utc).strip() not in ("", "nan", "NaT", "None"):
        try:
            ts = pd.Timestamp(str(kickoff_utc))
            ts = ts.tz_localize(config.ET) if ts.tzinfo is None else ts.tz_convert(config.ET)
            return ts.hour + ts.minute / 60
        except (ValueError, TypeError):
            pass
    m = _TIME_RE.search(str(tip_et or ""))
    if not m:
        return None
    hour, minute, half = int(m.group(1)) % 12, int(m.group(2)), m.group(3).lower()
    return (hour + (12 if half == "p" else 0)) + minute / 60


def _slate(kickoff_utc, tip_et) -> str:
    """Which Saturday slate a kickoff belongs to; "tbd" when there is no time at all.

    "tbd" is a bucket rather than a silent drop. Games a week out often have no announced
    kickoff, and a slate filter that hid them would quietly shrink the board with no way to
    tell it had happened.
    """
    hour = _et_hour(kickoff_utc, tip_et)
    if hour is None:
        return "tbd"
    for key, _label, lo, hi in config.slates():
        if lo <= hour < hi:
            return key
    return "late"          # a 24:00 rounding artefact belongs at the end, not nowhere


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

    # FBS-vs-FCS games are labelled but rank on the same footing as everything else: they are
    # staked on the same edge thresholds, so demoting them would hide plays the system took.
    df["fcs"] = ~df["fbs"].astype(bool) if "fbs" in df.columns else False
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
    df["slate"] = [_slate(k, t) for k, t in
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


def _slate_tabs(picks: list[dict]) -> list[dict]:
    """Slate chips in kickoff order, with counts, and only for slates that have games.

    An empty chip is worse than no chip: it invites a click that blanks the board. TBD sorts
    last and is only offered when something is actually unannounced.
    """
    counts: dict[str, int] = {}
    for p in picks:
        k = p.get("slate") or "tbd"
        counts[k] = counts.get(k, 0) + 1
    order = [(key, label) for key, label, _lo, _hi in config.slates()] + [("tbd", "Time TBD")]
    return [{"key": k, "label": lab, "n": counts[k]} for k, lab in order if counts.get(k)]


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
        days=_days(picks), slates=_slate_tabs(picks),
        updated=metrics.get("updated", ""),
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
