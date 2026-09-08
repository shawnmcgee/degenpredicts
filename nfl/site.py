"""Render the static NFL site into docs/nfl/ (GitHub Pages serves the docs folder).

    python -m nfl.site

No Flask, no server, no uploads. The workflow commits docs/ and Pages publishes it. The NFL
page lives in a subfolder so it and the college page can share one Pages deployment without
either overwriting the other's index.html.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import config

log = logging.getLogger("nfl.site")
TEMPLATES = Path(__file__).resolve().parent / "templates"  # ships with the package


def _mult(ask, prob):
    """Turn an exchange ask into what a punter actually reads: a payout multiple.

    You pay `ask` plus the taker fee for a contract that settles at $1, so the return per unit
    risked is 1/(ask+fee). "Fair" is 1/prob - what the multiple would have to be for the bet to
    break even at our estimated probability. Edge is the ratio, i.e. the ROI.
    """
    from .sources.kalshi import fee
    try:
        ask = float(ask)
        prob = float(prob)
    except (TypeError, ValueError):
        return None, None, None
    if not (0 < ask < 1) or not (0 < prob < 1):
        return None, None, None
    cost = ask + fee(ask)
    if cost <= 0 or cost >= 1:
        return None, None, None
    return round(1.0 / cost, 2), round(1.0 / prob, 2), round(100 * (prob / cost - 1), 1)


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


def _context(g) -> str:
    """The one-line 'why this game is different' strip under the matchup.

    Deliberately only the NFL-specific situational facts - a new starting quarterback, a rest
    mismatch, a long trip, weather - because those are the things a reader cannot get from the
    line itself and the things this model adds over the college one.
    """
    bits = []
    if g.get("h_qb_new") == 1:
        bits.append(f"new {g.get('home_team')} QB" + (f" ({g['home_qb']})" if g.get("home_qb") else ""))
    if g.get("a_qb_new") == 1:
        bits.append(f"new {g.get('away_team')} QB" + (f" ({g['away_qb']})" if g.get("away_qb") else ""))
    rd = g.get("rest_diff")
    if rd is not None and rd == rd and abs(rd) >= 3:
        side = g.get("home_team") if rd > 0 else g.get("away_team")
        bits.append(f"{side} +{abs(int(rd))} days rest")
    tz = g.get("a_tz_shift")
    if tz is not None and tz == tz and abs(tz) >= 2:
        bits.append(f"{g.get('away_team')} crosses {abs(int(tz))} time zones")
    km = g.get("a_travel_km")
    if km is not None and km == km and km >= 3000:
        bits.append(f"{int(km):,} km trip")
    w = g.get("wind")
    if not g.get("is_indoor") and w is not None and w == w and w >= 15:
        bits.append(f"wind {int(w)} mph")
    if g.get("div_game"):
        bits.append("divisional")
    return " · ".join(bits)


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
    # Everything below adds columns one at a time; copying once here keeps pandas from
    # warning about a fragmented frame on every build.
    df = df.sort_values("prediction_date").drop_duplicates("game_id", keep="last").copy()

    # Rank by the model's largest disagreement with the line, but a tradeable exchange edge
    # outranks it - that is the only number here tied to a price you could actually pay.
    df["_rank"] = df[["total_disagree", "margin_disagree"]].abs().max(axis=1).fillna(0)
    playable = pd.Series(False, index=df.index)
    for c in ("kt_pick", "ks_pick", "ml_pick"):
        if c in df:
            playable |= df[c].notna()
    df["has_play"] = playable
    df.loc[playable, "_rank"] = df.loc[playable, "_rank"] + 100
    df = df.sort_values("_rank", ascending=False)

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

    df["day"] = pd.to_datetime(df["date"]).dt.strftime("%a %b %-d")
    df["day_key"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["time_label"] = [("Time TBD" if (not t or str(t) in ("nan", "")) else str(t))
                        for t in df.get("tip_et", pd.Series([""] * len(df)))]
    _blank = pd.Series([""] * len(df), index=df.index)
    df["kick_sort"] = [_kick_key(k, t) for k, t in
                       zip(df["kickoff_utc"] if "kickoff_utc" in df else _blank,
                           df["tip_et"] if "tip_et" in df else _blank)]
    div_h = df["home_div"] if "home_div" in df else pd.Series([""] * len(df), index=df.index)
    div_a = df["away_div"] if "away_div" in df else pd.Series([""] * len(df), index=df.index)
    df["search"] = (df["home_team"].fillna("") + " " + df["away_team"].fillna("") + " "
                    + div_h.fillna("") + " " + div_a.fillna("") + " "
                    + df.get("home_qb", pd.Series([""] * len(df), index=df.index)).fillna("")
                    + " "
                    + df.get("away_qb", pd.Series([""] * len(df), index=df.index)).fillna("")
                    ).str.lower()

    # pandas cannot hold None inside a float64 column, so `where(notna, None)` leaves NaN in
    # place - and bool(nan) is True, which makes every Jinja `{% if %}` pass and renders the
    # literal string "nan" on the page. Casting to object first is what actually clears them.
    df = df.astype(object).where(pd.notna(df), None)
    records = df.to_dict("records")
    for r in records:
        r["context"] = _context(r)
    return records, week


def _metrics() -> dict:
    if config.METRICS.exists():
        return json.loads(config.METRICS.read_text())
    return {}


def _model_note() -> dict:
    """What the last training run concluded, surfaced on the page rather than buried in JSON.

    If the model does not beat the closing line, the page should say so where a reader will
    see it. Publishing the number while hiding that finding would be the dishonest version of
    this project.
    """
    path = config.MODEL_DIR / "meta.json"
    if not path.exists():
        return {}
    try:
        meta = json.loads(path.read_text())
    except (ValueError, OSError):
        return {}
    e = (meta.get("eval") or {}).get("margin_market") or {}
    if e.get("skipped") or not e:
        return {}
    return {"beats_market": e.get("beats_market"), "mae_model": e.get("mae_model"),
            "mae_market": e.get("mae_market_baseline"), "ats_rate": e.get("ats_rate"),
            "ats_stderr": e.get("ats_stderr"), "ats_n": e.get("ats_n"),
            "shrink": e.get("shrink"), "seasons": e.get("test_seasons")}


def _backtest_buckets() -> dict:
    """The out-of-sample disagreement table from the last training run.

    The page's live "cover rate by model disagreement" table is built from graded picks, so
    it is empty until games have actually been played and stays thin for months afterwards.
    Meanwhile the walk-forward already measured the same thing across ~1,700 out-of-sample
    games, and it was sitting unread in models/meta.json. This surfaces it, clearly labelled
    as a backtest, with the significance correction attached - a bucket that looks strong
    here has usually not survived the number of segments the report looked at.
    """
    path = config.MODEL_DIR / "meta.json"
    if not path.exists():
        return {}
    try:
        meta = json.loads(path.read_text())
    except (ValueError, OSError):
        return {}
    ev = (meta.get("eval") or {}).get("margin_market") or {}
    tot = (meta.get("eval") or {}).get("total_market") or {}
    if ev.get("skipped") or not ev.get("ats_by_disagreement"):
        return {}
    by_total = {b.get("disagreement"): b for b in tot.get("ats_by_disagreement") or []}
    rows = []
    for b in ev["ats_by_disagreement"]:
        rows.append({"bucket": b.get("disagreement"),
                     "spread": b,
                     "total": by_total.get(b.get("disagreement"))})
    sig = ev.get("significance") or {}
    return {"rows": rows,
            "seasons": ev.get("test_seasons") or [],
            "n": ev.get("n_test_total"),
            "z_required": sig.get("z_required"),
            "verdict": sig.get("verdict"),
            "break_even": ev.get("break_even_pct")}


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
        results=_recent_results(), venue=config.VENUE, model=_model_note(),
        support_url=config.SUPPORT_URL, support_label=config.SUPPORT_LABEL,
        days=_days(picks), updated=metrics.get("updated", ""),
        backtest=_backtest_buckets(),
        total_min=config.TOTAL_EDGE_MIN, spread_min=config.SPREAD_EDGE_MIN,
    )
    (config.DOCS / "index.html").write_text(html)
    (config.DOCS / ".nojekyll").touch()
    for src in (config.PICKS, config.METRICS, config.RESULTS):
        if src.exists():
            shutil.copy(src, config.DOCS / src.name)
    log.info("built %s: week %s, %d games", config.DOCS / "index.html", week, len(picks))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    build()
