"""Render the static NFL site into docs/nfl/ (GitHub Pages serves the docs folder).

    python -m nfl.site

No Flask, no server, no uploads. The workflow commits docs/ and Pages publishes it. The NFL
page lives in a subfolder so it and the college page can share one Pages deployment without
either overwriting the other's index.html.

The page is a prediction board, not a bet sheet. Every game gets the side of the spread and
the total the model thinks is likelier to happen, the score it projects, and how far it sits
from the line. Stakes, prices, EV and play/pass are still computed by predict.py and kept in
picks.csv and results.csv; they are just not what this page is for.
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


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _gap(disagree) -> float | None:
    """How many points the model sits from the line, toward the side it picked.

    This is what each card shows as the strength of a lean, rather than a win probability.
    The calibrated probability (`*_p_win`, the line nudged a tenth of the way toward the
    model) sits between 50% and 51.5% on nine picks in ten, so printed per card it reads as
    "50%" on almost every game and tells a reader nothing; the honest calibration is stated
    once at the top of the page instead. The side is always the direction of the gap, so the
    absolute value is the whole story.
    """
    d = _num(disagree)
    return round(abs(d), 1) if d is not None else None


def _projected(r: dict) -> dict:
    """The model's own projected final score, read off its raw total and home margin.

    The raw numbers, not the shrunk ones: the shrunk margin is the line nudged a tenth of the
    way toward the model, so a score built from it would just be the line restated. The side
    each pick takes comes from the same raw numbers, so the score and the picks always agree.
    """
    total = _num(r.get("total_raw"))
    if total is None:
        total = _num(r.get("total_pred"))
    margin = _num(r.get("margin_raw"))
    if margin is None:
        margin = _num(r.get("margin_pred"))
    if total is None or margin is None:
        return {}
    home, away = (total + margin) / 2, (total - margin) / 2
    h, a = r.get("home_team"), r.get("away_team")
    first = [(h, home), (a, away)] if home >= away else [(a, away), (h, home)]
    return {"proj": " – ".join(f"{t} {round(p)}" for t, p in first)}


def _spread_label(team, number) -> str:
    n = _num(number)
    if not team or n is None:
        return ""
    return f"{team} PK" if n == 0 else f"{team} {n:+g}"


def _model_margin(r: dict) -> str:
    """'Model: GB by 1.7', said from whichever team the model has winning."""
    m = _num(r.get("margin_raw"))
    if m is None:
        m = _num(r.get("margin_pred"))
    if m is None:
        return ""
    if abs(m) < 0.05:
        return "Model: dead even"
    team = r.get("home_team") if m > 0 else r.get("away_team")
    return f"Model: {team} by {abs(m):.1f}"


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
    """Sort key for kickoff order: the ISO timestamp, or "" if unannounced.

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
    """This week's games in kickoff order, freshest prediction per game, with a spread pick
    and a total pick on every one."""
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

    df["day"] = pd.to_datetime(df["date"]).dt.strftime("%a %b %-d")
    df["day_key"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["time_label"] = [("Time TBD" if (not t or str(t) in ("nan", "")) else str(t))
                        for t in df.get("tip_et", pd.Series([""] * len(df)))]
    _blank = pd.Series([""] * len(df), index=df.index)
    df["kick_sort"] = [_kick_key(k, t) for k, t in
                       zip(df["kickoff_utc"] if "kickoff_utc" in df else _blank,
                           df["tip_et"] if "tip_et" in df else _blank)]
    # Kickoff order with scripts off too; unannounced kickoffs go last. Parsed, not compared
    # as text: the feed stamps -04:00 before the clocks change and -05:00 after.
    df["_at"] = pd.to_datetime(df["kick_sort"].replace("", None), utc=True, errors="coerce")
    df = df.sort_values(["_at", "date"], na_position="last").drop(columns="_at")
    div_h = df["home_div"] if "home_div" in df else _blank
    div_a = df["away_div"] if "away_div" in df else _blank
    df["search"] = (df["home_team"].fillna("") + " " + df["away_team"].fillna("") + " "
                    + div_h.fillna("") + " " + div_a.fillna("") + " "
                    + df.get("home_qb", _blank).fillna("")
                    + " "
                    + df.get("away_qb", _blank).fillna("")
                    ).str.lower()

    # pandas cannot hold None inside a float64 column, so `where(notna, None)` leaves NaN in
    # place - and bool(nan) is True, which makes every Jinja `{% if %}` pass and renders the
    # literal string "nan" on the page. Casting to object first is what actually clears them.
    df = df.astype(object).where(pd.notna(df), None)
    records = df.to_dict("records")
    for r in records:
        r["context"] = _context(r)
        r.update(_projected(r))
        r["spread_label"] = _spread_label(r.get("spread_side"), r.get("spread_number")) \
            or (r.get("spread_pick") or "")
        r["model_margin"] = _model_margin(r)
        r["spread_gap"] = _gap(r.get("margin_disagree"))
        r["total_gap"] = _gap(r.get("total_disagree"))
        r["lean"] = max(r["spread_gap"] or 0.0, r["total_gap"] or 0.0)
    return records, week


def _wlp(results: pd.Series) -> dict:
    w, l, p = int((results == "win").sum()), int((results == "loss").sum()), \
        int((results == "push").sum())
    return {"wins": w, "losses": l, "pushes": p,
            "pct": round(100 * w / (w + l), 1) if w + l else None}


def _graded_season() -> pd.DataFrame:
    """This season's graded games: every game, since every game gets a pick."""
    if not config.RESULTS.exists():
        return pd.DataFrame()
    df = pd.read_csv(config.RESULTS, dtype={"game_id": str})
    if df.empty or "season" not in df:
        return pd.DataFrame()
    season = df[df["season"] == config.season_of(config.today_et())]
    return season if len(season) else df[df["season"] == df["season"].max()]


def _records(df: pd.DataFrame) -> dict:
    """Season and week-by-week records for both markets, newest week first."""
    if df.empty:
        return {}
    weeks = [{"week": int(wk), "spread": _wlp(g["spread_result"]),
              "total": _wlp(g["total_result"])}
             for wk, g in df.groupby("week")]
    weeks.sort(key=lambda w: w["week"], reverse=True)
    return {"spread": _wlp(df["spread_result"]), "total": _wlp(df["total_result"]),
            "weeks": weeks, "last": weeks[0] if weeks else None}


def _final(r: dict) -> str:
    """'SEA 20 – KC 3', winner first."""
    hp, ap = _num(r.get("home_points")), _num(r.get("away_points"))
    if hp is None or ap is None:
        return ""
    h, a = (r.get("home_team"), int(hp)), (r.get("away_team"), int(ap))
    first, second = (h, a) if hp >= ap else (a, h)
    return f"{first[0]} {first[1]} – {second[0]} {second[1]}"


def _last_week(df: pd.DataFrame) -> tuple[int | None, list[dict]]:
    """Every graded game from the most recent graded week, in kickoff order: the pick on each
    market, the final score, and whether the pick hit."""
    if df.empty:
        return None, []
    wk = int(df["week"].max())
    g = df[df["week"] == wk].copy()
    if "kickoff_utc" in g:
        g["_at"] = pd.to_datetime(g["kickoff_utc"], utc=True, errors="coerce")
        g = g.sort_values("_at", na_position="last").drop(columns="_at")
    g = g.astype(object).where(pd.notna(g), None)
    rows = []
    for r in g.to_dict("records"):
        line = _num(r.get("total_line"))
        rows.append({
            "matchup": f"{r.get('away_team')} at {r.get('home_team')}",
            "final": _final(r),
            "spread_label": _spread_label(r.get("spread_side"), r.get("spread_number"))
            or (r.get("spread_pick") or ""),
            "spread_result": r.get("spread_result"),
            "total_label": f"{r.get('total_pick')} {line:g}" if line is not None else "",
            "total_result": r.get("total_result"),
        })
    return wk, rows


def _model_note() -> dict:
    """How the model's side has done out of sample, for the 'how to read this' note.

    The page should say plainly how often the likelier side actually hit across the
    walk-forward, so a reader calibrates a 52% against what 52% has meant in the past.
    """
    path = config.MODEL_DIR / "meta.json"
    if not path.exists():
        return {}
    try:
        meta = json.loads(path.read_text())
    except (ValueError, OSError):
        return {}
    ev = meta.get("eval") or {}
    sp, tot = ev.get("margin_market") or {}, ev.get("total_market") or {}
    if sp.get("skipped") or not sp.get("ats_rate"):
        return {}
    seasons = sp.get("test_seasons") or []
    return {"spread_rate": sp.get("ats_rate"), "spread_n": sp.get("ats_n"),
            "total_rate": tot.get("ats_rate"), "total_n": tot.get("ats_n"),
            "first": seasons[0] if seasons else None, "last": seasons[-1] if seasons else None}


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
    picks, week = _board()
    graded = _graded_season()
    results_week, results = _last_week(graded)
    published = str(config.today_et())
    html = env.get_template("index.html").render(
        title=config.SITE_TITLE, picks=picks, week=week, updated=published,
        days=_days(picks), model=_model_note(), rec=_records(graded),
        results=results, results_week=results_week,
        support_url=config.SUPPORT_URL, support_label=config.SUPPORT_LABEL,
    )
    (config.DOCS / "index.html").write_text(html)
    # the chooser's card reads this; metrics.json's `updated` only moves when grading runs
    (config.DOCS / "published.json").write_text(json.dumps({"published": published}) + "\n")
    (config.DOCS / ".nojekyll").touch()
    for src in (config.PICKS, config.METRICS, config.RESULTS):
        if src.exists():
            shutil.copy(src, config.DOCS / src.name)
    log.info("built %s: week %s, %d games", config.DOCS / "index.html", week, len(picks))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    build()
