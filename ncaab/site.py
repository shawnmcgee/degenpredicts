"""Render the college basketball board into docs/ncaab/ (GitHub Pages serves docs/).

    python -m ncaab.site

The same card as the NFL board, game for game: the model's projected score, then a spread tile
and a total tile, each with the side of the book's line the model thinks is likelier, the
model's own number underneath, and how many points it sits from the line. What differs is the
amber context strip - the things a college line has to price that the ratings cannot see: AP
ranks, neutral courts, holiday tournaments, back-to-backs - and the filters a 100-game night
needs: Top 25, power conferences, neutral site.

Below the record sits the bake-off: every model that was tried for each market, how far off it
was on seasons it never saw, and which one the board is running.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import config
from .models import LABELS, SIMPLICITY
from .train import load_meta

log = logging.getLogger("ncaab.site")
TEMPLATES = Path(__file__).resolve().parent / "templates"


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _gap(d) -> float | None:
    """Points the model sits from the line, toward its side. The side is the direction of the
    gap, so the absolute value is the whole story."""
    d = _num(d)
    return round(abs(d), 1) if d is not None else None


def _rank(v) -> int | None:
    r = _num(v)
    return int(r) if r is not None and 1 <= r <= 25 else None


def _conf(v) -> str:
    return config.CONFERENCES.get(str(v or "").split(".")[0], "")


def _projected(r: dict) -> str:
    """The model's own final score, from its raw total and margin - the numbers the picks come
    from, so the score and the picks always agree."""
    total, margin = _num(r.get("total_raw")), _num(r.get("margin_raw"))
    if total is None or margin is None:
        return ""
    home, away = (total + margin) / 2, (total - margin) / 2
    h, a = r.get("home_team"), r.get("away_team")
    first = [(h, home), (a, away)] if home >= away else [(a, away), (h, home)]
    return " – ".join(f"{t} {round(p)}" for t, p in first)


def _model_margin(r: dict) -> str:
    m = _num(r.get("margin_raw"))
    if m is None:
        return ""
    if abs(m) < 0.05:
        return "Model: dead even"
    team = r.get("home_team") if m > 0 else r.get("away_team")
    return f"Model: {team} by {abs(m):.1f}"


def _context(r: dict) -> str:
    """The amber strip: only what the line has to price and the ratings cannot see."""
    bits = []
    note = str(r.get("note") or "").strip()
    if note:
        bits.append(note if len(note) <= 60 else note[:57].rstrip() + "…")
    if r.get("neutral_site") in (True, "True", 1, "1") and not note:
        bits.append("neutral court")
    hr, ar = _rank(r.get("home_rank")), _rank(r.get("away_rank"))
    if hr and ar:
        bits.append("ranked matchup")
    for side, rest in (("home_team", "h_rest"), ("away_team", "a_rest")):
        d = _num(r.get(rest))
        if d is not None and d <= 1:
            bits.append(f"{r.get(side)} on a back-to-back")
    for side, new in (("home_team", "h_new"), ("away_team", "a_new")):
        if _num(r.get(new)) == 1:
            bits.append(f"{r.get(side)}: first season in the ratings")
    return " · ".join(bits)


def _spread_label(team, number) -> str:
    n = _num(number)
    if not team or n is None:
        return ""
    return f"{team} PK" if n == 0 else f"{team} {n:+g}"


def _board() -> list[dict]:
    """Games still to tip, freshest prediction per game, in tip-off order."""
    if not config.PICKS.exists():
        return []
    df = pd.read_csv(config.PICKS, dtype={"game_id": str, "home_conf": str, "away_conf": str})
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] >= config.today_et()]
    if df.empty:
        return []
    df = df.sort_values("predicted_at").drop_duplicates("game_id", keep="last").copy()
    df["_at"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")
    df = df.sort_values(["_at", "game_id"], na_position="last").drop(columns="_at")
    df["day"] = pd.to_datetime(df["date"]).dt.strftime("%a %b %-d")
    df["day_key"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    # NaN is truthy in Jinja and prints as "nan"; object dtype is what lets None replace it
    df = df.astype(object).where(pd.notna(df), None)
    rows = df.to_dict("records")
    for r in rows:
        r["home_rank"], r["away_rank"] = _rank(r.get("home_rank")), _rank(r.get("away_rank"))
        r["home_conf_name"], r["away_conf_name"] = _conf(r.get("home_conf")), _conf(r.get("away_conf"))
        r["ranked"] = bool(r["home_rank"] or r["away_rank"])
        r["power"] = any(str(r.get(c) or "").split(".")[0] in config.POWER
                         for c in ("home_conf", "away_conf"))
        r["neutral"] = r.get("neutral_site") in (True, "True", 1, "1")
        r["proj"] = _projected(r)
        r["context"] = _context(r)
        r["model_margin"] = _model_margin(r)
        r["spread_label"] = _spread_label(r.get("spread_side"), r.get("spread_number"))
        tl = _num(r.get("total_line"))
        r["total_label"] = f"{r.get('total_pick')} {tl:g}" if tl is not None and r.get("total_pick") else ""
        r["spread_gap"] = _gap(r.get("margin_disagree")) if r["spread_label"] else None
        r["total_gap"] = _gap(r.get("total_disagree")) if r["total_label"] else None
        r["lean"] = max(r["spread_gap"] or 0.0, r["total_gap"] or 0.0)
        r["search"] = " ".join(str(x or "") for x in (
            r.get("home_team"), r.get("away_team"), r["home_conf_name"], r["away_conf_name"],
            r.get("home_abbr"), r.get("away_abbr"), r.get("note"))).lower()
    return rows


def _days(picks: list[dict]) -> list[dict]:
    seen: dict[str, str] = {}
    for p in picks:
        if p.get("day_key") and p["day_key"] not in seen:
            seen[p["day_key"]] = p.get("day") or p["day_key"]
    return [{"key": k, "label": v} for k, v in sorted(seen.items())]


def _graded() -> pd.DataFrame:
    if not config.RESULTS.exists():
        return pd.DataFrame()
    df = pd.read_csv(config.RESULTS, dtype={"game_id": str})
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    season = config.season_of(config.today_et())
    cur = df[df["date"].map(config.season_of) == season]
    return cur if len(cur) else df[df["date"].map(config.season_of) == df["date"].map(config.season_of).max()]


def _final(r: dict) -> str:
    hp, ap = _num(r.get("home_points")), _num(r.get("away_points"))
    if hp is None or ap is None:
        return ""
    h, a = (r.get("home_team"), int(hp)), (r.get("away_team"), int(ap))
    first, second = (h, a) if hp >= ap else (a, h)
    return f"{first[0]} {first[1]} – {second[0]} {second[1]}"


def _last_day(df: pd.DataFrame) -> tuple[str | None, list[dict]]:
    """Every graded game from the most recent graded day."""
    if df.empty:
        return None, []
    df = df[df["spread_result"].isin(["win", "loss", "push"]) |
            df["total_result"].isin(["win", "loss", "push"])]
    if df.empty:
        return None, []
    day = max(df["date"])
    g = df[df["date"] == day].copy()
    g["_at"] = pd.to_datetime(g["kickoff_utc"], utc=True, errors="coerce")
    g = g.sort_values("_at", na_position="last").drop(columns="_at")
    g = g.astype(object).where(pd.notna(g), None)
    rows = []
    for r in g.to_dict("records"):
        tl = _num(r.get("total_line"))
        rows.append({"matchup": f"{r.get('away_team')} at {r.get('home_team')}",
                     "final": _final(r),
                     "spread_label": _spread_label(r.get("spread_side"), r.get("spread_number")),
                     "spread_result": r.get("spread_result"),
                     "total_label": f"{r.get('total_pick')} {tl:g}" if tl is not None else "",
                     "total_result": r.get("total_result")})
    return pd.Timestamp(day).strftime("%a %b %-d"), rows


def _metrics() -> dict:
    try:
        return json.loads(config.METRICS.read_text())
    except (OSError, ValueError):
        return {}


def _bakeoff(meta: dict) -> dict:
    """The leaderboard for the two numbers the board publishes, side by side."""
    jobs = meta.get("jobs") or {}
    sp, tot = jobs.get("margin_market") or {}, jobs.get("total_market") or {}
    if (not sp or sp.get("skipped")) and (not tot or tot.get("skipped")):
        return {}
    by = {}
    for key, job in (("spread", sp), ("total", tot)):
        for r in job.get("leaderboard") or []:
            # the line's own row has an error but no side; every row gets the same keys
            by.setdefault(r["model"], {})[key] = {"hit_rate": None, **r}
    order = [m for m in SIMPLICITY if m in by] + (["line"] if "line" in by else [])
    rows = [{"model": m, "label": LABELS.get(m, m), "spread": by[m].get("spread"),
             "total": by[m].get("total"), "reference": m == "line"} for m in order]
    seasons = sp.get("test_seasons") or tot.get("test_seasons") or []
    lab = lambda y: f"{y}-{(y + 1) % 100:02d}"   # 2025 -> "2025-26"
    span = (lab(seasons[0]) if len(seasons) == 1 else f"{lab(seasons[0])} to {lab(seasons[-1])}") \
        if seasons else ""
    return {"rows": rows, "spread_chosen": sp.get("chosen"), "total_chosen": tot.get("chosen"),
            "spread_best": sp.get("best"), "total_best": tot.get("best"),
            "span": span,
            "n_spread": sp.get("n"), "n_total": tot.get("n"),
            "spread_hit": sp.get("hit_rate"), "total_hit": tot.get("hit_rate"),
            "spread_hit_n": sp.get("hit_n"), "total_hit_n": tot.get("hit_n"),
            "spread_by_gap": sp.get("by_gap") or [], "total_by_gap": tot.get("by_gap") or [],
            "rule": meta.get("select_rule", "1se"), "trained_at": (meta.get("trained_at") or "")[:10]}


def build() -> None:
    config.ensure_dirs()
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    picks = _board()
    graded = _graded()
    results_day, results = _last_day(graded)
    published = str(config.today_et())
    html = env.get_template("index.html").render(
        title=config.SITE_TITLE, picks=picks, updated=published, days=_days(picks),
        m=_metrics(), bake=_bakeoff(load_meta()), results=results, results_day=results_day,
        support_url=config.SUPPORT_URL, support_label=config.SUPPORT_LABEL)
    (config.DOCS / "index.html").write_text(html)
    # the chooser's card reads this; metrics.json's `updated` only moves when grading runs
    (config.DOCS / "published.json").write_text(json.dumps({"published": published}) + "\n")
    (config.DOCS / ".nojekyll").touch()
    for src in (config.PICKS, config.METRICS, config.RESULTS):
        if src.exists():
            shutil.copy(src, config.DOCS / src.name)
    log.info("built %s: %d games", config.DOCS / "index.html", len(picks))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    build()
