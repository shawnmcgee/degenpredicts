"""Render the static EPL site into docs/epl/ (GitHub Pages serves the docs folder).

    python -m epl.site

No Flask, no server, no uploads. The workflow commits docs/ and Pages publishes it. The board
lives in its own subfolder so every sport can share one Pages deployment without overwriting
another's index.html.

The page is laid out for football rather than ported from the gridiron one. Each match shows a
**probability strip** - home, draw, away, and the likeliest scoreline - above three markets
priced off the same distribution. That ordering is the point: the distribution is the model, and
the three prices are three ways of reading it, so the reader sees what we think before they see
what it is worth.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import config

log = logging.getLogger("epl.site")
TEMPLATES = Path(__file__).resolve().parent / "templates"


def _recent_results(limit: int = 12) -> list[dict]:
    if not config.RESULTS.exists():
        return []
    df = pd.read_csv(config.RESULTS, dtype={"game_id": str}, low_memory=False)
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date", ascending=False).head(limit)
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def _context(g) -> str:
    """The one-line 'why this match is different' strip under the fixture.

    Only the situational facts a reader cannot get from the price itself, and only the ones
    this model actually uses: who is newly promoted, who is carrying European midweeks, whether
    it is a derby, and whether anyone is on short rest. Everything here corresponds to a real
    feature, which is deliberate - a context line that mentions things the model ignores
    teaches the reader to trust it for the wrong reasons.
    """
    bits = []
    if g.get("h_promoted") == 1:
        bits.append(f"{g.get('home_team')} newly promoted")
    if g.get("a_promoted") == 1:
        bits.append(f"{g.get('away_team')} newly promoted")
    if g.get("is_derby"):
        bits.append("local derby")
    for side, key in (("home_team", "h_in_europe"), ("away_team", "a_in_europe")):
        if g.get(key) == 1:
            bits.append(f"{g.get(side)} in Europe")
    rd = g.get("rest_diff")
    if rd is not None and rd == rd and abs(rd) >= 3:
        side = g.get("home_team") if rd > 0 else g.get("away_team")
        bits.append(f"{side} +{abs(int(rd))} days rest")
    km = g.get("travel_km")
    if km is not None and km == km and km >= 350:
        bits.append(f"{int(km)} km trip")
    if g.get("no_crowd"):
        bits.append("behind closed doors")
    return " · ".join(bits)


def _pct(v):
    return None if v is None or v != v else round(100 * float(v), 1)


def _board() -> tuple[list[dict], int | None]:
    """This round's fixtures, freshest prediction per match, ready for the template."""
    if not config.PICKS.exists():
        return [], None
    df = pd.read_csv(config.PICKS, dtype={"game_id": str}, low_memory=False)
    if df.empty:
        return [], None
    df["date"] = pd.to_datetime(df["date"]).dt.date
    upcoming = df[df["date"] >= config.today_uk()]
    df = upcoming if len(upcoming) else df[df["matchweek"] == df["matchweek"].max()]
    week = int(df["matchweek"].mode().iloc[0]) if len(df) else None
    df = df.sort_values("prediction_date").drop_duplicates("game_id", keep="last").copy()

    # Rank by the model's largest disagreement with the market, but a positive-EV price on a
    # real quote outranks it - that is the only number here tied to something you could pay.
    have = [c for c in ("sup_disagree", "total_disagree") if c in df]
    df["_rank"] = (df[have].abs().max(axis=1).fillna(0) if have
                   else pd.Series(0.0, index=df.index))
    playable = pd.Series(False, index=df.index)
    for c in ("ah_ev", "total_ev", "x2_ev"):
        if c in df:
            playable |= df[c].fillna(-1) > 0
    df["has_play"] = playable
    df.loc[playable, "_rank"] = df.loc[playable, "_rank"] + 100
    df = df.sort_values("_rank", ascending=False)

    for c in ("p_home", "p_draw", "p_away", "p_btts", "ah_p_win", "total_p_win", "x2_p_win",
              "x2_mkt_p", "mkt_p_home", "mkt_p_draw", "mkt_p_away", "score_prob"):
        if c in df:
            df[c + "_pct"] = [_pct(v) for v in df[c]]

    df["day"] = pd.to_datetime(df["date"]).dt.strftime("%a %-d %b")
    df["day_key"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["time_label"] = [("Time TBC" if (not t or str(t) in ("nan", "")) else str(t))
                        for t in df.get("kickoff", pd.Series([""] * len(df)))]
    df["kick_sort"] = df["day_key"].fillna("") + " " + df.get(
        "kickoff", pd.Series([""] * len(df), index=df.index)).fillna("")
    df["search"] = (df["home_team"].fillna("") + " " + df["away_team"].fillna("")).str.lower()

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


def _meta() -> dict:
    path = config.MODEL_DIR / "meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def _model_note() -> dict:
    """What the last training run concluded, surfaced where a reader will see it.

    If the model does not beat the market, the page says so. Publishing the number while hiding
    that finding would be the dishonest version of this project.
    """
    meta = _meta()
    e = (meta.get("eval") or {}).get("sup_market") or {}
    if e.get("skipped") or not e:
        return {}
    pr = meta.get("probability") or {}
    return {"beats_market": e.get("beats_market"), "mae_model": e.get("mae_model"),
            "mae_market": e.get("mae_market_baseline"), "cover_rate": e.get("cover_rate"),
            "cover_stderr": e.get("cover_stderr"), "cover_n": e.get("cover_n"),
            "shrink": e.get("shrink"), "seasons": e.get("test_seasons"),
            "log_loss_model": pr.get("log_loss_model"),
            "log_loss_market": pr.get("log_loss_market"),
            "log_loss_edge": pr.get("log_loss_edge"),
            "beats_log_loss": pr.get("beats_market_log_loss"),
            "draw": pr.get("draw") or {}}


def _backtest() -> dict:
    """The out-of-sample disagreement table from the last training run.

    The live table is built from graded picks, so it is empty until matches have been played
    and stays thin for months. Meanwhile the walk-forward already measured the same thing
    across thousands of out-of-sample matches. This surfaces it, clearly labelled as a
    backtest, with the significance correction attached.
    """
    meta = _meta()
    ev = (meta.get("eval") or {}).get("sup_market") or {}
    tot = (meta.get("eval") or {}).get("total_market") or {}
    if ev.get("skipped") or not ev.get("cover_by_disagreement"):
        return {}
    by_total = {b.get("disagreement"): b for b in tot.get("cover_by_disagreement") or []}
    rows = [{"bucket": b.get("disagreement"), "spread": b,
             "total": by_total.get(b.get("disagreement"))}
            for b in ev["cover_by_disagreement"]]
    sig = ev.get("significance") or {}
    return {"rows": rows, "seasons": ev.get("test_seasons") or [],
            "n": ev.get("n_test_total"), "z_required": sig.get("z_required"),
            "verdict": sig.get("verdict"), "break_even": ev.get("break_even_pct")}


def _calibration() -> list[dict]:
    return ((_meta().get("probability") or {}).get("calibration") or [])


def _days(picks: list[dict]) -> list[dict]:
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
        title=config.SITE_TITLE, league=config.LEAGUE_NAME, picks=picks, m=metrics, week=week,
        results=_recent_results(), venue=config.VENUE, model=_model_note(),
        support_url=config.SUPPORT_URL, support_label=config.SUPPORT_LABEL,
        days=_days(picks), updated=metrics.get("updated", ""),
        backtest=_backtest(), calibration=_calibration(),
        sup_min=config.SUP_EDGE_MIN, total_min=config.GOALS_EDGE_MIN,
    )
    (config.DOCS / "index.html").write_text(html)
    (config.DOCS / ".nojekyll").touch()
    for src in (config.PICKS, config.METRICS, config.RESULTS):
        if src.exists():
            shutil.copy(src, config.DOCS / src.name)
    log.info("built %s: matchweek %s, %d fixtures", config.DOCS / "index.html", week, len(picks))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    build()
