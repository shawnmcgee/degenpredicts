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


def _board() -> tuple[list[dict], int | None]:
    """This week's games, freshest prediction per game."""
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
    # Rank by the model's largest disagreement with the line - that's the reason to look at
    # this page at all. Kickoff order is available via the filter chips.
    df["_rank"] = df[["total_disagree", "margin_disagree"]].abs().max(axis=1)
    # A tradeable Kalshi edge outranks a big model disagreement - it is the only number here
    # that reflects a price you could actually pay.
    if "kalshi_home_ev" in df:
        best_ev = df[["kalshi_home_ev", "kalshi_away_ev"]].max(axis=1)
        df["kalshi_best_ev"] = best_ev.round(3)
        df["kalshi_ev_cents"] = (best_ev * 100).round(1)
        df["_rank"] = df["_rank"] + np.where(
            df.get("kalshi_tradeable", False).fillna(False).astype(bool) & (best_ev > 0),
            50 + best_ev * 100, 0)
    df = df.sort_values("_rank", ascending=False)
    for col, out in (("total_p_win", "total_cents"), ("spread_p_win", "spread_cents"),
                     ("p_home_win", "home_win_cents"), ("p_away_win", "away_win_cents")):
        if col in df:
            df[out] = df[col].map(_prob_cents)
    # tier flag drives the G5 filter chip
    P4 = {"SEC", "Big Ten", "Big 12", "ACC"}
    if "home_conf" in df and "away_conf" in df:
        df["is_g5"] = ~(df["home_conf"].isin(P4) & df["away_conf"].isin(P4))
    else:
        df["is_g5"] = False
    df = df.astype(object).where(pd.notna(df), None)
    return df.to_dict("records"), week


def _metrics() -> dict:
    if config.METRICS.exists():
        return json.loads(config.METRICS.read_text())
    return {}


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
