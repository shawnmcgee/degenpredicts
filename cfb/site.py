"""Render the static site into docs/ (GitHub Pages serves that folder).

    python -m cfb.site

No Flask, no server, no uploads. The workflow commits docs/ and Pages publishes it.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import config

log = logging.getLogger("cfb.site")
TEMPLATES = Path(__file__).resolve().parent / "templates"  # ships with the package


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
    df = (df.sort_values("prediction_date").drop_duplicates("game_id", keep="last")
            .sort_values(["date", "tip_et"]))
    df = df.where(pd.notna(df), None)
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
