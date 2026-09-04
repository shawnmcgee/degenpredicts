"""Render the static site into docs/ (GitHub Pages serves that folder).

    python -m ncaab.site

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

log = logging.getLogger("ncaab.site")
TEMPLATES = Path(__file__).resolve().parent / "templates"  # ships with the package


def _picks_for_today() -> list[dict]:
    if not config.PICKS.exists():
        return []
    df = pd.read_csv(config.PICKS)
    if df.empty:
        return []
    latest = df["prediction_date"].max()
    df = df[df["prediction_date"] == latest].copy()
    df = df.replace({float("nan"): None}).where(pd.notna(df), None)
    return df.sort_values("tip_et").to_dict("records")


def _metrics() -> dict:
    if config.METRICS.exists():
        return json.loads(config.METRICS.read_text())
    return {}


def build() -> None:
    config.ensure_dirs()
    env = Environment(loader=FileSystemLoader(TEMPLATES),
                      autoescape=select_autoescape(["html"]))
    env.filters["money"] = lambda v: ("+" if (v or 0) >= 0 else "") + f"{v or 0:.2f}"
    picks = _picks_for_today()
    metrics = _metrics()
    html = env.get_template("index.html").render(
        title=config.SITE_TITLE, picks=picks, m=metrics,
        updated=metrics.get("updated", ""),
        total_min=config.TOTAL_EDGE_MIN, spread_min=config.SPREAD_EDGE_MIN,
    )
    (config.DOCS / "index.html").write_text(html)
    (config.DOCS / ".nojekyll").touch()
    # expose the raw data so the page (or you) can fetch it
    for src in (config.PICKS, config.METRICS, config.RESULTS):
        if src.exists():
            shutil.copy(src, config.DOCS / src.name)
    log.info("built %s with %d picks", config.DOCS / "index.html", len(picks))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    build()
