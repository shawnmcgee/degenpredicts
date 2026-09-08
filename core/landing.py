"""Render the site's front door: a chooser for the published sport boards.

    python -m core.landing

This is the one page that belongs to no sport, which is why it lives in ``core``. It imports
nothing from ``cfb``, ``nfl`` or ``ncaab`` - it reads the files those pipelines have already
published into ``docs/<slug>/`` and renders a card for each. That keeps the isolation rule
intact: a sport can break, or be removed entirely, and the worst that happens here is its card
stops appearing.

Each publishing workflow calls this after building its own board, so the front page refreshes
whenever any sport does, and self-heals if one run is skipped.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

log = logging.getLogger("core.landing")

ROOT = Path(os.environ.get("DEGEN_ROOT", Path(__file__).resolve().parent.parent))
DOCS = Path(os.environ.get("DEGEN_DOCS", ROOT / "docs"))
TEMPLATES = Path(__file__).resolve().parent / "templates"

SITE_TITLE = os.environ.get("DEGEN_SITE_TITLE", "DegenPredicts")
SUPPORT_URL = os.environ.get("DEGEN_SUPPORT_URL",
                             os.environ.get("DEGEN_COFFEE_URL",
                                            "https://buymeacoffee.com/smcgee"))
SUPPORT_LABEL = os.environ.get("DEGEN_SUPPORT_LABEL", "Buy me a coffee")

# The sports this site knows how to show, in the order they appear. A slug with no published
# index.html is skipped, so basketball can sit here dormant until November and appear on its
# own the first time its pipeline publishes.
SPORTS = [
    {"slug": "cfb", "name": "College Football",
     "blurb": "Every FBS game, spreads and totals."},
    {"slug": "nfl", "name": "NFL",
     "blurb": "Spreads and totals, with quarterback, rest, travel and weather context."},
    {"slug": "ncaab", "name": "College Basketball",
     "blurb": "Built and dormant until November."},
]


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _board_summary(picks: Path, today: str) -> dict:
    """Week number and game count for the games still to be played.

    Reads the published picks file directly with the csv module rather than importing a
    sport's own loader - this page must not depend on any pipeline's code.
    """
    if not picks.exists():
        return {}
    try:
        with picks.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return {}
    if not rows:
        return {}
    # picks.csv accumulates history and can hold several predictions per game; keep the last
    # row per game_id, then look only at games that have not kicked off yet.
    latest = {r.get("game_id"): r for r in rows}
    upcoming = [r for r in latest.values() if (r.get("date") or "") >= today]
    if not upcoming:
        return {"games": 0}
    weeks = [r.get("week") for r in upcoming if r.get("week")]
    week = max(set(weeks), key=weeks.count) if weeks else None
    plays = sum(1 for r in upcoming
                if r.get("total_strength") in ("play", "bold")
                or r.get("spread_strength") in ("play", "bold"))
    try:
        week = int(float(week)) if week is not None else None
    except (TypeError, ValueError):
        week = None
    return {"games": len(upcoming), "week": week, "plays": plays}


def _record(metrics: dict, key: str) -> dict:
    """Season record for one market, preferring the all-games figure.

    `all_games` rather than `season`: the staked-only record is empty by design whenever the
    edge thresholds sit above what the model can prove, and an empty card reads as broken
    rather than as honest.
    """
    block = (metrics.get(key) or {}).get("all_games") or {}
    if not block.get("n"):
        return {}
    return {"n": block.get("n"), "units": block.get("units"),
            "win_pct": block.get("win_pct"), "clv": block.get("clv")}


def collect(docs: Path | None = None, today: str | None = None) -> list[dict]:
    """One card per sport that has actually published a board."""
    docs = docs or DOCS
    today = today or date.today().isoformat()
    cards = []
    for sport in SPORTS:
        folder = docs / sport["slug"]
        if not (folder / "index.html").exists():
            log.info("skipping %s: nothing published at %s", sport["slug"], folder)
            continue
        metrics = _json(folder / "metrics.json")
        card = dict(sport)
        card["href"] = f"{sport['slug']}/"
        card["updated"] = metrics.get("updated")
        card["board"] = _board_summary(folder / "picks.csv", today)
        card["totals"] = _record(metrics, "totals")
        card["spreads"] = _record(metrics, "spreads")
        card["has_data"] = bool(card["board"].get("games"))
        cards.append(card)
    return cards


def build(docs: Path | None = None) -> Path:
    docs = docs or DOCS
    docs.mkdir(parents=True, exist_ok=True)
    env = Environment(loader=FileSystemLoader(TEMPLATES),
                      autoescape=select_autoescape(["html"]))
    env.filters["money"] = lambda v: ("+" if (v or 0) >= 0 else "") + f"{v or 0:.2f}"
    cards = collect(docs)
    html = env.get_template("landing.html").render(
        title=SITE_TITLE, sports=cards,
        support_url=SUPPORT_URL, support_label=SUPPORT_LABEL,
        updated=max((c["updated"] for c in cards if c.get("updated")), default=""),
    )
    out = docs / "index.html"
    out.write_text(html)
    (docs / ".nojekyll").touch()
    log.info("built %s with %d sport(s): %s", out, len(cards),
             ", ".join(c["slug"] for c in cards) or "none")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    build()
