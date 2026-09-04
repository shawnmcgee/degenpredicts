"""Central configuration.

Everything is repo-relative so the same code runs on your laptop and in GitHub Actions.
Secrets come from the environment (Actions secrets / a local .env you never commit).
"""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(os.environ.get("DEGEN_ROOT", Path(__file__).resolve().parent.parent))
DATA = Path(os.environ.get("DEGEN_DATA", ROOT / "data"))
DOCS = Path(os.environ.get("DEGEN_DOCS", ROOT / "docs"))   # GitHub Pages serves this folder

# --- data files (all committed to the repo; that IS the database) -----------------
GAMES = DATA / "games.csv"                 # every final D1 game we've seen
LINES = DATA / "lines.csv"                 # every line snapshot we've ever pulled
PICKS = DATA / "picks.csv"                 # every published pick
RESULTS = DATA / "results.csv"             # every graded pick
TORVIK = DATA / "torvik_daily.csv"         # as-of-date tempo/efficiency snapshots
MODEL_DIR = DATA / "models"
METRICS = DATA / "metrics.json"

# --- external services ------------------------------------------------------------
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
NCAA_API_BASE = os.environ.get("NCAA_API_BASE", "https://ncaa-api.henrygd.me").rstrip("/")
NCAA_API_KEY = os.environ.get("NCAA_API_KEY", "")     # sent as the x-ncaa-key header
TORVIK_BASE = os.environ.get("TORVIK_BASE", "https://barttorvik.com")

HTTP_TIMEOUT = float(os.environ.get("DEGEN_HTTP_TIMEOUT", "30"))
HTTP_RETRIES = int(os.environ.get("DEGEN_HTTP_RETRIES", "4"))
NCAA_RATE_LIMIT = float(os.environ.get("DEGEN_NCAA_SLEEP", "0.25"))  # public API: 5 req/s per IP

# Books to prefer for "the line". First present wins, else median of all books.
PREFERRED_BOOKS = ["Pinnacle", "DraftKings", "FanDuel", "BetMGM", "Caesars", "Bovada"]
MARKETS = "totals,spreads"

# --- time & season ------------------------------------------------------------------
ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(ET)


def today_et() -> date:
    return now_et().date()


def season_of(d) -> int:
    """Season label = year the season starts in. Nov 2026 -> 2026, Mar 2027 -> 2026."""
    d = d if isinstance(d, date) else d.date()
    return d.year if d.month >= 8 else d.year - 1


def season_start(season: int) -> date:
    return date(season, 10, 20)


def season_end(season: int) -> date:
    return date(season + 1, 4, 15)


FIRST_SEASON = int(os.environ.get("DEGEN_FIRST_SEASON", "2021"))

# --- modelling / betting knobs -------------------------------------------------------
MIN_GAMES = int(os.environ.get("DEGEN_MIN_GAMES", "3"))      # thin-data guard
TOTAL_EDGE_MIN = float(os.environ.get("DEGEN_TOTAL_EDGE", "3.0"))
SPREAD_EDGE_MIN = float(os.environ.get("DEGEN_SPREAD_EDGE", "2.0"))
BOLD_MULT = 2.0                                              # bold = 2x the minimum edge
KELLY_FRACTION = float(os.environ.get("DEGEN_KELLY", "0.25"))  # quarter Kelly
BANKROLL_UNITS = 100.0

# Model output is shrunk toward the market before an edge is computed. 1.0 = trust the model
# completely, 0.0 = never disagree with the book. Calibrated from holdout; see train.py.
DEFAULT_SHRINK = float(os.environ.get("DEGEN_SHRINK", "0.5"))

SITE_TITLE = os.environ.get("DEGEN_SITE_TITLE", "DegenPredicts")


def ensure_dirs() -> None:
    for p in (DATA, DOCS, MODEL_DIR):
        p.mkdir(parents=True, exist_ok=True)
