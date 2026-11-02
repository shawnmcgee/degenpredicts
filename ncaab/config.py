"""Configuration for the college basketball pipeline.

Same shape as the other sports: everything is repo-relative, the repo is the database, and the
board publishes into its own folder so it can never overwrite another sport's page.

**Why the paths moved.** The first version of this package predates the multi-sport layout and
wrote to ``data/`` and ``docs/`` at the root - so its first run would have replaced the sport
chooser at ``docs/index.html`` with a basketball page. It now owns ``data/ncaab/`` and
``docs/ncaab/`` like every other board, and a test pins it there.

**Why every knob is ``DEGEN_NCAAB_*``.** The other sports read unprefixed names like
``DEGEN_FIRST_SEASON`` and ``DEGEN_BOARD_DAYS``. A repo variable set for one of them would
silently retune this board too, so nothing here shares a name with them.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _env(name: str, default: str) -> str:
    """Empty counts as unset: GitHub Actions passes an unconfigured repo variable as ''."""
    return os.environ.get(name, "").strip() or default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


ROOT = Path(os.environ.get("DEGEN_ROOT", Path(__file__).resolve().parent.parent))
DATA = Path(os.environ.get("DEGEN_DATA", ROOT / "data")) / "ncaab"
_NCAAB_DOCS = _env("DEGEN_NCAAB_DOCS", "")
DOCS = Path(_NCAAB_DOCS) if _NCAAB_DOCS else \
    Path(os.environ.get("DEGEN_DOCS", ROOT / "docs")) / "ncaab"

# --- data files (all committed; the repo is the database) --------------------------
GAMES = DATA / "games.csv"          # every final D1 game: score, ESPN's closing line, context
DAYS = DATA / "days.csv"            # which scoreboard days have been fetched for good
PICKS = DATA / "picks.csv"          # one row per game: the latest pre-tip prediction
RESULTS = DATA / "results.csv"      # every graded game
METRICS = DATA / "metrics.json"
MODEL_DIR = DATA / "models"         # meta.json: the bake-off leaderboard and the chosen models

# --- the source ----------------------------------------------------------------------
# One ESPN endpoint gives everything: finals, the slate, ESPN's partner-book line (also on
# finished games, which is the closing line the models train against), AP rank, neutral site
# and conference. groups=50 is all of Division I - without it ESPN returns a curated handful.
ESPN_SCOREBOARD = _env(
    "DEGEN_NCAAB_SCOREBOARD",
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard")
ESPN_PARAMS = {"groups": "50", "limit": "1000"}
ESPN_SLEEP = _env_float("DEGEN_NCAAB_SLEEP", 0.2)
MAX_SPREAD = 55.0                   # anything bigger in a spread slot is a moneyline, not a line

HTTP_TIMEOUT = _env_float("DEGEN_HTTP_TIMEOUT", 30.0)
HTTP_RETRIES = _env_int("DEGEN_HTTP_RETRIES", 4)

# --- time & season --------------------------------------------------------------------
ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(ET)


def today_et() -> date:
    return now_et().date()


def season_of(d) -> int:
    """Season label = the year it starts in. Nov 2026 -> 2026, Mar 2027 -> 2026."""
    d = d if isinstance(d, date) and not isinstance(d, datetime) else pd_date(d)
    return d.year if d.month >= 8 else d.year - 1


def pd_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.fromisoformat(str(d)[:10]).date()


def season_start(season: int) -> date:
    """First day scanned. Opening night is the first Monday in November; 2020 started on
    Nov 25 - scanning from late October costs a few empty days and misses nothing."""
    return date(season, 10, 25)


def season_end(season: int) -> date:
    return date(season + 1, 4, 12)


def in_season(d: date) -> bool:
    s = season_of(d)
    return season_start(s) <= d <= season_end(s)


# 2018-19 onward: ESPN's feed carries a line on most D1 games from there. The first season is a
# warm-up - it is replayed so the ratings start the next one from something, but no model is
# trained or scored on it.
FIRST_SEASON = _env_int("DEGEN_NCAAB_FIRST_SEASON", 2018)
WARMUP_SEASONS = _env_int("DEGEN_NCAAB_WARMUP_SEASONS", 1)
WALK_SEASONS = _env_int("DEGEN_NCAAB_WALK_SEASONS", 5)

# --- the board ------------------------------------------------------------------------
BOARD_DAYS = _env_int("DEGEN_NCAAB_BOARD_DAYS", 1)       # today plus this many days ahead
MIN_GAMES = _env_int("DEGEN_NCAAB_MIN_GAMES", 4)         # below this a side is "early season"
# A team that plays fewer games than this in a season is not Division I - it is a D2/NAIA
# visitor on a D1 schedule. Those games still move the D1 team's ratings, but no model is
# trained on them: there is no line, and the visitor has no rating worth the name.
D1_MIN_GAMES = _env_int("DEGEN_NCAAB_D1_MIN_GAMES", 10)

# --- the bake-off ---------------------------------------------------------------------
# "1se": the simplest model whose error is within one standard error of the best one - the
# leader on a few thousand games is usually ahead by noise. "lowest": whatever scored best.
SELECT = _env("DEGEN_NCAAB_SELECT", "1se").lower()
# Comma-separated candidate names to skip, e.g. "mlp,rf" to make the weekly job faster.
SKIP_MODELS = {m.strip() for m in _env("DEGEN_NCAAB_SKIP_MODELS", "").split(",") if m.strip()}
DEFAULT_SHRINK = _env_float("DEGEN_NCAAB_SHRINK", 0.5)
BREAK_EVEN = 52.38                  # a -110 side has to hit this often to break even

SITE_TITLE = _env("DEGEN_SITE_TITLE", "DegenPredicts")
# Same as every other board: set to blank to hide the button.
COFFEE_URL = os.environ.get("DEGEN_COFFEE_URL", "https://buymeacoffee.com/smcgee")
SUPPORT_URL = os.environ.get("DEGEN_SUPPORT_URL", COFFEE_URL)
SUPPORT_LABEL = _env("DEGEN_SUPPORT_LABEL", "Buy me a coffee")

# --- conferences ----------------------------------------------------------------------
# ESPN's group ids, as they appear in the scoreboard's team.conferenceId. An id that is not
# here simply shows no conference name; nothing downstream depends on the map being complete.
CONFERENCES = {
    "1": "America East", "2": "ACC", "3": "Atlantic 10", "4": "Big East", "5": "Big Sky",
    "6": "Big South", "7": "Big Ten", "8": "Big 12", "9": "Big West", "10": "CAA",
    "11": "Conference USA", "12": "Ivy League", "13": "MAAC", "14": "MAC", "16": "MEAC",
    "18": "Missouri Valley", "19": "NEC", "20": "OVC", "21": "Pac-12", "22": "Patriot",
    "23": "SEC", "24": "SoCon", "25": "Southland", "26": "SWAC", "27": "Sun Belt", "29": "WCC",
    "30": "WAC", "44": "Mountain West", "45": "Horizon", "46": "ASUN", "49": "Summit League",
    "62": "American",
}
POWER = {"2", "4", "7", "8", "23"}  # ACC, Big East, Big Ten, Big 12, SEC


def ensure_dirs() -> None:
    for p in (DATA, DOCS, MODEL_DIR):
        p.mkdir(parents=True, exist_ok=True)
