"""Configuration for the NHL pipeline.

Deliberately parallel to ``nfl/config.py`` and ``epl/config.py`` - same knob names, same env-var
prefix - so the pipelines stay readable side by side. What differs here differs because hockey
is neither gridiron nor association football, and each divergence is spelled out below:

* **The unit is goals, about six a game**, so the markets are priced, not pointed. A puck line
  is almost always +/-1.5 and a total almost always 5.5, 6 or 6.5; what moves is the PRICE.
  The thresholds are therefore EV thresholds against the offered price, not "points off the
  line" - a 0.3-goal disagreement means nothing until it is turned into a probability and set
  against what the book is paying.
* **The calendar wraps a new year and has one broken season.** A season runs October to June
  and is named for the year it starts in, except 2019-20, whose playoffs ran into late
  September 2020. Games carry the NHL's own season id, so dates are never used to assign one.
* **Most totals lines are whole numbers now.** 41% of 2023-26 totals were 6.0, which pushes on
  exactly six goals. Every price, EV and grade carries a push leg.
* **Odds are American at the books** but are aggregated and priced in decimal only. The median
  of -115 and +105 in American odds is -5, which reads as a 20x payout; that is not a
  hypothetical, it is what the first backtest of this pipeline did.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _env(name: str, default: str) -> str:
    """Read an environment variable, treating empty as unset.

    ``os.environ.get(name, default)`` returns "" for a variable that is SET but empty, not the
    default - and GitHub Actions passes an unconfigured repo variable as exactly that. For the
    numeric knobs that is worse than blank: float("") raises, and the daily job dies on a
    variable nobody ever set.
    """
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
DATA = Path(os.environ.get("DEGEN_DATA", ROOT / "data")) / "nhl"
_NHL_DOCS = os.environ.get("DEGEN_NHL_DOCS", "")
DOCS = Path(_NHL_DOCS) if _NHL_DOCS else \
    Path(os.environ.get("DEGEN_DOCS", ROOT / "docs")) / "nhl"

GAMES = DATA / "games.csv"          # results, starting goalies, shots - one row per game
LINES = DATA / "lines.csv"          # historical closing lines, and our own pre-game snapshots
SNAPSHOTS = DATA / "snapshots.csv"  # every live pull, for closing-line value
STARTERS = DATA / "starters.csv"    # every starting-goalie pull: who, and how sure
PICKS = DATA / "picks.csv"
RESULTS = DATA / "results.csv"
METRICS = DATA / "metrics.json"
MODEL_DIR = DATA / "models"

# --- credentials --------------------------------------------------------------------
# The NHL's own APIs need no key. ODDS_API_KEY is what turns the board into prices you can bet;
# without it the model still predicts every game, but nothing can be staked because there is
# no price to stake against.
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
# Consensus is formed across whatever of these the feed returns. Named books are preferred for
# the quoted price only when present on every side of a market; see sources/odds.py.
ODDS_BOOKS = ["DraftKings", "FanDuel", "BetMGM", "Caesars", "BetRivers", "Bovada"]

HTTP_TIMEOUT = float(os.environ.get("DEGEN_NHL_HTTP_TIMEOUT",
                                    os.environ.get("DEGEN_HTTP_TIMEOUT", "60")))
HTTP_RETRIES = int(os.environ.get("DEGEN_NHL_HTTP_RETRIES",
                                  os.environ.get("DEGEN_HTTP_RETRIES", "3")))

# The NHL Stats REST API. One call returns every game in a season, scheduled ones included, and
# one more returns every goalie's line for every game in it. That is the whole data spine.
NHL_STATS_API = _env("DEGEN_NHL_STATS_API", "https://api.nhle.com/stats/rest/en").rstrip("/")

# Starting goalies. The NHL names a starter only once the puck drops; Daily Faceoff tracks the
# day's news and labels each one Confirmed, Likely or Unconfirmed (see sources/starters.py).
# DEGEN_NHL_STARTERS=0 runs the board on the model's own guess from recent starts instead.
STARTERS_URL = _env("DEGEN_NHL_STARTERS_URL",
                    "https://www.dailyfaceoff.com/starting-goalies/{date}")
STARTERS_ON = _env("DEGEN_NHL_STARTERS", "1").lower() not in ("0", "false", "no", "off")
# How much of the board's belief about who starts each label carries; the rest stays on the
# model's own guess. Every pull is logged to starters.csv so these can be checked against who
# actually started.
STARTER_WEIGHT = {"confirmed": 1.0, "likely": 0.85, "projected": 0.5}
# A pick is staked only once both starters are confirmed or likely - whenever the feed is up at
# all. With the feed down the board falls back to the model's guess and stakes as before.
REQUIRE_STARTERS = _env("DEGEN_NHL_REQUIRE_STARTERS", "1").lower() not in ("0", "false", "no",
                                                                            "off")

# Historical lines are imported once and committed (see sources/history.py). Both archives are
# static files on raw.githubusercontent.com - the host nflverse and the EPL archive are served
# from - so the import can be re-run from a GitHub runner if it ever needs rebuilding.
HISTORY_SBR_URL = _env(
    "DEGEN_NHL_HISTORY_SBR",
    "https://raw.githubusercontent.com/ethanbell528-cmd/fda-project-1/main/data/nhl.csv")
HISTORY_GOALIES_URL = _env(
    "DEGEN_NHL_HISTORY_GOALIES",
    "https://raw.githubusercontent.com/ethanbell528-cmd/fda-project-1/main/data/nhl_goalie_starts.csv")
HISTORY_ODDSAPI_URL = _env(
    "DEGEN_NHL_HISTORY_ODDSAPI",
    "https://raw.githubusercontent.com/nielsenz/odds-api-current-save/main/odds-data/historical/"
    "odds_{date}.csv")

# --- time / season -------------------------------------------------------------------
ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(ET)


def today_et() -> date:
    return now_et().date()


def season_of(d) -> int:
    """Season = the calendar year it starts in, with an OCTOBER boundary.

    Only ever used to name the season TODAY falls in. Games carry the NHL's own season id, and
    that is the one to trust: 2019-20 finished its playoffs in the bubble on 28 September 2020,
    which no date boundary can file correctly while also filing an October opener correctly. An
    October boundary puts late-September preseason games in the previous season, which is
    harmless because preseason games are never loaded.
    """
    d = d if isinstance(d, date) else d.date()
    return d.year if d.month >= 10 else d.year - 1


def season_id(season: int) -> int:
    """The NHL's eight-digit season id: 2026 -> 20262027."""
    return season * 10000 + season + 1


def season_end(season: int) -> date:
    """The Stanley Cup final ends in June; July 1 is a safe upper bound (2019-20 aside)."""
    return date(season + 1, 7, 1)


GAMES_PER_SEASON = 1312            # 32 teams x 82 / 2
# 2005-06 is the first season of the modern rules: the shootout, the two-line pass and the
# goalie trapezoid all arrived together after the lockout. Earlier hockey is a different game.
# Closing lines start in 2007-08, so 2005 and 2006 are warm-up: they advance the ratings and
# the goalie tracker but emit no training rows.
FIRST_SEASON = _env_int("DEGEN_FIRST_SEASON", 2007)
WARMUP_SEASONS = _env_int("DEGEN_WARMUP_SEASONS", 2)
LOAD_FROM_SEASON = FIRST_SEASON - WARMUP_SEASONS
# Hockey is played nearly every day. The board is today's and tomorrow's games; the evening run
# refreshes the same slate with prices that know the starting goalies.
BOARD_DAYS = _env_int("DEGEN_BOARD_DAYS", 1)

# --- modelling / betting --------------------------------------------------------------
# Games a team must have played THIS season before its picks are staked. Ratings carry over
# between seasons (0.6-0.8 of the way) but rosters and, above all, goaltending turn over; five
# games is about ten days into October.
MIN_GAMES = _env_int("DEGEN_MIN_GAMES", 5)

# Selection is on EXPECTED VALUE at the offered price, per unit staked, with pushes returned.
# Set these from the `roi_by_ev` tables in models/meta.json, not from intuition. The defaults
# sit above the zero-EV line the backtest measured because that backtest ran against noon
# prices for three seasons only; see README for exactly what it did and did not show.
SPREAD_EV_MIN = _env_float("DEGEN_NHL_SPREAD_EV", 0.03)     # puck line
TOTAL_EV_MIN = _env_float("DEGEN_NHL_TOTAL_EV", 0.05)       # over/under
# Kept under the shared names too, so the reporting helpers read identically across sports.
SPREAD_EDGE_MIN = SPREAD_EV_MIN
TOTAL_EDGE_MIN = TOTAL_EV_MIN
BOLD_MULT = 2.0
# Eighth Kelly, matching the NFL and the Premier League. Kelly assumes you know your edge;
# against a market this close to efficient, overestimating it is the expensive mistake.
KELLY_FRACTION = _env_float("DEGEN_KELLY", 0.125)
BANKROLL_UNITS = 100.0

# --- venue / cost model ---------------------------------------------------------------
# NHL sides and totals are bet at sportsbooks at posted American prices, so there is no single
# break-even: EV is computed against each price. BREAK_EVEN is the -110 reference, used only by
# the shared reporting helpers and the page's hit-rate colouring.
VENUE = os.environ.get("DEGEN_VENUE", "sportsbook")
BREAK_EVEN = _env_float("DEGEN_BREAK_EVEN", 0) or 52.38

# How far off the market the published number is allowed to move, and the defaults used when
# too few priced games exist to fit them. The split (who wins, by how much) and the total are
# shrunk separately because the backtest says they deserve different amounts of trust: the
# totals market was the one the model never beat.
DEFAULT_SPLIT_SHRINK = _env_float("DEGEN_NHL_SPLIT_SHRINK", 0.5)
DEFAULT_TOTAL_SHRINK = _env_float("DEGEN_NHL_TOTAL_SHRINK", 0.25)
SHRINK_CAP = _env_float("DEGEN_SHRINK_CAP", 0.8)

# The over price the SBR archive does not carry. 2023-26 de-vigged over prices averaged 0.505
# at every common line (sd 0.02), so a juice-less closing total is read as that.
P_OVER_DEFAULT = 0.505

SITE_TITLE = os.environ.get("DEGEN_SITE_TITLE", "DegenPredicts")
COFFEE_URL = os.environ.get("DEGEN_COFFEE_URL", "https://buymeacoffee.com/smcgee")
SUPPORT_URL = os.environ.get("DEGEN_SUPPORT_URL", COFFEE_URL)
SUPPORT_LABEL = os.environ.get("DEGEN_SUPPORT_LABEL", "Buy me a coffee")


def ensure_dirs() -> None:
    for p in (DATA, DOCS, MODEL_DIR):
        p.mkdir(parents=True, exist_ok=True)
