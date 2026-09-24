"""Configuration for the NBA pipeline.

Deliberately parallel to ``nfl/config.py`` and ``nhl/config.py`` - same knob names, same env-var
prefix - so the pipelines stay readable side by side. What differs here differs because
basketball is neither gridiron nor hockey, and each divergence is spelled out below:

* **Who plays is most of the story.** One player can be worth five points of spread, and the
  books move within minutes of an injury report. The ratings therefore carry a player layer
  (see :mod:`nba.players`), the board reads the day's injury report, and a pick whose line
  still hangs on a questionable star waits for the news rather than betting into it.
* **The unit is points, about 230 a game, at roughly -110 either side** - so spreads and totals
  are pointed like the NFL's, not priced like the NHL's puck line. Thresholds are points of
  disagreement with the line; the moneyline, which is priced, uses expected value.
* **The calendar crosses a new year and has one broken season.** A season runs October to June
  and is named for the year it starts in. 2019-20 finished in the Orlando bubble in October 2020
  and 2020-21 started that December, so games carry the season the data source assigns them and
  dates are never used to file one.
* **The free odds quota is already spoken for.** Four other boards share one Odds API key, and
  in their overlapping months they use most of its free 500 credits. So the NBA board reads the
  free ESPN scoreboard's prices by default and uses The Odds API only when asked
  (``DEGEN_NBA_ODDS=oddsapi``).
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
DATA = Path(os.environ.get("DEGEN_DATA", ROOT / "data")) / "nba"
_NBA_DOCS = os.environ.get("DEGEN_NBA_DOCS", "")
DOCS = Path(_NBA_DOCS) if _NBA_DOCS else \
    Path(os.environ.get("DEGEN_DOCS", ROOT / "docs")) / "nba"

GAMES = DATA / "games.csv"              # schedule, results and possessions - one row per game
PLAYERS = DATA / "players"              # minutes and box-score value, one file per season
PLAYER_NAMES = DATA / "player_names.csv"
LINES = DATA / "lines.csv"              # closing lines, and our own pre-tip snapshots
SNAPSHOTS = DATA / "snapshots.csv"      # every live price pull, for closing-line value
INJURIES = DATA / "injuries.csv"        # every injury-report pull: who, and how likely to sit
PICKS = DATA / "picks.csv"
RESULTS = DATA / "results.csv"
METRICS = DATA / "metrics.json"
MODEL_DIR = DATA / "models"

# --- credentials and sources --------------------------------------------------------
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
# "espn" (default): the free ESPN scoreboard's lines and prices, one book, no quota.
# "oddsapi": The Odds API's consensus across US books - 3 credits a run, from the shared key.
# "none": no prices at all; the board still predicts every game.
ODDS_SOURCE = _env("DEGEN_NBA_ODDS", "espn").lower()
ODDS_BOOKS = ["DraftKings", "FanDuel", "BetMGM", "Caesars", "BetRivers", "Bovada"]

HTTP_TIMEOUT = float(os.environ.get("DEGEN_NBA_HTTP_TIMEOUT",
                                    os.environ.get("DEGEN_HTTP_TIMEOUT", "60")))
HTTP_RETRIES = int(os.environ.get("DEGEN_NBA_HTTP_RETRIES",
                                  os.environ.get("DEGEN_HTTP_RETRIES", "3")))

# hoopR (sportsdataverse) republishes ESPN's NBA schedules, team box scores and player box scores
# as static files on GitHub releases, every morning of the season - the NBA's nflverse. No key.
HOOPR_RELEASES = _env("DEGEN_HOOPR_RELEASES",
                      "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
                      ).rstrip("/")
# Today's slate, tip times, game state and the ESPN book's prices; and the league injury report.
ESPN_SCOREBOARD = _env("DEGEN_NBA_ESPN_SCOREBOARD",
                       "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard")
ESPN_INJURIES = _env("DEGEN_NBA_ESPN_INJURIES",
                     "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries")
INJURIES_ON = _env("DEGEN_NBA_INJURIES", "1").lower() not in ("0", "false", "no", "off")

# Historical lines are imported once and committed (see sources/history.py). All three archives
# are static files on raw.githubusercontent.com, so the import can be re-run from a runner.
HISTORY_CLOSE_URL = _env(
    "DEGEN_NBA_HISTORY_CLOSE",
    "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba/betting_lines/"
    "closing_lines_odds_api.parquet")
HISTORY_ARCHIVE_URL = _env(
    "DEGEN_NBA_HISTORY_ARCHIVE",
    "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba/betting_lines/"
    "games-archive.json")
HISTORY_ML_URL = _env(
    "DEGEN_NBA_HISTORY_ML",
    "https://raw.githubusercontent.com/kyleskom/NBA-Machine-Learning-Sports-Betting/master/Data/"
    "OddsData.sqlite")

# --- time / season -------------------------------------------------------------------
ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(ET)


def today_et() -> date:
    return now_et().date()


def season_of(d) -> int:
    """Season = the calendar year it starts in, with an AUGUST boundary.

    Only ever used to name the season TODAY falls in. Games carry the season the data source
    gave them: the 2019-20 Finals were played in October 2020, which no date boundary can file
    correctly while also filing the next October's opener correctly.
    """
    d = d if isinstance(d, date) else d.date()
    return d.year if d.month >= 8 else d.year - 1


def season_end(season: int) -> date:
    """The Finals end in June; July 1 is a safe upper bound (the 2019-20 bubble aside, which
    finished in October 2020 - long before anything here looks at it)."""
    return date(season + 1, 7, 1)


def season_label(season: int) -> str:
    return f"{season}-{str(season + 1)[-2:]}"


GAMES_PER_SEASON = 1230            # 30 teams x 82 / 2
# 2003-04 is loaded as warm-up: the ratings and the player book need a couple of seasons of
# history before a training row means anything. Rows start in 2007-08, the first season with
# a closing moneyline in the archive, and a year after the league last changed shape (the
# 2004-05 hand-check rules, the Bobcats joining to make thirty).
FIRST_SEASON = _env_int("DEGEN_FIRST_SEASON", 2007)
WARMUP_SEASONS = _env_int("DEGEN_WARMUP_SEASONS", 4)
LOAD_FROM_SEASON = FIRST_SEASON - WARMUP_SEASONS
# Basketball is played nearly every day. The board is today's and tomorrow's games.
BOARD_DAYS = _env_int("DEGEN_BOARD_DAYS", 1)

# The 2019-20 restart was played in a single Florida bubble with no crowds: every one of its 172
# games is a neutral-site game, whatever the schedule calls the home side. 2020-21 was played
# mostly in empty or capped arenas - home margin fell to +1.2 from a +2.2 to +3.3 norm - so it
# carries a flag rather than being dropped.
BUBBLE = (date(2020, 7, 1), date(2020, 10, 15))
NO_CROWD_SEASONS = {2020}

# --- modelling / betting --------------------------------------------------------------
# Games a team must have played THIS season before its picks are staked. Rosters carry over
# better than in college sports, but trades, rookies and new rotations take a couple of weeks
# to show up in anything a box score can measure.
MIN_GAMES = _env_int("DEGEN_MIN_GAMES", 5)

# Thresholds on the market-aware model's RAW disagreement with the line (|model - line|, in
# points), the same quantity `ats_by_disagreement` in models/meta.json is bucketed on - set them
# from that table, not from intuition. The walk-forward found NO bucket that clears break-even
# (its largest were 2-3 points on spreads and 4.5+ on totals, both below it), so these sit at or
# above all of them: almost nothing is flagged, and every game is still graded and CLV-tracked.
SPREAD_EDGE_MIN = _env_float("DEGEN_NBA_SPREAD_EDGE", 3.0)
TOTAL_EDGE_MIN = _env_float("DEGEN_NBA_TOTAL_EDGE", 5.0)
# The moneyline is priced, so its bar is expected value per unit at the posted price. It is the
# one bucket in the backtest that showed anything: at 10%+ EV - almost always an underdog the
# closing spread rated better than its moneyline did - the model's side returned about +16% on
# ~790 bets, positive in six of seven seasons, while the 5-10% bucket lost. That is 1.8 standard
# errors and does not survive correction for the looks taken; the bar sits at the bucket that
# showed something, eighth Kelly sizes it, and closing-line value will say whether it is real.
ML_EV_MIN = _env_float("DEGEN_NBA_ML_EV", 0.10)
BOLD_MULT = 2.0
# Eighth Kelly, matching the NFL, the NHL and the Premier League. Kelly assumes you know your
# edge; against the NBA close you do not, and overestimating it is the expensive mistake.
KELLY_FRACTION = _env_float("DEGEN_KELLY", 0.125)
BANKROLL_UNITS = 100.0

# --- injuries ---------------------------------------------------------------------------
# The chance each report status means the player sits. "Day-To-Day" is ESPN's catch-all for a
# player who has not been ruled out; the league's own report grades the same players questionable
# or probable closer to tip. Every pull is logged, so these can be checked against who played.
MISS_PROB = {"out": 1.0, "suspension": 1.0, "doubtful": 0.75, "questionable": 0.5,
             "day-to-day": 0.4, "probable": 0.1, "available": 0.0}
# A pick is staked only once no unresolved report entry could still move its team by this many
# points. Until then a pick that clears the bar is shown as waiting on injury news.
WAIT_POINTS = _env_float("DEGEN_NBA_WAIT_POINTS", 1.5)
REQUIRE_NEWS = _env("DEGEN_NBA_REQUIRE_NEWS", "1").lower() not in ("0", "false", "no", "off")

# --- venue / cost model ---------------------------------------------------------------
VENUE = os.environ.get("DEGEN_VENUE", "sportsbook")
BREAK_EVEN = _env_float("DEGEN_BREAK_EVEN", 0) or 52.38

# How far off the market the published number moves toward the model. Fitted walk-forward and
# clamped; the default is used only when too few closing lines exist to fit it. Expect the fit
# to land near zero - that is the market being right, not the fitter failing.
DEFAULT_SHRINK = _env_float("DEGEN_SHRINK", 0.10)
SHRINK_CAP = _env_float("DEGEN_SHRINK_CAP", 0.5)

SITE_TITLE = os.environ.get("DEGEN_SITE_TITLE", "DegenPredicts")
COFFEE_URL = os.environ.get("DEGEN_COFFEE_URL", "https://buymeacoffee.com/smcgee")
SUPPORT_URL = os.environ.get("DEGEN_SUPPORT_URL", COFFEE_URL)
SUPPORT_LABEL = os.environ.get("DEGEN_SUPPORT_LABEL", "Buy me a coffee")


def ensure_dirs() -> None:
    for p in (DATA, DOCS, MODEL_DIR, PLAYERS):
        p.mkdir(parents=True, exist_ok=True)
