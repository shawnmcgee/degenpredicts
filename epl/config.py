"""Configuration for the Premier League pipeline.

Deliberately parallel to ``nfl/config.py`` - same knob names, same env-var prefix - because the
pipelines are meant to stay readable side by side. What is genuinely different here is
different because association football is not gridiron, and each divergence is spelled out
below:

* **The calendar wraps a new year.** A season runs August to May and is named for the year it
  kicks off in, so ``season_of(date(2026, 2, 1))`` is 2025.
* **The unit is goals, not points**, and there are about 2.8 of them per match. Every threshold,
  cap and sigma downstream is on that scale; a 0.5-goal disagreement is a big claim.
* **Draws are a real outcome**, roughly a quarter of matches, so the market is three-way and the
  vig has to be divided three ways. See :mod:`epl.poisson` and :func:`epl.odds_math.devig_three`.
* **Three clubs are replaced every season.** Promotion and relegation means ~15% of the league
  each year has no top-flight history at all, which is a preseason problem the NFL never has.
* **Odds are decimal**, because that is what every source for this sport quotes.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
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
DATA = Path(os.environ.get("DEGEN_DATA", ROOT / "data")) / "epl"
_EPL_DOCS = os.environ.get("DEGEN_EPL_DOCS", "")
DOCS = Path(_EPL_DOCS) if _EPL_DOCS else \
    Path(os.environ.get("DEGEN_DOCS", ROOT / "docs")) / "epl"

GAMES = DATA / "games.csv"          # results, shots, cards - one row per match
LINES = DATA / "lines.csv"          # closing 1X2, Asian handicap and over/under 2.5
SNAPSHOTS = DATA / "snapshots.csv"  # our own live pulls, for closing-line value
STRENGTH = DATA / "team_strength.csv"   # opponent-adjusted prior-season attack/defence
FIXTURES = DATA / "fixtures.csv"    # forthcoming matches (football-data publishes these too)
LOWER_GAMES = DATA / "lower_games.csv"  # the division below, cached so retrains do not refetch it
PICKS = DATA / "picks.csv"
RESULTS = DATA / "results.csv"
METRICS = DATA / "metrics.json"
MODEL_DIR = DATA / "models"

# --- credentials --------------------------------------------------------------------
# football-data.co.uk needs none: it is static CSVs over plain HTTPS, the same property that
# made nflverse fit this no-server design. There is no key to rotate and no monthly quota.
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")   # optional: live prices for EV/Kelly
ODDS_BOOKS = ["Pinnacle", "Betfair", "William Hill", "Bet365", "Unibet", "1xBet"]

# Connect and read timeouts are separate, and both are short. This pipeline makes far more
# requests per run than the other two - a first backfill is ~50 files - so the per-request
# budget multiplies. A generous retry policy against a host that is simply not answering turned
# a 6-file schema check into a 25-minute job and would have made a first retrain a THREE-HOUR
# one: 5 attempts x 45s plus backoff is 4.1 minutes of dead time per file, spent 50 times over.
#
# The connect timeout is the one that matters. A host that refuses or resets answers instantly;
# a host whose packets are being dropped by a firewall answers never, and the connect timeout is
# the only thing that bounds it. 8 seconds is far more than a static file host needs to accept
# a TCP connection and far less than the 45 it was costing.
HTTP_CONNECT_TIMEOUT = float(os.environ.get("DEGEN_EPL_CONNECT_TIMEOUT", "8"))
# 15 seconds is already generous for a 30 KB static file. It is deliberately NOT the main
# defence, though - see PRIMARY_TIME_BUDGET below for why a socket timeout cannot be one.
HTTP_READ_TIMEOUT = float(os.environ.get("DEGEN_EPL_HTTP_TIMEOUT",
                                         os.environ.get("DEGEN_HTTP_TIMEOUT", "15")))
HTTP_TIMEOUT = (HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT)
HTTP_RETRIES = int(os.environ.get("DEGEN_EPL_HTTP_RETRIES",
                                  os.environ.get("DEGEN_HTTP_RETRIES", "1")))
# After this many consecutive failures the primary host is treated as down for the rest of the
# run and every later fetch goes straight to the mirror. Retrying a host that has already failed
# twice, once per file, for fifty files, is the difference between a slow run and a stuck one.
PRIMARY_FAILURE_LIMIT = int(os.environ.get("DEGEN_EPL_PRIMARY_FAILURES", "2"))
# ...and this many seconds of WASTED wall-clock against the primary host, whichever comes first.
#
# This is the bound that actually holds, and the failure count is the weaker of the two. Tuning
# retry counts and socket timeouts assumes you know HOW a host will fail, and you do not: this
# one turned out to accept the TCP connection and then stall, so the connect timeout never fired
# and the read timeout did - three attempts at 25s was 75 seconds per file, not the 24 the
# connect path would have cost. Worse, `read` in requests is a per-socket-read timeout rather
# than a deadline for the whole response, so a host trickling one byte at a time can exceed any
# value of it indefinitely and no retry setting will save you.
#
# Wall-clock is invariant to all of that. Thirty seconds of nothing from a static file host is
# all the evidence needed. Only time from FAILED requests counts, so a merely slow-but-working
# host is never abandoned.
PRIMARY_TIME_BUDGET = float(os.environ.get("DEGEN_EPL_PRIMARY_BUDGET", "30"))

# --- data source ---------------------------------------------------------------------
# football-data.co.uk publishes one CSV per league per season, in a stable layout, carrying
# results AND the bookmakers' closing prices. That second half is the whole reason it is the
# spine here rather than a scores feed: it is what lets the market-aware models train from day
# one instead of after a season of self-logging.
FOOTBALL_DATA = _env("DEGEN_FOOTBALL_DATA", "https://www.football-data.co.uk/mmz4281").rstrip("/")
# A results-only mirror on GitHub, used when the primary host is unreachable. It carries no
# odds columns, so a run served entirely from the mirror can still rate teams and predict but
# cannot train or score the market-aware models. Logged loudly when it happens.
FOOTBALL_DATA_MIRROR = _env(
    "DEGEN_FOOTBALL_DATA_MIRROR",
    "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets").rstrip("/")

# The league this pipeline predicts. football-data uses the same column layout for every
# division it publishes, so pointing this at E1 gives a Championship board from the same code.
# That is not a hypothetical: the honest expectation is that the Premier League close is too
# sharp to beat and that whatever edge exists in English football is a division or two down.
LEAGUE = _env("DEGEN_EPL_LEAGUE", "E0")
LEAGUE_NAMES = {"E0": "Premier League", "E1": "Championship", "E2": "League One",
                "E3": "League Two", "EC": "National League"}
LEAGUE_NAME = LEAGUE_NAMES.get(LEAGUE, LEAGUE)
# The division directly below, used to give a newly promoted club a prior instead of a blank.
LEAGUE_BELOW = {"E0": "E1", "E1": "E2", "E2": "E3", "E3": "EC"}.get(LEAGUE, "")
# Mirror path segments, for the fallback source only.
MIRROR_SLUGS = {"E0": "premier-league", "E1": "championship", "E2": "league-one",
                "E3": "league-two"}

# --- time / season -------------------------------------------------------------------
# Kickoffs are published in UK local time. The board is rendered in the same zone: this is an
# English league and its audience reads 3pm Saturday, not 10am Eastern.
UK = ZoneInfo("Europe/London")
ET = UK          # name kept so the shared site helpers read the same as the other sports


def now_uk() -> datetime:
    return datetime.now(UK)


def today_uk() -> date:
    return now_uk().date()


# Aliases, so code ported from the other pipelines keeps working unchanged.
now_et, today_et = now_uk, today_uk


def season_of(d) -> int:
    """Season = the calendar year it kicks off in. A season runs August to May, so anything
    from July onward belongs to the season starting that year and January-June belongs to the
    season that started the previous year. July is the boundary because it is the only month
    with no fixtures in it."""
    d = d if isinstance(d, date) else d.date()
    return d.year if d.month >= 7 else d.year - 1


def season_code(season: int) -> str:
    """football-data's four-digit season folder: 2024 -> "2425"."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def season_start(season: int) -> date:
    """Opening weekend is mid-August; July 1 is a safe lower bound."""
    return date(season, 7, 1)


def season_end(season: int) -> date:
    """The final round is in May; July 1 is a safe upper bound."""
    return date(season + 1, 7, 1)


MATCHDAYS = 38          # a 20-team round robin
TEAMS_IN_LEAGUE = 20
PROMOTED_PER_SEASON = 3

# Matches played behind closed doors. Unlike the NFL's single flagged season this is a DATE
# window, because the disruption started mid-season: the 2019-20 season was suspended in March
# 2020 and its last 92 matches were played empty from June 17, then essentially all of 2020-21
# was too. Flagging whole seasons would wrongly mark the 288 matches of 2019-20 that were
# played in front of full grounds. Home advantage collapsed inside this window - the league's
# home win rate fell from ~45% to ~36% - so it is both a feature and a suppression of the
# home term in the rating replay.
NO_CROWD_FROM = date(2020, 6, 1)
NO_CROWD_TO = date(2021, 5, 16)


def no_crowd(d) -> bool:
    if d is None:
        return False
    d = d if isinstance(d, date) else d.date()
    return NO_CROWD_FROM <= d <= NO_CROWD_TO


def matchweek(d, season: int | None = None) -> int:
    """Approximate matchweek from the date.

    football-data does not publish a round number, and unlike the NFL there is no clean one to
    publish: midweek rounds, cup replays and televised rearrangements mean two clubs can be
    three fixtures apart in the same calendar week. This is used for display and for bucketing
    only - never as a model feature. What the model sees is `games_played`, which is the honest
    version of the same question and is per club rather than per league.
    """
    d = d if isinstance(d, date) else d.date()
    season = season if season is not None else season_of(d)
    start = date(season, 8, 1)
    return max(1, min(MATCHDAYS + 4, (d - start).days // 7 + 1))


def current_week(games, today: date | None = None) -> tuple[int, int]:
    today = today or today_uk()
    season = season_of(today)
    return season, matchweek(today, season)


# football-data carries results from 1993-94, but the odds columns are what make this worth
# training on and they thin out fast going back: 1X2 prices start in 2000-01, Asian handicaps
# in 2006-07, and explicit CLOSING prices only in 2019-20. 2005 is where enough of the market
# picture exists to be worth the rows; see sources/footballdata.py for the era handling.
FIRST_SEASON = _env_int("DEGEN_FIRST_SEASON", 2005)
# Seasons loaded before the training window purely to warm the rating engine up. Two rather
# than the NFL's one because a football season is 38 matches of low-scoring evidence: a single
# warm-up year leaves attack and defence ratings still visibly shrunk toward the league mean
# when the first training season kicks off.
WARMUP_SEASONS = _env_int("DEGEN_WARMUP_SEASONS", 2)
LOAD_FROM_SEASON = FIRST_SEASON - WARMUP_SEASONS
# A league round is Saturday-Monday, with midweek rounds through the winter. 8 days covers a
# full round plus the midweek fixtures from any weekday.
BOARD_DAYS = _env_int("DEGEN_BOARD_DAYS", 8)

# --- modelling / betting --------------------------------------------------------------
GAMES_PER_SEASON = 380
# Matches before a club's rating is trusted. Higher than the NFL's 3 in absolute terms but
# lower as a share of the season (5/38 against 3/17), because promotion and relegation means
# three clubs every year have no top-flight rating at all and the prior-division prior carrying
# them is weaker than anything the NFL has to work with.
MIN_GAMES = _env_int("DEGEN_MIN_GAMES", 5)

# Thresholds are in GOALS, on the model's raw disagreement with the market's number - the same
# quantity ats_by_disagreement in models/meta.json is bucketed on, so set them from that table
# rather than from intuition. They are deliberately large: 0.6 of a goal of supremacy is an
# enormous disagreement with a Pinnacle closing handicap, and the point of starting here is
# that nothing gets staked until the backtest gives a reason to come down.
SUP_EDGE_MIN = _env_float("DEGEN_SUP_EDGE", 0.60)       # Asian handicap / supremacy
GOALS_EDGE_MIN = _env_float("DEGEN_GOALS_EDGE", 0.70)   # over/under total goals
# Kept under the old names too, so the shared reporting helpers read identically across sports.
SPREAD_EDGE_MIN = SUP_EDGE_MIN
TOTAL_EDGE_MIN = GOALS_EDGE_MIN
BOLD_MULT = 2.0
# Eighth Kelly, matching the NFL rather than college. Pinnacle's closing Asian handicap is
# generally reckoned the most efficient price in any sport; against it, an overestimated edge
# is the expensive direction to be wrong in.
KELLY_FRACTION = _env_float("DEGEN_KELLY", 0.125)

# --- venue / cost model ---------------------------------------------------------------
# Football is priced in decimal odds and the vig is quoted as an overround. A two-way market at
# 1.95/1.95 is a 2.6% overround (break-even 51.28%); a typical 1X2 market runs 4-6% across
# three outcomes. Exchange venues charge commission on winnings instead.
VENUE = os.environ.get("DEGEN_VENUE", "sportsbook")
FEE_COEF = {"sportsbook": None, "kalshi_taker": 0.07, "kalshi_maker": 0.0175,
            "exchange_zero": 0.0}
# The two-way price a "sportsbook" break-even assumes. 1.95 is the standard Asian handicap and
# over/under price at a sharp book - notably better than the -110 (1.909) the American markets
# quote, which is why this pipeline's break-even is lower than the NFL's.
SPORTSBOOK_DECIMAL = _env_float("DEGEN_SPORTSBOOK_DECIMAL", 1.95)


def break_even_pct(venue: str | None = None, price: float = 0.50) -> float:
    """Win rate needed to break even at `venue` on a contract priced at `price`."""
    venue = venue or VENUE
    coef = FEE_COEF.get(venue, None)
    if coef is None:                      # sportsbook at the configured decimal price
        return 100.0 / SPORTSBOOK_DECIMAL
    fee = coef * price * (1 - price)
    cost = price + fee
    return 100 * cost / ((1 - cost) + cost)


BREAK_EVEN = _env_float("DEGEN_BREAK_EVEN", 0) or break_even_pct()
BANKROLL_UNITS = 100.0
DEFAULT_SHRINK = _env_float("DEGEN_SHRINK", 0.25)

# --- the scoreline model ----------------------------------------------------------------
# Dixon-Coles low-score dependence. Independent Poisson marginals understate 0-0 and 1-1 and
# overstate 1-0 and 0-1, which matters because those four scorelines are about a fifth of all
# Premier League matches and they are exactly the ones that decide a draw. Negative rho lifts
# the drawn low scores. Dixon and Coles fitted -0.13 on English football of the early nineties;
# the modern, higher-scoring league fits nearer -0.04.
DC_RHO = _env_float("DEGEN_DC_RHO", -0.04)
# The scoreline grid is truncated here. 0-10 each side covers >99.99% of the distribution and
# the tail beyond it is renormalised back in.
MAX_GOALS = _env_int("DEGEN_MAX_GOALS", 10)

# --- Kalshi guards ---------------------------------------------------------------------
KALSHI_MIN_EV = _env_float("DEGEN_KALSHI_MIN_EV", 0.07)
KALSHI_PROB_MIN = _env_float("DEGEN_KALSHI_PROB_MIN", 0.15)
KALSHI_PROB_MAX = _env_float("DEGEN_KALSHI_PROB_MAX", 0.80)
KALSHI_MAX_BOOK_GAP = _env_float("DEGEN_KALSHI_MAX_GAP", 1.0)   # goals, not points

# Kalshi's football series tickers. NOT confirmed against the live API from the machine this
# was written on - the exchange was unreachable - so they are env-overridable and every Kalshi
# path degrades to "no exchange prices" rather than failing. Run
# `python -m epl.sources.kalshi --discover` once to confirm them.
KALSHI_SERIES_MONEYLINE = _env("DEGEN_KALSHI_EPL_SERIES", "KXEPLGAME")
KALSHI_SERIES_TOTAL = _env("DEGEN_KALSHI_EPL_TOTAL_SERIES", "KXEPLTOTAL")

SITE_TITLE = os.environ.get("DEGEN_SITE_TITLE", "DegenPredicts")
COFFEE_URL = os.environ.get("DEGEN_COFFEE_URL", "https://buymeacoffee.com/smcgee")
SUPPORT_URL = os.environ.get("DEGEN_SUPPORT_URL", COFFEE_URL)
SUPPORT_LABEL = os.environ.get("DEGEN_SUPPORT_LABEL", "Buy me a coffee")


def ensure_dirs() -> None:
    for p in (DATA, DOCS, MODEL_DIR):
        p.mkdir(parents=True, exist_ok=True)
