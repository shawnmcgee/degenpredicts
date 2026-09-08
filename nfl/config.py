"""Configuration for the NFL pipeline.

Deliberately parallel to ``cfb/config.py`` - same knob names, same env-var prefix - because the
two pipelines are meant to stay readable side by side. Four settings are genuinely different
and each one is different for a reason spelled out below: the season/week calendar, the size of
home-field advantage, the training window, and how much we are willing to bet.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(os.environ.get("DEGEN_ROOT", Path(__file__).resolve().parent.parent))
DATA = Path(os.environ.get("DEGEN_DATA", ROOT / "data")) / "nfl"
# The CFB site owns docs/index.html. NFL gets its own folder so both can publish from the
# same Pages branch without one overwriting the other.
_NFL_DOCS = os.environ.get("DEGEN_NFL_DOCS", "")
DOCS = Path(_NFL_DOCS) if _NFL_DOCS else \
    Path(os.environ.get("DEGEN_DOCS", ROOT / "docs")) / "nfl"

GAMES = DATA / "games.csv"          # nflverse schedule: results, closing lines, venue, QBs
LINES = DATA / "lines.csv"          # closing spread/total/moneyline, split out of games.csv
SNAPSHOTS = DATA / "snapshots.csv"  # our own live pulls, for closing-line value
EPA_PRIOR = DATA / "team_epa.csv"   # opponent-adjusted EPA/play by season - used from season-1
CONTINUITY = DATA / "continuity.csv"  # snap-weighted roster continuity by season
PICKS = DATA / "picks.csv"
RESULTS = DATA / "results.csv"
METRICS = DATA / "metrics.json"
MODEL_DIR = DATA / "models"

# --- credentials --------------------------------------------------------------------
# nflverse needs none. That is the whole reason it fits this no-server design: the data is
# static files on GitHub releases, so the pipeline has no key to rotate and no quota to blow.
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")   # optional: live prices for EV/Kelly
ODDS_BOOKS = ["Pinnacle", "DraftKings", "FanDuel", "BetMGM", "Caesars", "Bovada"]

NFLVERSE_SCHEDULE = os.environ.get(
    "DEGEN_NFLVERSE_SCHEDULE",
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv")
NFLVERSE_RELEASES = os.environ.get(
    "DEGEN_NFLVERSE_RELEASES",
    "https://github.com/nflverse/nflverse-data/releases/download").rstrip("/")

# --- time / season -------------------------------------------------------------------
ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(ET)


def today_et() -> date:
    return now_et().date()


def season_of(d) -> int:
    """NFL season = the calendar year it kicks off in. January and February playoff games
    belong to the previous season, and the Super Bowl is in February."""
    d = d if isinstance(d, date) else d.date()
    return d.year - 1 if d.month <= 2 else d.year


def season_start(season: int) -> date:
    """Week 1 Thursday is the Thursday after Labor Day; Sep 1 is a safe lower bound."""
    return date(season, 9, 1)


def season_end(season: int) -> date:
    """The Super Bowl is the second Sunday in February. March 1 is a safe upper bound."""
    return date(season + 1, 3, 1)


# The regular season went from 16 games (weeks 1-17) to 17 games (weeks 1-18) in 2021.
# That moved every playoff round's week number by one, so a raw `week` is NOT comparable
# across the change - week 18 is the Wild Card round in 2019 and a regular-season game in
# 2022. Anything the model sees has to be normalised; see features.week_features().
LAST_16_GAME_SEASON = 2020
REG_WEEKS_BEFORE = 17
REG_WEEKS_AFTER = 18


def reg_weeks(season: int) -> int:
    return REG_WEEKS_BEFORE if season <= LAST_16_GAME_SEASON else REG_WEEKS_AFTER


PLAYOFF_ROUNDS = {"WC": 1, "DIV": 2, "CON": 3, "SB": 4}


def playoff_round(game_type: str) -> int:
    """0 for a regular-season game, 1-4 for the four playoff rounds."""
    return PLAYOFF_ROUNDS.get(str(game_type).strip().upper(), 0)


def current_week(games, today: date | None = None) -> tuple[int, int]:
    """Infer (season, week) from the schedule: the week containing the next unplayed game."""
    today = today or today_et()
    season = season_of(today)
    s = games[games["season"] == season]
    if s.empty:
        return season, 1
    upcoming = s[s["date"] >= today]
    if upcoming.empty:
        return season, int(s["week"].max())
    return season, int(upcoming.sort_values("date")["week"].iloc[0])


# 2010+ ("modern era"). Two reasons this is where the line is drawn rather than 1999:
# nflverse's team-level EPA (stats_team_week) starts at 2010, and the pre-2010 game is
# different enough - lower scoring, different pass-interference and QB-contact rules - that
# those seasons would be teaching the model about a sport that no longer exists.
FIRST_SEASON = int(os.environ.get("DEGEN_FIRST_SEASON", "2010"))
# Snap counts (and therefore roster continuity) only exist from 2013, so continuity is NaN
# for games before 2014. That is fine - the trees handle a missing feature - but it is why
# you will see the column empty in the early seasons.
FIRST_SNAP_SEASON = 2013
# An NFL week is Thursday to Monday; 7 days ahead covers the full slate from any weekday.
BOARD_DAYS = int(os.environ.get("DEGEN_BOARD_DAYS", "7"))

# --- modelling / betting --------------------------------------------------------------
# 272 regular-season games a season against college football's ~800. Everything downstream
# that depends on sample size - shrink stability, segment minimums, how many seasons the
# walk-forward pools - has to be looser here, and the honest read is that a single NFL season
# tells you almost nothing.
GAMES_PER_SEASON = 272
MIN_GAMES = int(os.environ.get("DEGEN_MIN_GAMES", "2"))     # thin-data guard, weeks 1-2

# NFL closing lines are the sharpest market in sports. These thresholds are on the model's
# RAW disagreement with the line (|model - line|), the same quantity ats_by_disagreement in
# models/meta.json is bucketed on - so set them from that table, not from intuition. They
# start higher than the CFB defaults because a 3-point disagreement with a college number and
# a 3-point disagreement with an NFL number are not the same claim.
TOTAL_EDGE_MIN = float(os.environ.get("DEGEN_TOTAL_EDGE", "6.0"))
SPREAD_EDGE_MIN = float(os.environ.get("DEGEN_SPREAD_EDGE", "5.0"))
BOLD_MULT = 2.0
# Eighth Kelly, not the quarter used for college. Kelly sizing assumes you know your edge;
# against the NFL close you do not, and the cost of overestimating it compounds. Halving the
# fraction costs a little growth and buys a lot of survival.
KELLY_FRACTION = float(os.environ.get("DEGEN_KELLY", "0.125"))

# --- venue / cost model ---------------------------------------------------------------
# Same machinery as CFB: sportsbooks bake margin into the price (-110 => 52.38% break-even),
# exchanges charge an explicit fee instead. Fee schedules change - verify at
# kalshi.com/fee-schedule before sizing anything.
VENUE = os.environ.get("DEGEN_VENUE", "kalshi_taker")
FEE_COEF = {"sportsbook": None, "kalshi_taker": 0.07, "kalshi_maker": 0.0175,
            "exchange_zero": 0.0}


def break_even_pct(venue: str | None = None, price: float = 0.50) -> float:
    """Win rate needed to break even at `venue` on a contract priced at `price`."""
    venue = venue or VENUE
    coef = FEE_COEF.get(venue, 0.07)
    if coef is None:                      # sportsbook at -110
        return 52.38
    fee = coef * price * (1 - price)
    cost = price + fee
    return 100 * cost / ((1 - cost) + cost)


BREAK_EVEN = float(os.environ.get("DEGEN_BREAK_EVEN", "0")) or break_even_pct()
BANKROLL_UNITS = 100.0
# Expect the fitted shrink to come out low here - lower than college. That is the market
# telling you it is already right, not the fitter failing.
DEFAULT_SHRINK = float(os.environ.get("DEGEN_SHRINK", "0.25"))

# --- Kalshi guards ---------------------------------------------------------------------
# Same winner's-curse problem as CFB, but the NFL board is far more liquid and far more
# efficiently priced, so the EV bar is higher: a 5c edge against an NFL contract is much more
# likely to be model error than a real mispricing.
KALSHI_MIN_EV = float(os.environ.get("DEGEN_KALSHI_MIN_EV", "0.07"))
KALSHI_PROB_MIN = float(os.environ.get("DEGEN_KALSHI_PROB_MIN", "0.25"))
KALSHI_PROB_MAX = float(os.environ.get("DEGEN_KALSHI_PROB_MAX", "0.75"))
KALSHI_MAX_BOOK_GAP = float(os.environ.get("DEGEN_KALSHI_MAX_GAP", "6.0"))

# Kalshi's NFL series tickers. These follow the same KX<LEAGUE><MARKET> pattern as the
# college series, which were confirmed against live payloads. The NFL ones are NOT confirmed
# here - the exchange was unreachable from the machine this was written on - so they are env
# overridable and every Kalshi path degrades to "no exchange prices" rather than failing.
# Run `python -m nfl.sources.kalshi --discover` once to confirm them against the live API.
KALSHI_SERIES_MONEYLINE = os.environ.get("DEGEN_KALSHI_ML_SERIES", "KXNFLGAME")
KALSHI_SERIES_SPREAD = os.environ.get("DEGEN_KALSHI_SPREAD_SERIES", "KXNFLSPREAD")
KALSHI_SERIES_TOTAL = os.environ.get("DEGEN_KALSHI_TOTAL_SERIES", "KXNFLTOTAL")

SITE_TITLE = os.environ.get("DEGEN_SITE_TITLE", "DegenPredicts")
COFFEE_URL = os.environ.get("DEGEN_COFFEE_URL", "https://buymeacoffee.com/smcgee")
SUPPORT_URL = os.environ.get("DEGEN_SUPPORT_URL", COFFEE_URL)
SUPPORT_LABEL = os.environ.get("DEGEN_SUPPORT_LABEL", "Buy me a coffee")


def ensure_dirs() -> None:
    for p in (DATA, DOCS, MODEL_DIR):
        p.mkdir(parents=True, exist_ok=True)
