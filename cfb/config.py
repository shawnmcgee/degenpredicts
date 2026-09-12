"""Configuration for the college-football pipeline."""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(os.environ.get("DEGEN_ROOT", Path(__file__).resolve().parent.parent))
DATA = Path(os.environ.get("DEGEN_DATA", ROOT / "data")) / "cfb"
# GitHub Pages serves the docs folder. Its ROOT is a sport chooser rendered by
# core.landing; each sport publishes into its own subfolder, so neither can overwrite the
# other and adding a sport needs no coordination.
_CFB_DOCS = os.environ.get("DEGEN_CFB_DOCS", "")
DOCS = Path(_CFB_DOCS) if _CFB_DOCS else \
    Path(os.environ.get("DEGEN_DOCS", ROOT / "docs")) / "cfb"

GAMES = DATA / "games.csv"            # every FBS game, final or scheduled
LINES = DATA / "lines.csv"            # CFBD historical + current lines (open and current)
SNAPSHOTS = DATA / "snapshots.csv"    # our own live pulls, for closing-line value
RETURNING = DATA / "returning.csv"    # returning production by team/season (preseason-known)
SP_PRIOR = DATA / "sp_ratings.csv"    # SP+ by season - only ever used from season-1
PICKS = DATA / "picks.csv"
RESULTS = DATA / "results.csv"
METRICS = DATA / "metrics.json"
MODEL_DIR = DATA / "models"

# --- credentials -------------------------------------------------------------------
CFBD_KEY = os.environ.get("CFBD_API_KEY", "")
CFBD_BASE = os.environ.get("CFBD_BASE", "https://api.collegefootballdata.com").rstrip("/")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")   # optional: live prices for EV/Kelly

# CFBD line providers, best first. CFBD returns several per game.
PREFERRED_PROVIDERS = ["Bovada", "DraftKings", "ESPN Bet", "consensus", "teamrankings",
                       "numberfire", "William Hill (New Jersey)"]
ODDS_BOOKS = ["Pinnacle", "DraftKings", "FanDuel", "BetMGM", "Caesars", "Bovada"]

# --- time / season -------------------------------------------------------------------
ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(ET)


def today_et() -> date:
    return now_et().date()


def season_of(d) -> int:
    """CFB season = calendar year. A January bowl/playoff game belongs to the prior season."""
    d = d if isinstance(d, date) else d.date()
    return d.year - 1 if d.month <= 2 else d.year


def season_start(season: int) -> date:
    """Week 0 Saturday is the last Saturday of August; Aug 20 is a safe lower bound."""
    return date(season, 8, 20)


def season_end(season: int) -> date:
    return date(season + 1, 2, 1)


def current_week(games, today: date | None = None) -> tuple[int, int]:
    """Infer (season, week) from the schedule: the week containing the next unplayed game."""
    today = today or today_et()
    season = season_of(today)
    s = games[(games["season"] == season)]
    if s.empty:
        return season, 1
    upcoming = s[s["date"] >= today]
    if upcoming.empty:
        return season, int(s["week"].max())
    return season, int(upcoming.sort_values("date")["week"].iloc[0])


# Seasons played to empty or near-empty stadiums. Non-neutral FBS home margin was +2.13 in
# 2020 against +4.62 the year before and +3.68 the year after, so crediting those 499 games
# the usual 2.5-point home edge pushes every 2020 home team's rating down by an advantage
# that was not there. `nfl` and `epl` both already suppress it; this is the same rule.
NO_CROWD_SEASONS = {int(s) for s in os.environ.get("DEGEN_NO_CROWD", "2020").split(",")
                    if s.strip()}

FIRST_SEASON = int(os.environ.get("DEGEN_FIRST_SEASON", "2015"))
# Games this many days ahead go on the board. CFB weeks run Thu-Mon.
BOARD_DAYS = int(os.environ.get("DEGEN_BOARD_DAYS", "7"))

# Saturday kickoff slates, as ET hours. A CFB Saturday runs ~50 games and reading them as one
# list is the problem these solve. Boundaries are taken from where the schedule actually
# clusters rather than from round numbers: a typical board sits at 12:00, 15:30-16:15,
# 19:00-20:00 and 22:30+, which is the familiar noon / 3:30-4 / primetime / west-coast-late
# shape. 6pm counts as night, not afternoon - it is an evening kickoff by any normal reading,
# and the afternoon window is the 3:30-4:00 block.
_SLATE_LABELS = (("early", "Noon"), ("afternoon", "Afternoon"),
                 ("night", "Night"), ("late", "Late"))
_SLATE_DEFAULT = (15.0, 18.0, 22.5)


def slates() -> tuple[tuple[str, str, float, float], ...]:
    """(key, label, start_hour, end_hour) per slate, start-inclusive and end-exclusive.

    Three cut points make four buckets. `DEGEN_CFB_SLATES` overrides them as a comma-separated
    list of ET hours ("15,18,22.5"); anything unparseable falls back to the defaults rather
    than crashing the daily site build over a typo in an env var.
    """
    cuts = _SLATE_DEFAULT
    raw = os.environ.get("DEGEN_CFB_SLATES", "").strip()
    if raw:
        try:
            parsed = tuple(sorted(float(x) for x in raw.split(",")))
            if len(parsed) == 3 and all(0 < c < 24 for c in parsed):
                cuts = parsed
        except ValueError:
            pass
    edges = (0.0,) + cuts + (24.0,)
    return tuple((key, label, edges[i], edges[i + 1])
                 for i, (key, label) in enumerate(_SLATE_LABELS))


# --- modelling / betting -------------------------------------------------------------
MIN_GAMES = int(os.environ.get("DEGEN_MIN_GAMES", "2"))     # thin-data guard (weeks 1-2)
# NOTE: these thresholds apply to the model's RAW disagreement with the line
# (|model - line|), NOT to the shrunk display edge. Set them from the
# ats_by_disagreement table in models/meta.json - pick the smallest bucket whose
# cover_pct clears 52.4 by more than ~2 stderr on a decent sample.
TOTAL_EDGE_MIN = float(os.environ.get("DEGEN_TOTAL_EDGE", "5.0"))
SPREAD_EDGE_MIN = float(os.environ.get("DEGEN_SPREAD_EDGE", "4.0"))
BOLD_MULT = 2.0
KELLY_FRACTION = float(os.environ.get("DEGEN_KELLY", "0.25"))

# --- venue / cost model --------------------------------------------------------------
# Sportsbooks bake their margin into the line (-110 => 52.38% break-even). Prediction-market
# exchanges (Kalshi, Polymarket, DK/FanDuel Predictions) instead sell a 0/100 contract at its
# implied probability and charge an explicit fee, which is materially cheaper. Kalshi's
# published taker formula is roundup(0.07 * contracts * P * (1-P)); several sports series carry
# maker fees at a quarter of that when a resting order fills.
#
# Fee schedules change - verify at kalshi.com/fee-schedule before sizing anything.
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
    # A 0/100 contract costs `cost` and returns 1.00, so you break even at a win rate of
    # exactly `cost`. The old expression divided by ((1 - cost) + cost), which is 1 - the
    # right answer, but written as though it were doing something.
    return 100 * cost


BREAK_EVEN = float(os.environ.get("DEGEN_BREAK_EVEN", "0")) or break_even_pct()
BANKROLL_UNITS = 100.0
DEFAULT_SHRINK = float(os.environ.get("DEGEN_SHRINK", "0.4"))

# --- Kalshi ladder guards -------------------------------------------------------------
# Picking the best-EV rung out of ~20 per game is a maximum over noisy estimates: it returns a
# positive number almost every time even when the model has no edge (winner's curse). These
# guards exist to stop that, and they are deliberately strict.
KALSHI_MIN_EV = float(os.environ.get("DEGEN_KALSHI_MIN_EV", "0.05"))   # 5c after fees
# Only price rungs where the model is not extrapolating into the tail. A normal approximation
# is poor out there - football margins cluster on 3/7/10/14 and have thinner tails than a
# Gaussian - so tail probabilities are overstated and cheap longshots look mispriced.
KALSHI_PROB_MIN = float(os.environ.get("DEGEN_KALSHI_PROB_MIN", "0.20"))
KALSHI_PROB_MAX = float(os.environ.get("DEGEN_KALSHI_PROB_MAX", "0.80"))
# Ignore rungs far from the sportsbook number; those are the tails by another name.
KALSHI_MAX_BOOK_GAP = float(os.environ.get("DEGEN_KALSHI_MAX_GAP", "7.0"))

SITE_TITLE = os.environ.get("DEGEN_SITE_TITLE", "DegenPredicts")
COFFEE_URL = os.environ.get("DEGEN_COFFEE_URL", "https://buymeacoffee.com/smcgee")
# Optional support link shown at the top of the site. Set the repo variable DEGEN_SUPPORT_URL
# to your Buy Me a Coffee page; leave unset and the button simply doesn't render.
SUPPORT_URL = os.environ.get("DEGEN_SUPPORT_URL", COFFEE_URL)
SUPPORT_LABEL = os.environ.get("DEGEN_SUPPORT_LABEL", "Buy me a coffee")
# Visual theme: "ticker" (light, dense, tabular), "scoreboard" (dark, amber, oversized
# numerals), "field" (light, chalk green, position strips).
THEME = os.environ.get("DEGEN_THEME", "ticker")
# Optional support link, e.g. https://buymeacoffee.com/yourname . Blank hides it.


def ensure_dirs() -> None:
    for p in (DATA, DOCS, MODEL_DIR):
        p.mkdir(parents=True, exist_ok=True)
