"""Feature construction for the NFL.

Built by replaying the schedule in order, so a feature attached to game N only ever saw games
1..N-1. That guarantee is what the whole project rests on and it is verified in CI by
rebuilding from a truncated schedule and asserting identical output.

Three groups of features, in order of how much work they do:

**The league-agnostic core** - ratings, form, rest - is the same shape as the college model.

**The preseason carry.** In weeks 1-4 there is barely any in-season data, and unlike college
football there is no SP+ to lean on. Two things fill the gap, both legitimately known before
week 1 and both joined only from season-1:

* ``prev_off_epa`` / ``prev_def_epa`` - opponent-adjusted EPA per play. Note that **defence is
  signed so that lower is better**: it is EPA allowed.
* ``cont_off`` / ``cont_def`` - snap-weighted roster continuity, the share of last season's
  snaps still on the roster.

**The NFL-specific ones**, which is where this genuinely diverges from the college model. These
are the things that actually move an NFL line and have no college analogue worth modelling:

* **Quarterback.** One position is worth several points of spread. ``qb_new`` flags a starter
  who did not start the team's last game; ``qb_starts`` counts their prior starts. Both are
  computed from the schedule itself, chronologically, so they cannot leak - and nflverse
  publishes the projected starters for unplayed games, so they exist on the board too.
* **Rest.** Short weeks (Thursday games) and byes are scheduled, extreme and known months
  ahead. ``rest_diff`` is the one that matters; the raw values are there for interactions.
* **Travel.** Distance and time-zone crossings, computed from the stadium each team actually
  played its home games in that season - so relocations and London games need no special case.
  A west-coast team at a 1pm Eastern kickoff is the classic version of this.
* **Venue and weather.** Dome vs outdoors, temperature and wind. Wind is the single largest
  weather effect on totals; temperature mostly proxies for late-season northern games.
* **Divisional.** Division rivals play twice a year, know each other, and historically produce
  closer games than the ratings imply.

**Week numbering is normalised, not used raw.** The regular season went from 17 weeks to 18 in
2021, which shifted every playoff round's week number by one - week 18 is the Wild Card round
in 2019 and a regular-season game in 2022. Feeding the raw number to the model would teach it
something false about the calendar. ``week_frac`` and ``playoff_round`` are era-comparable.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .ratings import LEAGUE_PPG, RatingBook, RatingConfig
from .teams import haversine_km, tz_shift, venue

log = logging.getLogger(__name__)

BASE_FEATURES = [
    # calendar, era-normalised
    "neutral_site", "div_game", "week_frac", "playoff_round", "is_early_season", "no_crowd",
    # ratings
    "exp_margin", "exp_total", "exp_home_points", "exp_away_points",
    "h_margin", "a_margin", "h_off", "a_off", "h_def", "a_def",
    "h_games", "a_games",
    # form
    "h_form_margin", "a_form_margin", "h_form_total", "a_form_total",
    # rest
    "h_rest", "a_rest", "rest_diff", "h_bye", "a_bye", "h_short_week", "a_short_week",
    # preseason carry: prior-season EPA (def is EPA allowed - lower is better)
    "h_prev_off_epa", "a_prev_off_epa", "h_prev_def_epa", "a_prev_def_epa",
    "epa_margin_est", "epa_total_est",
    # preseason carry: roster continuity
    "h_cont_off", "a_cont_off", "h_cont_def", "a_cont_def",
    # quarterback
    "h_qb_new", "a_qb_new", "h_qb_starts", "a_qb_starts",
    # travel and body clock
    "h_travel_km", "a_travel_km", "travel_diff_km", "h_tz_shift", "a_tz_shift",
    # venue and weather
    "is_indoor", "is_grass", "temp", "wind",
]

# nflverse publishes the CLOSING number only - there is no opening line in the feed - so the
# line-movement features the college model uses have no historical counterpart here. Rather
# than train on a column that is zero for every historical row and non-zero live (a
# train/serve mismatch that would quietly poison the market models), they are simply absent.
# The moneyline is a genuine extra: it is a second, independent market view of the same game.
MARKET_FEATURES = ["total_line", "spread_home", "total_vs_model", "spread_vs_model",
                   "ml_prob_home"]

REST_CAP = 21
SHORT_WEEK = 5          # Thursday off a Sunday game
BYE_REST = 12
EARLY_WEEKS = 4


def _text(v) -> str:
    """Text field from a CSV round-trip. An empty cell comes back as float NaN, and NaN has
    no .strip(), so every text feature has to go through this."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v).strip()


def _rest(last, gdate):
    if last is None or gdate is None:
        return np.nan
    return float(min((gdate - last).days, REST_CAP))


def hfa_suppressed(season: int, neutral: bool) -> bool:
    """True when home-field advantage should be treated as absent for this game.

    Two separate causes, same consequence: an actual neutral-site game, and a game played
    without a crowd. They are kept apart as *features* - `neutral_site` still means the venue
    and `no_crowd` still means the season - but both mean the rating engine must not credit a
    home edge. Applying 1.7 points of home advantage to all 269 games of 2020, when the
    league's actual home margin that year was +0.14, would push every home team's rating down
    by an edge that was not there.
    """
    return bool(neutral) or int(season) in config.NO_CROWD_SEASONS


def week_features(season: int, week: int, playoff_round: int) -> tuple[float, int]:
    """(week_frac, is_early_season), comparable across the 17->18 week change in 2021.

    ``week_frac`` runs 0..1 across the regular season and continues past 1 into the playoffs,
    so "late season" means the same thing in 2015 and 2025.
    """
    reg = config.reg_weeks(season)
    if playoff_round:
        return float(1.0 + playoff_round / 4.0), 0
    return float(week) / reg, int(week <= EARLY_WEEKS)


def _american_prob(price) -> float:
    if price is None or price != price:
        return np.nan
    p = float(price)
    return 100 / (p + 100) if p > 0 else -p / (-p + 100)


def _devig_home(home_ml, away_ml) -> float:
    """Market P(home wins) with the vig divided out proportionally."""
    h, a = _american_prob(home_ml), _american_prob(away_ml)
    if h != h or a != a or (h + a) <= 0:
        return np.nan
    return h / (h + a)


class QBTracker:
    """Who started each team's last game, and how many starts they have.

    Replayed in schedule order alongside the ratings, so it is leak-free by the same argument:
    the state used for game N is the state after game N-1. nflverse fills the starter in for
    unplayed games too, which is what makes this usable on the board rather than only in
    training.
    """

    def __init__(self):
        self.last: dict[str, str] = {}          # team -> QB who started their last game
        self.starts: dict[tuple[str, str], int] = {}   # (team, QB) -> starts so far

    def look(self, team: str, qb: str) -> tuple[float, float]:
        if not qb:
            return np.nan, np.nan
        prev = self.last.get(team)
        # No prior game on record (week 1, or the first season in the window) is *unknown*,
        # not "new starter" - filling it with 1 would flag all 32 week-1 starters as changes.
        is_new = np.nan if prev is None else float(qb != prev)
        return is_new, float(self.starts.get((team, qb), 0))

    def update(self, team: str, qb: str) -> None:
        if not qb:
            return
        self.starts[(team, qb)] = self.starts.get((team, qb), 0) + 1
        self.last[team] = qb


def _travel(homes: dict, season: int, home: str, away: str, stadium_id: str,
            neutral: bool) -> dict:
    """Distance each side travelled and how many time zones they crossed.

    The venue is the game's stadium; each side's origin is the stadium it plays its home games
    in that season. For a normal game the home side travels zero. For a neutral-site game -
    London, Munich, Mexico City - both sides travel, which is the situation that most needs
    this feature and the one a naive "away team travels" rule gets wrong.
    """
    from .sources.nflverse import origin_stadium

    v = venue(stadium_id)
    out = {"h_travel_km": np.nan, "a_travel_km": np.nan, "travel_diff_km": np.nan,
           "h_tz_shift": np.nan, "a_tz_shift": np.nan}
    if v is None:
        if stadium_id:
            log.debug("unmapped stadium %s - travel features empty for this game", stadium_id)
        return out
    v_ll, v_tz = (v[0], v[1]), v[2]
    for side, team in (("h", home), ("a", away)):
        origin = origin_stadium(homes, season, team)
        o = venue(origin)
        if o is None:
            continue
        if side == "h" and not neutral:
            out["h_travel_km"], out["h_tz_shift"] = 0.0, 0.0
            continue
        out[f"{side}_travel_km"] = round(haversine_km((o[0], o[1]), v_ll), 1)
        shift = tz_shift(o[2], v_tz)
        if shift == shift:
            out[f"{side}_tz_shift"] = round(shift, 1)
    if out["h_travel_km"] == out["h_travel_km"] and out["a_travel_km"] == out["a_travel_km"]:
        out["travel_diff_km"] = round(out["a_travel_km"] - out["h_travel_km"], 1)
    return out


def is_primetime(weekday, kickoff_utc) -> int:
    """A standalone nationally televised game: Thursday, Monday, or a Sunday night kickoff.

    Not a model feature - a segmentation axis for market_softness. The Sunday 1pm window puts
    nine games on at once and splits the betting public's attention nine ways; Sunday Night
    Football is one game with the whole market watching it. If public-data edge exists anywhere
    in the NFL it is far likelier to be in the 1pm window than in the primetime window, and
    this is how the training report checks that rather than assuming it.
    """
    day = _text(weekday).lower()
    if day in ("thursday", "monday", "friday", "saturday"):
        return 1
    iso = _text(kickoff_utc)
    if day == "sunday" and iso:
        try:
            return int(pd.Timestamp(iso).hour >= 19)
        except (ValueError, TypeError):
            return 0
    return 0


INDOOR_ROOFS = {"dome", "closed"}


def _venue_weather(roof: str, surface: str, temp, wind) -> dict:
    """Indoor/outdoor, surface, and weather - with the dome convention made explicit.

    nflverse leaves temperature and wind blank for indoor games rather than recording the
    controlled conditions. Left as NaN the trees would have to learn "missing means dome",
    which they can, but stating it directly is both more honest and one fewer thing to get
    wrong: indoors is 68F and no wind.
    """
    r = _text(roof).lower()
    indoor = r in INDOOR_ROOFS
    t, w = float(temp) if temp == temp else np.nan, float(wind) if wind == wind else np.nan
    if indoor:
        t = 68.0 if t != t else t
        w = 0.0 if w != w else w
    return {"is_indoor": int(indoor),
            "is_grass": int(_text(surface).lower().startswith("grass")),
            "temp": t, "wind": w}


def _row(book, qbs, homes, epa_prev, cont, season, week, playoff_round, home, away, gdate,
         neutral, div_game, rec) -> dict:
    h, a = book.get(season, home), book.get(season, away)
    no_hfa = hfa_suppressed(season, neutral)
    e = book.expect(season, home, away, no_hfa)
    he, ae = epa_prev.get((season - 1, home), {}), epa_prev.get((season - 1, away), {})
    hc, ac = cont.get((season, home), {}), cont.get((season, away), {})

    h_off_epa, a_off_epa = he.get("off", np.nan), ae.get("off", np.nan)
    h_def_epa, a_def_epa = he.get("def", np.nan), ae.get("def", np.nan)
    # A crude points translation of the prior-season ratings, so the model gets the
    # interaction pre-computed rather than having to discover it from four columns.
    # ~62 plays a side per game; EPA is already in points, so plays * net EPA is points.
    have_epa = not any(x != x for x in (h_off_epa, a_off_epa, h_def_epa, a_def_epa))
    plays = 62.0
    epa_margin = (plays * ((h_off_epa - a_def_epa) - (a_off_epa - h_def_epa))
                  + (0 if no_hfa else book.cfg.hfa)) if have_epa else np.nan
    epa_total = (2 * LEAGUE_PPG + plays * ((h_off_epa + a_def_epa) + (a_off_epa + h_def_epa))) \
        if have_epa else np.nan

    # Rest comes from nflverse where it exists (it is authoritative and handles the bye
    # correctly); the replayed date gap is the fallback for a board game with none posted.
    h_rest = rec.get("home_rest", np.nan)
    a_rest = rec.get("away_rest", np.nan)
    h_rest = float(min(h_rest, REST_CAP)) if h_rest == h_rest else _rest(h.last_date, gdate)
    a_rest = float(min(a_rest, REST_CAP)) if a_rest == a_rest else _rest(a.last_date, gdate)

    h_qb_new, h_qb_starts = qbs.look(home, _text(rec.get("home_qb")))
    a_qb_new, a_qb_starts = qbs.look(away, _text(rec.get("away_qb")))
    week_frac, early = week_features(season, week, playoff_round)

    row = {
        "neutral_site": int(bool(neutral)), "div_game": int(bool(div_game)),
        "week_frac": week_frac, "playoff_round": int(playoff_round),
        "is_early_season": early,
        "no_crowd": int(int(season) in config.NO_CROWD_SEASONS),
        "exp_margin": e["exp_margin"], "exp_total": e["exp_total"],
        "exp_home_points": e["exp_home_points"], "exp_away_points": e["exp_away_points"],
        "h_margin": h.margin, "a_margin": a.margin,
        "h_off": h.off, "a_off": a.off, "h_def": h.deff, "a_def": a.deff,
        "h_games": h.games, "a_games": a.games,
        "h_form_margin": float(np.mean(h.recent_margin)) if h.recent_margin else 0.0,
        "a_form_margin": float(np.mean(a.recent_margin)) if a.recent_margin else 0.0,
        "h_form_total": float(np.mean(h.recent_total)) if h.recent_total else 0.0,
        "a_form_total": float(np.mean(a.recent_total)) if a.recent_total else 0.0,
        "h_rest": h_rest, "a_rest": a_rest,
        "rest_diff": (h_rest - a_rest) if h_rest == h_rest and a_rest == a_rest else np.nan,
        "h_bye": int(h_rest >= BYE_REST) if h_rest == h_rest else 0,
        "a_bye": int(a_rest >= BYE_REST) if a_rest == a_rest else 0,
        "h_short_week": int(h_rest <= SHORT_WEEK) if h_rest == h_rest else 0,
        "a_short_week": int(a_rest <= SHORT_WEEK) if a_rest == a_rest else 0,
        "h_prev_off_epa": h_off_epa, "a_prev_off_epa": a_off_epa,
        "h_prev_def_epa": h_def_epa, "a_prev_def_epa": a_def_epa,
        "epa_margin_est": epa_margin, "epa_total_est": epa_total,
        "h_cont_off": hc.get("off", np.nan), "a_cont_off": ac.get("off", np.nan),
        "h_cont_def": hc.get("def", np.nan), "a_cont_def": ac.get("def", np.nan),
        "h_qb_new": h_qb_new, "a_qb_new": a_qb_new,
        "h_qb_starts": h_qb_starts, "a_qb_starts": a_qb_starts,
    }
    row.update(_travel(homes, season, home, away, _text(rec.get("stadium_id")), neutral))
    row.update(_venue_weather(rec.get("roof", ""), rec.get("surface", ""),
                              rec.get("temp", np.nan), rec.get("wind", np.nan)))
    return row


def _market(row: dict, total_line, spread_home, home_ml=np.nan, away_ml=np.nan) -> dict:
    tl = float(total_line) if total_line == total_line and total_line is not None else np.nan
    sh = float(spread_home) if spread_home == spread_home and spread_home is not None else np.nan
    row["total_line"] = tl
    row["spread_home"] = sh
    row["total_vs_model"] = row["exp_total"] - tl
    # spread_home is the CFBD convention: -6.5 means home favoured by 6.5, so the market's
    # expected home margin is -spread_home. The sign flip out of nflverse happens once, in
    # sources/nflverse._to_spread_home.
    row["spread_vs_model"] = row["exp_margin"] - (-sh)
    row["ml_prob_home"] = _devig_home(home_ml, away_ml)
    return row


def build(games: pd.DataFrame, upcoming: pd.DataFrame | None = None,
          lines: pd.DataFrame | None = None, epa: pd.DataFrame | None = None,
          continuity: pd.DataFrame | None = None, cfg: RatingConfig | None = None,
          first_train_season: int | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame, RatingBook]:
    """Replay the schedule and emit one feature row per completed game.

    Seasons before `first_train_season` are **warm-up**: they advance the ratings and the
    quarterback tracker but emit no training rows. That is what gives week 1 of the first
    training season a real regressed prior instead of a league of identically-rated teams.
    """
    first_train = config.FIRST_SEASON if first_train_season is None else first_train_season
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"]).dt.date
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)

    epa_prev, cont = {}, {}
    if epa is not None and len(epa):
        epa_prev = {(int(r.season), r.team): {"off": r.off_epa_adj, "def": r.def_epa_adj}
                    for r in epa.itertuples()}
    if continuity is not None and len(continuity):
        cont = {(int(r.season), r.team): {"off": r.cont_off, "def": r.cont_def}
                for r in continuity.itertuples()}

    line_map = {}
    if lines is not None and len(lines):
        for r in lines.itertuples():
            line_map[str(r.game_id)] = (r.spread_home, r.total_line,
                                        getattr(r, "home_ml", np.nan),
                                        getattr(r, "away_ml", np.nan))

    homes = {}
    try:
        from .sources.nflverse import home_stadiums
        homes = home_stadiums(games)
    except Exception as e:                      # travel is a nice-to-have, not a dependency
        log.warning("could not derive home stadiums (%s) - travel features will be empty", e)

    book, qbs = RatingBook(cfg), QBTracker()
    rows = []
    for g in games.itertuples(index=False):
        if not bool(getattr(g, "completed", False)):
            continue  # scheduled-but-unplayed rows live in `upcoming`, not training
        season, week = int(g.season), int(g.week)
        neutral = bool(getattr(g, "neutral_site", False))
        rec = {k: getattr(g, k, np.nan) for k in
               ("home_rest", "away_rest", "roof", "surface", "temp", "wind", "stadium_id",
                "home_qb", "away_qb")}
        if season < first_train:
            # Warm-up: advance the state, emit nothing. Skipping the row build is also the
            # expensive part, so warm-up seasons are close to free.
            book.update(season, g.home_team, g.away_team, g.home_points, g.away_points,
                        g.date, hfa_suppressed(season, neutral))
            qbs.update(g.home_team, _text(rec.get("home_qb")))
            qbs.update(g.away_team, _text(rec.get("away_qb")))
            continue
        r = _row(book, qbs, homes, epa_prev, cont, season, week,
                 int(getattr(g, "playoff_round", 0)), g.home_team, g.away_team, g.date,
                 neutral, bool(getattr(g, "div_game", False)), rec)
        sh, tl, hml, aml = line_map.get(str(g.game_id), (np.nan, np.nan, np.nan, np.nan))
        r = _market(r, tl, sh, hml, aml)
        r.update({"game_id": g.game_id, "date": g.date, "season": season, "week": week,
                  "home_team": g.home_team, "away_team": g.away_team,
                  "total_points": g.total_points, "home_margin": g.home_margin,
                  "home_div": getattr(g, "home_div", ""), "away_div": getattr(g, "away_div", ""),
                  "season_type": getattr(g, "season_type", "REG"),
                  "weekday": _text(getattr(g, "weekday", "")),
                  "is_primetime": is_primetime(getattr(g, "weekday", ""),
                                               getattr(g, "kickoff_utc", "")),
                  "roof": _text(rec.get("roof")),
                  "stadium_id": _text(rec.get("stadium_id"))})
        rows.append(r)
        # State advances only AFTER the row is recorded. This ordering is the leak-free
        # guarantee, and test_leak_free rebuilds from a truncated schedule to prove it.
        book.update(season, g.home_team, g.away_team, g.home_points, g.away_points, g.date,
                    hfa_suppressed(season, neutral))
        qbs.update(g.home_team, _text(rec.get("home_qb")))
        qbs.update(g.away_team, _text(rec.get("away_qb")))

    train_rows = pd.DataFrame(rows)

    up_rows = []
    if upcoming is not None and len(upcoming):
        for rec in upcoming.to_dict("records"):
            gdate = pd.Timestamp(rec["date"]).date()
            season = int(rec.get("season") or config.season_of(gdate))
            r = _row(book, qbs, homes, epa_prev, cont, season, int(rec.get("week", 1)),
                     int(rec.get("playoff_round", 0) or 0), rec["home_team"], rec["away_team"],
                     gdate, bool(rec.get("neutral_site", False)),
                     bool(rec.get("div_game", False)), rec)
            r = _market(r, rec.get("total_line", np.nan), rec.get("spread_home", np.nan),
                        rec.get("home_ml", np.nan), rec.get("away_ml", np.nan))
            r.update(rec)
            up_rows.append(r)
    return train_rows, pd.DataFrame(up_rows), book


def nfl_teams(games: pd.DataFrame, season: int) -> set[str]:
    """The name list the odds and exchange matchers resolve feed spellings against.

    Returns an empty set rather than raising on a frame with no schedule in it. The callers
    wrap this in a try/except that logs "exchange unavailable", so a KeyError here used to
    look like a third-party outage and silently cost every exchange column.
    """
    if games is None or not len(games) or "season" not in games.columns:
        return set()
    m = games["season"].isin([season, season - 1])
    return set(games.loc[m, "home_team"]) | set(games.loc[m, "away_team"])
