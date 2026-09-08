"""Feature construction for the Premier League.

Built by replaying the season in date order, so a feature attached to match N only ever saw
matches 1..N-1. That guarantee is what the whole project rests on and it is verified in CI by
rebuilding from a truncated schedule and asserting identical output.

**The targets are supremacy and total goals**, not margin and points. Everything downstream
consumes them as a pair and hands them to :mod:`epl.poisson`, which turns them into a full
scoreline distribution. That is the structural difference from the other two sports and the
reason a 1X2 price, an Asian handicap and an over/under can all be quoted off one model without
contradicting each other.

Three groups of features.

**The league-agnostic core** - ratings, form, rest - is the same shape as the other pipelines,
in goals rather than points.

**The preseason carry.** For the first month there is barely any in-season evidence, and unlike
college football there is no SP+ to lean on. Prior-season opponent-adjusted strength fills the
gap, joined only from season-1 so it cannot leak, in two flavours:

* ``prev_att`` / ``prev_def`` from goals
* ``prev_att_shots`` / ``prev_def_shots`` from shots on target

Both are supplied because the shot version is the better predictor over a 38-match sample and
the goal version carries information the shot version misses (finishing quality is real, it is
just smaller than people think). Note that **defence is signed so that higher is worse**: it is
a multiplier on goals conceded.

**The football-specific ones**, which is where this genuinely diverges:

* **Promotion.** Three of twenty clubs are new every season and have no top-flight history at
  all. ``promoted`` flags them and their prior-season row is carried up from the division below,
  scaled. Nothing in the NFL behaves like this - the same 32 franchises come back every year.
* **European commitments.** A club in the Champions or Europa League plays midweek, often
  abroad, for most of the season. The league schedule alone cannot see those matches, so
  ``in_europe`` is derived from prior-season finishing position - legitimately known before a
  ball is kicked, and the honest version of a fixture-congestion feature given what this data
  actually contains.
* **Derbies and travel.** England has no time zones, so what remains is a genuine north-south
  haul at one end and, at the other, the local derby - a fixture that reliably produces more
  cards and fewer goals than the ratings imply.
* **Crowds.** The behind-closed-doors window of 2020-21 is flagged, and home advantage is
  suppressed in the rating replay for it.

**Matchweek is deliberately NOT a feature.** football-data publishes no round number, and the
one you would reconstruct from dates is a lie: postponements and European fixtures mean two
clubs in the same calendar week can be three matches apart. What the model sees instead is
``h_games`` / ``a_games`` - matches actually played by that club - which is the honest version
of the same question and is per club rather than per league.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .ratings import FORM_WINDOW, LEAGUE_GPG, RatingBook, RatingConfig
from .teams import is_derby, travel_km

log = logging.getLogger(__name__)

BASE_FEATURES = [
    # context
    "no_crowd", "is_derby", "season_frac", "h_games", "a_games", "is_early_season",
    # ratings (log-goals space, straight from the replay)
    "exp_sup", "exp_total", "exp_home_goals", "exp_away_goals",
    "h_att", "a_att", "h_def", "a_def",
    # form
    "h_form_sup", "a_form_sup", "h_form_total", "a_form_total",
    # rest and congestion
    "h_rest", "a_rest", "rest_diff", "h_short_rest", "a_short_rest",
    # preseason carry: prior-season opponent-adjusted strength (def: higher is worse)
    "h_prev_att", "a_prev_att", "h_prev_def", "a_prev_def",
    "h_prev_att_shots", "a_prev_att_shots", "h_prev_def_shots", "a_prev_def_shots",
    "h_prev_ppg", "a_prev_ppg", "h_prev_pos", "a_prev_pos",
    "prior_sup_est", "prior_total_est",
    # promotion and Europe
    "h_promoted", "a_promoted", "h_in_europe", "a_in_europe",
    # geography
    "travel_km",
]

# The market's own view, in the model's units. `mkt_sup` and `mkt_total` are the Asian handicap
# and over/under inverted through the same scoreline model the predictions come out of, so
# "the market says +1.2 goals" and "we say +1.5 goals" are the same quantity and the difference
# is meaningful. The three de-vigged 1X2 probabilities are a second, independent market view
# of the same match - the analogue of the NFL model's moneyline feature.
MARKET_FEATURES = ["ah_home", "total_line", "mkt_sup", "mkt_total",
                   "sup_vs_model", "total_vs_model",
                   "mkt_p_home", "mkt_p_draw", "mkt_p_away"]

REST_CAP = 21
SHORT_REST = 4          # a Saturday-Tuesday-Friday run; the congested-fixture case
EARLY_MATCHES = 6       # roughly the first six weeks, before ratings have settled
EUROPE_POSITIONS = 6    # prior-season top six is a good proxy for European commitments


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if f == f else np.nan


def _rest(last, gdate):
    if last is None or gdate is None:
        return np.nan
    try:
        d = (gdate - last).days
    except TypeError:
        return np.nan
    return float(min(d, REST_CAP))


def hfa_suppressed(gdate, neutral: bool = False) -> bool:
    """True when home advantage should be treated as absent for this match.

    The behind-closed-doors window is the whole reason this exists. English home advantage in
    that period fell to almost nothing - home win rate went from ~45% to ~36% - and crediting
    those matches a normal home edge would push every home club's attack rating down by an
    advantage that was not there. Unlike the NFL's version this is a DATE test rather than a
    season test, because the disruption started mid-season: 2019-20 was played in front of full
    grounds until March and empty from June.
    """
    return bool(neutral) or config.no_crowd(gdate)


def _strength_lookup(strength: pd.DataFrame | None) -> dict:
    """(season, club) -> prior-season strength row, keyed for a season-1 join."""
    if strength is None or not len(strength):
        return {}
    out = {}
    for r in strength.itertuples():
        out[(int(r.season), r.team)] = {
            "att": _num(getattr(r, "att_goals", np.nan)),
            "def": _num(getattr(r, "def_goals", np.nan)),
            "att_shots": _num(getattr(r, "att_shots", np.nan)),
            "def_shots": _num(getattr(r, "def_shots", np.nan)),
            "ppg": _num(getattr(r, "ppg", np.nan)),
            "position": _num(getattr(r, "position", np.nan)),
            "promoted": int(getattr(r, "promoted", 0) or 0),
        }
    return out


def _row(book, prev, season, home, away, gdate, rec) -> dict:
    h, a = book.get(season, home), book.get(season, away)
    no_hfa = hfa_suppressed(gdate)
    e = book.expect(season, home, away, no_hfa)

    hp = prev.get((season - 1, home), {})
    ap = prev.get((season - 1, away), {})

    # A goals translation of the two prior-season ratings, so the model is handed the
    # interaction pre-computed rather than having to find it across eight columns. Same
    # multiplicative form the rating engine and the scoreline model both use, so it is on the
    # same scale as exp_sup and directly comparable to it.
    h_att, a_att = hp.get("att", np.nan), ap.get("att", np.nan)
    h_dfn, a_dfn = hp.get("def", np.nan), ap.get("def", np.nan)
    have = not any(x != x for x in (h_att, a_att, h_dfn, a_dfn))
    if have:
        lh = LEAGUE_GPG * h_att * a_dfn * (1.0 if no_hfa else 1.11)
        la = LEAGUE_GPG * a_att * h_dfn * (1.0 if no_hfa else 0.90)
        prior_sup, prior_total = lh - la, lh + la
    else:
        prior_sup = prior_total = np.nan

    h_rest = _rest(h.last_date, gdate)
    a_rest = _rest(a.last_date, gdate)
    h_pos, a_pos = hp.get("position", np.nan), ap.get("position", np.nan)

    return {
        "no_crowd": int(config.no_crowd(gdate)),
        "is_derby": is_derby(home, away),
        "season_frac": min(1.0, max(h.games, a.games) / config.MATCHDAYS),
        "h_games": h.games, "a_games": a.games,
        "is_early_season": int(min(h.games, a.games) < EARLY_MATCHES),
        "exp_sup": e["exp_sup"], "exp_total": e["exp_total"],
        "exp_home_goals": e["exp_home_goals"], "exp_away_goals": e["exp_away_goals"],
        "h_att": h.att, "a_att": a.att, "h_def": h.deff, "a_def": a.deff,
        "h_form_sup": float(np.mean(h.recent_sup)) if h.recent_sup else 0.0,
        "a_form_sup": float(np.mean(a.recent_sup)) if a.recent_sup else 0.0,
        "h_form_total": float(np.mean(h.recent_total)) if h.recent_total else 0.0,
        "a_form_total": float(np.mean(a.recent_total)) if a.recent_total else 0.0,
        "h_rest": h_rest, "a_rest": a_rest,
        "rest_diff": (h_rest - a_rest) if h_rest == h_rest and a_rest == a_rest else np.nan,
        "h_short_rest": int(h_rest <= SHORT_REST) if h_rest == h_rest else 0,
        "a_short_rest": int(a_rest <= SHORT_REST) if a_rest == a_rest else 0,
        "h_prev_att": h_att, "a_prev_att": a_att,
        "h_prev_def": h_dfn, "a_prev_def": a_dfn,
        "h_prev_att_shots": hp.get("att_shots", np.nan),
        "a_prev_att_shots": ap.get("att_shots", np.nan),
        "h_prev_def_shots": hp.get("def_shots", np.nan),
        "a_prev_def_shots": ap.get("def_shots", np.nan),
        "h_prev_ppg": hp.get("ppg", np.nan), "a_prev_ppg": ap.get("ppg", np.nan),
        "h_prev_pos": h_pos, "a_prev_pos": a_pos,
        "prior_sup_est": prior_sup, "prior_total_est": prior_total,
        # A club with no prior-division row at all is promoted too - that is what "no row"
        # means once the strength table covers the division below as well.
        "h_promoted": int(hp.get("promoted", 1) if hp else 1),
        "a_promoted": int(ap.get("promoted", 1) if ap else 1),
        "h_in_europe": int(h_pos <= EUROPE_POSITIONS) if h_pos == h_pos else 0,
        "a_in_europe": int(a_pos <= EUROPE_POSITIONS) if a_pos == a_pos else 0,
        "travel_km": travel_km(home, away),
    }


def _market(row: dict, rec: dict) -> dict:
    """Attach the market's view and the model's disagreement with it.

    ``ah_home`` follows the repo-wide convention that a negative number means the home side is
    favoured, so the market's expected supremacy is ``-ah_home``. That is the same relationship
    ``spread_home`` has to ``exp_margin`` in the other two pipelines, and it is why no sign
    flip lives in this file - the one flip that could be needed was not, and the test suite
    pins that.
    """
    ah = _num(rec.get("ah_home"))
    tl = _num(rec.get("total_line"))
    mkt_sup = _num(rec.get("mkt_sup"))
    mkt_total = _num(rec.get("mkt_total"))
    if mkt_sup != mkt_sup and ah == ah:
        mkt_sup = -ah
    row["ah_home"] = ah
    row["total_line"] = tl
    row["mkt_sup"] = mkt_sup
    row["mkt_total"] = mkt_total
    row["sup_vs_model"] = row["exp_sup"] - mkt_sup
    row["total_vs_model"] = row["exp_total"] - mkt_total
    row["mkt_p_home"] = _num(rec.get("mkt_p_home"))
    row["mkt_p_draw"] = _num(rec.get("mkt_p_draw"))
    row["mkt_p_away"] = _num(rec.get("mkt_p_away"))
    return row


def build(games: pd.DataFrame, upcoming: pd.DataFrame | None = None,
          lines: pd.DataFrame | None = None, strength: pd.DataFrame | None = None,
          cfg: RatingConfig | None = None, first_train_season: int | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame, RatingBook]:
    """Replay the season and emit one feature row per completed match.

    Seasons before `first_train_season` are **warm-up**: they advance the ratings but emit no
    training rows. That is what gives the opening weekend of the first training season a real
    regressed prior instead of a league of twenty identically-rated clubs.
    """
    first_train = config.FIRST_SEASON if first_train_season is None else first_train_season
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"]).dt.date
    # Sorted by DATE, not by round: a match postponed from August to February must be replayed
    # in February, where the ratings actually stood when it was played. game_id breaks ties so
    # the order is deterministic across runs.
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)

    prev = _strength_lookup(strength)
    line_map = {}
    if lines is not None and len(lines):
        keep = [c for c in ("ah_home", "total_line", "mkt_sup", "mkt_total", "mkt_p_home",
                            "mkt_p_draw", "mkt_p_away", "price_home", "price_draw",
                            "price_away", "price_over", "price_under", "price_ah_home",
                            "price_ah_away", "is_closing", "odds_source")
                if c in lines.columns]
        for r in lines.to_dict("records"):
            line_map[str(r["game_id"])] = {k: r.get(k) for k in keep}

    book = RatingBook(cfg)
    rows = []
    for g in games.itertuples(index=False):
        season = int(g.season)
        gdate = g.date
        if not bool(getattr(g, "completed", False)):
            continue
        if season < first_train:
            book.update(season, g.home_team, g.away_team, g.home_goals, g.away_goals,
                        gdate, hfa_suppressed(gdate))
            continue
        r = _row(book, prev, season, g.home_team, g.away_team, gdate, {})
        r = _market(r, line_map.get(str(g.game_id), {}))
        r.update({"game_id": g.game_id, "date": gdate, "season": season,
                  "home_team": g.home_team, "away_team": g.away_team,
                  "matchweek": int(getattr(g, "matchweek", 0) or 0),
                  "home_goals": g.home_goals, "away_goals": g.away_goals,
                  "supremacy": g.supremacy, "total_goals": g.total_goals,
                  "result": getattr(g, "result", ""),
                  "referee": getattr(g, "referee", ""),
                  "is_closing": bool(line_map.get(str(g.game_id), {}).get("is_closing", False))})
        rows.append(r)
        # State advances only AFTER the row is recorded. This ordering IS the leak-free
        # guarantee, and test_leak_free rebuilds from a truncated schedule to prove it.
        book.update(season, g.home_team, g.away_team, g.home_goals, g.away_goals, gdate,
                    hfa_suppressed(gdate))

    train_rows = pd.DataFrame(rows)

    up_rows = []
    if upcoming is not None and len(upcoming):
        for rec in upcoming.to_dict("records"):
            gdate = pd.Timestamp(rec["date"]).date()
            season = int(rec.get("season") or config.season_of(gdate))
            r = _row(book, prev, season, rec["home_team"], rec["away_team"], gdate, rec)
            r = _market(r, rec)
            r.update(rec)
            up_rows.append(r)
    return train_rows, pd.DataFrame(up_rows), book


def epl_clubs(games: pd.DataFrame, season: int) -> set[str]:
    m = games["season"].isin([season, season - 1])
    return set(games.loc[m, "home_team"]) | set(games.loc[m, "away_team"])
