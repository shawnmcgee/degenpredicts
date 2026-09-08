"""Offline tests for the Premier League pipeline. No network, no API key.

Structured like ``tests/test_nfl.py``: thresholds are asserted as ABSOLUTE values rather than
compared against another sport's config, so retuning the college or NFL pipelines cannot fail
this suite. CI runs one job per sport, so a red check names the sport that broke.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Sandbox every path before any epl module is imported or re-read.

    The modules read ``config.NAME`` at call time precisely so this works; a module that bound
    its paths at import would quietly read and write the real repo data instead.
    """
    monkeypatch.setenv("DEGEN_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("DEGEN_DOCS", str(tmp_path / "docs"))
    monkeypatch.setenv("DEGEN_ROOT", str(ROOT))
    import importlib
    from epl import config as cfg
    importlib.reload(cfg)
    for mod in ("epl.teams", "epl.ratings", "epl.poisson", "epl.odds_math", "epl.features",
                "epl.sources.footballdata", "epl.sources.odds", "epl.train", "epl.predict",
                "epl.grade", "epl.site"):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    yield tmp_path


# ---------------------------------------------------------------------------------
# Clubs: the guardrail that matters most
# ---------------------------------------------------------------------------------
def test_canonical_map_resolves_and_refuses():
    from epl.teams import UnknownClub, canon, is_known

    # every spelling of the same club lands on one canonical name
    for spelling in ("Man United", "Manchester United", "Man Utd", "MUN", "manchester utd"):
        assert canon(spelling) == "Man United"
    for spelling in ("Nott'm Forest", "Nottingham Forest", "Nottm Forest", "NFO"):
        assert canon(spelling) == "Nott'm Forest"
    assert canon("Spurs") == canon("Tottenham Hotspur") == "Tottenham"
    assert canon("Wolverhampton Wanderers") == "Wolves"

    # an unknown club raises rather than passing through and inventing a rating history
    with pytest.raises(UnknownClub):
        canon("Real Madrid")
    with pytest.raises(UnknownClub):
        canon(None)
    assert not is_known("Real Madrid")


def test_ambiguous_short_names_refuse_rather_than_guess():
    """"Sheffield" is two clubs. Resolving it either way merges two rating histories, and
    nothing downstream can detect that it happened."""
    from epl.teams import UnknownClub, canon

    for ambiguous in ("Sheffield", "Bristol", "Manchester", "United", "City"):
        with pytest.raises(UnknownClub) as e:
            canon(ambiguous)
        # the error has to name the alternatives, or it is not actionable
        assert "ambiguous" in str(e.value).lower()

    # the FULL names are of course fine and must stay distinct
    assert canon("Sheffield United") != canon("Sheffield Weds")


def test_every_premier_league_club_has_a_ground():
    """Travel and derby features need coordinates. An unmapped club is silent at runtime -
    NaN travel, which the trees absorb - so it has to be loud here instead."""
    from epl.teams import PREMIER_LEAGUE_CLUBS, ground

    missing = [c for c in PREMIER_LEAGUE_CLUBS if ground(c) is None]
    assert not missing, f"clubs with no ground mapped: {missing}"


def test_derby_detection():
    from epl.teams import is_derby, travel_km

    assert is_derby("Arsenal", "Tottenham")          # north London
    assert is_derby("Everton", "Liverpool")          # Merseyside
    assert is_derby("Man City", "Man United")        # Manchester
    assert not is_derby("Newcastle", "Bournemouth")  # the length of the country
    assert travel_km("Bournemouth", "Newcastle") > 400


# ---------------------------------------------------------------------------------
# The scoreline model
# ---------------------------------------------------------------------------------
def test_grid_is_a_probability_distribution():
    from epl.poisson import grid, match_odds

    for sup, tot in ((0.0, 2.8), (0.3, 2.75), (1.5, 3.2), (-2.0, 2.1)):
        m = grid(sup, tot)
        assert m.sum() == pytest.approx(1.0, abs=1e-9)
        assert (m >= 0).all()
        h, d, a = match_odds(m)
        assert h + d + a == pytest.approx(1.0, abs=1e-9)


def test_grid_recovers_its_inputs():
    """The whole design rests on (supremacy, total) round-tripping through the grid: the models
    predict that pair and every market is read back off it."""
    from epl.poisson import expected, grid

    for sup, tot in ((0.0, 2.8), (0.35, 2.75), (1.2, 3.1)):
        e_sup, e_tot = expected(grid(sup, tot))
        assert e_sup == pytest.approx(sup, abs=0.01)
        assert e_tot == pytest.approx(tot, abs=0.01)


def test_draw_probability_is_real_and_peaks_at_level():
    """The reason this sport does not use a Gaussian margin model. A continuous density puts
    zero mass on an exact draw; here it is a quarter of all matches."""
    from epl.poisson import grid, match_odds

    draws = [match_odds(grid(s, 2.8))[1] for s in (0.0, 0.5, 1.0, 2.0)]
    assert 0.22 < draws[0] < 0.30, draws[0]
    assert draws == sorted(draws, reverse=True), "draw rate must fall as one side pulls ahead"


def test_dixon_coles_lifts_the_low_draws():
    """Independent Poisson understates 0-0 and 1-1 and overstates 1-0 and 0-1. Those four
    scorelines are about a fifth of matches and they decide whether a match is drawn."""
    from epl.poisson import grid

    indep = grid(0.3, 2.75, rho=0.0)
    dc = grid(0.3, 2.75, rho=-0.04)
    assert dc[0, 0] > indep[0, 0]
    assert dc[1, 1] > indep[1, 1]
    assert dc[1, 0] < indep[1, 0]
    assert dc[0, 1] < indep[0, 1]


def test_quarter_handicaps_split_across_two_lines():
    """A -0.75 line is half at -0.5 and half at -1.0. Half the stake can win while the other
    half pushes, which no single-outcome formula expresses - and quarter lines are quoted on
    most Premier League matches, so getting this wrong misprices the main market."""
    from epl.poisson import asian_handicap, grid

    m = grid(0.4, 2.8)
    w_half, p_half, l_half = asian_handicap(m, -0.5)
    w_whole, p_whole, l_whole = asian_handicap(m, -1.0)
    w_q, p_q, l_q = asian_handicap(m, -0.75)

    assert w_q == pytest.approx((w_half + w_whole) / 2, abs=1e-9)
    assert p_q == pytest.approx((p_half + p_whole) / 2, abs=1e-9)
    assert p_half == pytest.approx(0.0, abs=1e-12), "a half line can never push"
    assert p_whole > 0.05, "a whole line pushes when the match lands exactly on it"
    for legs in ((w_half, p_half, l_half), (w_whole, p_whole, l_whole), (w_q, p_q, l_q)):
        assert sum(legs) == pytest.approx(1.0, abs=1e-9)


def test_whole_number_goal_lines_have_a_push_leg():
    """Football quotes whole-number totals as readily as halves, and 'over 3 goals' pushes on
    exactly three - about 15% of matches. Folding that into either side misprices the bet."""
    from epl.poisson import grid, over_under

    m = grid(0.2, 2.9)
    o, p, u = over_under(m, 2.5)
    assert p == pytest.approx(0.0, abs=1e-12)
    assert o + u == pytest.approx(1.0, abs=1e-9)

    o3, p3, u3 = over_under(m, 3.0)
    assert p3 > 0.10
    assert o3 + p3 + u3 == pytest.approx(1.0, abs=1e-9)


def test_away_side_is_priced_off_the_mirrored_grid():
    """Backing the away side is the mirror of the home handicap, not the home leg's loss
    column - which differs on a quarter line, where part of the stake pushes on both sides."""
    from epl.poisson import asian_handicap, grid

    m = grid(0.5, 2.8)
    hw, hp, hl = asian_handicap(m, -0.75)          # home gives 0.75
    aw, ap, al = asian_handicap(m.T, 0.75)         # away receives 0.75
    assert hw + hp + hl == pytest.approx(1.0, abs=1e-9)
    assert aw + ap + al == pytest.approx(1.0, abs=1e-9)
    # the two sides of the same bet: home's loss is away's win, push is shared
    assert aw == pytest.approx(hl, abs=1e-9)
    assert ap == pytest.approx(hp, abs=1e-9)


# ---------------------------------------------------------------------------------
# Odds mathematics
# ---------------------------------------------------------------------------------
def test_three_way_devig_sums_to_one_and_corrects_longshot_bias():
    """Proportional de-vigging inflates longshots, which is exactly where a model talks itself
    into a bet. Shin shrinks them and lifts the favourite."""
    from epl.odds_math import devig_three, overround

    prices = (1.25, 6.00, 12.00)
    assert overround(*prices) == pytest.approx(1.05, abs=0.005)

    prop = devig_three(*prices, method="proportional")
    shin = devig_three(*prices, method="shin")
    for p in (prop, shin):
        assert sum(p) == pytest.approx(1.0, abs=1e-9)
    assert shin[0] > prop[0], "Shin must lift the favourite"
    assert shin[2] < prop[2], "Shin must shrink the longshot"


def test_devig_rejects_a_book_that_is_not_a_book():
    """An overround under 1 is an arbitrage, which does not happen at a real book and always
    means a column was read from the wrong place."""
    from epl.odds_math import devig_three

    assert all(p != p for p in devig_three(1.1, 1.1, 1.1))      # 2.7 overround
    assert all(p != p for p in devig_three(4.0, 4.0, 4.0))      # sums to 0.75
    assert all(p != p for p in devig_three(None, 3.0, 3.0))


def test_market_numbers_invert_back_through_the_same_model():
    """`mkt_sup` and `mkt_total` must be on the model's scale, or 'we differ by 0.4 goals' is
    comparing two different quantities."""
    from epl.odds_math import supremacy_from_prices, total_from_prices
    from epl.poisson import grid, match_odds, over_under

    for sup in (0.0, 0.4, 1.3):
        h, d, a = match_odds(grid(sup, 2.8))
        assert supremacy_from_prices(1 / h, 1 / d, 1 / a,
                                     method="proportional") == pytest.approx(sup, abs=0.02)
    for tot in (2.3, 2.8, 3.3):
        o, _p, u = over_under(grid(0.0, tot), 2.5)
        assert total_from_prices(1 / o, 1 / u, 2.5,
                                 method="proportional") == pytest.approx(tot, abs=0.02)


def test_decimal_payout_is_price_minus_one():
    from epl.odds_math import payout

    assert payout(2.50) == pytest.approx(1.50)
    assert payout(1.95) == pytest.approx(0.95)
    assert payout(1.0) != payout(1.0)      # NaN: not a real price


# ---------------------------------------------------------------------------------
# The source: dates, signs and the era change
# ---------------------------------------------------------------------------------
def test_dates_are_parsed_day_first():
    """football-data writes British dates. Month-first, 01/02/2024 becomes 2 January - which
    raises nothing, reorders a third of the season, and breaks the leak-free replay."""
    from epl.sources.footballdata import _dates

    d = _dates(pd.DataFrame({"Date": ["01/02/2024", "26/12/2023"]}))
    assert list(d.dt.date) == [date(2024, 2, 1), date(2023, 12, 26)]


def test_iso_dates_from_the_mirror_also_parse():
    from epl.sources.footballdata import _dates

    d = _dates(pd.DataFrame({"Date": ["2024-08-16", "2024-12-26"]}))
    assert list(d.dt.date) == [date(2024, 8, 16), date(2024, 12, 26)]


def _raw_row(**extra):
    base = dict(Date="16/08/2024", HomeTeam="Man City", AwayTeam="Luton", FTHG=3, FTAG=0,
                HS=20, AS=4, HST=9, AST=1, HC=8, AC=2, HY=1, AY=3, HR=0, AR=0)
    base.update(extra)
    return pd.DataFrame([base])


BETBRAIN_ERA = dict(B365H=1.20, B365D=7.00, B365A=15.0,
                    BbAvH=1.22, BbAvD=6.80, BbAvA=14.0,
                    BbAHh=-2.0, BbAvAHH=1.95, BbAvAHA=1.90,
                    **{"BbAv>2.5": 1.55, "BbAv<2.5": 2.45})
MODERN_ERA = dict(B365H=1.20, B365D=7.00, B365A=15.0,
                  AvgH=1.22, AvgD=6.80, AvgA=14.0, PSCH=1.18, PSCD=7.20, PSCA=16.0,
                  AHh=-2.0, AvgAHH=1.95, AvgAHA=1.90, AHCh=-2.25, PCAHH=1.98, PCAHA=1.92,
                  **{"Avg>2.5": 1.55, "Avg<2.5": 2.45, "PC>2.5": 1.52, "PC<2.5": 2.52})


def test_odds_resolve_in_both_column_eras():
    """The columns changed shape in 2019-20: Betbrain aggregates out, Avg/Max plus a whole
    parallel set of CLOSING columns in. Reading one era's spelling does not raise - it yields
    NaN for every row in the other era, and the market models silently train on half of
    history."""
    from epl.sources.footballdata import parse_season

    _g, old = parse_season(_raw_row(**BETBRAIN_ERA), 2015)
    assert len(old) == 1
    assert not bool(old.is_closing.iloc[0])
    assert "BbAHh" in old.odds_source.iloc[0]
    assert old.ah_home.iloc[0] == pytest.approx(-2.0)

    _g, new = parse_season(_raw_row(**MODERN_ERA), 2024)
    assert len(new) == 1
    assert bool(new.is_closing.iloc[0]), "must prefer the explicit closing columns"
    assert "PSCH" in new.odds_source.iloc[0], "must prefer Pinnacle over the panel average"
    assert new.ah_home.iloc[0] == pytest.approx(-2.25), "must use the CLOSING handicap"


def test_asian_handicap_sign_convention():
    """football-data quotes the handicap on the home team: -2.25 means the home side gives
    2.25 goals. That already matches this repo's `spread_home` convention (negative = home
    favoured), so unlike the NFL feed there is NO flip - and this pins that."""
    from epl.sources.footballdata import parse_season

    _g, ln = parse_season(_raw_row(**MODERN_ERA), 2024)
    assert ln.ah_home.iloc[0] < 0, "home favourite must carry a negative handicap"
    assert ln.mkt_sup.iloc[0] == pytest.approx(2.25), \
        "market supremacy is -ah_home: a positive number means the HOME side is favoured"
    assert ln.mkt_p_home.iloc[0] > ln.mkt_p_away.iloc[0]


def test_game_id_is_stable_when_a_match_is_postponed():
    """English football rearranges constantly. A date-keyed id would mint a NEW id when a
    postponed match is finally played, so its published pick would never grade."""
    from epl.sources.footballdata import make_game_id, parse_season

    august = parse_season(_raw_row(Date="16/08/2024", **MODERN_ERA), 2024)[0]
    february = parse_season(_raw_row(Date="04/02/2025", **MODERN_ERA), 2024)[0]
    assert august.game_id.iloc[0] == february.game_id.iloc[0]
    assert make_game_id(2024, "Man City", "Luton") != make_game_id(2024, "Luton", "Man City")


def test_unknown_club_is_skipped_loudly_not_invented(caplog):
    from epl.sources.footballdata import parse_season

    g, _ln = parse_season(_raw_row(HomeTeam="Real Madrid"), 2024)
    assert len(g) == 0
    assert any("skipping" in r.message.lower() or "unmapped" in r.message.lower()
               for r in caplog.records)


# ---------------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------------
def test_home_advantage_is_football_sized():
    """Measured at +0.30 goals across 2015-16 to 2024-25 excluding the empty-stadium seasons.
    Asserted as an absolute band so retuning another sport cannot fail this."""
    from epl.ratings import RatingBook

    e = RatingBook().expect(2024, "Arsenal", "Chelsea")
    assert 0.20 <= e["exp_sup"] <= 0.40, e["exp_sup"]
    assert 2.5 <= e["exp_total"] <= 3.1, e["exp_total"]


def test_promoted_clubs_are_seeded_not_defaulted():
    """Three of twenty clubs are new each season. Entering them at league average hands them
    about a third of a goal a match they have not earned - larger than any feature in the
    model."""
    from epl.ratings import RatingBook

    book = RatingBook()
    book.update(2023, "Arsenal", "Chelsea", 2, 1, date(2023, 8, 12))
    promoted = book.get(2024, "Luton")
    assert promoted.att < 0, "a promoted club must be seeded below average in attack"
    assert promoted.deff > 0, "and above average in goals conceded"
    assert book.is_new(2024, "Luton")
    assert not book.is_new(2024, "Arsenal")

    # the very first club ever rated is the start of history, NOT a promotion
    fresh = RatingBook()
    first = fresh.get(2015, "Arsenal")
    assert first.att == 0.0 and first.deff == 0.0


def test_ratings_stay_anchored_to_the_league_mean():
    """Nothing forces mean attack and mean defence to zero, so in a high-scoring season every
    club's ratings drift up together. Left alone that breaks the season rollover, which
    regresses toward zero - no longer the mean - and quietly deflates scoring each August."""
    from epl.ratings import RatingBook

    book = RatingBook()
    clubs = [f"Club{i}" for i in range(10)]
    # a full season of matches scoring well above the anchor rate
    for season in (2020, 2021):
        d = date(season, 8, 15)
        for _ in range(12):
            for i in range(0, 10, 2):
                book.update(season, clubs[i], clubs[i + 1], 3, 2, d)
                d += timedelta(days=3)
    # The rollover centres the ratings the NEXT season starts from - that is the point, since
    # it is the regression toward the mean that needs the mean to actually be zero.
    nxt = book.season(2022)
    assert abs(np.mean([t.att for t in nxt.values()])) < 1e-9
    assert abs(np.mean([t.deff for t in nxt.values()])) < 1e-9
    # the level absorbed it instead
    assert book.league_log > 0.05, "a high-scoring league must raise the LEVEL, not every club"


def test_ratings_track_a_dominant_side():
    from epl.ratings import RatingBook

    book = RatingBook()
    d = date(2024, 8, 17)
    for _ in range(12):
        book.update(2024, "Arsenal", "Chelsea", 3, 0, d)
        d += timedelta(days=7)
    assert book.get(2024, "Arsenal").att > book.get(2024, "Chelsea").att
    assert book.get(2024, "Arsenal").deff < book.get(2024, "Chelsea").deff
    assert book.expect(2024, "Arsenal", "Chelsea")["exp_sup"] > 1.0


# ---------------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------------
def _synthetic_games(seasons=(2020, 2021, 2022), clubs=None, seed=0):
    """A deterministic fake league: a full double round robin per season."""
    clubs = clubs or ["Arsenal", "Chelsea", "Everton", "Fulham", "Liverpool", "Man City"]
    rng = np.random.default_rng(seed)
    rows = []
    from epl.sources.footballdata import make_game_id
    for s in seasons:
        d = date(s, 8, 15)
        for h in clubs:
            for a in clubs:
                if h == a:
                    continue
                hg, ag = int(rng.poisson(1.6)), int(rng.poisson(1.2))
                rows.append({"game_id": make_game_id(s, h, a), "season": s, "league": "E0",
                             "date": d, "kickoff": "15:00", "kickoff_uk": "", "matchweek": 1,
                             "home_team": h, "away_team": a,
                             "home_goals": hg, "away_goals": ag,
                             "supremacy": hg - ag, "total_goals": hg + ag,
                             "result": "H" if hg > ag else ("D" if hg == ag else "A"),
                             "ht_home_goals": 0, "ht_away_goals": 0,
                             "home_shots": 12, "away_shots": 9, "home_sot": 5, "away_sot": 3,
                             "home_corners": 5, "away_corners": 4,
                             "home_cards": 1, "away_cards": 1, "referee": "R Ref",
                             "no_crowd": 0, "completed": True})
                d += timedelta(days=4)
    return pd.DataFrame(rows)


def test_leak_free():
    """A feature on match N may only ever have seen matches 1..N-1. Rebuilt from a truncated
    schedule, the surviving rows must be byte-identical."""
    from epl.features import BASE_FEATURES, build

    games = _synthetic_games()
    full, _u, _b = build(games, first_train_season=2020)
    cut = int(len(games) * 0.6)
    part, _u, _b = build(games.iloc[:cut], first_train_season=2020)

    common = set(part.game_id) & set(full.game_id)
    assert len(common) > 50
    a = full[full.game_id.isin(common)].sort_values("game_id")[BASE_FEATURES].reset_index(drop=True)
    b = part[part.game_id.isin(common)].sort_values("game_id")[BASE_FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)


def test_warmup_seasons_advance_ratings_without_emitting_rows():
    """Without a warm-up the opening weekend of the first training season runs on a league of
    twenty identically-rated clubs, and every fixture that day shares one exp_sup."""
    from epl.features import build

    games = _synthetic_games(seasons=(2020, 2021, 2022))
    warm, _u, _b = build(games, first_train_season=2022)
    cold, _u, _b = build(games[games.season == 2022], first_train_season=2022)

    assert set(warm.season.unique()) == {2022}
    assert len(warm) == len(cold)
    first_day = cold[cold.date == cold.date.min()]
    assert first_day.exp_sup.nunique() == 1, "cold start: every club identical, as expected"
    warm_first = warm[warm.date == warm.date.min()]
    assert warm_first.exp_sup.nunique() >= 1
    assert warm.exp_sup.std() > cold.exp_sup.std() * 0.5


def test_matches_behind_closed_doors_lose_home_advantage():
    """The window is a DATE range, not a season: 2019-20 was played in front of full grounds
    until March and empty from June, so flagging whole seasons would mislabel 288 matches."""
    from epl import config
    from epl.features import hfa_suppressed
    from epl.ratings import RatingBook

    assert hfa_suppressed(date(2020, 11, 7))       # inside the window
    assert not hfa_suppressed(date(2019, 11, 7))   # before it
    assert not hfa_suppressed(date(2022, 11, 7))   # after it
    assert config.no_crowd(date(2021, 1, 2))

    book = RatingBook()
    normal = book.expect(2022, "Arsenal", "Chelsea", no_hfa=False)["exp_sup"]
    empty = book.expect(2020, "Arsenal", "Chelsea", no_hfa=True)["exp_sup"]
    assert empty == pytest.approx(0.0, abs=1e-9)
    assert normal > 0.2


def test_promotion_and_europe_flags_reach_the_features():
    from epl.features import build

    games = _synthetic_games(seasons=(2021, 2022))
    strength = pd.DataFrame([
        {"season": 2021, "league": "E0", "team": "Arsenal", "played": 38, "att_goals": 1.3,
         "def_goals": 0.8, "att_shots": 1.2, "def_shots": 0.85, "ppg": 2.1, "gd_per_game": 1.0,
         "position": 2, "promoted": 0},
        {"season": 2021, "league": "E1", "team": "Fulham", "played": 46, "att_goals": 0.9,
         "def_goals": 1.1, "att_shots": 0.95, "def_shots": 1.05, "ppg": 1.2,
         "gd_per_game": -0.1, "position": 14, "promoted": 1},
    ])
    tr, _u, _b = build(games, strength=strength, first_train_season=2022)
    ars = tr[tr.home_team == "Arsenal"].iloc[0]
    assert ars.h_promoted == 0
    assert ars.h_in_europe == 1          # finished 2nd
    ful = tr[tr.home_team == "Fulham"].iloc[0]
    assert ful.h_promoted == 1
    assert ful.h_in_europe == 0
    # a club with no prior row at all is treated as promoted, not as average
    eve = tr[tr.home_team == "Everton"].iloc[0]
    assert eve.h_promoted == 1


def test_matchweek_is_not_a_model_feature():
    """Reconstructed from dates it is a lie - postponements and European fixtures put two
    clubs three matches apart in the same calendar week. The honest version is games played."""
    from epl.features import BASE_FEATURES

    assert "matchweek" not in BASE_FEATURES
    assert "h_games" in BASE_FEATURES and "a_games" in BASE_FEATURES


# ---------------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("handicap,supremacy,expected,frac", [
    (-0.5, 1, "win", 1.0),
    (-0.5, 0, "loss", -1.0),
    (-1.0, 1, "push", 0.0),
    (0.0, 0, "push", 0.0),
    (-0.75, 2, "win", 1.0),
    (-0.75, 1, "half win", 0.5),
    (-0.75, 0, "loss", -1.0),
    (-0.25, 1, "win", 1.0),
    (-0.25, 0, "half loss", -0.5),
])
def test_quarter_line_settlement(handicap, supremacy, expected, frac):
    """A -0.75 backer whose side wins by one wins half the stake and gets the rest back.
    Collapsing that to a plain win or loss misstates the return on the main market."""
    from epl.grade import _ah_result

    assert _ah_result(handicap, supremacy, took_home=True) == (expected, frac)


def test_away_side_grading_mirrors_the_handicap():
    from epl.grade import _ah_result

    # home -1.0, match ends 2-0: home covers, away does not
    assert _ah_result(-1.0, 2, took_home=True)[0] == "win"
    assert _ah_result(-1.0, 2, took_home=False)[0] == "loss"
    # exactly on the line: both push
    assert _ah_result(-1.0, 1, took_home=True)[0] == "push"
    assert _ah_result(-1.0, 1, took_home=False)[0] == "push"


# ---------------------------------------------------------------------------------
# Betting discipline
# ---------------------------------------------------------------------------------
def test_stake_sizing_is_eighth_kelly():
    from epl import config
    from epl.predict import kelly

    assert config.KELLY_FRACTION == pytest.approx(0.125)
    # a 55% shot at even money is a 10% full-Kelly edge -> 1.25% at an eighth
    assert kelly(0.55, 1.0) == pytest.approx(0.10 * 0.125, abs=1e-9)
    assert kelly(0.40, 1.0) == 0.0, "no stake on a negative edge"
    assert kelly(float("nan"), 1.0) == 0.0


def test_betting_knobs_are_conservative():
    """Absolute thresholds, deliberately not compared against another sport's config, so
    retuning the college or NFL pipelines cannot fail this suite."""
    from epl import config

    assert config.SUP_EDGE_MIN >= 0.5, "handicap threshold must stay above what is provable"
    assert config.GOALS_EDGE_MIN >= 0.6
    assert config.KELLY_FRACTION <= 0.125
    assert config.MIN_GAMES >= 5
    # a sharp two-way football price is better than -110, so break-even is LOWER than the
    # American pipelines' 52.38 - which is a real difference, not a typo
    assert 50.0 < config.BREAK_EVEN < 52.4


def test_push_leg_is_returned_not_lost():
    from epl.predict import _ev

    # 50% win, 20% push, 30% loss at even money -> 0.5 - 0.3 = +0.2
    assert _ev(0.5, 0.2, 1.0) == pytest.approx(0.2)
    # ignoring the push would give 0.5 - 0.5 = 0.0, a materially different number
    assert _ev(0.5, 0.0, 1.0) == pytest.approx(0.0)


def test_empty_env_vars_fall_back_to_defaults():
    """An unset GitHub Actions repo variable arrives as an empty string. float("") raises, and
    the daily job would die on a variable nobody ever set."""
    import importlib

    for var in ("DEGEN_SUP_EDGE", "DEGEN_KELLY", "DEGEN_MIN_GAMES", "DEGEN_EPL_LEAGUE",
                "DEGEN_FIRST_SEASON", "DEGEN_DC_RHO"):
        os.environ[var] = ""
    try:
        from epl import config as cfg
        importlib.reload(cfg)
        assert cfg.SUP_EDGE_MIN == 0.60
        assert cfg.KELLY_FRACTION == 0.125
        assert cfg.MIN_GAMES == 5
        assert cfg.LEAGUE == "E0"
        assert cfg.FIRST_SEASON == 2005
        assert cfg.DC_RHO == -0.04
    finally:
        for var in ("DEGEN_SUP_EDGE", "DEGEN_KELLY", "DEGEN_MIN_GAMES", "DEGEN_EPL_LEAGUE",
                    "DEGEN_FIRST_SEASON", "DEGEN_DC_RHO"):
            os.environ.pop(var, None)
        importlib.reload(cfg)


def test_season_boundary_wraps_the_calendar_year():
    """A season runs August to May and is named for the year it starts in - the one calendar
    fact the American pipelines never have to handle."""
    from epl import config

    assert config.season_of(date(2024, 8, 16)) == 2024
    assert config.season_of(date(2025, 2, 1)) == 2024
    assert config.season_of(date(2025, 5, 25)) == 2024
    assert config.season_of(date(2025, 8, 15)) == 2025
    assert config.season_code(2024) == "2425"
    assert config.season_code(1999) == "9900"


# ---------------------------------------------------------------------------------
# Reporting honesty
# ---------------------------------------------------------------------------------
def test_segments_are_corrected_for_the_number_of_looks():
    """Across ~20 segments the chance one clears |z| >= 2 by luck is about 60%. The report has
    to say what bar a segment actually needs."""
    from epl.train import _apply_multiple_comparisons

    segs = [{"segment": f"s{i}", "vs_break_even_se": 2.1} for i in range(20)]
    out = _apply_multiple_comparisons(segs)
    assert out["comparisons"] == 20
    assert out["z_required"] > 2.5, "a 20-look family needs far more than 2 s.e."
    assert all(s["significant"] is False for s in segs)
    assert "no segment survives" in out["verdict"]

    strong = [{"segment": "real", "vs_break_even_se": 4.5}]
    out2 = _apply_multiple_comparisons(strong)
    assert strong[0]["significant"] is True
    assert "1 of 1" in out2["verdict"]


def test_segments_report_per_season_stability():
    """A real edge shows up in most seasons; a fluke is two bad years and four ordinary ones."""
    from epl.train import _seg

    df = pd.DataFrame({"right": ([True] * 120 + [False] * 80) * 2,
                       "season": [2020] * 200 + [2021] * 200})
    row = _seg(df, "test", min_n=100)
    assert row["seasons_measured"] == 2
    assert "season_cover_pct" in row
    assert row["n"] == 400


def test_probability_report_scores_the_distribution_not_just_the_point():
    """Two models can have identical supremacy MAE while disagreeing completely about how often
    matches are drawn. Log loss on 1X2 is the test that catches that."""
    from epl.train import _calibration

    p = pd.DataFrame({"p_home": [0.5] * 100, "p_draw": [0.25] * 100, "p_away": [0.25] * 100,
                      "outcome": [0] * 50 + [1] * 25 + [2] * 25})
    cal = _calibration(p)
    assert cal, "calibration table must not be empty"
    for row in cal:
        assert 0 <= row["actual_pct"] <= 100
        assert row["n"] >= 50


# ---------------------------------------------------------------------------------
# Isolation and integration
# ---------------------------------------------------------------------------------
def test_epl_module_does_not_import_other_sports():
    """Each sport is a self-contained package. The duplication is deliberate: these are
    independently scheduled jobs committing to main on their own crons, and the thing worth
    optimising is blast radius, not line count."""
    import ast

    offenders = []
    for path in (ROOT / "epl").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for n in names:
                if n.split(".")[0] in ("cfb", "nfl", "ncaab", "core"):
                    offenders.append(f"{path.name}: {n}")
    assert not offenders, f"cross-sport imports found: {offenders}"


def test_source_modules_read_config_at_call_time():
    """Binding paths at import would freeze them, so a DEGEN_DATA override - which this whole
    suite relies on - would be ignored by whichever module imported first, and the tests would
    read and write the real repo data while believing they were sandboxed."""
    import re

    for name in ("footballdata.py", "odds.py"):
        src = (ROOT / "epl" / "sources" / name).read_text()
        bad = re.findall(r"^from \.\.config import .*\b(GAMES|LINES|PICKS|SNAPSHOTS|STRENGTH)\b",
                         src, re.M)
        assert not bad, f"{name} binds a config path at import: {bad}"


def test_landing_page_lists_the_premier_league():
    from core import landing

    assert any(s["slug"] == "epl" for s in landing.SPORTS)


def test_the_epl_page_is_reachable_from_the_site_root(tmp_path):
    """core.landing must render an EPL card from published files alone, importing nothing from
    the epl package."""
    from core import landing

    docs = tmp_path / "docs"
    (docs / "epl").mkdir(parents=True)
    (docs / "epl" / "index.html").write_text("<html></html>")
    (docs / "epl" / "metrics.json").write_text(json.dumps({
        "updated": "2026-09-08", "sport": "epl",
        "spreads": {"all_games": {"n": 40, "units": 1.5, "win_pct": 52.5, "clv": 0.02}},
        "totals": {"all_games": {"n": 40, "units": -0.5, "win_pct": 48.0, "clv": -0.01}},
    }))
    (docs / "epl" / "picks.csv").write_text(
        "game_id,date,week,matchweek,total_strength,spread_strength\n"
        "2025_arsenal_chelsea,2099-01-01,5,5,play,pass\n")
    cards = landing.collect(docs, today="2026-09-08")
    epl = [c for c in cards if c["slug"] == "epl"]
    assert len(epl) == 1
    assert epl[0]["totals"]["n"] == 40
    assert epl[0]["board"]["games"] == 1


def test_site_renders_without_any_data():
    """A fresh clone with no picks must still produce a page rather than a traceback."""
    from epl import config, site

    config.ensure_dirs()
    site.build()
    html = (config.DOCS / "index.html").read_text()
    assert "Premier League" in html
    assert "No fixtures on the board" in html


def test_workflows_exist_and_are_wired():
    wf = ROOT / ".github" / "workflows"
    for name in ("epl-train.yml", "epl-predict.yml", "epl-grade.yml"):
        assert (wf / name).exists(), f"missing workflow {name}"
    predict = (wf / "epl-predict.yml").read_text()
    assert "python -m epl.predict" in predict
    assert "python -m epl.site" in predict
    assert "python -m core.landing" in predict, "the chooser must refresh after each sport"
    test_wf = (wf / "test.yml").read_text()
    assert "epl" in test_wf, "CI must run the EPL suite as its own job"
