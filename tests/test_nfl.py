"""Offline end-to-end test for the NFL pipeline.

No network. Builds synthetic seasons with realistic-looking lines, then runs
train -> predict -> grade -> site and checks the wiring plus the leak-free guarantee.

On top of the college pipeline's checks there are four NFL-specific guardrails, each one
pinning something that failed silently while this was being written:

* the canonical team map, which must resolve every code in the committed data and raise on
  anything else rather than inventing a franchise;
* the spread sign convention, flipped exactly once on the way out of nflverse;
* week normalisation across the 2021 change from a 17-week to an 18-week regular season;
* time-zone shift wrapped across the date line, so a trip to Australia is 6 hours and not 18.
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("dpnfl")
    os.environ["DEGEN_ROOT"] = str(root)
    os.environ["DEGEN_DATA"] = str(root / "data")
    os.environ["DEGEN_DOCS"] = str(root / "docs")
    os.environ.pop("DEGEN_NFL_DOCS", None)
    os.environ["DEGEN_FIRST_SEASON"] = "2018"
    import importlib
    from nfl import config
    importlib.reload(config)
    config.ensure_dirs()
    return config


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """This suite must not touch the network.

    predict.run() reaches for Kalshi, and without this the whole file waits on connection
    retries to a host CI may not be able to reach - turning an offline test into a slow,
    flaky one. Returning no markets exercises the same path a real outage would.
    """
    from nfl.sources import kalshi
    monkeypatch.setattr(kalshi, "fetch_markets", lambda *a, **k: [], raising=False)
    monkeypatch.setattr(kalshi, "list_series", lambda *a, **k: [], raising=False)


from nfl import config as _c  # noqa: E402
THIS = _c.season_of(_c.today_et())
# Six seasons, because the walk-forward pools six and an NFL season is only 272 games.
SEASONS = tuple(THIS - n for n in range(6, 0, -1))

TEAMS32 = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
           "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
           "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"]
STADIA = {t: s for t, s in zip(TEAMS32, [
    "PHO00", "ATL97", "BAL00", "BUF00", "CAR00", "CHI98", "CIN00", "CLE00", "DAL00", "DEN00",
    "DET00", "GNB00", "HOU00", "IND00", "JAX00", "KAN00", "LAX01", "LAX01", "VEG00", "MIA00",
    "MIN01", "BOS00", "NOR00", "NYC01", "NYC01", "PHI00", "PIT00", "SEA00", "SFO01", "TAM00",
    "NAS00", "WAS00"])}


def synth() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A synthetic league whose market is deliberately near-perfect.

    The line knows the true expectation plus a little noise, which is the honest analogue of
    the real NFL: the model should NOT beat it, and the test asserts that rather than hoping
    for the opposite.
    """
    rng = np.random.default_rng(7)
    power = {t: rng.normal(0, 6) for t in TEAMS32}
    scoring = {t: rng.normal(22.5, 3) for t in TEAMS32}
    qbs = {t: [f"{t} QB1", f"{t} QB2"] for t in TEAMS32}
    games, lines, gid = [], [], 0
    for s in SEASONS:
        first = date(s, 9, 8)
        for wk in range(1, _c.reg_weeks(s) + 1):
            day = first + timedelta(days=7 * (wk - 1))
            order = rng.permutation(TEAMS32)
            for i in range(0, len(order) - 1, 2):
                h, a = str(order[i]), str(order[i + 1])
                exp_m = power[h] - power[a] + 1.7
                exp_t = scoring[h] + scoring[a]
                margin = rng.normal(exp_m, 13)
                total = max(20, rng.normal(exp_t, 10))
                hp = int(round((total + margin) / 2))
                ap = int(round((total - margin) / 2))
                games.append(dict(
                    game_id=f"{s}_{wk:02d}_{a}_{h}", season=s, week=wk, season_type="REG",
                    playoff_round=0, date=day, tip_et=f"Sun Sep {wk}, 1:00 PM",
                    kickoff_utc=f"{day}T17:00:00+00:00", start_time_tbd=False, weekday="Sunday",
                    home_team=h, away_team=a, home_points=float(hp), away_points=float(ap),
                    total_points=float(hp + ap), home_margin=float(hp - ap),
                    neutral_site=False, div_game=bool(i % 4 == 0), home_div="AFC East",
                    away_div="NFC West", home_rest=7.0, away_rest=7.0 if i % 8 else 4.0,
                    roof="dome" if i % 5 == 0 else "outdoors", surface="grass",
                    temp=np.nan if i % 5 == 0 else 55.0, wind=np.nan if i % 5 == 0 else 8.0,
                    stadium_id=STADIA[h],
                    # a starter change every ~10 games, so h_qb_new is neither constant nor noise
                    home_qb=qbs[h][0] if (gid % 10) else qbs[h][1],
                    away_qb=qbs[a][0], completed=True))
                lines.append(dict(
                    game_id=f"{s}_{wk:02d}_{a}_{h}", season=s, week=wk, date=day,
                    home_team=h, away_team=a, provider="nflverse_close",
                    # spread_home is the CFBD convention: negative means home favoured
                    spread_home=round(-(exp_m + rng.normal(0, 1.2)) * 2) / 2,
                    spread_open=np.nan, total_open=np.nan,
                    total_line=round((exp_t + rng.normal(0, 1.6)) * 2) / 2,
                    home_ml=-150.0, away_ml=130.0, n_providers=1))
                gid += 1
    return pd.DataFrame(games), pd.DataFrame(lines)


# ---------------------------------------------------------------------------------
# Guardrail: the canonical team map
# ---------------------------------------------------------------------------------
def test_canonical_map_resolves_and_refuses():
    from nfl.teams import RELOCATIONS, TEAMS, UnknownTeam, canon
    assert len(TEAMS) == 32
    # every alias resolves to one of the 32 current franchises
    from nfl.teams import TEAM_ALIASES
    assert set(TEAM_ALIASES.values()) <= set(TEAMS)
    # the minefield spellings, all of them
    for raw, want in [("OAK", "LV"), ("LVR", "LV"), ("SD", "LAC"), ("SDG", "LAC"),
                      ("STL", "LA"), ("SL", "LA"), ("LAR", "LA"),
                      ("WSH", "WAS"), ("WFT", "WAS"), ("JAC", "JAX"),
                      ("GNB", "GB"), ("KAN", "KC"), ("NWE", "NE"), ("SFO", "SF")]:
        assert canon(raw) == want, f"{raw} -> {canon(raw)}, expected {want}"
    assert canon("  lv  ") == "LV"          # whitespace and case
    # relocations collapse, so a franchise's rating carries across the move
    for old, new in RELOCATIONS.items():
        assert canon(old) == canon(new)
    # and anything unknown RAISES rather than passing through - a silently wrong team code
    # splits or merges rating histories and nothing downstream can detect it
    for bad in ("XYZ", "", None, "Dallas Cowboys"):
        with pytest.raises(UnknownTeam):
            canon(bad)


def test_every_team_code_in_the_committed_data_is_mapped():
    """The committed schedule must contain no code the canonical map does not know."""
    from pathlib import Path
    from nfl.teams import UnknownTeam, canon
    games = Path(__file__).resolve().parent.parent / "data" / "nfl" / "games.csv"
    if not games.exists():
        pytest.skip("no committed NFL games yet")
    df = pd.read_csv(games, dtype={"game_id": str})
    unmapped = set()
    for code in set(df["home_team"]) | set(df["away_team"]):
        try:
            canon(code)
        except UnknownTeam:
            unmapped.add(code)
    assert not unmapped, f"unmapped team codes in games.csv: {sorted(unmapped)}"


def test_every_stadium_in_the_committed_data_is_mapped():
    """A new or renamed stadium must fail CI, not silently empty the travel features.

    At runtime an unmapped stadium yields NaN travel and the model copes. Here it is an error,
    so a new venue is noticed the week it appears rather than a season later.
    """
    from pathlib import Path
    from nfl.teams import STADIUMS
    games = Path(__file__).resolve().parent.parent / "data" / "nfl" / "games.csv"
    if not games.exists():
        pytest.skip("no committed NFL games yet")
    df = pd.read_csv(games, dtype={"game_id": str, "stadium_id": str})
    have = {str(s).strip().upper() for s in df["stadium_id"].dropna() if str(s).strip()}
    missing = sorted(have - set(STADIUMS))
    assert not missing, f"stadium_ids with no coordinates: {missing}"


# ---------------------------------------------------------------------------------
# Guardrail: the spread sign convention
# ---------------------------------------------------------------------------------
def test_spread_sign_convention():
    """nflverse quotes +3 = home favoured by 3; this pipeline uses -3 = home favoured by 3.

    The flip happens exactly once, on the way out of the source module. Getting it backwards
    would not raise anywhere - it would just make every spread pick the wrong side, and the
    ATS rate would come out near 48% instead of near 52%, which is subtle enough to miss.
    """
    from nfl.sources.nflverse import _to_spread_home
    assert _to_spread_home(3.0) == -3.0        # home favoured by 3
    assert _to_spread_home(-4.5) == 4.5        # home a 4.5-point underdog
    assert _to_spread_home(0) == 0
    assert _to_spread_home(None) != _to_spread_home(None)   # NaN

    # and the whole way through: a home favourite that wins by more than the number covers
    from nfl.features import _market
    row = _market({"exp_margin": 6.0, "exp_total": 44.0}, 44.0, -3.0)
    assert row["spread_vs_model"] == pytest.approx(3.0)     # model 6, market 3 -> +3 on home
    home_margin, spread_home = 7.0, -3.0
    assert home_margin + spread_home > 0                    # a 7-point win covers -3


def test_market_probability_is_devigged():
    from nfl.features import _devig_home
    p = _devig_home(-150, 130)
    assert 0.5 < p < 0.65
    # the two sides must sum to exactly 1 once the vig is divided out
    assert _devig_home(-150, 130) + _devig_home(130, -150) == pytest.approx(1.0)
    assert _devig_home(None, 130) != _devig_home(None, 130)  # NaN


# ---------------------------------------------------------------------------------
# Guardrail: week numbering across the 17 -> 18 week change
# ---------------------------------------------------------------------------------
def test_week_normalisation_survives_the_18_game_season():
    """Week 18 is the Wild Card round in 2019 and a regular-season game in 2022.

    Feeding the raw number to the model would teach it something false about the calendar, so
    the feature is a fraction of that season's regular season plus a separate playoff round.
    """
    from nfl.config import reg_weeks
    from nfl.features import week_features
    assert reg_weeks(2019) == 17 and reg_weeks(2022) == 18

    # the last regular-season game of either era is the same point in the season
    assert week_features(2019, 17, 0)[0] == pytest.approx(1.0)
    assert week_features(2022, 18, 0)[0] == pytest.approx(1.0)
    # a mid-season game lands in the same place either side of the change
    assert week_features(2019, 9, 0)[0] == pytest.approx(9 / 17)
    assert week_features(2022, 9, 0)[0] == pytest.approx(9 / 18)
    # playoffs sort strictly after any regular-season game, in round order, in both eras
    wc19, sb19 = week_features(2019, 18, 1)[0], week_features(2019, 21, 4)[0]
    wc22, sb22 = week_features(2022, 19, 1)[0], week_features(2022, 22, 4)[0]
    assert wc19 == wc22 and sb19 == sb22
    assert 1.0 < wc19 < sb19
    # and the early-season flag never fires in the playoffs
    assert week_features(2022, 1, 0)[1] == 1
    assert week_features(2022, 19, 1)[1] == 0


# ---------------------------------------------------------------------------------
# Guardrail: time zones wrap across the date line
# ---------------------------------------------------------------------------------
def test_timezone_shift_wraps_across_the_date_line():
    """A Los Angeles team playing in Melbourne shifts 6 hours, not 18.

    Plain offset subtraction gives 18, a value no other game in the data comes close to, on
    exactly the international games where the effect is meant to matter most.
    """
    from nfl.teams import tz_shift
    assert tz_shift("America/Los_Angeles", "Australia/Melbourne") == pytest.approx(-6)
    assert tz_shift("America/New_York", "Europe/London") == pytest.approx(5)
    assert tz_shift("America/Los_Angeles", "America/New_York") == pytest.approx(3)
    assert tz_shift("America/New_York", "America/Los_Angeles") == pytest.approx(-3)
    assert tz_shift("America/Chicago", "America/Chicago") == 0
    for tz in ("America/New_York", "Europe/London", "Australia/Melbourne"):
        for other in ("America/Los_Angeles", "Europe/Berlin", "America/Sao_Paulo"):
            assert -12 <= tz_shift(tz, other) <= 12


def test_travel_is_measured_from_where_the_team_actually_plays():
    """Relocations and neutral sites need no special case if the origin comes from the
    schedule: the Raiders travel from Oakland in 2019 and Las Vegas in 2021."""
    from nfl.sources.nflverse import home_stadiums, origin_stadium
    games = pd.DataFrame([
        {"season": 2019, "home_team": "LV", "neutral_site": False, "stadium_id": "OAK00"},
        {"season": 2019, "home_team": "LV", "neutral_site": False, "stadium_id": "OAK00"},
        {"season": 2021, "home_team": "LV", "neutral_site": False, "stadium_id": "VEG00"},
        # a "home" game in London must not become the franchise's home venue
        {"season": 2021, "home_team": "LV", "neutral_site": True, "stadium_id": "LON00"},
    ])
    homes = home_stadiums(games)
    assert origin_stadium(homes, 2019, "LV") == "OAK00"
    assert origin_stadium(homes, 2021, "LV") == "VEG00"
    # a franchise with no home game on record falls back to its default, not to nothing
    assert origin_stadium(homes, 2021, "GB") == "GNB00"


def test_neutral_site_makes_both_sides_travel():
    from nfl.features import _travel
    homes = {(2025, "SF"): "SFO01", (2025, "LA"): "LAX01"}
    normal = _travel(homes, 2025, "LA", "SF", "LAX01", neutral=False)
    assert normal["h_travel_km"] == 0.0 and normal["a_travel_km"] > 400
    away_in_melbourne = _travel(homes, 2025, "LA", "SF", "MEL00", neutral=True)
    assert away_in_melbourne["h_travel_km"] > 10000
    assert away_in_melbourne["a_travel_km"] > 10000
    assert abs(away_in_melbourne["a_tz_shift"]) <= 12
    # an unmapped stadium empties the travel columns instead of raising
    assert _travel(homes, 2025, "LA", "SF", "ZZZ99", neutral=False)["a_travel_km"] != \
        _travel(homes, 2025, "LA", "SF", "ZZZ99", neutral=False)["a_travel_km"]


def test_dome_games_get_stated_conditions():
    """nflverse leaves temp/wind blank indoors. Saying '68F, no wind' beats making the trees
    infer that a missing value means a roof."""
    from nfl.features import _venue_weather
    dome = _venue_weather("dome", "fieldturf", np.nan, np.nan)
    assert dome["is_indoor"] == 1 and dome["temp"] == 68.0 and dome["wind"] == 0.0
    assert dome["is_grass"] == 0
    closed = _venue_weather("closed", "grass", np.nan, np.nan)
    assert closed["is_indoor"] == 1 and closed["is_grass"] == 1
    out = _venue_weather("outdoors", "grass ", 41.0, 17.0)
    assert out["is_indoor"] == 0 and out["wind"] == 17.0 and out["is_grass"] == 1
    # unknown roof (a game not yet assigned one) must not read as indoors
    assert _venue_weather("", "grass", np.nan, np.nan)["is_indoor"] == 0


# ---------------------------------------------------------------------------------
# Core model behaviour
# ---------------------------------------------------------------------------------
def test_leak_free(env):
    """Rebuilding from a chronological PREFIX of the schedule must reproduce the same rows.

    The cut has to be a real point in time, not an arbitrary row index. `build()` sorts by
    (date, game_id), so slicing the raw frame mid-day leaves a partial day whose surviving
    games get re-ordered against the full build - which fails the comparison for a reason that
    has nothing to do with leakage. Cutting at a season boundary asks the question actually
    worth asking: does a feature on game N depend on anything after game N?
    """
    from nfl.features import build
    g, ln = synth()
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    cut_season = SEASONS[len(SEASONS) // 2]
    prefix = g[g["season"] < cut_season]
    assert 0 < len(prefix) < len(g)

    full, _, _ = build(g, lines=ln)
    half, _, _ = build(prefix, lines=ln)
    cols = ["exp_margin", "exp_total", "h_margin", "a_margin", "h_off", "a_def",
            "h_qb_new", "h_qb_starts", "a_qb_starts", "h_form_margin", "h_rest"]
    pd.testing.assert_frame_equal(full.iloc[: len(half)][cols].reset_index(drop=True),
                                  half[cols].reset_index(drop=True), atol=1e-9)
    # and the prefix really is a prefix: same games, same order
    assert list(full["game_id"][: len(half)]) == list(half["game_id"])


def test_warmup_season_advances_ratings_without_emitting_rows(env):
    """A warm-up season must teach the ratings and then stay out of the training set.

    Without it every team enters week 1 of the first training season rated identically - the
    real 2010 week 1 had ONE distinct `h_margin` across all 16 games - and the whole first
    season runs on ratings that started from zero.
    """
    from nfl.features import build
    g, ln = synth()
    first_train = SEASONS[1]                      # treat SEASONS[0] as warm-up

    cold, _, _ = build(g, lines=ln, first_train_season=SEASONS[0])
    warm, _, _ = build(g, lines=ln, first_train_season=first_train)

    # the warm-up season contributes no training rows
    assert set(warm["season"]) == set(SEASONS[1:])
    assert len(warm) < len(cold)

    # ...but its results are in the ratings: week 1 of the first training season is no longer
    # a league of identical teams
    w1 = warm[(warm["season"] == first_train) & (warm["week"] == 1)]
    assert len(w1) > 1
    assert w1["h_margin"].nunique() == len(w1), "every team still rated identically"
    assert w1["h_margin"].abs().max() > 0

    # and the cold build's first week is exactly the degenerate case being fixed
    c1 = cold[(cold["season"] == SEASONS[0]) & (cold["week"] == 1)]
    assert c1["h_margin"].nunique() == 1

    # the rows that DO survive are unchanged by the warm-up cut being drawn elsewhere:
    # same games, same order
    overlap = cold[cold["season"] >= first_train]
    assert list(overlap["game_id"]) == list(warm["game_id"])


def test_no_crowd_season_is_flagged_and_loses_home_field(env):
    """2020 was played without crowds and home-field collapsed to +0.14 points.

    Keeping the season is right - it is 269 games of real football - but applying a normal
    home edge to it is not: it would push every 2020 home team's rating down by an advantage
    that did not exist.
    """
    from nfl.features import hfa_suppressed
    from nfl.ratings import RatingBook

    assert 2020 in env.NO_CROWD_SEASONS
    assert hfa_suppressed(2020, False) is True        # no crowd
    assert hfa_suppressed(2019, True) is True         # genuine neutral site
    assert hfa_suppressed(2019, False) is False
    assert hfa_suppressed(2021, False) is False       # crowds came back

    # the two causes stay separate as features even though both suppress home field
    b = RatingBook()
    assert b.expect(2020, "KC", "BUF", hfa_suppressed(2020, False))["exp_margin"] == 0.0
    assert b.expect(2021, "KC", "BUF", hfa_suppressed(2021, False))["exp_margin"] > 0

    from nfl.features import build
    g, ln = synth()
    tr, _, _ = build(g, lines=ln)
    if 2020 in set(tr["season"]):
        assert (tr.loc[tr["season"] == 2020, "no_crowd"] == 1).all()
        assert (tr.loc[tr["season"] != 2020, "no_crowd"] == 0).all()
        # a no-crowd game gets no home edge in the rating expectation
        wk1 = tr[(tr["season"] == 2020) & (tr["week"] == 1)]
        assert (wk1["exp_margin"].abs() < 1e-9).all()


def test_qb_tracker_is_chronological(env):
    from nfl.features import QBTracker
    q = QBTracker()
    # first sighting is unknown, not "new" - otherwise all 32 week-1 starters read as changes
    is_new, starts = q.look("KC", "Mahomes")
    assert is_new != is_new and starts == 0
    q.update("KC", "Mahomes")
    assert q.look("KC", "Mahomes") == (0.0, 1.0)
    q.update("KC", "Mahomes")
    assert q.look("KC", "Mahomes") == (0.0, 2.0)
    # a different starter is flagged, and carries his own (zero) start count
    assert q.look("KC", "Backup") == (1.0, 0.0)
    q.update("KC", "Backup")
    # ...and when the starter returns he is flagged too, with his history intact
    assert q.look("KC", "Mahomes") == (1.0, 2.0)
    # a blank quarterback is unknown, never a change
    assert all(v != v for v in q.look("KC", ""))


def test_ratings_sane(env):
    """A team that wins every game by 20 should end up rated well above average, and NFL
    ratings must regress harder between seasons than the college ones do."""
    from nfl.ratings import SEASON_CARRY, RatingBook
    # Asserted as an absolute range, not against cfb.ratings. NFL rosters churn harder than
    # college ones so this must stay well below college's ~0.72 - but importing that constant
    # to say so would mean a tuning change in the college pipeline failing the NFL suite,
    # which is exactly the coupling this repo keeps the sports separate to avoid.
    assert 0.45 <= SEASON_CARRY <= 0.65
    b = RatingBook()
    d = date(2024, 9, 8)
    for i in range(17):
        b.update(2024, "Good", f"Foe{i}", 31, 11, d + timedelta(days=7 * i))
    good = b.get(2024, "Good").margin
    assert good > 8
    assert b.get(2024, "Foe0").margin < 0
    assert b.get(2025, "Good").margin == pytest.approx(good * SEASON_CARRY)


def test_home_field_is_nfl_sized(env):
    """NFL home-field is ~1.5-2 points, not college's 2.5-3. Overstating it biases every
    home side on every game."""
    from nfl.ratings import HFA, RatingBook
    assert 1.0 <= HFA <= 2.2
    b = RatingBook()
    e = b.expect(2025, "KC", "BUF")
    assert e["exp_margin"] == pytest.approx(HFA)
    assert b.expect(2025, "KC", "BUF", neutral=True)["exp_margin"] == 0.0


def test_stake_sizing_is_eighth_kelly(env):
    from nfl.predict import american_payout, kelly
    # Eighth Kelly, stated absolutely. Kelly sizing assumes you know your edge; against the
    # NFL close you do not, so this is half the fraction the college pipeline uses.
    assert env.KELLY_FRACTION == pytest.approx(0.125)
    assert abs(american_payout(-110) - 0.909) < 0.01
    assert kelly(0.50, 0.909) == 0.0
    assert kelly(0.60, 0.909) > 0
    # half the fraction, half the stake, on the same edge
    from nfl import predict as nfl_predict
    full = (0.60 * 1.909 - 1) / 0.909
    assert kelly(0.60, 0.909) == pytest.approx(full * 0.125)


def test_epa_opponent_adjustment(env):
    """The adjustment has to move a rating toward what the schedule justifies.

    The fixture needs a CONNECTED schedule graph. If the team that plays only elite defences
    and the team that plays only weak ones share no opponent, the two halves of the league are
    separate components and no adjustment can compare across them - the ratings would be
    equal, which is correct and useless. So a filler team plays every defence and establishes
    which are strong.
    """
    from nfl.sources.nflverse import season_epa

    def row(team, opp, epa, wk):
        return dict(season=2024, week=wk, season_type="REG", team=team, opponent_team=opp,
                    attempts=30, sacks_suffered=2, carries=28,
                    passing_epa=epa, rushing_epa=0.0)

    strong, weak = ["BAL", "PIT"], ["CAR", "NYG"]
    rows = []
    for wk in range(1, 9):
        # FILLER plays every defence and does badly against the strong ones: that is what
        # makes them strong, and it is what connects the two halves of the schedule.
        for d in strong:
            rows += [row("SF", d, -12.0, wk), row(d, "SF", 0.0, wk)]
        for d in weak:
            rows += [row("SF", d, 12.0, wk), row(d, "SF", 0.0, wk)]
        # ARI and ATL post IDENTICAL raw offence against opposite halves of the league
        rows += [row("ARI", strong[wk % 2], 0.0, wk), row(strong[wk % 2], "ARI", 0.0, wk)]
        rows += [row("ATL", weak[wk % 2], 0.0, wk), row(weak[wk % 2], "ATL", 0.0, wk)]

    out = season_epa(2024, pd.DataFrame(rows)).set_index("team")
    # the fixture is built so the two raw offences are identical...
    assert out.loc["ARI", "off_epa_play"] == pytest.approx(out.loc["ATL", "off_epa_play"])
    # ...and the strong defences really do allow less
    assert out.loc["BAL", "def_epa_play"] < out.loc["CAR", "def_epa_play"]
    # ...so the adjustment must separate them, crediting the harder schedule
    assert out.loc["ARI", "off_epa_adj"] > out.loc["ATL", "off_epa_adj"]
    # defence is EPA ALLOWED, so lower stays better after adjustment too
    assert out.loc["BAL", "def_epa_adj"] < out.loc["CAR", "def_epa_adj"]
    assert set(out.columns) >= {"off_epa_adj", "def_epa_adj", "net_epa_adj", "plays"}
    # a season with no data must return an empty frame, not raise
    assert season_epa(1999, pd.DataFrame()).empty


def test_roster_continuity_uses_the_gsis_crosswalk(env):
    """Joining snaps to rosters on pfr_id looks like it works and drops nearly every offensive
    lineman, because roster pfr_id is ~30% null and the nulls are concentrated there. The
    crosswalk through players.csv is what makes the number mean anything."""
    from nfl.sources.nflverse import season_continuity
    snaps = pd.DataFrame([
        # a lineman with no pfr_id on the roster, and a back who has one
        dict(game_type="REG", team="KC", pfr_player_id="P_LINE", offense_snaps=700,
             defense_snaps=0),
        dict(game_type="REG", team="KC", pfr_player_id="P_BACK", offense_snaps=300,
             defense_snaps=0),
        dict(game_type="REG", team="KC", pfr_player_id="P_GONE", offense_snaps=0,
             defense_snaps=500),
    ])
    roster = pd.DataFrame([
        dict(team="KC", gsis_id="G_LINE", pfr_id=None),
        dict(team="KC", gsis_id="G_BACK", pfr_id="P_BACK"),
    ])
    xw = {"P_LINE": "G_LINE", "P_BACK": "G_BACK", "P_GONE": "G_GONE"}
    out = season_continuity(2025, snaps, roster, xw).set_index("team")
    # both offensive players are back: 100%, not the 30% a pfr_id join would report
    assert out.loc["KC", "cont_off"] == pytest.approx(1.0)
    assert out.loc["KC", "cont_def"] == pytest.approx(0.0)


def test_pipeline(env, monkeypatch):
    from nfl import grade, predict, site, train
    from nfl.sources import nflverse, odds

    games, lines = synth()
    env.ensure_dirs()
    games.to_csv(env.GAMES, index=False)
    lines.to_csv(env.LINES, index=False)
    monkeypatch.setattr(nflverse, "update_games", lambda *a, **k: nflverse.load_games())
    monkeypatch.setattr(nflverse, "update_lines", lambda *a, **k: nflverse.load_lines())
    monkeypatch.setattr(nflverse, "update_epa", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(nflverse, "update_continuity", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(odds, "snapshot", lambda *a, **k: pd.DataFrame())

    train.main(["--no-fetch"])
    meta = json.loads((env.MODEL_DIR / "meta.json").read_text())
    assert {"total_market", "margin_market"} <= set(meta["models"])
    assert meta["sport"] == "nfl"
    ev = meta["eval"]["margin_market"]
    # the market in this fixture is deliberately near-perfect, so the model should NOT beat it
    assert ev["mae_market_baseline"] <= ev["mae_model"] * 1.2
    # walk-forward must never hold out the in-progress season
    assert THIS not in ev["test_seasons"]
    assert ev["n_test_total"] > 400
    # shrink is capped and can't be pushed to 1.0 by a lucky slice
    assert 0.0 <= ev["shrink"] <= train.SHRINK_CAP
    assert "ats_stderr" in ev and "beats_market" in ev
    buckets = ev["ats_by_disagreement"]
    assert buckets and all(b["n"] >= train.MIN_BUCKET for b in buckets)
    assert all("cover_pct" in b and "roi_pct" in b for b in buckets)
    assert 50.0 <= ev["break_even_pct"] <= 53.0
    soft = ev["market_softness"]
    # the NFL-specific segmentation, not the college one
    assert "by_tv_window" in soft and "by_rest" in soft and "by_divisional" in soft
    for rows in soft.values():
        for r in rows:
            assert r["n"] >= 60 and "vs_break_even_se" in r
    # line-movement features have no historical counterpart in nflverse and must be absent
    assert "spread_move" not in meta["market_features"]
    assert "ml_prob_home" in meta["market_features"]

    # Week 1 of the CURRENT season: no games played yet, exactly the real week-1 situation
    upcoming = games.iloc[:16].copy()
    upcoming["game_id"] = ["u" + g for g in upcoming["game_id"]]
    upcoming["completed"] = False
    upcoming["season"] = THIS
    upcoming["week"] = 1
    upcoming["date"] = env.today_et() + timedelta(days=2)
    for c in ("home_points", "away_points", "total_points", "home_margin"):
        upcoming[c] = np.nan
    up_lines = lines.iloc[:16].copy()
    up_lines["game_id"] = list(upcoming["game_id"])
    up_lines["season"] = THIS
    up_lines["week"] = 1
    pd.concat([games, upcoming]).to_csv(env.GAMES, index=False)
    pd.concat([lines, up_lines]).to_csv(env.LINES, index=False)

    out = predict.run()
    assert len(out) == 16
    for c in ("total_pred", "total_edge", "total_disagree", "total_p_win", "total_stake",
              "margin_pred", "margin_edge", "margin_disagree", "spread_pick", "spread_stake",
              "h_qb_new", "rest_diff", "a_travel_km"):
        assert c in out.columns, c
    assert out["total_p_win"].between(0, 1).all()
    assert (out["total_stake"] >= 0).all()
    # the published side must agree with the model's own sign, even when the rounded edge
    # lands on 0.0 - which the NFL's small shrink makes common
    assert ((out["total_pick"] == "Over") == (out["total_side_val"] > 0)).all()
    assert ((out["spread_side"] == out["home_team"]) == (out["margin_side_val"] > 0)).all()
    # week 1 => no current-season games played => every pick flagged thin, none staked as play
    assert out["thin_data"].all()
    assert set(out["total_strength"]) <= {"pass", "thin"}
    assert set(out["spread_strength"]) <= {"pass", "thin"}

    # finalise those games so grading has scores
    finished = upcoming.copy()
    finished["home_points"] = 27.0
    finished["away_points"] = 20.0
    finished["total_points"] = 47.0
    finished["home_margin"] = 7.0
    finished["completed"] = True
    pd.concat([games, finished]).to_csv(env.GAMES, index=False)

    done = grade.grade()
    assert len(done) == 16
    assert set(done["total_result"]) <= {"win", "loss", "push"}
    assert set(done["spread_result"]) <= {"win", "loss", "push"}
    # grading must reproduce the side predict published, not re-derive it from a rounded column
    home_covered = (done["home_margin"] + done["spread_home"]) > 0
    took_home = done["spread_side"] == done["home_team"]
    expected = np.where(took_home == home_covered, "win", "loss")
    pushed = (done["home_margin"] + done["spread_home"]) == 0
    assert (done.loc[~pushed, "spread_result"] == expected[~pushed]).all()

    m = grade.metrics(done)
    assert m["totals"]["all_games"]["n"] == 16
    assert m["sport"] == "nfl"
    env.METRICS.write_text(json.dumps(m, default=str))

    site.build()
    html = (env.DOCS / "index.html").read_text()
    assert "NFL" in html
    assert "nan" not in html.lower().replace("finance", ""), \
        "empty fields must not render as 'nan'"


def test_odds_matcher():
    from nfl.sources.odds import build_matcher
    m = build_matcher()
    cases = {
        "Kansas City Chiefs": "KC", "Los Angeles Rams": "LA",
        "Los Angeles Chargers": "LAC", "San Francisco 49ers": "SF",
        "Washington Commanders": "WAS", "Washington Football Team": "WAS",
        "Oakland Raiders": "LV", "Las Vegas Raiders": "LV",
        "New York Giants": "NYG", "New York Jets": "NYJ",
        # bare codes in any feed's spelling go through the canonical map
        "LAR": "LA", "WSH": "WAS", "JAC": "JAX", "OAK": "LV", "SD": "LAC",
        # nicknames
        "Chiefs": "KC", "49ers": "SF", "Bucs": "TB",
    }
    for raw, want in cases.items():
        assert m(raw) == want, f"{raw} -> {m(raw)}, expected {want}"
    assert not m.unmatched
    # unlike canon(), the price-feed matcher reports a miss instead of raising: a bad match
    # here costs one game's price, not a franchise's rating history
    assert m("Toledo Mud Hens") == "Toledo Mud Hens"
    assert m.unmatched == {"Toledo Mud Hens"}


# ---------------------------------------------------------------------------------
# Kalshi. The NFL series tickers could not be confirmed against the live API when this was
# written, so these fixtures use the NCAAF payload shape with NFL wordings. They pin the
# parsing and the guards; they cannot pin the ticker, which is why every Kalshi path in
# predict.py degrades to "no exchange prices" rather than failing.
# ---------------------------------------------------------------------------------
KALSHI_SAMPLE = [
    {"event_ticker": "KXNFLGAME-26SEP13DALNYG", "ticker": "KXNFLGAME-26SEP13DALNYG-DAL",
     "title": "Dallas Cowboys wins", "yes_sub_title": "Dallas Cowboys",
     "market_type": "binary", "yes_bid_dollars": "0.0800", "yes_ask_dollars": "0.8100",
     "yes_ask_size_fp": "131.00", "yes_bid_size_fp": "538.00", "volume_fp": "0.00",
     "open_interest_fp": "0.00", "last_price_dollars": "0.0000",
     "rules_primary": "If Dallas Cowboys wins the Dallas Cowboys vs New York Giants NFL "
                      "football game originally scheduled for Sep 13, 2026, then the market "
                      "resolves to Yes."},
    {"event_ticker": "KXNFLGAME-26SEP13DALNYG", "ticker": "KXNFLGAME-26SEP13DALNYG-NYG",
     "title": "New York Giants wins", "yes_sub_title": "New York Giants",
     "market_type": "binary", "yes_bid_dollars": "0.4200", "yes_ask_dollars": "0.4400",
     "yes_ask_size_fp": "318.00", "yes_bid_size_fp": "143.00", "volume_fp": "766.09",
     "open_interest_fp": "696.28", "last_price_dollars": "0.4300",
     "rules_primary": "If New York Giants wins the Dallas Cowboys vs New York Giants NFL "
                      "football game originally scheduled for Sep 13, 2026, then the market "
                      "resolves to Yes."},
]


def test_kalshi_parses_multiword_team_names(env):
    """Regression: a permissive sport pattern swallowed team names.

    With a lazy home-team group, "the Dallas vs New York Giants NFL football game" parsed the
    home team as "New", because "York Giants NFL football" matched a permissive sport phrase.
    Every multi-word club would have been truncated and nothing about the output would have
    looked wrong.
    """
    from datetime import date as _date
    from nfl.sources import kalshi
    rows = [kalshi.parse_market(m) for m in KALSHI_SAMPLE]
    assert all(r for r in rows)
    dal = rows[0]
    assert dal["kalshi_team"] == "Dallas Cowboys"
    assert dal["away_raw"] == "Dallas Cowboys" and dal["home_raw"] == "New York Giants"
    assert dal["date"] == _date(2026, 9, 13)
    assert dal["yes_ask"] == 0.81
    # 73c spread, zero volume -> must NOT be considered tradeable
    assert dal["tradeable"] is False
    nyg = rows[1]
    assert nyg["home_raw"] == "New York Giants"
    assert round(nyg["quote_spread"], 2) == 0.02
    assert nyg["tradeable"] is True          # 2c spread, real size and volume

    # the sport phrase varies by series; all three wordings must parse identically
    for phrase in ("NFL football", "pro football", "football"):
        hit = kalshi.RULES_RE.search(
            f"If Green Bay Packers wins the Green Bay Packers vs Tampa Bay Buccaneers "
            f"{phrase} game originally scheduled for Sep 13, 2026, then...")
        assert hit and hit.group("a") == "Green Bay Packers"
        assert hit.group("b") == "Tampa Bay Buccaneers"


def test_kalshi_board_matches_names(env, monkeypatch):
    from nfl.sources import kalshi, odds
    monkeypatch.setattr(kalshi, "fetch_markets", lambda *a, **k: KALSHI_SAMPLE)
    df = kalshi.moneyline_board(odds.build_matcher())
    assert len(df) == 2
    assert set(df["team"]) == {"DAL", "NYG"}
    assert set(df["home_team"]) == {"NYG"} and set(df["away_team"]) == {"DAL"}
    assert int(df["tradeable"].sum()) == 1


def test_kalshi_fee_and_ev(env):
    from nfl.sources import kalshi
    # published taker formula: 0.07 * P * (1-P); peaks at 1.75c on a 50c contract
    assert abs(kalshi.fee(0.50, 0.07) - 0.0175) < 1e-9
    assert kalshi.fee(0.90, 0.07) < kalshi.fee(0.50, 0.07)
    ev, roi = kalshi.contract_ev(0.50, 0.50)
    assert ev < 0                                  # no edge -> negative once the fee is paid
    ev, roi = kalshi.contract_ev(0.58, 0.50)
    assert ev > 0.05 and roi > 0.1                 # a real 8-point edge survives the fee
    assert all(v != v for v in kalshi.contract_ev(None, 0.5))


KALSHI_TOTAL = [
    {"event_ticker": "KXNFLTOTAL-26SEP13DALNYG", "ticker": "KXNFLTOTAL-26SEP13DALNYG-52",
     "floor_strike": 51.5, "strike_type": "greater", "market_type": "binary",
     "yes_bid_dollars": "0.0900", "yes_ask_dollars": "0.1700", "yes_ask_size_fp": "150.00",
     "volume_fp": "0.00", "open_interest_fp": "0.00",
     "rules_primary": "If the teams collectively score more than 51.5 points in the Dallas "
                      "Cowboys vs New York Giants NFL football game originally scheduled for "
                      "Sep 13, 2026, then the market resolves to Yes."},
    {"event_ticker": "KXNFLTOTAL-26SEP13DALNYG", "ticker": "KXNFLTOTAL-26SEP13DALNYG-45",
     "floor_strike": 44.5, "strike_type": "greater", "market_type": "binary",
     "yes_bid_dollars": "0.4600", "yes_ask_dollars": "0.4900", "yes_ask_size_fp": "150.00",
     "volume_fp": "0.00", "open_interest_fp": "0.00",
     "rules_primary": "If the teams collectively score more than 44.5 points in the Dallas "
                      "Cowboys vs New York Giants NFL football game originally scheduled for "
                      "Sep 13, 2026, then the market resolves to Yes."},
]

KALSHI_SPREAD = [
    {"event_ticker": "KXNFLSPREAD-26SEP13DALNYG", "ticker": "KXNFLSPREAD-26SEP13DALNYG-NYG9",
     "floor_strike": 8.5, "strike_type": "greater", "market_type": "binary",
     "yes_bid_dollars": "0.0700", "yes_ask_dollars": "0.9000", "yes_ask_size_fp": "701.00",
     "volume_fp": "0.00", "open_interest_fp": "0.00",
     "rules_primary": "If New York Giants wins by more than 8.5 points in the Dallas Cowboys "
                      "vs New York Giants NFL football game originally scheduled for "
                      "Sep 13, 2026, then the market resolves to Yes."},
    {"event_ticker": "KXNFLSPREAD-26SEP13DALNYG", "ticker": "KXNFLSPREAD-26SEP13DALNYG-NYG5",
     "floor_strike": 4.5, "strike_type": "greater", "market_type": "binary",
     "yes_bid_dollars": "0.0700", "yes_ask_dollars": "0.1700", "yes_ask_size_fp": "150.00",
     "volume_fp": "0.00", "open_interest_fp": "0.00",
     "rules_primary": "If New York Giants wins by more than 4.5 points in the Dallas Cowboys "
                      "vs New York Giants NFL football game originally scheduled for "
                      "Sep 13, 2026, then the market resolves to Yes."},
]


def test_kalshi_ladder_parse(env):
    from datetime import date as _date
    from nfl.sources import kalshi
    t = kalshi.parse_ladder(KALSHI_TOTAL[0], "total")
    assert t["strike"] == 51.5 and t["date"] == _date(2026, 9, 13)
    assert t["away_raw"] == "Dallas Cowboys" and t["home_raw"] == "New York Giants"
    assert t["yes_ask"] == 0.17
    s = kalshi.parse_ladder(KALSHI_SPREAD[0], "spread")
    assert s["strike"] == 8.5 and s["team_raw"] == "New York Giants"
    # the matchup is phrased away-first: "Dallas vs New York Giants" => the Giants host
    assert s["away_raw"] == "Dallas Cowboys" and s["home_raw"] == "New York Giants"


def test_kalshi_monotonicity_detects_incoherent_ladder(env, monkeypatch):
    """P(win by >8.5) cannot exceed P(win by >4.5)."""
    from nfl.sources import kalshi
    monkeypatch.setattr(kalshi, "fetch_markets", lambda *a, **k: KALSHI_SPREAD)
    breaks = kalshi.monotonicity_breaks(kalshi.ladder_board("spread"))
    assert len(breaks) == 1
    assert breaks[0]["lower_strike"] == 4.5 and breaks[0]["higher_strike"] == 8.5
    assert breaks[0]["higher_ask"] > breaks[0]["lower_ask"]
    # a coherent ladder must produce no breaks
    monkeypatch.setattr(kalshi, "fetch_markets", lambda *a, **k: KALSHI_TOTAL)
    assert kalshi.monotonicity_breaks(kalshi.ladder_board("total")) == []


def test_ladder_guards_reject_thin_and_tails(env, monkeypatch):
    """Week 1 (no games played) must produce zero exchange picks, and rungs far from the book
    number are refused. The NFL guards are strictly tighter than the college ones."""
    from datetime import date as _date
    from nfl import config as C, predict
    from nfl.sources import kalshi, odds
    # Absolute bars, not relative to the college pipeline's: the NFL exchange board is more
    # liquid and more efficiently priced, so a small edge is likelier to be model error.
    assert C.KALSHI_MIN_EV >= 0.07
    assert C.KALSHI_MAX_BOOK_GAP <= 6.0
    assert 0.20 < C.KALSHI_PROB_MIN < C.KALSHI_PROB_MAX < 0.80

    day = _date(2026, 9, 13)
    out = pd.DataFrame([{
        "date": day, "home_team": "NYG", "away_team": "DAL",
        "total_pred": 47.0, "total_sigma": 13.0, "total_line": 47.0,
        "margin_pred": 3.0, "margin_sigma": 12.8, "spread_home": -3.0,
        "p_home_win": 0.58, "p_away_win": 0.42,
        "thin_data": True,          # <- week 1
    }])
    ladder = pd.DataFrame([{
        "kind": "total", "ticker": "T-52", "event_ticker": "E", "strike": 51.5,
        "team": None, "home_team": "NYG", "away_team": "DAL", "date": day,
        "yes_bid": 0.09, "yes_ask": 0.17, "quote_spread": 0.02, "ask_size": 500.0,
        "volume": 900.0, "tradeable": True,
    }])
    monkeypatch.setattr(kalshi, "ladder_board",
                        lambda kind, matcher=None: ladder if kind == "total" else ladder.iloc[0:0])
    monkeypatch.setattr(odds, "build_matcher", lambda teams=None: (lambda n: n))
    monkeypatch.setattr(predict, "nfl_teams", lambda g, s: {"NYG", "DAL"})

    res = predict._price_ladders(out.copy(), pd.DataFrame())
    assert res["kt_pick"].isna().all(), "thin-data games must never produce an exchange pick"

    out2 = out.copy()
    out2.loc[0, "thin_data"] = False
    res2 = predict._price_ladders(out2.copy(), pd.DataFrame())
    if res2["kt_pick"].notna().any():
        assert res2["kt_ev"].iloc[0] >= C.KALSHI_MIN_EV
        assert C.KALSHI_PROB_MIN <= res2["kt_prob"].iloc[0] <= C.KALSHI_PROB_MAX
        assert abs(res2["kt_strike"].iloc[0] - 47.0) <= C.KALSHI_MAX_BOOK_GAP

    # A rung far from the book number must be refused outright.
    far = ladder.copy()
    far.loc[0, "strike"] = 75.0
    monkeypatch.setattr(kalshi, "ladder_board",
                        lambda kind, matcher=None: far if kind == "total" else far.iloc[0:0])
    res3 = predict._price_ladders(out2.copy(), pd.DataFrame())
    assert res3["kt_pick"].isna().all()
    assert res3["kt_rungs"].fillna(0).iloc[0] == 0


def test_source_modules_read_config_at_call_time(env):
    """Data paths must be read as `config.NAME`, not imported by value.

    `from ..config import GAMES` freezes the path at import. Whichever module happened to be
    imported before a `DEGEN_DATA` override took effect would then keep pointing at the real
    repo, and this suite silently trained on committed production data while believing it was
    sandboxed. Nothing failed - the numbers were just quietly wrong.
    """
    import re
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / "nfl" / "sources"
    frozen = {"GAMES", "LINES", "EPA_PRIOR", "CONTINUITY", "PICKS", "RESULTS", "METRICS",
              "MODEL_DIR", "SNAPSHOTS", "DATA", "DOCS", "ODDS_API_KEY", "FIRST_SEASON"}
    for path in sorted(src.glob("*.py")):
        for line in path.read_text().splitlines():
            m = re.match(r"\s*from \.\.config import (.+)", line)
            if not m:
                continue
            names = {n.strip().strip("()") for n in m.group(1).split(",")}
            bad = names & frozen
            assert not bad, (f"{path.name} imports {sorted(bad)} by value from config; "
                             "reference them as config.NAME so DEGEN_* overrides apply")

    # and the sandbox really is in force for this run
    from nfl.sources import nflverse
    assert str(env.DATA) in str(nflverse.config.GAMES)
    assert "degenpredicts/data/nfl/games.csv" not in str(nflverse.config.GAMES)


def test_nfl_betting_knobs_are_conservative(env):
    """Every knob that decides how much we bet, pinned to an absolute floor.

    An earlier version asserted these against `cfb.config` - "stricter than college". That
    read well and was wrong for this repo: it made a tuning change in the live college
    pipeline fail the NFL suite, which is the cross-sport breakage the modules are kept
    separate to prevent. The values below encode the same intent without the coupling.
    """
    assert env.TOTAL_EDGE_MIN >= 6.0
    assert env.SPREAD_EDGE_MIN >= 5.0
    assert env.KELLY_FRACTION <= 0.125
    assert env.DEFAULT_SHRINK <= 0.25
    assert env.MIN_GAMES >= 3          # weeks 1-3 flagged and unstaked
    from nfl.train import SHRINK_CAP
    assert SHRINK_CAP <= 0.45
    from nfl.ratings import HFA, SEASON_CARRY
    assert 1.0 <= HFA <= 2.2           # NFL home field, not college's 2.5-3
    assert 0.45 <= SEASON_CARRY <= 0.65


def test_nfl_module_does_not_import_other_sports(env):
    """nfl/ must not reach into cfb/, ncaab/ or the shared core/ HTTP layer.

    The point of keeping the sports separate is blast radius: a change made for one sport
    must not be able to break another. That guarantee is only real if it is enforced, so this
    walks the package and fails on any cross-sport import.
    """
    import re
    from pathlib import Path
    pkg = Path(__file__).resolve().parent.parent / "nfl"
    offenders = []
    for path in sorted(pkg.rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.match(r"\s*(from|import)\s+(cfb|ncaab|core)\b", line):
                offenders.append(f"{path.relative_to(pkg.parent)}:{i}: {line.strip()}")
    assert not offenders, "nfl/ imports another sport or the shared core: " + "; ".join(offenders)
