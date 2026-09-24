"""Offline tests for the NBA pipeline. No network, no API key.

Structured like ``tests/test_nhl.py`` and ``tests/test_nfl.py``: thresholds are asserted as
ABSOLUTE values rather than against another sport's config, so retuning another pipeline cannot
fail this suite. CI runs one job per sport, so a red check names the sport that broke.

The guardrails below each pin something that went wrong, or nearly did, while this was built:

* "available" once meant "logged minutes", and garbage time hands minutes to the end of the
  bench only in blowouts - boosted trees read the result off it. Availability now comes from
  rotation players at their usual minutes, and a test moves only the bench's minutes and
  asserts nothing in the row moves;
* the SBR-derived archive prints every spread as a positive number, so its spreads are never
  imported and the closing lines' sign is checked against the favourite;
* ESPN names the All-Star game's sides "EAST" and "WEST" and unfilled knockout slots "TBD", so
  unknown codes are refused rather than rated;
* NBA slices hold thousands of games, where a plain 50% cover rate is "significantly" below
  break-even - so the significance test is one-sided and a test pins it;
* four other boards share one Odds API key, so the NBA board must default to the free source.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
NBA_MODULES = ("nba.teams", "nba.odds_math", "nba.linear", "nba.ratings", "nba.players",
               "nba.features", "nba.sources.hoopr", "nba.sources.espn", "nba.sources.odds",
               "nba.sources.history", "nba.train", "nba.predict", "nba.grade", "nba.site")


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Sandbox every path before any nba module is re-read.

    The modules read ``config.NAME`` at call time precisely so this works; a module that bound
    its paths at import would quietly read and write the real repo data instead.
    """
    monkeypatch.setenv("DEGEN_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("DEGEN_DOCS", str(tmp_path / "docs"))
    monkeypatch.setenv("DEGEN_ROOT", str(ROOT))
    for var in ("DEGEN_NBA_DOCS", "ODDS_API_KEY", "DEGEN_NBA_ODDS"):
        monkeypatch.delenv(var, raising=False)
    import importlib
    import sys
    from nba import config as cfg
    importlib.reload(cfg)
    for mod in NBA_MODULES:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    # Offline: every HTTP call is a failure unless a test hands it a payload.
    from nba import http
    monkeypatch.setattr(http, "get", lambda *a, **k: None)
    monkeypatch.setattr(http, "get_json", lambda *a, **k: None)
    for mod in ("nba.sources.hoopr", "nba.sources.espn", "nba.sources.odds", "nba.sources.history"):
        m = sys.modules.get(mod) or importlib.import_module(mod)
        for name in ("get", "get_json"):
            if hasattr(m, name):
                monkeypatch.setattr(m, name, lambda *a, **k: None)
    cfg.ensure_dirs()
    yield cfg


def _rendered_text(html: str) -> str:
    body = re.sub(r"<script\b.*?</script>", "", html, flags=re.S | re.I)
    return re.sub(r"<style\b.*?</style>", "", body, flags=re.S | re.I).lower()


# ---------------------------------------------------------------------------------
# Teams: the guardrail that matters most
# ---------------------------------------------------------------------------------
def test_canonical_map_resolves_and_refuses():
    from nba.teams import TEAMS, UnknownTeam, canon, from_name, is_known
    assert len(TEAMS) == 30
    for raw, want in [("GS", "GSW"), ("NY", "NYK"), ("NO", "NOP"), ("SA", "SAS"), ("UTAH", "UTA"),
                      ("WSH", "WAS"), ("NJ", "BKN"), ("SEA", "OKC"), (" gsw ", "GSW"), ("PHO", "PHX")]:
        assert canon(raw) == want, f"{raw} -> {canon(raw)}"
    for bad in ("EAST", "WEST", "TBD", "CHK", "SHQ", "STARS", "USA", None):
        with pytest.raises(UnknownTeam):
            canon(bad)
    assert not is_known("WORLD")
    for name, want in [("Los Angeles Clippers", "LAC"), ("LA Clippers", "LAC"),
                       ("Los Angeles Lakers", "LAL"), ("Philadelphia 76ers", "PHI"),
                       ("Portland Trail Blazers", "POR"), ("Seattle SuperSonics", "OKC"),
                       ("New Jersey Nets", "BKN"), ("Charlotte Bobcats", "CHA")]:
        assert from_name(name) == want
    with pytest.raises(UnknownTeam):
        from_name("Los Angeles")          # two clubs: refused, never guessed


def test_every_team_and_venue_in_the_committed_data_is_mapped():
    """A silently wrong code splits or merges a franchise's rating history; an unmapped venue
    silently zeroes every travel feature for the games played there."""
    from nba.teams import CITIES, TEAMS
    path = ROOT / "data" / "nba" / "games.csv"
    if not path.exists():
        pytest.skip("no committed NBA games")
    g = pd.read_csv(path, usecols=["home_team", "away_team", "venue_city"])
    assert not (set(g["home_team"]) | set(g["away_team"])) - set(TEAMS)
    cities = set(g["venue_city"].dropna())
    assert not cities - set(CITIES), f"unmapped venues: {cities - set(CITIES)}"


# ---------------------------------------------------------------------------------
# Prices and probabilities
# ---------------------------------------------------------------------------------
def test_american_and_decimal_convert_both_ways():
    from nba.odds_math import american_to_decimal, decimal_to_american, devig_two, ev
    assert american_to_decimal(-110) == pytest.approx(1.9091, abs=1e-4)
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert decimal_to_american(1.8) == pytest.approx(-125)
    assert np.isnan(american_to_decimal(50))
    assert devig_two(1.91, 1.91) == pytest.approx(0.5)
    assert ev(0.45, 0.10, 2.0) == pytest.approx(0.0)      # a push returns the stake


def test_books_are_aggregated_in_decimal_on_the_modal_line():
    """-115 and +105 have an American median of -5 - a "20x payout"."""
    from nba.sources.odds import parse_event

    def book(title, sp, hp, ap, tl=224.5):
        return {"title": title, "markets": [
            {"key": "spreads", "outcomes": [{"name": "Boston Celtics", "price": hp, "point": sp},
                                            {"name": "New York Knicks", "price": ap, "point": -sp}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": 1.91, "point": tl},
                                           {"name": "Under", "price": 1.91, "point": tl}]},
            {"key": "h2h", "outcomes": [{"name": "Boston Celtics", "price": 1.4},
                                        {"name": "New York Knicks", "price": 3.1}]}]}

    ev = {"id": "x", "commence_time": "2026-11-03T00:30:00Z", "home_team": "Boston Celtics",
          "away_team": "New York Knicks",
          "bookmakers": [book("A", -6.5, 1.87, 2.05), book("B", -6.5, 2.05, 1.87),
                         book("C", -7.0, 1.95, 1.95, 225.5)]}
    r = parse_event(ev)
    assert r["spread_home"] == -6.5 and r["total_line"] == 224.5     # modal, not the odd one out
    assert 1.8 < r["spread_home_dec"] < 2.1
    assert r["p_spread_home"] == pytest.approx(0.5, abs=0.01)
    assert r["home_team"] == "BOS" and r["date"] == date(2026, 11, 2)   # the Eastern date
    assert parse_event({**ev, "home_team": "Seattle Storm"}) is None


def test_margins_are_integers_that_are_never_zero_and_whole_lines_push():
    from nba.odds_math import cover, margin_from_prob, over_under, win_prob
    h, p, a = cover([0.0], 13.0, [0.0])
    assert p[0] == 0.0 and h[0] == pytest.approx(0.5), "no ties in basketball: a pick'em cannot push"
    h, p, a = cover([7.0], 13.0, [-7.0])
    assert p[0] > 0.02 and h[0] + p[0] + a[0] == pytest.approx(1.0)
    h, p, a = cover([7.0], 13.0, [-6.5])
    assert p[0] == 0.0 and h[0] > 0.5
    o, pu, u = over_under([224.0], 18.0, [224.5])
    assert pu[0] == 0.0 and o[0] + u[0] == pytest.approx(1.0) and o[0] < 0.5
    o, pu, u = over_under([224.0], 18.0, [224.0])
    assert pu[0] > 0.015
    assert win_prob([0.0], 12.0)[0] == pytest.approx(0.5)
    for mu in (-9.0, -2.5, 3.0, 11.0):
        assert margin_from_prob(win_prob([mu], 12.0), 12.0)[0] == pytest.approx(mu, abs=0.05)


def test_spread_sign_convention():
    """-6.5 means home favoured by 6.5, so the market's expected home margin is +6.5. Getting it
    backwards raises nothing - it picks the wrong side of every game."""
    from nba.features import market_view
    v = market_view(pd.DataFrame({"spread_home": [-6.5, 3.0], "total_line": [224.5, 230.0]}))
    assert list(v["mkt_margin"]) == [6.5, -3.0]
    assert list(v["mkt_total"]) == [224.5, 230.0]


# ---------------------------------------------------------------------------------
# Sources, against payloads shaped like the real ones
# ---------------------------------------------------------------------------------
def _sched(rows):
    base = {"season_type": 2, "status_type_completed": True, "neutral_site": False,
            "status_period": 4, "venue_address_city": "Boston"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_hoopr_schedule_parse():
    from nba.sources.hoopr import parse_schedule
    df = _sched([
        {"game_id": 401, "season": 2026, "date": "2025-10-22T23:30Z", "game_date": "2025-10-22",
         "home_abbreviation": "BOS", "away_abbreviation": "NY", "home_score": 110, "away_score": 99},
        {"game_id": 402, "season": 2026, "date": "2026-02-16T01:00Z", "game_date": "2026-02-15",
         "home_abbreviation": "EAST", "away_abbreviation": "WEST", "home_score": 150, "away_score": 160},
        {"game_id": 403, "season": 2027, "date": "2026-10-21T02:00Z", "game_date": "2026-10-20",
         "home_abbreviation": "GS", "away_abbreviation": "UTAH", "home_score": 0, "away_score": 0,
         "status_type_completed": False, "status_period": 0, "venue_address_city": "San Francisco"},
        {"game_id": 404, "season": 2026, "date": "2025-10-01T23:00Z", "game_date": "2025-10-01",
         "season_type": 1, "home_abbreviation": "BOS", "away_abbreviation": "NY",
         "home_score": 100, "away_score": 90},
        {"game_id": 405, "season": 2026, "date": "2026-04-15T23:00Z", "game_date": "2026-04-15",
         "season_type": 5, "home_abbreviation": "SA", "away_abbreviation": "WSH",
         "home_score": 101, "away_score": 103, "status_period": 5},
    ])
    g = parse_schedule(df).set_index("game_id")
    assert "402" not in g.index, "the All-Star game is not a game between two clubs"
    assert "404" not in g.index, "preseason is dropped"
    assert g.loc["401", "season"] == 2025, "hoopR's 2026 is the season that STARTS in 2025"
    assert g.loc["401", "home_team"] == "BOS" and g.loc["401", "away_team"] == "NYK"
    assert str(g.loc["403", "date"]) == "2026-10-20", "the Eastern date, not the UTC one"
    assert not bool(g.loc["403", "completed"]) and np.isnan(g.loc["403", "home_points"])
    assert g.loc["405", "game_type"] == "I" and g.loc["405", "periods"] == 5
    assert (g.loc["403", "home_team"], g.loc["403", "away_team"]) == ("GSW", "UTA")


def test_possessions_use_player_turnovers():
    """Some seasons' total_turnovers count every team turnover twice (30 for a side that
    committed 15). Possessions are FGA + 0.44 FTA - offensive rebounds + turnovers."""
    from nba.sources.hoopr import parse_team_box
    box = pd.DataFrame([{"game_id": 1, "team_home_away": "home", "field_goals_attempted": 80,
                         "free_throws_attempted": 25, "offensive_rebounds": 10, "turnovers": 14,
                         "total_turnovers": 28}])
    assert parse_team_box(box)["poss"].iloc[0] == pytest.approx(80 + 11 - 10 + 14)


def test_player_box_keeps_only_minutes_played():
    from nba.sources.hoopr import parse_player_box
    pb = pd.DataFrame([
        {"game_id": 1, "athlete_id": 7, "athlete_display_name": "A Star", "team_abbreviation": "BOS",
         "minutes": 36, "points": 30, "field_goals_made": 11, "field_goals_attempted": 20,
         "free_throws_made": 6, "free_throws_attempted": 7, "offensive_rebounds": 1,
         "defensive_rebounds": 7, "assists": 5, "steals": 1, "blocks": 1, "fouls": 2, "turnovers": 3},
        {"game_id": 1, "athlete_id": 8, "athlete_display_name": "Hurt Guy", "team_abbreviation": "BOS",
         "minutes": np.nan},
        {"game_id": 1, "athlete_id": 9, "athlete_display_name": "Game One", "team_abbreviation": "SHQ",
         "minutes": 20}])
    rows, names = parse_player_box(pb)
    assert list(rows["athlete_id"]) == [7]
    r = rows.iloc[0]
    assert r["gs"] == pytest.approx(30 + 4.4 - 14 - 0.4 + 0.7 + 2.1 + 1 + 3.5 + 0.7 - 0.8 - 3, abs=0.05)
    assert r["os"] < r["gs"] + 5
    assert set(names["name"]) == {"A Star"}, "a name comes with minutes played for a real club"


def test_a_bad_refresh_never_erases_history(env):
    from nba.sources.hoopr import GAME_COLS, _merge, _save_players, players_path
    old = pd.DataFrame([{"game_id": "1", "season": 2025, "date": "2025-11-01", "completed": True,
                         "home_points": 110.0, "away_points": 100.0, "poss": 99.0}]).reindex(columns=GAME_COLS)
    new = pd.DataFrame([{"game_id": "1", "season": 2025, "date": "2025-11-01", "completed": False,
                         "home_points": np.nan, "away_points": np.nan}]).reindex(columns=GAME_COLS)
    m = _merge(old, new).iloc[0]
    assert bool(m["completed"]) and m["home_points"] == 110 and m["poss"] == 99
    rows = pd.DataFrame({"game_id": ["1", "1"], "team": ["BOS", "BOS"], "athlete_id": [1, 2],
                         "minutes": [30.0, 20.0], "gs": [10.0, 5.0], "os": [8.0, 4.0]})
    assert _save_players(2025, rows)
    assert not _save_players(2025, rows.head(1)), "a smaller refresh must not replace the cache"
    assert len(pd.read_csv(players_path(2025))) == 2


def test_espn_scoreboard_odds_parse_both_shapes():
    """Written without reaching ESPN: both known shapes of its odds object must come through,
    and the flat shape's sign must come from who is favoured, never from `spread` alone."""
    from nba.sources.espn import parse_scoreboard

    def event(gid, home, away, odds, state="pre"):
        return {"id": gid, "date": "2026-11-03T00:30Z", "shortName": f"{away} @ {home}",
                "competitions": [{"date": "2026-11-03T00:30Z", "status": {"type": {"state": state}},
                                  "competitors": [{"homeAway": "home", "team": {"abbreviation": home}},
                                                  {"homeAway": "away", "team": {"abbreviation": away}}],
                                  "odds": odds}]}

    nested = [{"provider": {"name": "DraftKings"},
               "pointSpread": {"home": {"open": {"line": "-6", "odds": "-110"},
                                        "close": {"line": "-6.5", "odds": "-108"}},
                               "away": {"close": {"line": "+6.5", "odds": "-112"}}},
               "total": {"over": {"open": {"line": "o223.5", "odds": "-110"},
                                  "close": {"line": "o224.5", "odds": "-110"}},
                         "under": {"close": {"line": "u224.5", "odds": "-110"}}},
               "moneyline": {"home": {"close": {"odds": "-250"}}, "away": {"close": {"odds": "+205"}}}}]
    flat_home_dog = [{"provider": {"name": "ESPN BET"}, "details": "NY -3.5", "overUnder": 219.0,
                      "spread": 3.5, "homeTeamOdds": {"favorite": False, "moneyLine": 140},
                      "awayTeamOdds": {"favorite": True, "moneyLine": -165}}]
    s = parse_scoreboard({"events": [event("1", "BOS", "NY", nested), event("2", "BKN", "NY", flat_home_dog),
                                     event("3", "EAST", "WEST", nested), event("4", "LAL", "GS", [], "in")]})
    s = s.set_index("game_id")
    assert "3" not in s.index
    a, b = s.loc["1"], s.loc["2"]
    assert (a["spread_home"], a["total_line"], a["open_spread_home"], a["open_total"]) == (-6.5, 224.5, -6.0, 223.5)
    assert a["home_dec"] == pytest.approx(1.4) and a["spread_home_dec"] == pytest.approx(1 + 100 / 108)
    assert b["spread_home"] == 3.5, "the home side is the underdog here: +3.5, whatever 'spread' says"
    assert b["total_line"] == 219.0 and b["away_dec"] == pytest.approx(1 + 100 / 165)
    assert s.loc["4", "state"] == "in"


def test_injury_report_parses_by_status_and_skips_the_unknown():
    from nba.sources.espn import parse_injuries, status
    j = {"injuries": [
        {"displayName": "Boston Celtics", "injuries": [
            {"status": "Out", "athlete": {"id": "11", "displayName": "Star One"},
             "details": {"type": "Knee", "side": "Right"}},
            {"status": "Day-To-Day", "athlete": {"id": "12", "displayName": "Guard Two"}},
            {"status": "Something New", "athlete": {"id": "13", "displayName": "Mystery"}}]},
        {"displayName": "Seattle Storm", "injuries": [
            {"status": "Out", "athlete": {"id": "99", "displayName": "Wrong League"}}]},
        {"displayName": "Utah Jazz", "injuries": [
            {"type": {"description": "questionable"}, "athlete": {"id": "21", "displayName": "Big Three"}}]}]}
    r = parse_injuries(j).set_index("athlete_id")
    assert set(r.index) == {11, 12, 21}
    assert r.loc[11, "status"] == "out" and r.loc[11, "p_out"] == 1.0 and r.loc[11, "team"] == "BOS"
    assert r.loc[12, "status"] == "day-to-day" and 0 < r.loc[12, "p_out"] < 1
    assert r.loc[21, "status"] == "questionable" and r.loc[21, "team"] == "UTA"
    assert status("INJURY_STATUS_OUT") == "out" and status("GTD") == "questionable"


def test_a_failed_evening_read_keeps_the_mornings_report(env):
    from nba.sources import espn
    today = env.today_et()
    stamp = datetime.combine(today, time(15, 0), env.ET).astimezone().isoformat()
    pd.DataFrame([{"pulled_at": stamp, "team": "BOS", "athlete_id": 11, "name": "Star One",
                   "status": "questionable", "p_out": 0.5, "detail": ""}]).to_csv(env.INJURIES, index=False)
    got = espn.latest_injuries(None)
    assert list(got["athlete_id"]) == [11]
    fresh = pd.DataFrame(columns=espn.INJURY_COLS)
    assert espn.latest_injuries(fresh).empty, "a fresh empty report is news: nobody is hurt"


def test_closing_history_refuses_unsigned_or_flipped_rows(env):
    import io
    from nba.sources import history
    games = pd.DataFrame({"game_id": ["1", "2", "3", "4"], "season": 2023,
                          "date": ["2023-11-01", "2023-11-01", "2023-11-02", "2023-11-02"],
                          "home_team": ["BOS", "LAL", "NYK", "DEN"], "away_team": ["MIA", "GSW", "PHI", "UTA"]})
    close = pd.DataFrame({"game_id": [1, 2, 3], "home_point": [-5.5, 3.0, -2.0],
                          "home_favorite": [True, False, False], "over_under": [220.5, 230.0, 225.0],
                          "n_books": [12, 12, 12], "match_method": ["exact", "exact", "exact"]})
    buf = io.BytesIO()
    try:
        close.to_parquet(buf)
    except ImportError:
        pytest.skip("pyarrow is only needed for the one-time line import")
    got = history.oddsapi_close(buf.getvalue(), games).set_index("game_id")
    assert list(got.index) == ["1", "2"], "a home 'favourite' getting points has its sign wrong"
    assert got.loc["1", "spread_home"] == -5.5


def test_sbr_moneylines_are_read_and_spreads_are_not(env, tmp_path):
    import sqlite3
    from nba.sources.history import sbr_moneylines
    db = tmp_path / "odds.sqlite"
    con = sqlite3.connect(db)
    pd.DataFrame({"Date": ["2019-11-01"], "Home": ["Boston Celtics"], "Away": ["Miami Heat"],
                  "OU": [215.0], "Spread": [5.0], "ML_Home": [-220], "ML_Away": [180]}).to_sql(
        "odds_2019-20_new", con, index=False)
    pd.DataFrame({"Date": ["2019-20-1101"], "Home": ["Boston Celtics"], "Away": ["Miami Heat"],
                  "ML_Home": [999], "ML_Away": [999]}).to_sql("odds_2019-20", con, index=False)
    con.close()
    games = pd.DataFrame({"game_id": ["9"], "season": [2019], "date": ["2019-11-01"],
                          "home_team": ["BOS"], "away_team": ["MIA"]})
    got = sbr_moneylines(db.read_bytes(), games)
    assert len(got) == 1 and "spread_home" not in got
    assert got["p_home"].iloc[0] == pytest.approx((220 / 320) / (220 / 320 + 100 / 280), abs=1e-4)


# ---------------------------------------------------------------------------------
# The player layer and the replay
# ---------------------------------------------------------------------------------
TEAMS8 = ["BOS", "NYK", "MIA", "PHI", "LAL", "GSW", "DEN", "UTA"]
CITY8 = {"BOS": "Boston", "NYK": "New York", "MIA": "Miami", "PHI": "Philadelphia",
         "LAL": "Los Angeles", "GSW": "San Francisco", "DEN": "Denver", "UTA": "Salt Lake City"}


def _league(seasons, teams=TEAMS8, days=40, seed=3, lines_from=None, absent=0.08):
    """A small league with rosters, minutes, box scores, absences and (optionally) a market."""
    rng = np.random.default_rng(seed)
    # player 0 is a genuine star, so his nights off are worth measuring
    quality = {t: [14.0 + rng.normal(0, 1)] + [rng.normal(0, 6) for _ in range(8)] for t in teams}
    mins = [36, 34, 32, 30, 28, 24, 20, 16, 10]
    games, players, lines, gid = [], [], [], 0
    for s in seasons:
        day = date(s, 10, 24)
        for _ in range(days):
            order = list(rng.permutation(teams))
            for i in range(0, len(order), 2):
                h, a = order[i], order[i + 1]
                gid += 1
                out = {}
                strength = {}
                for t in (h, a):
                    out[t] = {k for k in range(9) if rng.random() < absent}
                    strength[t] = sum(quality[t][k] * mins[k] / 240 for k in range(9) if k not in out[t])
                exp_m = 2.5 + strength[h] - strength[a]
                poss = rng.normal(99, 4)
                hp = round(poss * 1.12 + exp_m / 2 + rng.normal(0, 8))
                ap = round(hp - exp_m - rng.normal(0, 12))
                if hp == ap:
                    hp += 1
                g_id = f"4{s}{gid:05d}"
                games.append({"game_id": g_id, "season": s, "game_type": "R", "date": str(day),
                              "tip_utc": f"{day}T23:30:00Z", "home_team": h, "away_team": a,
                              "home_points": float(hp), "away_points": float(ap), "completed": True,
                              "neutral_site": False, "periods": 4.0, "venue_city": CITY8[h],
                              "home_poss": poss, "away_poss": poss, "poss": poss, "source": "test"})
                for t in (h, a):
                    for k in range(9):
                        if k in out[t]:
                            continue
                        m = mins[k] + rng.normal(0, 2)
                        players.append({"game_id": g_id, "team": t,
                                        "athlete_id": 1000 * (teams.index(t) + 1) + k,
                                        "minutes": round(m, 1), "gs": round(m / 36 * (10 + quality[t][k]) + rng.normal(0, 4), 1),
                                        "os": round(m / 36 * (8 + quality[t][k] / 2) + rng.normal(0, 3), 1)})
                if lines_from is not None and s >= lines_from:
                    mkt = exp_m + rng.normal(0, 0.8)
                    p = 1 / (1 + np.exp(-mkt / 7.5))
                    lines.append({"game_id": g_id, "season": s, "date": str(day), "home_team": h,
                                  "away_team": a, "source": "oddsapi_close", "is_closing": True,
                                  "spread_home": -round(mkt * 2) / 2, "total_line": round((hp + ap) / 2 + 110 + rng.normal(0, 3)),
                                  "n_books": 10, "ml_source": "sbr_close", "home_dec": 1 / (p * 1.02),
                                  "away_dec": 1 / ((1 - p) * 1.02), "p_home": p})
            day += timedelta(days=1)
    return pd.DataFrame(games), pd.DataFrame(players), pd.DataFrame(lines)


def test_leak_free(env):
    """A row for game N is identical whether or not games after N exist."""
    from nba.features import build
    g, p, _ = _league([2021, 2022])
    full, _, _, _ = build(g, p, first_train_season=2022, today=date(2030, 1, 1))
    cut = g.iloc[: len(g) // 2 + 7]
    part, _, _, _ = build(cut, p[p["game_id"].isin(cut["game_id"])], first_train_season=2022,
                          today=date(2030, 1, 1))
    cols = ["game_id", "e_margin", "e_total", "h_d_net", "a_d_off", "h_miss_top", "h_rest",
            "a_b2b", "h_games", "altitude"]
    merged = part[cols].merge(full[cols], on="game_id", suffixes=("", "_full"))
    assert len(merged) == len(part) > 0
    for c in cols[1:]:
        assert np.allclose(merged[c].astype(float), merged[f"{c}_full"].astype(float),
                           equal_nan=True), c


def test_garbage_time_cannot_move_a_pregame_row(env):
    """The leak that made boosted trees look 0.15 points better than they were: who got into the
    game says how the game went. Changing ONLY this game's score and bench minutes - adding a
    garbage-time cameo, trimming a starter - must not move a single feature of its own row."""
    from nba.features import BASE_FEATURES, build
    g, p, _ = _league([2021, 2022])
    base, _, _, _ = build(g, p, first_train_season=2022, today=date(2030, 1, 1))
    last = g.iloc[-1]
    g2, p2 = g.copy(), p.copy()
    g2.loc[g2.index[-1], ["home_points", "away_points"]] = [150.0, 90.0]
    mine = p2["game_id"] == last["game_id"]
    p2.loc[mine & (p2["athlete_id"] % 1000 == 0), "minutes"] = 20.0        # the star sat the 4th
    cameo = {"game_id": last["game_id"], "team": last["home_team"], "athlete_id": 999999,
             "minutes": 6.0, "gs": 4.0, "os": 3.0}
    p2 = pd.concat([p2, pd.DataFrame([cameo])], ignore_index=True)
    moved, _, _, _ = build(g2, p2, first_train_season=2022, today=date(2030, 1, 1))
    a = base[base["game_id"] == last["game_id"]][BASE_FEATURES].astype(float).values
    b = moved[moved["game_id"] == last["game_id"]][BASE_FEATURES].astype(float).values
    assert np.allclose(a, b, equal_nan=True)


def test_a_missing_star_counts_and_a_returning_one_counts_back():
    from nba.players import PlayerBook
    b = PlayerBook()
    for i in range(30):
        b.update("BOS", [(1, 36, 30.0, 26.0), (2, 30, 10.0, 8.0), (3, 8, 2.0, 1.0)], date(2025, 11, 1))
    assert b.is_rotation(1) and not b.is_rotation(3)
    full = b.delta("BOS", b.played_target([1, 2, 3]))
    without = b.delta("BOS", b.played_target([2, 3]))
    assert without["net"] < full["net"] - 3 and without["top"] > 3
    assert b.delta("BOS", b.played_target([1, 2]))["net"] == pytest.approx(full["net"]), \
        "a deep-bench player is not a rotation player: his absence is not news"
    for _ in range(12):                      # the star misses three weeks
        b.update("BOS", [(2, 34, 12.0, 9.0), (3, 20, 5.0, 3.0)], date(2025, 12, 1))
    back = b.delta("BOS", b.played_target([1, 2, 3]))
    assert back["net"] > full["net"] + 2, "his return lifts the team above its recent self"


def test_board_availability_follows_the_report_the_roster_and_the_trade_wire():
    from nba.players import PlayerBook
    b = PlayerBook()
    for _ in range(20):
        b.update("BOS", [(1, 36, 30.0, 26.0), (2, 32, 15.0, 12.0), (4, 28, 12.0, 9.0)], date(2025, 11, 1))
        b.update("MIA", [(5, 34, 20.0, 15.0)], date(2025, 11, 1))
    b.update("MIA", [(4, 30, 12.0, 9.0), (5, 34, 20.0, 15.0)], date(2025, 11, 2))   # 4 was traded
    t, notes = b.board_target("BOS", report={1: "out", 2: "questionable"})
    assert 1 not in t and 4 not in t, "ruled out, and traded away"
    assert t[2] == pytest.approx(b.exp_min[2] / 48 * 0.5)
    assert [n["status"] for n in notes] == ["out", "questionable"]
    assert b.pending_points(notes) == pytest.approx(notes[1]["impact"]), "only the unresolved one"
    t2, _ = b.board_target("MIA", report={}, roster={5, 4, 77})
    assert 4 in t2 and 5 in t2 and 77 not in t2
    t3, _ = b.board_target("BOS", report={}, roster={2})
    assert set(t3) == {2}, "off the current roster means gone"


def test_last_aprils_rest_days_are_not_this_octobers_absences():
    from nba.players import PlayerBook
    b = PlayerBook()
    b.rollover(2025)
    for _ in range(20):
        b.update("BOS", [(1, 36, 30.0, 26.0), (2, 30, 12.0, 9.0)], date(2026, 3, 1))
    for _ in range(4):                      # the star rests through the last week of the season
        b.update("BOS", [(2, 34, 12.0, 9.0)], date(2026, 4, 10))
    _, notes = b.board_target("BOS")
    assert [n["status"] for n in notes] == ["absent"], "mid-season, three missed games is news"
    b.rollover(2026)
    t, notes = b.board_target("BOS")
    assert 1 in t and not notes, "on opening night nobody has missed a game of this season"


def test_bubble_games_have_no_home_court(env):
    from nba.features import build
    g, p, _ = _league([2019, 2020], days=10)
    g.loc[g.index[-5:], "date"] = "2020-08-05"          # the Orlando restart
    rows, _, _, _ = build(g, p, first_train_season=2019, today=date(2030, 1, 1))
    bub = rows[pd.to_datetime(rows["date"]).dt.date == date(2020, 8, 5)]
    assert len(bub) == 5 and (bub["H"] == 0).all() and (bub["no_crowd"] == 1).all()


def test_unplayed_past_games_do_not_tire_anyone():
    from nba.features import schedule_features
    g = pd.DataFrame({"game_id": ["1", "2", "3"], "season": 2025,
                      "date": ["2025-11-01", "2025-11-02", "2025-11-03"],
                      "home_team": ["BOS", "BOS", "BOS"], "away_team": ["NYK", "MIA", "DEN"],
                      "completed": [True, False, True], "venue_city": ["Boston"] * 3})
    s = schedule_features(g, today=date(2025, 12, 1)).set_index("game_id")
    assert "2" not in s.index
    assert s.loc["3", "h_rest"] == 2 and s.loc["3", "h_b2b"] == 0
    assert s.loc["3", "a_km"] == 0.0                 # DEN's first game: no trip measured


# ---------------------------------------------------------------------------------
# Models and knobs
# ---------------------------------------------------------------------------------
def test_ridge_round_trips_through_json():
    from nba.linear import Ridge
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=2000), "x2": rng.normal(size=2000)})
    df["y"] = 3 + 2 * df["x1"] - df["x2"] + rng.normal(0, 0.1, 2000)
    m = Ridge(["x1", "x2"], alpha=1.0).fit(df, "y")
    assert m.predict(pd.DataFrame({"x1": [0.0], "x2": [0.0]}))[0] == pytest.approx(3, abs=0.05)
    m2 = Ridge.from_dict(json.loads(json.dumps(m.to_dict())))
    assert np.allclose(m2.predict(df.head(20)), m.predict(df.head(20)))


def test_significance_is_one_sided():
    """An NBA slice holds thousands of games; at that size 50% is 'significantly' below the 52.4%
    break-even. That is the vig, not a finding, and must never be reported as one."""
    from nba.train import _significance
    rows = [{"segment": "big and ordinary", "vs_break_even_se": -4.5},
            {"segment": "real", "vs_break_even_se": 4.5}, {"segment": "noise", "vs_break_even_se": 1.0}]
    sig = _significance(rows)
    assert sig["significant_segments"] == ["real"]


def test_betting_knobs_are_conservative(env):
    """Pinned to absolute floors, never to another sport's config."""
    assert env.SPREAD_EDGE_MIN >= 3.0
    assert env.TOTAL_EDGE_MIN >= 5.0
    assert env.ML_EV_MIN >= 0.10, "the 5-10% moneyline bucket lost in the backtest"
    assert env.KELLY_FRACTION <= 0.125
    assert env.MIN_GAMES >= 5
    assert env.SHRINK_CAP <= 0.5
    assert env.REQUIRE_NEWS


def test_the_shared_odds_quota_is_opt_in(env):
    """Four other boards share one Odds API key; an NBA board that spent it by default would
    take their prices down mid-month."""
    assert env.ODDS_SOURCE == "espn"
    wf = (ROOT / ".github" / "workflows" / "nba-predict.yml").read_text()
    assert "DEGEN_NBA_ODDS: ${{ vars.DEGEN_NBA_ODDS }}" in wf


def test_empty_env_vars_fall_back_to_defaults(monkeypatch):
    """An unset GitHub Actions repo variable arrives as an empty string, not as absent."""
    import importlib
    from nba import config
    for var in ("DEGEN_NBA_SPREAD_EDGE", "DEGEN_NBA_TOTAL_EDGE", "DEGEN_NBA_ML_EV", "DEGEN_KELLY",
                "DEGEN_MIN_GAMES", "DEGEN_BOARD_DAYS", "DEGEN_FIRST_SEASON", "DEGEN_NBA_ODDS",
                "DEGEN_NBA_WAIT_POINTS"):
        monkeypatch.setenv(var, "")
    importlib.reload(config)
    assert config.SPREAD_EDGE_MIN == 3.0 and config.ML_EV_MIN == 0.10 and config.KELLY_FRACTION == 0.125
    assert config.MIN_GAMES == 5 and config.FIRST_SEASON == 2007 and config.ODDS_SOURCE == "espn"


# ---------------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------------
def test_pipeline(env, monkeypatch):
    from nba import grade, predict, site, train
    from nba.sources import espn, hoopr, odds

    this = env.season_of(env.today_et())
    seasons = list(range(this - 7, this))
    games, players, lines = _league(seasons, days=45, lines_from=this - 4)
    games.to_csv(env.GAMES, index=False)
    for s in seasons:
        players[players["game_id"].isin(games.loc[games["season"] == s, "game_id"])].to_csv(
            env.PLAYERS / f"{s}.csv", index=False)
    lines.to_csv(env.LINES, index=False)
    monkeypatch.setattr(env, "FIRST_SEASON", this - 6)
    monkeypatch.setattr(env, "LOAD_FROM_SEASON", this - 7)
    for k, v in (("MIN_TRAIN_GAMES", 200), ("MIN_MARKET_TRAIN", 200), ("MIN_TEST_GAMES", 100),
                 ("MIN_SHRINK_GAMES", 300), ("MIN_BUCKET", 10), ("MIN_SEGMENT", 30)):
        monkeypatch.setattr(train, k, v)

    train.main(["--no-fetch"])
    meta = json.loads((env.MODEL_DIR / "meta.json").read_text())
    assert set(meta["models"]) == {"margin_market", "margin_nomarket", "total_market", "total_nomarket"}
    assert meta["sport"] == "nba"
    ev = meta["eval"]
    mk = ev["margin_market"]
    assert this not in mk["test_seasons"], "the in-progress season must never be the holdout"
    assert set(mk["shrink_by_season"].values()) <= set(np.round(np.arange(0, 0.55, 0.05), 2))
    assert 0 <= meta["shrink"]["margin"] <= env.SHRINK_CAP
    assert "ats_by_disagreement" in mk and "significance" in mk and "market_softness" in mk
    assert 8 <= meta["sigma"]["win"] <= 16
    assert not ev["moneyline"].get("skipped")

    # an upcoming slate: today and tomorrow, one team on a back-to-back, one star ruled out
    today = env.today_et()
    monkeypatch.setattr(env, "now_et", lambda: datetime.combine(today, time(11, 0), env.ET))
    up = []
    for i in range(0, 8, 2):
        d = today if i < 4 else today + timedelta(days=1)
        h = TEAMS8[i] if i != 4 else TEAMS8[0]          # BOS plays today and tomorrow
        a = TEAMS8[i + 1]
        up.append({"game_id": f"9{i}", "season": this, "game_type": "R", "date": str(d),
                   "tip_utc": f"{d}T23:30:00Z", "home_team": h, "away_team": a, "completed": False,
                   "neutral_site": False, "venue_city": CITY8[h]})
    allg = pd.concat([games, pd.DataFrame(up)], ignore_index=True)
    monkeypatch.setattr(hoopr, "update", lambda *a, **k: allg)
    monkeypatch.setattr(hoopr, "rosters", lambda *a, **k: {})
    snap = pd.DataFrame([{"pulled_at": f"{today}T15:00:00+00:00", "source": "espn",
                          "game_id": u["game_id"], "event_id": u["game_id"],
                          "commence_time": f"{u['date']}T23:30:00+00:00", "date": pd.Timestamp(u["date"]).date(),
                          "home_team": u["home_team"], "away_team": u["away_team"], "provider": "DraftKings",
                          "spread_home": -4.5, "spread_home_dec": 1.91, "spread_away_dec": 1.91,
                          "p_spread_home": 0.5, "n_spread": 1, "total_line": 222.5, "over_dec": 1.91,
                          "under_dec": 1.91, "p_over": 0.5, "n_total": 1, "home_dec": 1.55,
                          "away_dec": 2.55, "p_home": 0.62, "n_ml": 1} for u in up[:-1]])
    monkeypatch.setattr(odds, "snapshot", lambda *a, **k: snap)
    report = pd.DataFrame([{"pulled_at": f"{today}T15:00:00+00:00", "team": "BOS",
                            "athlete_id": 1000, "name": "Boston Star", "status": "out", "p_out": 1.0,
                            "detail": "Knee"},
                           {"pulled_at": f"{today}T15:00:00+00:00", "team": TEAMS8[2],
                            "athlete_id": 3000, "name": "Miami Star", "status": "questionable",
                            "p_out": 0.5, "detail": ""}])
    monkeypatch.setattr(espn, "injuries", lambda *a, **k: report)

    out = predict.run()
    assert len(out) == 4
    for c in ("spread_pick", "spread_p_win", "spread_ev", "total_pick", "ml_pick", "ml_ev",
              "p_home_win", "pub_margin", "nm_margin", "first_spread_home", "h_notes", "tip_label"):
        assert c in out.columns, c
    assert np.allclose(out["p_home_win"] + out["p_away_win"], 1.0, atol=1e-3)
    # a new season: nobody has played, so nothing may be staked
    assert out["thin_data"].all()
    for k in ("spread", "total", "ml"):
        assert set(out[f"{k}_strength"]) <= {"pass", "thin", "wait"}
        assert (out[f"{k}_stake"] == 0).all()
    by = out.set_index("game_id")
    assert "Boston Star" in by.loc["90", "h_notes"] and '"out"' in by.loc["90", "h_notes"]
    assert bool(by.loc["92", "injury_pending"]), "a questionable star leaves the number unsettled"
    unpriced = by.loc["96"]
    assert unpriced["spread_pick"] == "" and unpriced["spread_strength"] == "pass"
    assert int(by.loc["94", "h_b2b"]) == 1, "tonight's game makes tomorrow's a back-to-back"
    # the ruled-out star is worth points: without him Boston is priced worse than with him
    monkeypatch.setattr(espn, "injuries", lambda *a, **k: report.iloc[1:])
    healthy = predict.run(dry_run=True).set_index("game_id")
    assert healthy.loc["90", "h_d_net"] > by.loc["90", "h_d_net"] + 3
    assert healthy.loc["90", "nm_margin"] > by.loc["90", "nm_margin"]

    # the evening run rewrites the rows; the number first published must survive it
    monkeypatch.setattr(espn, "injuries", lambda *a, **k: report)
    monkeypatch.setattr(odds, "snapshot", lambda *a, **k: snap.assign(spread_home=-5.5,
                                                                        pulled_at=f"{today}T22:15:00+00:00"))
    monkeypatch.setattr(env, "now_et", lambda: datetime.combine(today, time(18, 15), env.ET))
    predict.run()
    picks = pd.read_csv(env.PICKS, dtype={"game_id": str})
    assert len(picks) == 4
    row = picks.set_index("game_id").loc["90"]
    assert row["first_spread_home"] == -4.5 and row["spread_home"] == -5.5

    site.build()
    html = (env.DOCS / "index.html").read_text()
    assert "NBA" in html and 'href="../"' in html and 'href="../nhl/"' in html
    assert "Boston Star" in html or "Star" in html
    assert "nan" not in _rendered_text(html), "empty fields must not render as 'nan'"
    assert html.count('class="game ') + html.count('class="game"') == 4

    # finish the slate and grade it: home by 5 everywhere
    fin = allg.copy()
    m = fin["game_id"].isin([u["game_id"] for u in up])
    fin.loc[m, ["home_points", "away_points", "completed"]] = [110.0, 105.0, True]
    monkeypatch.setattr(hoopr, "update", lambda *a, **k: fin)
    done = grade.grade()
    assert len(done) == 4
    g90 = done.set_index("game_id").loc["90"]
    side_margin = 5 if g90["spread_side"] == "home" else -5
    want = "win" if side_margin + g90["spread_line"] > 0 else ("push" if side_margin + g90["spread_line"] == 0 else "loss")
    assert g90["spread_result"] == want, "grading must settle the side we published"
    assert g90["spread_clv"] == pytest.approx(1.0 if g90["spread_side"] == "home" else -1.0), \
        "the line moved a point toward the home side between the two runs"
    assert set(done["ml_result"]) <= {"win", "loss", ""}
    met = grade.metrics(done)
    assert met["sport"] == "nba" and met["spreads"]["all_games"]["n"] == 3
    assert met["spreads"]["season"]["n"] == 0, "nothing was staked, so there is no staked record"
    env.METRICS.write_text(json.dumps(met, default=str))
    site.build()
    html = (env.DOCS / "index.html").read_text()
    assert "nan" not in _rendered_text(html)
    # every graded game sits in its own bucket, so each of the three priced games shows its
    # season record on all three bets - through the real predict, grade and site, not a fixture
    assert html.count('<div class="hit">Hit ') == 9
    assert html.count(" pts off</div>") == 6 and html.count(" EV, priced at ") == 3
    assert "Record by disagreement" in html


def _edge(lo, hi, wins, losses, expected=None):
    return {"bucket": f"{lo}-{hi}", "lo": lo, "hi": hi, "n": wins + losses, "wins": wins,
            "losses": losses, "pushes": 0, "win_pct": round(100 * wins / (wins + losses), 1),
            "units": 0.0, "expected_pct": expected}


def test_each_pick_carries_the_season_hit_rate_at_its_own_disagreement(env):
    """As on the college football board: each card shows how often a pick at its own gap from
    the line (or, for the moneyline, its own EV) has landed this season, so the number is read
    where the pick is rather than in a table below a board of a dozen games."""
    from nba import grade, site

    metrics = {"updated": "2026-11-20", "break_even": 52.38,
               "spreads": {"by_edge": [_edge(0.5, 1, 30, 20), _edge(3, 99, 3, 1)]},
               "totals": {"by_edge": [_edge(0, 1, 40, 45), _edge(4.5, 99, 0, 2)]},
               "moneyline": {"by_edge": [_edge(-1.0, 0.0, 31, 21, expected=57.9),
                                         _edge(0.10, 99, 9, 31, expected=19.4)]}}
    # a bucket is [lo, hi) exactly as the grader cut it, spreads and totals by size, EV by sign
    assert site._hit_for(metrics, "spreads", -0.7)["wins"] == 30
    assert site._hit_for(metrics, "spreads", 1.0) is None, "1-2 has no graded games yet"
    assert site._hit_for(metrics, "spreads", 12.0)["label"] == "3+"
    assert site._hit_for(metrics, "moneyline", -0.04)["label"] == "negative"
    assert site._hit_for(metrics, "moneyline", 0.04) is None, "EV is signed, not folded"
    assert site._hit_for(metrics, "moneyline", 1.7)["label"] == "10%+", "a stale price still lands"
    for blank in (None, float("nan"), "", "nan"):
        assert site._hit_for(metrics, "totals", blank) is None
    # coloured only once a bucket has the games to mean something: spreads and totals against
    # break-even, the moneyline against the rate its own prices implied
    assert site._hit_for(metrics, "spreads", 0.6)["tone"] == "up"
    assert site._hit_for(metrics, "totals", 0.2)["tone"] == "down"
    assert site._hit_for(metrics, "spreads", 3.5)["tone"] == "", "3-1 is not evidence"
    assert site._hit_for(metrics, "moneyline", -0.02)["tone"] == "up", "57.9% priced, 59.6% won"
    assert site._hit_for(metrics, "moneyline", 0.2)["tone"] == "up", "19.4% priced, 22.5% won"
    # the grader's own edges are open-ended at the top, so no pick can fall outside them all
    for kind, edges in grade.EDGE_BUCKETS.items():
        assert edges[-1][1] >= 99, kind
        assert all(a[1] == b[0] for a, b in zip(edges, edges[1:])), kind

    today = env.today_et()
    pd.DataFrame([{
        "prediction_date": str(today), "game_id": "401", "season": 2026, "date": str(today),
        "tip_label": "Fri Nov 20, 7:30 PM", "home_team": "BOS", "away_team": "NYK",
        "spread_pick": "NYK +4.5", "spread_strength": "pass", "spread_disagree": 0.7,
        "spread_p_win": 0.51, "spread_p_push": 0.0, "spread_ev": -0.02, "spread_dec": 1.91,
        "total_pick": "Under 224.5", "total_strength": "pass", "total_disagree": -0.4,
        "total_p_win": 0.5, "total_p_push": 0.0, "total_ev": -0.04, "total_dec": 1.91,
        "ml_pick": "NYK to win", "ml_strength": "play", "ml_ev": 0.12, "ml_dec": 4.8,
        "ml_p_win": 0.233, "ml_mkt_p": 0.2, "p_home_win": 0.767, "p_away_win": 0.233,
        "pub_margin": 4.3, "nm_margin": 5.1, "spread_home": -4.5, "total_line": 224.5,
        "pub_total": 224.1, "nm_total": 226.0, "thin_data": False,
    }]).to_csv(env.PICKS, index=False)
    env.METRICS.write_text(json.dumps(metrics))
    site.build()
    html = (env.DOCS / "index.html").read_text()
    assert 'Hit <b class="up">60.0%</b> (30-20) this season at 0.5–1 pts off' in html
    assert 'Hit <b class="down">47.1%</b> (40-45) this season at 0–1 pts off' in html
    assert ('Hit <b class="up">22.5%</b> (9-31) this season at 10%+ EV, priced at 19.4%'
            in html)
    # the table below names the same buckets the cards do, in words rather than grader keys
    assert "<td>3+ pts</td>" in html and "<td>negative</td>" in html and "<td>10%+</td>" in html
    assert "-+" not in _rendered_text(html), "a raw grader key must never reach the page"
    assert "nan" not in _rendered_text(html), "empty fields must not render as 'nan'"


def test_no_hit_rate_before_anything_is_graded(env):
    """Opening night: nothing graded, so no card claims a record and no table is drawn."""
    from nba import site

    env.METRICS.write_text(json.dumps({"updated": "2026-10-20", "break_even": 52.38}))
    site.build()
    html = (env.DOCS / "index.html").read_text()
    assert 'class="hit"' not in html and "Record by disagreement" not in html


def test_no_injury_report_means_nothing_is_staked(env, monkeypatch):
    """No report is not "nobody is hurt": every pick waits until one can be read."""
    from nba import predict
    g = {"home_team": "BOS", "away_team": "NYK", "pub_margin": 9.0, "raw_margin": 9.0,
         "mkt_margin": 4.5, "nm_margin": 9.0, "spread_home": -4.5, "spread_home_dec": 1.91,
         "spread_away_dec": 1.91, "pub_total": 220.0, "raw_total": 220.0, "nm_total": 220.0,
         "total_line": 220.5, "over_dec": 1.91, "under_dec": 1.91, "home_dec": 1.4, "away_dec": 3.2,
         "thin_data": False}
    ready = predict.price_game({**g, "injury_pending": False}, 12.0, 18.0)
    waiting = predict.price_game({**g, "injury_pending": True}, 12.0, 18.0)
    assert ready["spread_strength"] in ("play", "bold") and ready["spread_stake"] > 0
    assert waiting["spread_strength"] == "wait" and waiting["spread_stake"] == 0
    assert waiting["spread_pick"] == ready["spread_pick"] == "BOS -4.5", "the pick still publishes"


def test_games_under_way_are_never_priced_in_play(env, monkeypatch):
    from nba import predict
    now = datetime(2026, 11, 7, 18, 15, tzinfo=env.ET)
    board = pd.DataFrame([{"commence_time": np.nan, "tip_utc": "2026-11-07T20:00:00Z"},    # matinee
                          {"commence_time": "2026-11-08T00:30:00+00:00", "tip_utc": ""},  # tonight
                          {"commence_time": np.nan, "tip_utc": ""}])                        # unknown
    assert predict.started(board, now).tolist() == [True, False, False]


# ---------------------------------------------------------------------------------
# The published site and the repo around it
# ---------------------------------------------------------------------------------
def test_the_nba_page_links_everywhere_and_everyone_links_back():
    tpl = (ROOT / "nba" / "templates" / "index.html").read_text()
    for href in ('href="../"', 'href="../cfb/"', 'href="../nfl/"', 'href="../epl/"', 'href="../nhl/"'):
        assert href in tpl, href
    for other in ("cfb", "nfl", "epl", "nhl"):
        t = (ROOT / other / "templates" / "index.html").read_text()
        assert 'href="../nba/"' in t, f"{other} does not link to the NBA board"


def test_landing_page_lists_the_nba():
    from core import landing
    assert any(s["slug"] == "nba" for s in landing.SPORTS)


def test_predict_runs_morning_and_evening_and_forwards_the_knobs():
    wf = (ROOT / ".github" / "workflows" / "nba-predict.yml").read_text()
    assert len(re.findall(r"- cron: '", wf)) == 2, "the evening run is the closing line"
    for var in ("DEGEN_NBA_SPREAD_EDGE", "DEGEN_NBA_TOTAL_EDGE", "DEGEN_NBA_ML_EV",
                "DEGEN_NBA_WAIT_POINTS", "DEGEN_KELLY"):
        assert f"{var}: ${{{{ vars.{var} }}}}" in wf, f"{var} is not forwarded"
    assert "python -m core.landing" in wf
    test = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    assert "nba" in test


def test_nba_module_does_not_import_other_sports():
    """nba/ must not reach into another sport or the shared core - blast radius of one."""
    pkg = ROOT / "nba"
    offenders = []
    for path in sorted(pkg.rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.match(r"\s*(from|import)\s+(cfb|nfl|epl|nhl|ncaab|core)\b", line):
                offenders.append(f"{path.relative_to(pkg.parent)}:{i}: {line.strip()}")
    assert not offenders, "nba/ imports another sport or the shared core: " + "; ".join(offenders)


def test_the_daily_jobs_never_need_pyarrow():
    """Only the one-time line import reads parquet. A daily job that needed pyarrow would add a
    dependency every other sport's pipeline would then run under."""
    reqs = (ROOT / "requirements.txt").read_text().lower()
    assert "pyarrow" not in reqs
    for mod in ("hoopr.py", "espn.py", "odds.py"):
        assert "parquet" not in (ROOT / "nba" / "sources" / mod).read_text()
