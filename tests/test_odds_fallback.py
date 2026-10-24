"""Offline tests for the Odds API credit reserve and the ESPN fallback. No network, no key.

Every board that prices off The Odds API shares one key and the free plan's 500 credits a month.
October 2026 ran it to 56 credits by the 22nd at ~19 a day. Each board now reads the balance
from a free call before spending; under ``DEGEN_ODDS_RESERVE`` (75) - or when the paid call
fails - it takes ESPN's free scoreboard prices instead until the credits reset on the 1st.

Pinned here, for every board that spends credits:

* under the reserve, the paid odds endpoint is never called and ESPN's prices come back in the
  board's own row shape;
* with credits to spare, The Odds API is used and ESPN is not;
* a failed paid call falls back to ESPN; no key at all changes nothing;
* ESPN's two odds shapes both parse, and a hockey moneyline in the ``details`` slot is not read
  as a 150-goal handicap;
* the four sport-local copies of the parser stay identical.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 10, 24, 13, 0, tzinfo=timezone.utc)
KICK = "2026-10-25T17:00Z"


# ---------------------------------------------------------------------------------------
# A fake HTTP layer: the free balance check, the paid odds call, and ESPN's scoreboard
# ---------------------------------------------------------------------------------------
class Resp:
    def __init__(self, code=200, body=None, headers=None):
        self.status_code, self._body, self.headers = code, body, headers or {}

    def json(self):
        return self._body


class FakeHTTP:
    def __init__(self, remaining, espn, oddsapi=None, odds_code=200):
        self.remaining, self.espn, self.oddsapi, self.odds_code = remaining, espn, oddsapi, odds_code
        self.urls = []

    def __call__(self, url, params=None, **kw):
        self.urls.append(url)
        if url.endswith("/v4/sports"):
            h = {} if self.remaining is None else {"x-requests-remaining": str(self.remaining)}
            return Resp(200, [], h)
        if "the-odds-api.com" in url:
            return Resp(self.odds_code, self.oddsapi or [], {"x-requests-remaining": "1"})
        if "espn.com" in url:
            return Resp(200, self.espn if params and params.get("dates") == "20261025" else {"events": []})
        raise AssertionError(f"unexpected URL {url}")

    def paid(self):
        return [u for u in self.urls if "the-odds-api.com" in u and not u.endswith("/v4/sports")]

    def espn_calls(self):
        return [u for u in self.urls if "espn.com" in u]


def team(abbr, name, location=None):
    return {"team": {"abbreviation": abbr, "displayName": name, "shortDisplayName": name.split()[-1],
                     "location": location or " ".join(name.split()[:-1]), "name": name.split()[-1]}}


def scoreboard(home, away, odds, state="pre", eid="401"):
    return {"events": [{"id": eid, "date": KICK, "competitions": [{
        "date": KICK, "status": {"type": {"state": state}},
        "competitors": [{"homeAway": "home", **home}, {"homeAway": "away", **away}],
        "odds": odds}]}]}


def nested(spread=-3.5, sp=(-110, -110), total=44.5, ou=(-105, -115), ml=(-175, 145), draw=None):
    out = {"provider": {"name": "DraftKings"},
           "pointSpread": {"home": {"close": {"line": f"{spread:+g}", "odds": f"{sp[0]:+d}"}},
                           "away": {"close": {"line": f"{-spread:+g}", "odds": f"{sp[1]:+d}"}}},
           "total": {"over": {"close": {"line": f"o{total}", "odds": f"{ou[0]:+d}"}},
                     "under": {"close": {"line": f"u{total}", "odds": f"{ou[1]:+d}"}}},
           "moneyline": {"home": {"close": {"odds": f"{ml[0]:+d}"}},
                         "away": {"close": {"odds": f"{ml[1]:+d}"}}}}
    if draw is not None:
        out["moneyline"]["draw"] = {"close": {"odds": f"{draw:+d}"}}
    return out


# ---------------------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------------------
def test_the_four_copies_of_the_parser_are_identical():
    texts = {s: (ROOT / s / "sources" / "espn_odds.py").read_text() for s in ("cfb", "nfl", "nhl", "epl")}
    assert len(set(texts.values())) == 1, "sport-local copies of espn_odds.py have drifted"


def test_american_prices_and_lines_read_every_way_espn_writes_them():
    from nfl.sources.espn_odds import american, line, to_decimal
    assert american("+150") == 150 and american("-110") == -110 and american("EVEN") == 100
    assert american("OFF") != american("OFF") and american(50) != american(50)
    assert line("o48.5") == 48.5 and line("-3.5") == -3.5 and line("PK") == 0.0
    assert to_decimal(-150) == pytest.approx(1 + 100 / 150) and to_decimal("+200") == 3.0


def test_nested_and_flat_shapes_both_parse():
    from nfl.sources.espn_odds import parse_odds
    a = parse_odds([nested()], "KC", "DEN")
    assert (a["spread_home"], a["spread_home_price"], a["total_line"], a["over_price"]) == (-3.5, -110, 44.5, -105)
    assert (a["home_ml"], a["away_ml"]) == (-175, 145) and a["provider"] == "DraftKings"
    flat = {"provider": {"name": "DraftKings"}, "details": "DEN -2.5", "overUnder": 41.5, "spread": 2.5,
            "overOdds": -110, "underOdds": -110,
            "homeTeamOdds": {"moneyLine": 120, "spreadOdds": -105, "favorite": False},
            "awayTeamOdds": {"moneyLine": -140, "spreadOdds": -115, "favorite": True}}
    b = parse_odds([flat], "KC", "DEN")
    assert b["spread_home"] == 2.5, "the away favourite's -2.5 is the home side +2.5"
    assert (b["total_line"], b["home_ml"], b["spread_away_price"]) == (41.5, 120, -115)


def test_a_hockey_moneyline_in_the_details_slot_is_not_a_handicap():
    from nhl.sources.espn_odds import parse_odds
    flat = {"details": "TOR -150", "overUnder": 6.5, "homeTeamOdds": {"moneyLine": -150},
            "awayTeamOdds": {"moneyLine": 130}}
    r = parse_odds([flat], "TOR", "BOS", max_spread=3.0)
    assert r["spread_home"] != r["spread_home"]
    assert (r["home_ml"], r["away_ml"], r["total_line"]) == (-150, 130, 6.5)


def test_anything_that_is_not_a_scoreboard_parses_to_nothing():
    from nfl.sources.espn_odds import parse_scoreboard
    assert parse_scoreboard(None) == [] and parse_scoreboard([{"id": "x"}]) == []
    assert parse_scoreboard({"events": [{"id": "1", "competitions": [{}]}]}) == []


# ---------------------------------------------------------------------------------------
# Each board: reserve, credits to spare, a failed call, no key
# ---------------------------------------------------------------------------------------
def _setup(monkeypatch, cfg, odds, http, key_attr_owner=None):
    owner = key_attr_owner or cfg
    monkeypatch.setattr(owner, "ODDS_API_KEY", "k")
    monkeypatch.setattr(cfg, "ODDS_RESERVE", 75)
    monkeypatch.setattr(cfg, "ODDS_FALLBACK", "espn")
    monkeypatch.setattr(cfg, "BOARD_DAYS", 1)
    monkeypatch.setattr(cfg, "today_et", lambda: NOW.date())
    monkeypatch.setattr(cfg, "now_et", lambda: NOW.astimezone(cfg.ET))
    monkeypatch.setattr(odds, "get", http)


NFL_BOARD = scoreboard(team("KC", "Kansas City Chiefs"), team("DEN", "Denver Broncos"), [nested()])


def test_nfl_under_the_reserve_never_spends_and_prices_off_espn(monkeypatch):
    from nfl import config as cfg
    from nfl.sources import odds
    http = FakeHTTP(60, NFL_BOARD)
    _setup(monkeypatch, cfg, odds, http)
    snap = odds.snapshot(odds.build_matcher())
    assert http.paid() == [], "no credits spent under the reserve"
    r = snap.iloc[0]
    assert (r["home_team"], r["away_team"], r["live_spread_home"], r["live_total"]) == ("KC", "DEN", -3.5, 44.5)
    assert r["spread_book"] == "DraftKings" and r["price_source"] == "espn"
    assert r["p_home_mkt"] == pytest.approx(0.5) and r["over_price"] == -105
    q = json.loads(r["quotes"])
    assert {(x["m"], x["side"], x["point"]) for x in q} == {
        ("spread", "home", -3.5), ("spread", "away", 3.5), ("total", "over", 44.5), ("total", "under", 44.5)}
    assert {x["key"] for x in q} == {"draftkings"}


def test_nfl_with_credits_to_spare_uses_the_odds_api(monkeypatch):
    from nfl import config as cfg
    from nfl.sources import odds
    http = FakeHTTP(300, NFL_BOARD, oddsapi=[])
    _setup(monkeypatch, cfg, odds, http)
    odds.snapshot()
    assert http.paid() and not http.espn_calls()


def test_nfl_a_failed_paid_call_falls_back_and_no_key_changes_nothing(monkeypatch):
    from nfl import config as cfg
    from nfl.sources import odds
    http = FakeHTTP(None, NFL_BOARD, odds_code=401)
    _setup(monkeypatch, cfg, odds, http)
    assert len(odds.snapshot()) == 1 and http.paid() and http.espn_calls()
    monkeypatch.setattr(cfg, "ODDS_FALLBACK", "none")
    http.urls.clear()
    assert odds.snapshot().empty and not http.espn_calls()
    monkeypatch.setattr(cfg, "ODDS_API_KEY", "")
    http.urls.clear()
    assert odds.snapshot().empty and http.urls == []


def test_nfl_games_under_way_are_never_priced_from_espn(monkeypatch):
    from nfl import config as cfg
    from nfl.sources import odds
    board = scoreboard(team("KC", "Kansas City Chiefs"), team("DEN", "Denver Broncos"), [nested()], state="in")
    http = FakeHTTP(10, board)
    _setup(monkeypatch, cfg, odds, http)
    assert odds.snapshot().empty


def test_cfb_falls_back_to_the_full_fbs_slate_and_the_bare_school_name(monkeypatch):
    from cfb import config as cfg
    from cfb.sources import odds
    board = scoreboard(team("APP", "App State Mountaineers", "App State"),
                       team("GASO", "Georgia Southern Eagles", "Georgia Southern"),
                       [nested(spread=-7.0, total=55.5)])
    http = FakeHTTP(10, board)
    _setup(monkeypatch, cfg, odds, http, key_attr_owner=odds)
    seen = []
    real_get = http

    def get(url, params=None, **kw):
        seen.append(params)
        return real_get(url, params)
    monkeypatch.setattr(odds, "get", get)
    m = odds.build_matcher(["Appalachian State", "Georgia Southern"])
    snap = odds.snapshot(m)
    assert not http.paid()
    espn = [p for p in seen if p and "dates" in p]
    assert espn and all(p.get("groups") == "80" for p in espn), "FBS, not ESPN's curated handful"
    r = snap.iloc[0]
    assert (r["home_team"], r["away_team"], r["live_spread_home"]) == ("Appalachian State", "Georgia Southern", -7.0)
    assert not m.unmatched


def test_nhl_falls_back_to_decimal_prices_and_team_codes(monkeypatch):
    from nhl import config as cfg
    from nhl.sources import odds
    board = scoreboard(team("TB", "Tampa Bay Lightning"), team("MTL", "Montréal Canadiens"),
                       [nested(spread=-1.5, sp=(170, -200), total=6.5, ou=(-110, -110), ml=(-150, 130))])
    http = FakeHTTP(10, board)
    _setup(monkeypatch, cfg, odds, http)
    snap = odds.snapshot()
    assert not http.paid()
    r = snap.iloc[0]
    assert (r["home_team"], r["away_team"], r["pl_home"], r["n_ml"]) == ("TBL", "MTL", -1.5, 1)
    assert r["home_dec"] == pytest.approx(1 + 100 / 150) and r["pl_home_dec"] == pytest.approx(2.7)
    assert 0.5 < r["p_home"] < 0.6 and r["event_id"] == "espn:401"


def test_epl_builds_its_board_from_espn_and_says_so(monkeypatch, tmp_path):
    from epl import config as cfg
    from epl.sources import odds
    board = scoreboard(team("ARS", "Arsenal", "Arsenal"), team("BHA", "Brighton & Hove Albion"),
                       [nested(spread=-1.0, sp=(-120, 100), total=2.5, ou=(-130, 110), ml=(-200, 550), draw=330)])
    http = FakeHTTP(10, board)
    _setup(monkeypatch, cfg, odds, http)
    snap = odds.snapshot(odds.build_matcher())
    assert not http.paid()
    r = snap.iloc[0]
    assert (r["home_team"], r["away_team"], r["price_source"]) == ("Arsenal", "Brighton", "espn")
    assert r["live_price_draw"] == pytest.approx(4.3)
    assert r["p_home_mkt"] + r["p_draw_mkt"] + r["p_away_mkt"] == pytest.approx(1.0)
    fx = odds.board(odds.build_matcher())
    assert fx["odds_source"].iloc[0] == "espn:DraftKings"
    # a fixture missing a leg of its 1X2 is no price at all
    http.espn = scoreboard(team("ARS", "Arsenal", "Arsenal"), team("CHE", "Chelsea", "Chelsea"),
                           [nested(ml=(-200, 550))])
    assert odds.snapshot(odds.build_matcher()).empty


def test_epl_snapshot_file_survives_a_new_column(monkeypatch, tmp_path):
    from epl import config as cfg
    from epl.sources import odds
    monkeypatch.setattr(cfg, "SNAPSHOTS", tmp_path / "snapshots.csv")
    odds.append_snapshot(pd.DataFrame([{"pulled_at": "a", "date": "2026-10-01", "live_total_line": 2.5}]))
    odds.append_snapshot(pd.DataFrame([{"pulled_at": "b", "date": "2026-10-02", "live_total_line": 3.5,
                                        "price_source": "espn"}]))
    back = pd.read_csv(cfg.SNAPSHOTS)
    assert list(back["pulled_at"]) == ["a", "b"] and list(back.columns) == ["pulled_at", "date", "live_total_line"]


def test_nba_opt_in_odds_api_steps_back_to_espn_under_the_reserve(monkeypatch):
    from nba import config as cfg
    from nba.sources import espn, odds
    monkeypatch.setattr(cfg, "ODDS_SOURCE", "oddsapi")
    monkeypatch.setattr(cfg, "ODDS_API_KEY", "k")
    monkeypatch.setattr(cfg, "ODDS_RESERVE", 75)
    http = FakeHTTP(40, {"events": []})
    monkeypatch.setattr(odds, "get", http)
    monkeypatch.setattr(espn, "odds_rows", lambda days=None: pd.DataFrame([{"game_id": "1"}]))
    assert list(odds.snapshot()["game_id"]) == ["1"] and not http.paid()


def test_every_board_that_spends_credits_forwards_the_reserve():
    for sport in ("cfb", "nfl", "nhl", "epl", "nba"):
        wf = (ROOT / ".github" / "workflows" / f"{sport}-predict.yml").read_text()
        assert "DEGEN_ODDS_RESERVE: ${{ vars.DEGEN_ODDS_RESERVE }}" in wf, sport
    assert "odds_fallback" in (ROOT / ".github" / "workflows" / "test.yml").read_text()
