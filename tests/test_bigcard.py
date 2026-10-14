"""Offline tests for the Big Game Card and the Kalshi ladder log.

No network. Synthetic history with known margins, a synthetic board row and ladder, then the
page itself. The checks that matter most are the ones a casual bettor would be hurt by if they
were wrong: the fee-inclusive cost of an exchange price, which side a Kalshi Yes or No really
is, and that "fair" never claims to know the main line better than the book.
"""
from __future__ import annotations

import importlib
import json
import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def cfg(tmp_path):
    os.environ["DEGEN_ROOT"] = str(tmp_path)
    os.environ["DEGEN_DATA"] = str(tmp_path / "data")
    os.environ["DEGEN_DOCS"] = str(tmp_path / "docs")
    os.environ["CFBD_API_KEY"] = "test"
    from cfb import config
    importlib.reload(config)
    config.ensure_dirs()
    return config


# ---------------------------------------------------------------------------------------
# odds arithmetic
# ---------------------------------------------------------------------------------------
def test_exchange_cost_includes_fee():
    from cfb import bigcard as bc
    # Kalshi 56c: 0.56 + 0.07 * 0.56 * 0.44 = 0.5772 -> about -137, not the -127 56c suggests
    cost = bc.cost_of("kalshi", contract=0.56)
    assert cost == pytest.approx(0.56 + 0.07 * 0.56 * 0.44)
    assert round(bc.prob_to_american(0.56)) == -127
    assert round(bc.prob_to_american(cost)) == -137
    # Novig charges nothing pre-game; a sportsbook's margin is already in its price
    assert bc.cost_of("novig", contract=0.56) == pytest.approx(0.56)
    assert bc.cost_of("draftkings", american=-110) == pytest.approx(110 / 210)
    # an exchange quoted through the odds feed as an American price still pays its fee
    assert bc.cost_of("kalshi", american=-110) > bc.cost_of("draftkings", american=-110)


def test_expected_value_and_push():
    from cfb import bigcard as bc
    # -110 on a true coin flip loses about 4.5 cents per dollar
    assert bc.expected_value(0.5, 0.0, 110 / 210) == pytest.approx(-0.04545, abs=1e-4)
    # a push refunds the stake: 45% win, 10% push at even money is -0.0, not -0.1
    assert bc.expected_value(0.45, 0.10, 0.5) == pytest.approx(-0.0)
    assert bc.fmt_point(-8.5) == "−8.5" and bc.fmt_point(3) == "+3" and bc.fmt_point(0) == "PK"
    assert bc.fmt_american(-136.6) == "−137" and bc.fmt_american(266.2) == "+266"


# ---------------------------------------------------------------------------------------
# history and anchoring
# ---------------------------------------------------------------------------------------
def _history(n=6000, seed=3):
    """Favourites of 3-14 points; margins pile on 3 and 7 like real football."""
    from cfb import bigcard as bc
    rng = np.random.default_rng(seed)
    L = rng.choice(np.arange(3, 14.5, 0.5), n)
    m = np.round(rng.normal(L, 14)).astype(int)
    # push 15% of margins near 3 and 7 onto the key number itself
    for k in (3, 7):
        near = (np.abs(m - k) <= 2) & (rng.random(n) < 0.35)
        m[near] = k
    spread = pd.DataFrame({"L": L, "M": m})
    tl = rng.choice(np.arange(42, 66, 0.5), n)
    total = pd.DataFrame({"L": tl, "T": np.round(rng.normal(tl, 16)).astype(int)})
    return bc.History(spread, total)


def test_anchor_pins_main_line_to_market():
    from cfb import bigcard as bc
    h = _history()
    for p_mkt in (0.5, 0.53, 0.47):
        a = bc.Anchored(h.spread_dist(8.5), 8.5, p_mkt)
        assert a.p_gt(8.5) == pytest.approx(p_mkt, abs=1e-9)
    a = bc.Anchored(h.spread_dist(8.5), 8.5, 0.5)
    ys = [-14.5, -3.5, 0, 2.5, 6.5, 7.5, 9.5, 13.5, 27.5]
    ps = [a.p_gt(y) for y in ys]
    assert all(x >= y for x, y in zip(ps, ps[1:])), "fair must fall as the handicap grows"
    # the key number shows up as a step: crossing 7 costs more than crossing 9
    assert a.p_gt(6.5) - a.p_gt(7.5) > a.p_gt(8.5) - a.p_gt(9.5)


def test_log_odds_anchor_moves_tails_less_than_the_middle():
    from cfb import bigcard as bc
    d = _history().spread_dist(8.5)
    base = d.p_gt(8.5)
    a = bc.Anchored(d, 8.5, base + 0.04)
    mid_move = a.p_gt(8.5) - d.p_gt(8.5)
    tail_move = a.p_gt(24.5) - d.p_gt(24.5)
    assert 0 < tail_move < mid_move


def test_covers_are_complementary():
    from cfb import bigcard as bc
    a = bc.Anchored(_history().spread_dist(7), 7, 0.5)
    for pt in (-9.5, -7, -3, 3, 7, 9.5):
        fw, fp = bc.fav_cover(a, pt)
        dw, dp = bc.dog_cover(a, -pt)
        assert fp == pytest.approx(dp)
        assert fw + fp + dw == pytest.approx(1.0, abs=1e-9)
    # a half-point line cannot push; a whole number can
    assert bc.fav_cover(a, -7.5)[1] == 0 and bc.fav_cover(a, -7)[1] > 0


# ---------------------------------------------------------------------------------------
# one card
# ---------------------------------------------------------------------------------------
def _game(**over):
    g = {"game_id": "1", "week": 7, "date": "2030-10-19", "tip_et": "Sat Oct 19, 3:30 PM",
         "kickoff_utc": "2030-10-19T19:30:00Z", "home_team": "Tennessee", "away_team": "Alabama",
         "neutral_site": False, "spread_home": 8.5, "total_line": 51.5,
         "spread_home_price": -110, "spread_away_price": -110, "over_price": -115,
         "under_price": -105, "spread_book": "DraftKings", "total_book": "DraftKings",
         "margin_pred": -8.4, "total_pred": 51.4, "kalshi_home_ask": 0.26, "kalshi_away_ask": 0.76,
         "kalshi_tradeable": True, "home_conf": "SEC", "away_conf": "SEC"}
    g.update(over)
    return pd.Series(g)


def _rung(kind, team, strike, bid, ask, tradeable=True):
    return {"game_id": "1", "kind": kind, "team": team, "strike": strike, "yes_bid": bid,
            "yes_ask": ask, "ask_size": 500, "volume": 500, "open_interest": 500,
            "tradeable": tradeable, "ticker": f"T-{team}-{strike}", "event_ticker": "E"}


def test_kalshi_yes_and_no_map_to_the_right_sides():
    from cfb import bigcard as bc
    rungs = pd.DataFrame([_rung("spread", "Alabama", 9.5, 0.49, 0.50),
                          _rung("spread", "Tennessee", 6.5, 0.13, 0.15),
                          _rung("total", None, 50.5, 0.54, 0.55)])
    c = bc.build_card(_game(), _history(), rungs, [], None, None)
    assert c["fav"] == "Alabama" and c["dog"] == "Tennessee" and c["line"] == "−8.5"
    by_y = {m["y"]: m for m in c["menu"]}
    # YES on "Alabama by over 9.5" is Alabama -9.5; its NO is Tennessee +9.5 at 1 - bid
    assert by_y[9.5]["fav"]["label"] == "Alabama −9.5"
    assert by_y[9.5]["dog"]["label"] == "Tennessee +9.5"
    assert by_y[9.5]["dog"]["cost"] == pytest.approx(0.51 + 0.07 * 0.51 * 0.49)
    # YES on "Tennessee by over 6.5" is Tennessee -6.5; its NO is Alabama +6.5
    assert by_y[-6.5]["dog"]["label"] == "Tennessee −6.5"
    assert by_y[-6.5]["fav"]["label"] == "Alabama +6.5"
    assert by_y[-6.5]["fav"]["cost"] == pytest.approx(0.87 + 0.07 * 0.87 * 0.13)
    # totals: YES = over, NO = under
    t = c["menu_tot"][0]
    assert t["over"]["label"] == "Over 50.5" and t["under"]["label"] == "Under 50.5"
    assert t["under"]["cost"] == pytest.approx(0.46 + 0.07 * 0.46 * 0.54)


def test_near_main_rungs_join_the_side_and_thin_never_ranks_first():
    from cfb import bigcard as bc
    rungs = pd.DataFrame([_rung("spread", "Alabama", 7.5, 0.30, 0.31, tradeable=False),
                          _rung("spread", "Alabama", 9.5, 0.49, 0.50),
                          _rung("spread", "Alabama", 20.5, 0.22, 0.25)])
    c = bc.build_card(_game(), _history(), rungs, [], None, None)
    fav_labels = [r["label"] for r in c["sides"]["fav"]]
    assert "Alabama −7.5" in fav_labels and "Alabama −9.5" in fav_labels
    assert "Alabama −20.5" not in fav_labels            # an alt line, not the main bet
    assert c["sides"]["fav"][0]["liquid"]                     # a thin 31c quote can't be "best"
    # the book row is priced at its own number: -110 on the market's line is about -4.5%
    book = next(r for r in c["sides"]["fav"] if r["venue"] == "DraftKings")
    assert book["ev"] == pytest.approx(-0.04545, abs=2e-3)


def test_quotes_replace_the_single_book_and_skip_duplicate_kalshi():
    from cfb import bigcard as bc
    quotes = [{"book": "FanDuel", "key": "fanduel", "m": "spread", "side": "away", "point": -8.5, "price": -105},
              {"book": "Novig", "key": "novig", "m": "spread", "side": "home", "point": 8.5, "price": -102},
              {"book": "Kalshi", "key": "kalshi", "m": "spread", "side": "away", "point": -8.5, "price": -108},
              {"book": "DraftKings", "key": "draftkings", "m": "total", "side": "over", "point": 51.5, "price": -110}]
    rungs = pd.DataFrame([_rung("spread", "Alabama", 9.5, 0.49, 0.50)])
    c = bc.build_card(_game(), _history(), rungs, quotes, None, None)
    venues = {r["venue"] for r in c["sides"]["fav"]}
    assert "FanDuel" in venues and "DraftKings" not in venues
    # the feed's Kalshi quote is dropped when the ladder prices Kalshi rung by rung
    assert sum(r["venue"] == "Kalshi" and r["label"] == "Alabama −8.5" for r in c["sides"]["fav"]) == 0
    novig = c["sides"]["dog"][0]
    assert novig["venue"] == "Novig" and novig["cost"] == pytest.approx(102 / 202)
    # FanDuel -105 beats DraftKings-style -110 on the same number
    assert c["sides"]["fav"][0]["venue"] == "FanDuel"


def test_cheap_needs_more_than_noise():
    from cfb import bigcard as bc
    # 2% under fair but inside the history's sampling error: shown, not painted green
    row = bc._row("Kalshi", "kalshi", "x", 0.40, 0.0, 0.392, "39c", True, se=0.011)
    assert row["ev"] > 0 and row["verdict"] == "fair"
    row = bc._row("Kalshi", "kalshi", "x", 0.40, 0.0, 0.36, "35c", True, se=0.011)
    assert row["verdict"] == "cheap"
    row = bc._row("Kalshi", "kalshi", "x", 0.40, 0.0, 0.36, "35c", False, se=0.011)
    assert row["verdict"] != "cheap", "a thin quote is never called cheap"


# ---------------------------------------------------------------------------------------
# ladder log
# ---------------------------------------------------------------------------------------
def test_ladder_log_joins_board_and_dedupes(tmp_path):
    from core import ladders
    from datetime import date
    lad = pd.DataFrame([{**_rung("spread", "Alabama", 9.5, 0.49, 0.50), "date": date(2030, 10, 19),
                         "home_team": "Tennessee", "away_team": "Alabama"},
                        {**_rung("spread", "Ohio State", 3.5, 0.5, 0.52), "date": date(2030, 10, 19),
                         "home_team": "Michigan", "away_team": "Ohio State"}])
    board = pd.DataFrame([_game().to_dict()])
    board["date"] = date(2030, 10, 19)
    n = ladders.log_rungs([lad], board, tmp_path / "lad", "2030-10-14", pulled_at="2030-10-14T13:00:00+00:00")
    assert n == 1, "a rung for a game not on the board cannot be graded and is dropped"
    # same pull again: no duplicate rows; a later pull: kept alongside
    ladders.log_rungs([lad], board, tmp_path / "lad", "2030-10-14", pulled_at="2030-10-14T13:00:00+00:00")
    ladders.log_rungs([lad], board, tmp_path / "lad", "2030-10-14", pulled_at="2030-10-14T19:00:00+00:00")
    allr = ladders.load(tmp_path / "lad")
    assert len(allr) == 2 and set(allr["spread_home"]) == {8.5}
    last = ladders.latest(tmp_path / "lad")
    assert list(last["pulled_at"]) == ["2030-10-14T19:00:00+00:00"]
    # deterministic gzip: rewriting unchanged rows leaves identical bytes for git
    before = (tmp_path / "lad" / "2030-10-14.csv.gz").read_bytes()
    ladders.log_rungs([lad], board, tmp_path / "lad", "2030-10-14", pulled_at="2030-10-14T19:00:00+00:00")
    assert (tmp_path / "lad" / "2030-10-14.csv.gz").read_bytes() == before


def test_ladder_log_never_raises(tmp_path):
    from core import ladders
    assert ladders.log_rungs([], pd.DataFrame(), tmp_path, "2030-10-14") == 0
    assert ladders.log_rungs([pd.DataFrame({"x": [1]})], pd.DataFrame([_game().to_dict()]),
                             tmp_path, "2030-10-14") == 0


# ---------------------------------------------------------------------------------------
# odds feed: per-venue quotes
# ---------------------------------------------------------------------------------------
def test_cfb_quotes_keep_exchanges(cfg):
    from cfb.sources import odds
    importlib.reload(odds)
    ev = {"home_team": "Tennessee Volunteers", "away_team": "Alabama Crimson Tide", "bookmakers": [
        {"key": "novig", "title": "Novig", "markets": [{"key": "spreads", "outcomes": [
            {"name": "Tennessee Volunteers", "price": -102, "point": 8.5},
            {"name": "Alabama Crimson Tide", "price": -104, "point": -8.5}]}]},
        {"key": "betmgm", "title": "BetMGM", "markets": [{"key": "totals", "outcomes": [
            {"name": "Over", "price": -110, "point": 51.5}]}]}]}
    q = odds.quotes(ev)
    assert {x["key"] for x in q} == {"novig"}            # BetMGM sets the line, not a shop venue
    assert {(x["side"], x["point"]) for x in q} == {("home", 8.5), ("away", -8.5)}
    assert set(cfg.ODDS_BOOKMAKERS) >= {"kalshi", "polymarket", "novig", "draftkings"}
    assert len(cfg.ODDS_BOOKMAKERS) <= 10, "more than ten bookmakers bills as a second region"


def test_quotes_go_to_their_own_log_not_snapshots(cfg):
    from cfb.sources import odds
    from cfb import bigcard as bc
    importlib.reload(odds)
    q = [{"book": "Novig", "key": "novig", "m": "spread", "side": "home", "point": 8.5, "price": -102}]
    pull = pd.DataFrame([{"pulled_at": "2030-10-14T13:00:00", "event_id": "e", "date": "2030-10-19",
                          "home_team": "Tennessee", "away_team": "Alabama", "live_spread_home": 8.5,
                          "live_total": 51.5, "quotes": json.dumps(q)}])
    odds.append_snapshot(pull)
    odds.append_quotes(pull)
    odds.append_quotes(pull)                              # same pull twice: one row
    assert "quotes" not in pd.read_csv(cfg.SNAPSHOTS).columns
    log = odds.quotes_history()
    assert len(log) == 1
    assert bc._quotes_for(log, _game(date="2030-10-19"))[0]["key"] == "novig"


# ---------------------------------------------------------------------------------------
# which games get a card
# ---------------------------------------------------------------------------------------
def _polls():
    rows = []
    for week, poll, order in ((6, "AP Top 25", ["Georgia", "Alabama", "Ohio State", "Indiana"]),
                              (7, "AP Top 25", ["Georgia", "Ohio State", "Indiana", "Alabama", "Tennessee"]),
                              (7, "Coaches Poll", ["Alabama", "Georgia"]),
                              (7, "FCS Coaches Poll", ["Montana State"])):
        rows += [{"season": 2030, "season_type": "regular", "week": week, "poll": poll,
                  "rank": i + 1, "school": t} for i, t in enumerate(order)]
    rows.append({"season": 2030, "season_type": "regular", "week": 11,
                 "poll": "Playoff Committee Rankings", "rank": 1, "school": "Indiana"})
    return pd.DataFrame(rows)


def test_current_poll_prefers_newest_week_then_cfp_then_ap():
    from cfb import bigcard as bc
    ranks, poll = bc.current_poll(_polls(), 2030, 7)
    assert poll == "AP Top 25" and ranks["Tennessee"] == 5 and "Montana State" not in ranks
    # this week's poll not out yet: last week's stands
    ranks, _ = bc.current_poll(_polls(), 2030, 6)
    assert ranks == {"Georgia": 1, "Alabama": 2, "Ohio State": 3, "Indiana": 4}
    # once the committee has spoken it wins
    ranks, poll = bc.current_poll(_polls(), 2030, 12)
    assert poll == "Playoff Committee Rankings" and ranks == {"Indiana": 1}
    assert bc.current_poll(pd.DataFrame(), 2030, 7) == ({}, "")


def _board():
    rows = [("1", "Tennessee", "Alabama", 8.5), ("2", "Indiana", "Ohio State", 3.5),
            ("3", "Vanderbilt", "Arkansas", -6.0), ("4", "Toledo", "Georgia", 30.5),
            ("5", "Michigan", "Purdue", -10.5)]
    return pd.DataFrame([{**_game(game_id=g, home_team=h, away_team=a, spread_home=sp).to_dict()}
                         for g, h, a, sp in rows]).drop(columns=["home_conf", "away_conf"])


def _confs():
    conf = {"Tennessee": "SEC", "Alabama": "SEC", "Indiana": "Big Ten", "Ohio State": "Big Ten",
            "Vanderbilt": "SEC", "Arkansas": "SEC", "Toledo": "Mid-American", "Georgia": "SEC",
            "Michigan": "Big Ten", "Purdue": "Big Ten"}
    b = _board()
    return pd.DataFrame({"game_id": b["game_id"], "home_conf": b["home_team"].map(conf),
                         "away_conf": b["away_team"].map(conf)})


def test_ranked_matchups_best_first(monkeypatch):
    from cfb import bigcard as bc
    ranks, _ = bc.current_poll(_polls(), 2030, 7)
    monkeypatch.setattr(bc, "RANKED", "both")
    out = bc.select_games(_board(), _confs(), ranks)
    # #2 v #3 before #4 v #5; unranked Vanderbilt-Arkansas and Toledo-Georgia stay off
    assert list(out["home_team"]) == ["Indiana", "Tennessee"]
    assert out.loc[0, "away_rank"] == 2 and out.loc[0, "home_rank"] == 3
    monkeypatch.setattr(bc, "RANKED", "either")
    assert set(bc.select_games(_board(), _confs(), ranks)["home_team"]) == {"Indiana", "Tennessee", "Toledo"}


def test_no_poll_falls_back_to_conference_games(monkeypatch):
    from cfb import bigcard as bc
    monkeypatch.setattr(bc, "FALLBACK_CONFS", {"SEC"})
    out = bc.select_games(_board(), _confs(), {})
    assert set(out["home_team"]) == {"Tennessee", "Vanderbilt"}


def test_extra_conferences_and_teams_ride_along(monkeypatch):
    from cfb import bigcard as bc
    ranks, _ = bc.current_poll(_polls(), 2030, 7)
    monkeypatch.setattr(bc, "BIG_TEAMS", {"Michigan"})
    monkeypatch.setattr(bc, "FEATURE", {"Tennessee"})
    out = bc.select_games(_board(), _confs(), ranks)
    assert list(out["home_team"]) == ["Tennessee", "Indiana", "Michigan"]


def test_fetch_rankings_parses_and_drops_lower_divisions(cfg, monkeypatch):
    from cfb.sources import cfbd
    importlib.reload(cfbd)
    payload = [{"season": 2030, "seasonType": "regular", "week": 7, "polls": [
        {"poll": "AP Top 25", "ranks": [{"rank": 1, "school": "Georgia", "conference": "SEC", "points": 1550},
                                        {"rank": 2, "school": "Notre Dame", "conference": "FBS Independents"}]},
        {"poll": "FCS Coaches Poll", "ranks": [{"rank": 1, "school": "Montana State"}]},
        {"poll": "AFCA Division II Coaches Poll", "ranks": [{"rank": 1, "school": "Ferris State"}]}]}]
    monkeypatch.setattr(cfbd, "get_json", lambda *a, **k: payload)
    got = cfbd.fetch_rankings(2030)
    assert list(got["school"]) == ["Georgia", "Notre Dame"] and set(got["poll"]) == {"AP Top 25"}
    monkeypatch.setattr(cfbd, "get_json", lambda *a, **k: None)
    assert cfbd.fetch_rankings(2030).empty


# ---------------------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------------------
def test_build_renders_only_big_games(cfg):
    """No poll on disk yet, so the SEC fallback decides; ranked selection is tested above."""
    from cfb import bigcard as bc
    importlib.reload(bc)
    today = cfg.today_et()
    day = (pd.Timestamp(today) + pd.Timedelta(days=3)).date().isoformat()
    picks = pd.DataFrame([
        {**_game(game_id="1", date=day).to_dict(), "prediction_date": str(today)},
        {**_game(game_id="2", date=day, home_team="Toledo", away_team="Akron").to_dict(),
         "prediction_date": str(today)}])
    picks.drop(columns=["home_conf", "away_conf"]).to_csv(cfg.PICKS, index=False)
    games = pd.DataFrame([
        {"game_id": "1", "season": today.year, "week": 7, "date": day, "home_team": "Tennessee",
         "away_team": "Alabama", "home_conf": "SEC", "away_conf": "SEC", "completed": False,
         "home_margin": np.nan, "total_points": np.nan},
        {"game_id": "2", "season": today.year, "week": 7, "date": day, "home_team": "Toledo",
         "away_team": "Akron", "home_conf": "Mid-American", "away_conf": "Mid-American",
         "completed": False, "home_margin": np.nan, "total_points": np.nan}])
    # a little history so the fair prices have something to stand on
    h = _history(n=3000)
    past = pd.DataFrame({"game_id": [f"h{i}" for i in range(len(h.spread))], "season": 2025,
                         "week": 1, "date": "2025-09-01", "home_team": "H", "away_team": "A",
                         "home_conf": "X", "away_conf": "X", "completed": True,
                         "home_margin": -h.spread["M"].to_numpy(),
                         "total_points": h.total["T"].to_numpy()})
    pd.concat([games, past]).to_csv(cfg.GAMES, index=False)
    pd.DataFrame({"game_id": past["game_id"], "season": 2025, "week": 1, "date": "2025-09-01",
                  "home_team": "H", "away_team": "A", "spread_home": h.spread["L"].to_numpy(),
                  "spread_open": np.nan, "total_line": h.total["L"].to_numpy(), "total_open": np.nan,
                  "provider": "consensus", "n_providers": 1}).to_csv(cfg.LINES, index=False)

    out = bc.build()
    html = out.read_text()
    assert "Alabama" in html and "Tennessee" in html
    assert "Toledo" not in html, "a MAC game is not a big game"
    assert "costs" in html and "not a pick sheet" in html
    assert out == cfg.DOCS / "big" / "index.html"
