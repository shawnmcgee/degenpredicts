"""Offline tests for the NBA player-props board. No network, no API key.

The guardrails here are the NFL props suite's, carried over because the failure modes are the
same, plus the ones basketball adds:

* the replay must be leak-free, and a game's own box score must not reach its own row;
* "who else is out" counts core teammates only - a 13-minute man's absence is often game script;
* Kalshi names the two Los Angeles clubs "Los Angeles L" and "Los Angeles C", and the event
  ticker glues the two codes together ("26OCT20PHINYK"), so both are parsed and tested;
* a rung is matched on team and jersey and confirmed on surname, never guessed;
* a player who sits loses the over, so the injury report cuts every probability;
* the published price leans on Kalshi, and a value flag needs edge, band, liquidity and agreement.
"""
from __future__ import annotations

import importlib
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODULES = ("nba.teams", "nba.odds_math", "nba.players", "nba.features", "nba.sources.hoopr",
           "nba.sources.espn", "nba.sources.kalshi", "nba.props.features", "nba.props.model",
           "nba.props.board", "nba.props.grade", "nba.props.site", "nba.props.run")


@pytest.fixture(autouse=True)
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("DEGEN_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("DEGEN_DOCS", str(tmp_path / "docs"))
    monkeypatch.setenv("DEGEN_ROOT", str(ROOT))
    from nba import config
    importlib.reload(config)
    for mod in MODULES:
        importlib.reload(sys.modules[mod] if mod in sys.modules else importlib.import_module(mod))
    from nba import http
    monkeypatch.setattr(http, "get", lambda *a, **k: None)
    monkeypatch.setattr(http, "get_json", lambda *a, **k: None)
    for mod in ("nba.sources.hoopr", "nba.sources.espn", "nba.sources.kalshi"):
        m = sys.modules[mod]
        for name in ("get", "get_json"):
            if hasattr(m, name):
                monkeypatch.setattr(m, name, lambda *a, **k: None)
    config.ensure_dirs()
    yield config


TEAMS = ["BOS", "NYK", "MIA", "PHI", "LAL", "LAC"]


def _league(seasons=(2021, 2022), days=30, seed=5, absent=0.08):
    """A small league with box scores. Player k of each club plays mins[k] minutes."""
    rng = np.random.default_rng(seed)
    mins = [36, 34, 32, 30, 26, 22, 18, 13]
    games, players, gid = [], [], 0
    for s in seasons:
        day = date(s, 10, 24)
        for _ in range(days):
            order = list(rng.permutation(TEAMS))
            for i in range(0, len(order), 2):
                h, a = order[i], order[i + 1]
                gid += 1
                g_id = f"4{s}{gid:05d}"
                hp, ap = int(rng.normal(112, 10)), int(rng.normal(109, 10))
                hp += hp == ap
                games.append({"game_id": g_id, "season": s, "game_type": "R", "date": str(day),
                              "tip_utc": f"{day}T23:30:00Z", "home_team": h, "away_team": a,
                              "home_points": float(hp), "away_points": float(ap), "completed": True,
                              "neutral_site": False, "periods": 4.0, "venue_city": None,
                              "home_poss": 99.0, "away_poss": 99.0, "poss": 99.0, "source": "t"})
                for t in (h, a):
                    for k, m in enumerate(mins):
                        if rng.random() < absent:
                            continue
                        lam = m / 36
                        line = {c: int(rng.poisson(max(0.05, lam * mu * (1.6 if k == 0 else 1.0))))
                                for c, mu in (("pts", 14), ("fgm", 5), ("fga", 11), ("fg3m", 1.5),
                                              ("fg3a", 4), ("ftm", 2), ("fta", 3), ("oreb", 1),
                                              ("dreb", 4), ("ast", 3), ("stl", 1), ("blk", 0.5),
                                              ("tov", 1.5), ("pf", 2))}
                        players.append({"game_id": g_id, "team": t,
                                        "athlete_id": 1000 * (TEAMS.index(t) + 1) + k,
                                        "minutes": float(m + rng.normal(0, 2)), "gs": 10.0,
                                        "os": 8.0, "starter": int(k < 5), **line})
            day += timedelta(days=1)
    return pd.DataFrame(games), pd.DataFrame(players)


POS = {1000 * (i + 1) + k: ("G", "G", "F", "F", "C", "G", "F", "C")[k]
       for i in range(len(TEAMS)) for k in range(8)}


# ---------------------------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------------------------
def test_feature_replay_is_leak_free(cfg):
    from nba.props.features import FEATURES, build
    g, p = _league()
    full, _, _ = build(p, g, positions=POS, first_row_season=2022)
    cut = g.iloc[: len(g) * 3 // 4]
    part, _, _ = build(p[p["game_id"].isin(cut["game_id"])], cut, positions=POS,
                       first_row_season=2022)
    key = ["game_id", "athlete_id"]
    m = part[key + FEATURES].merge(full[key + FEATURES], on=key, suffixes=("", "_f"))
    assert len(m) == len(part) > 100
    for c in FEATURES:
        assert np.allclose(m[c].astype(float), m[f"{c}_f"].astype(float), equal_nan=True), c


def test_a_rows_features_do_not_contain_its_own_game(cfg):
    """Rewriting only the last game's box score moves nothing in that game's rows - except the
    targets themselves."""
    from nba.props.features import FEATURES, build
    g, p = _league()
    base, _, _ = build(p, g, positions=POS, first_row_season=2022)
    last = g.iloc[-1]["game_id"]
    p2 = p.copy()
    mine = p2["game_id"] == last
    p2.loc[mine, ["pts", "ast", "fg3m", "dreb"]] = [60, 20, 12, 25]
    moved, _, _ = build(p2, g, positions=POS, first_row_season=2022)
    a = base[base["game_id"] == last].sort_values("athlete_id")
    b = moved[moved["game_id"] == last].sort_values("athlete_id")
    assert np.allclose(a[FEATURES].astype(float), b[FEATURES].astype(float), equal_nan=True)
    assert (b["pts"].values == 60).all()


def test_missing_teammates_count_core_players_only(cfg):
    """A 30-minute teammate's absence shows up in every teammate's row; a 13-minute one's does
    not - whether the end of the bench plays is often decided by the score."""
    from nba.props.features import build
    g, p = _league(absent=0.0)
    last = g.iloc[-1]
    team = last["home_team"]
    base_id = 1000 * (TEAMS.index(team) + 1)
    for k, should_move in ((3, True), (7, False)):
        p2 = p[~((p["game_id"] == last["game_id"]) & (p["athlete_id"] == base_id + k))]
        rows, _, _ = build(p2, g, positions=POS, first_row_season=2022)
        r = rows[(rows["game_id"] == last["game_id"]) & (rows["athlete_id"] == base_id)].iloc[0]
        assert (r["miss_min"] > 0) == should_move, k


def test_the_board_reads_teammates_out_from_the_report(cfg):
    from nba.props.features import build
    g, p = _league(absent=0.0)
    nxt = pd.DataFrame([{"game_id": "9", "season": 2022, "game_type": "R", "date": "2023-01-30",
                         "tip_utc": "2023-01-31T00:00:00Z", "home_team": "BOS", "away_team": "NYK",
                         "completed": False, "neutral_site": False, "poss": np.nan,
                         "home_points": np.nan, "away_points": np.nan, "periods": np.nan}])
    g2 = pd.concat([g, nxt], ignore_index=True)
    up = pd.DataFrame([{"game_id": "9", "athlete_id": 1000, "team": "BOS"}])
    _, none, _ = build(p, g2, positions=POS, first_row_season=2022, upcoming=up)
    _, out, _ = build(p, g2, positions=POS, first_row_season=2022, upcoming=up,
                      news={"BOS": {1001: 1.0, 1002: 0.5}})
    assert none["miss_min"].iloc[0] == 0
    assert out["miss_min"].iloc[0] > 0 and out["n_miss"].iloc[0] == pytest.approx(1.5)


# ---------------------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------------------
def _market(ticker, title, rules, strike=None, bid=0.40, ask=0.42, size=100):
    return {"ticker": ticker, "event_ticker": "-".join(ticker.split("-")[:2]), "title": title,
            "rules_primary": rules, "floor_strike": strike, "yes_bid_dollars": str(bid),
            "yes_ask_dollars": str(ask), "yes_ask_size_fp": str(size), "volume_fp": "500",
            "open_interest_fp": "200", "close_time": "2026-10-21T04:00:00Z"}


def test_prop_rung_parses_the_known_shape(cfg):
    from nba.sources.kalshi import parse_rung
    m = _market("KXNBA3PT-26MAY09OKCLAL-OKCSGILGEOUSALEXANDER2-3",
                "Shai Gilgeous-Alexander: 3+ threes",
                "If Shai Gilgeous-Alexander records 3+ Three Pointers in the Oklahoma City at Los "
                "Angeles L professional basketball game originally scheduled for May 9, 2026, then "
                "the market resolves to Yes.", strike=2.5)
    r = parse_rung(m, "threes")
    assert r["k"] == 3.0 and r["jersey"] == 2 and r["seg_letters"] == "OKCSGILGEOUSALEXANDER"
    assert (r["date"], r["away_team"], r["home_team"]) == (date(2026, 5, 9), "OKC", "LAL")
    assert r["tradeable"]


def test_the_matchup_is_read_from_the_rules_when_the_ticker_has_none(cfg):
    from nba.sources.kalshi import parse_rung, place
    assert place("Los Angeles C") == "LAC" and place("Los Angeles L") == "LAL"
    assert place("Golden State") == "GSW" and place("Los Angeles") is None
    m = _market("ODD-TICKER", "Kawhi Leonard: 20+ points",
                "If Kawhi Leonard records 20+ points in the Los Angeles L at Los Angeles C "
                "professional basketball game originally scheduled for Oct 21, 2026.")
    r = parse_rung(m, "points")
    assert r["k"] == 20.0 and (r["away_team"], r["home_team"]) == ("LAL", "LAC")
    assert r["date"] == date(2026, 10, 21) and r["jersey"] is None


def test_event_ticker_codes_split_three_and_three_first(cfg):
    from nba.sources.kalshi import _event_game
    assert _event_game("KXNBAPTS-26OCT20PHINYK") == (date(2026, 10, 20), "PHI", "NYK")
    assert _event_game("KXNBAPTS-26NOV02SACCHA")[1:] == ("SAC", "CHA")
    assert _event_game("KXNBAPTS-26NOV02XXXYYY")[1:] == (None, None)


ROSTER = pd.DataFrame([
    {"team": "NYK", "athlete_id": 11, "name": "Jalen Brunson", "jersey": 11, "pos": "G"},
    {"team": "NYK", "athlete_id": 32, "name": "Karl-Anthony Towns", "jersey": 32, "pos": "C"},
    {"team": "PHI", "athlete_id": 21, "name": "Joel Embiid", "jersey": 21, "pos": "C"},
    {"team": "PHI", "athlete_id": 5, "name": "Jabari Smith", "jersey": 5, "pos": "F"},
    {"team": "PHI", "athlete_id": 6, "name": "Jalen Smith", "jersey": 6, "pos": "F"},
])


def _rung(player, seg, jersey, home="NYK", away="PHI"):
    return {"player_raw": player, "seg_letters": seg, "jersey": jersey, "home_team": home,
            "away_team": away}


def test_players_match_on_jersey_through_kalshis_team_code(cfg):
    from nba.sources.kalshi import match_players
    r = match_players(pd.DataFrame([_rung("Karl-Anthony Towns", "NYKKTOWNS", 32)]), ROSTER)
    assert r["athlete_id"].iloc[0] == 32 and r["team"].iloc[0] == "NYK" and r["match"].iloc[0] == "jersey"


def test_a_jersey_that_disagrees_with_the_name_falls_back_and_never_guesses(cfg):
    from nba.sources.kalshi import match_players
    r = match_players(pd.DataFrame([_rung("Joel Embiid", "PHIJEMBIID", 99),
                                    _rung("J. Smith", "XX", None)]), ROSTER)
    assert r["athlete_id"].iloc[0] == 21 and r["match"].iloc[0] == "name"
    assert pd.isna(r["athlete_id"].iloc[1]) and r["match"].iloc[1] == "ambiguous"


def test_attach_games_joins_on_date_and_both_clubs_either_way_round(cfg):
    from nba.sources.kalshi import attach_games
    games = pd.DataFrame([{"game_id": "1", "date": "2026-10-20", "home_team": "NYK", "away_team": "PHI"}])
    rungs = pd.DataFrame([{"date": date(2026, 10, 20), "home_team": "NYK", "away_team": "PHI"},
                          {"date": date(2026, 10, 20), "home_team": "PHI", "away_team": "NYK"},
                          {"date": date(2026, 10, 21), "home_team": "NYK", "away_team": "PHI"}])
    out = attach_games(rungs, games)
    assert out["game_id"].tolist()[:2] == ["1", "1"] and pd.isna(out["game_id"].iloc[2])
    assert out["home_team"].iloc[1] == "NYK"


# ---------------------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------------------
def test_published_price_leans_on_kalshi(cfg):
    from nba.props.board import publish
    p = float(publish(0.70, 0.50, 0.25))
    assert 0.50 < p < 0.56
    assert float(publish(0.70, np.nan, 0.25)) == pytest.approx(0.70)
    assert float(publish(0.70, 0.50, 0.0)) == pytest.approx(0.50)


def test_a_player_who_sits_loses_the_over(cfg):
    from nba.props.board import p_play
    assert p_play("out") == 0.0 and p_play("doubtful") == 0.25 and p_play("questionable") == 0.5
    assert p_play("probable") == 0.9 and p_play("") == 0.98 and p_play(None) == 0.98


def _priced(p_model_cond, bid, ask, size=100, p_play=0.98):
    from nba.props import board
    from nba.props.model import Dist

    class Fixed(Dist):
        def __init__(self, p):
            self.p = p

        def p_at_least(self, mu, k, group=None):
            return self.p

    rungs = pd.DataFrame([{"game_id": "1", "athlete_id": 7, "market": "points", "k": 20.0,
                           "ticker": "T", "yes_bid": bid, "yes_ask": ask, "quote_spread": ask - bid,
                           "ask_size": size, "volume": 500.0,
                           "tradeable": bool(ask - bid <= 0.06 and size >= 20), "team": "NYK",
                           "match": "jersey", "pulled_at": "x"}])
    proj = pd.DataFrame([{"game_id": "1", "athlete_id": 7, "market": "points", "position": "G",
                          "opponent": "PHI", "proj_mean": 22.0, "proj_median": 21.0,
                          "proj_avg": 22.0, "injury": "", "p_play": p_play}])
    return board.price(rungs, proj, {"points": Fixed(p_model_cond)}).iloc[0]


def test_value_flag_needs_edge_band_liquidity_and_agreement(cfg):
    assert _priced(0.62, 0.48, 0.50)["flag"] == ""                 # 25% of 12 points is no edge
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cfg, "PROPS_SHRINK", 1.0)
        assert _priced(0.62, 0.48, 0.50)["flag"] == "yes"          # the model's own number would
        assert _priced(0.62, 0.40, 0.50)["flag"] == ""             # a 10-cent quote is not a market
        assert _priced(0.62, 0.48, 0.50, size=5)["flag"] == ""     # nor is 5 contracts
        assert _priced(0.75, 0.40, 0.42)["flag"] == ""             # 34 points from Kalshi: news
        assert _priced(0.33, 0.44, 0.46)["flag"] == "no"


def test_the_injury_report_cuts_the_models_probability(cfg):
    r = _priced(0.60, 0.48, 0.50, p_play=0.5)
    assert r["p_model"] == pytest.approx(0.30) and r["p_cond"] == pytest.approx(0.60)


def test_distribution_keeps_the_skew(cfg):
    from nba.props.model import Dist
    rng = np.random.default_rng(1)
    mu = rng.uniform(5, 30, 20000)
    y = rng.gamma(4, mu / 4)                      # right-skewed around its mean
    d = Dist(mu, y, np.array(["G"] * len(y), object))
    assert d.quantile(20, 0.5, "G") < d.mean(20, "G")
    assert d.p_at_least(20, 20, "G") < 0.5


# ---------------------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------------------
def test_grading_counts_a_scratch_as_zero_and_scores_flags_at_their_price(cfg):
    from nba.props import grade
    cfg.PROPS_LADDERS.mkdir(parents=True, exist_ok=True)
    lad = pd.DataFrame([
        {"pulled_at": "2026-10-20T16:00:00+00:00", "game_id": "1", "athlete_id": 7,
         "player_name": "A", "market": "points", "k": 20.0, "ticker": "A20", "yes_bid": 0.48,
         "yes_ask": 0.50, "ask_size": 100, "volume": 1, "p_model": 0.6, "p_yes": 0.55,
         "p_play": 0.98, "proj_mean": 22, "gap": 0.1, "flag": "yes", "tip_utc": "2026-10-20T23:00:00Z"},
        {"pulled_at": "2026-10-20T16:00:00+00:00", "game_id": "1", "athlete_id": 8,
         "player_name": "B", "market": "rebounds", "k": 6.0, "ticker": "B6", "yes_bid": 0.48,
         "yes_ask": 0.50, "ask_size": 100, "volume": 1, "p_model": 0.6, "p_yes": 0.55,
         "p_play": 0.98, "proj_mean": 7, "gap": 0.1, "flag": "yes", "tip_utc": "2026-10-20T23:00:00Z"}])
    lad.to_csv(cfg.PROPS_LADDERS / "2026-10-20.csv.gz", index=False, compression="gzip")
    pd.DataFrame([{"game_id": "1", "season": 2026, "game_type": "R", "date": "2026-10-20",
                   "home_team": "NYK", "away_team": "PHI", "completed": True,
                   "neutral_site": False}]).to_csv(cfg.GAMES, index=False)
    pd.DataFrame([{"game_id": "1", "team": "NYK", "athlete_id": 7, "minutes": 34.0, "gs": 1,
                   "os": 1, "starter": 1, "pts": 25, "fgm": 9, "fga": 18, "fg3m": 2, "fg3a": 6,
                   "ftm": 5, "fta": 6, "oreb": 1, "dreb": 3, "ast": 4, "stl": 1, "blk": 0,
                   "tov": 2, "pf": 2}]).to_csv(cfg.PLAYERS / "2026.csv", index=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cfg, "today_et", lambda: date(2026, 10, 21))
        done = grade.grade().set_index("ticker")
    assert done.loc["A20", "hit"] == 1 and done.loc["A20", "units"] > 0
    assert done.loc["B6", "actual"] == 0 and done.loc["B6", "units"] < 0     # he never played


# ---------------------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------------------
def test_props_workflow_and_tab_are_wired(cfg):
    wf = (ROOT / ".github" / "workflows" / "nba-props.yml").read_text()
    assert "python -m nba.props.run" in wf and "group: nba-data" in wf
    for var in ("DEGEN_KALSHI_NBA_PTS_SERIES", "DEGEN_KALSHI_NBA_REB_SERIES",
                "DEGEN_KALSHI_NBA_AST_SERIES", "DEGEN_KALSHI_NBA_3PT_SERIES"):
        assert f"{var}: ${{{{ vars.{var} }}}}" in wf, var
    assert "nba_props" in (ROOT / ".github" / "workflows" / "test.yml").read_text()
    page = (ROOT / "nba" / "templates" / "index.html").read_text()
    assert 'href="props/"' in page
    props = (ROOT / "nba" / "templates" / "props.html").read_text()
    assert 'href="../"' in props and 'href="../../nfl/"' in props and "anytime" not in props


def test_the_props_page_renders_with_and_without_a_board(cfg):
    from nba.props import site
    site.build()
    html = (cfg.PROPS_DOCS / "index.html").read_text()
    assert "No props on the board" in html
    cfg.PROPS.mkdir(parents=True, exist_ok=True)
    row = {"game_id": "1", "date": "2099-10-20", "tip_utc": "2099-10-20T23:00:00Z",
           "tip_et": "Tue Oct 20, 7:00 PM", "away_team": "PHI", "home_team": "NYK", "team": "NYK",
           "opponent": "PHI", "athlete_id": 11, "player_name": "Jalen Brunson", "position": "G",
           "market": "points", "proj_mean": 25.1, "proj_median": 24.0, "proj_avg": 24.8,
           "p_cond": 0.6, "p_play": 0.98, "injury": "", "p_model": 0.59, "mid": 0.55, "p_yes": 0.56,
           "yes_bid": 0.54, "yes_ask": 0.56, "quote_spread": 0.02, "ask_size": 100, "volume": 9,
           "tradeable": True, "cost_yes": 0.58, "cost_no": 0.48, "edge_yes": -0.03,
           "edge_no": -0.08, "gap": 0.04, "flag": "", "match": "jersey", "ticker": "T",
           "pulled_at": "2026-10-20T16:00:00+00:00"}
    pd.DataFrame([{**row, "k": 25.0}, {**row, "k": 30.0, "ticker": "T2"}]).to_csv(cfg.PROPS_BOARD, index=False)
    site.build()
    html = (cfg.PROPS_DOCS / "index.html").read_text()
    assert "Jalen Brunson" in html and "25+" in html and "PHI @ NYK" in html
    # a board whose games have all tipped is not a board - a failed pull must not leave it up
    pd.DataFrame([{**row, "k": 25.0, "tip_utc": "2020-01-01T00:00:00Z"}]).to_csv(cfg.PROPS_BOARD, index=False)
    site.build()
    assert "No props on the board" in (cfg.PROPS_DOCS / "index.html").read_text()


def test_props_need_no_pyarrow_and_import_no_other_sport(cfg):
    for path in sorted((ROOT / "nba").rglob("*.py")):
        text = path.read_text()
        assert not re.search(r"^\s*(from|import)\s+(cfb|nfl|epl|nhl|ncaab|core)\b", text, re.M), path
    for mod in ("props/features.py", "props/board.py", "props/model.py", "sources/kalshi.py"):
        assert "parquet" not in (ROOT / "nba" / mod).read_text()
