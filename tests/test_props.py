"""Offline tests for the NFL player-props board. No network, no keys.

Each test pins something that would fail silently:

* the feature replay is leak-free (a row only ever sees earlier games);
* games a player was on the field for but recorded nothing in are rebuilt as zeros;
* Kalshi's prop rungs parse from the payload shape the live API actually sends;
* players are matched on team + jersey and confirmed on surname, and an ambiguous surname is
  refused rather than guessed;
* a distribution read off past projections keeps the skew (a median below the mean);
* the value flag respects every guard, and a player who did not play grades as zero.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def cfg(tmp_path):
    os.environ["DEGEN_ROOT"] = str(tmp_path)
    os.environ["DEGEN_DATA"] = str(tmp_path / "data")
    os.environ["DEGEN_DOCS"] = str(tmp_path / "docs")
    os.environ.pop("DEGEN_NFL_DOCS", None)
    import importlib
    from nfl import config
    importlib.reload(config)
    config.ensure_dirs()
    return config


# ---------------------------------------------------------------------------------------
# synthetic history
# ---------------------------------------------------------------------------------------
TEAMS = ["BUF", "KC", "LV", "LAC"]


def _history(seasons=(2022, 2023), weeks=6, seed=0):
    rng = np.random.default_rng(seed)
    games, players = [], []
    for s in seasons:
        start = date(s, 9, 10)
        for w in range(1, weeks + 1):
            d = start + timedelta(days=7 * (w - 1))
            pairs = [("BUF", "KC"), ("LV", "LAC")] if w % 2 else [("KC", "LV"), ("LAC", "BUF")]
            for away, home in pairs:
                gid = f"{s}_{w:02d}_{away}_{home}"
                games.append({"game_id": gid, "season": s, "week": w, "date": d,
                              "kickoff_utc": f"{d}T17:00:00+00:00", "tip_et": "",
                              "home_team": home, "away_team": away,
                              "home_qb": f"{home} Quarterback", "away_qb": f"{away} Quarterback",
                              "roof": "outdoors", "temp": 60.0, "wind": 5.0,
                              "spread_home": -3.0, "total_line": 45.0, "completed": True})
                for team, opp in ((home, away), (away, home)):
                    for pos, name in (("QB", "Quarterback"), ("RB", "Back"), ("WR", "Receiver"),
                                      ("TE", "Tightend")):
                        att = int(rng.integers(25, 40)) if pos == "QB" else 0
                        car = int(rng.integers(10, 22)) if pos == "RB" else (3 if pos == "QB" else 0)
                        tg = int(rng.integers(3, 11)) if pos in ("WR", "TE") else (3 if pos == "RB" else 0)
                        rec = int(rng.integers(0, tg + 1)) if tg else 0
                        players.append({
                            "season": s, "week": w, "season_type": "REG", "game_id": gid,
                            "team": team, "opponent_team": opp, "player_id": f"{team}-{pos}",
                            "player_name": f"{team} {name}", "position": pos,
                            "completions": int(att * 0.65), "attempts": att,
                            "passing_yards": att * 7.0, "passing_tds": 1, "passing_interceptions": 0,
                            "sacks_suffered": 2 if pos == "QB" else 0, "passing_air_yards": 0.0,
                            "carries": car, "rushing_yards": car * 4.3, "rushing_tds": int(rng.random() < 0.3),
                            "receptions": rec, "targets": tg, "receiving_yards": rec * 11.0,
                            "receiving_tds": int(rng.random() < 0.2), "receiving_air_yards": tg * 8.0,
                            "target_share": tg / 30, "air_yards_share": 0.2, "offense_snaps": 50,
                            "snap_pct": 0.8, "zero_filled": 0})
    g = pd.DataFrame(games)
    lines = g[["game_id", "spread_home", "total_line"]].copy()
    return pd.DataFrame(players), g, lines


# ---------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------
def test_feature_replay_is_leak_free(cfg):
    """Truncating the future must not change a single feature of the past."""
    from nfl.props.features import FEATURES, build
    players, games, lines = _history()
    full, _, _ = build(players, games, lines, first_row_season=2022)
    cut = games[~((games["season"] == 2023) & (games["week"] >= 4))]
    p_cut = players[players["game_id"].isin(cut["game_id"])]
    part, _, _ = build(p_cut, cut, lines, first_row_season=2022)
    key = ["game_id", "player_id"]
    a = full.merge(part[key], on=key).sort_values(key).reset_index(drop=True)
    b = part.sort_values(key).reset_index(drop=True)
    assert len(a) == len(b) > 0
    pd.testing.assert_frame_equal(a[FEATURES], b[FEATURES], check_dtype=False)


def test_a_rows_features_do_not_contain_its_own_game(cfg):
    """A player's first game has no history: his trailing numbers must be empty or zero."""
    from nfl.props.features import build
    players, games, lines = _history(seasons=(2022,), weeks=3)
    rows, _, _ = build(players, games, lines, first_row_season=2022)
    first = rows[rows["week"] == 1]
    assert (first["n_eff"] == 0).all()
    assert (first["ew_targets"] == 0).all()
    later = rows[(rows["week"] == 2) & (rows["position"] == "WR")]
    assert (later["ew_targets"] > 0).all()


def test_implied_team_total_reads_the_spread_the_right_way_round(cfg):
    """spread_home -3 means the home side is favoured: on a 45 total it is implied 24, the
    visitor 21. Backwards, every favourite is projected to throw like it is trailing."""
    from nfl.props.features import build
    players, games, lines = _history(seasons=(2022,), weeks=2)
    rows, _, _ = build(players, games, lines, first_row_season=2022)
    home = rows[rows["is_home"] == 1]["implied_team_total"].unique()
    away = rows[rows["is_home"] == 0]["implied_team_total"].unique()
    assert list(home) == [24.0] and list(away) == [21.0]


def test_starting_qb_comes_from_the_schedule_not_the_box_score(cfg):
    from nfl.props.features import build, eligible
    players, games, lines = _history(seasons=(2022,), weeks=3)
    rows, _, _ = build(players, games, lines, first_row_season=2022)
    qb = rows[(rows["position"] == "QB") & (rows["week"] == 3)]
    assert (qb["is_starter_qb"] == 1).all()
    assert all(eligible(r, "pass_yds") for r in qb.to_dict("records"))


# ---------------------------------------------------------------------------------------
# data: the zero games
# ---------------------------------------------------------------------------------------
def test_on_field_games_with_no_stat_line_become_zero_rows(cfg):
    """nflverse has no row for a tight end who played 30 snaps and was never targeted. Snap
    counts know he played; leaving the game out would tilt every under toward the over."""
    from nfl.props.data import trim
    stats = pd.DataFrame([{"season": 2025, "week": 1, "season_type": "REG",
                           "game_id": "2025_01_BUF_KC", "team": "KC", "opponent_team": "BUF",
                           "player_id": "00-1", "player_display_name": "Star Receiver",
                           "position": "WR", "targets": 9, "receptions": 7,
                           "receiving_yards": 95}])
    snaps = pd.DataFrame([
        {"game_id": "2025_01_BUF_KC", "pfr_player_id": "StarRe00", "player": "Star Receiver",
         "position": "WR", "team": "KC", "opponent": "BUF", "game_type": "REG", "season": 2025,
         "week": 1, "offense_snaps": 60, "offense_pct": 0.9},
        {"game_id": "2025_01_BUF_KC", "pfr_player_id": "QuieTi00", "player": "Quiet Tightend",
         "position": "TE", "team": "KC", "opponent": "BUF", "game_type": "REG", "season": 2025,
         "week": 1, "offense_snaps": 31, "offense_pct": 0.46}])
    out = trim(stats, snaps, {"StarRe00": "00-1", "QuieTi00": "00-2"})
    assert len(out) == 2
    te = out[out["player_id"] == "00-2"].iloc[0]
    assert te["zero_filled"] == 1 and te["receptions"] == 0 and te["receiving_yards"] == 0
    assert te["snap_pct"] == pytest.approx(0.46)
    star = out[out["player_id"] == "00-1"].iloc[0]
    assert star["zero_filled"] == 0 and star["snap_pct"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------------------
# Kalshi parsing and matching - payloads as the live API sent them on 2026-10-18
# ---------------------------------------------------------------------------------------
LAWRENCE = {
    "ticker": "KXNFLPASSYDS-26OCT18HOUJAC-JACTLAWRENCE16-225",
    "event_ticker": "KXNFLPASSYDS-26OCT18HOUJAC",
    "title": "Trevor Lawrence: 225+ passing yards", "yes_sub_title": "Trevor Lawrence: 225+",
    "rules_primary": ("If Trevor Lawrence records 225+ passing yards in the Houston vs Jacksonville "
                      "Pro Football game originally scheduled for Oct 18, 2026, then the market "
                      "resolves to Yes."),
    "floor_strike": 224.5, "strike_type": "greater",
    "yes_bid_dollars": "0.5200", "yes_ask_dollars": "0.5300", "yes_ask_size_fp": "300.00",
    "volume_fp": "300.82", "open_interest_fp": "200.00",
}
RODRIGUEZ = {
    "ticker": "KXNFLRSHYDS-26OCT18HOUJAC-JACCRODRIGUEZ24-65",
    "event_ticker": "KXNFLRSHYDS-26OCT18HOUJAC",
    "title": "Chris Rodriguez Jr.: 65+ rushing yards", "yes_sub_title": "Chris Rodriguez Jr.: 65+",
    "rules_primary": ("If Chris Rodriguez Jr. records 65+ rushing yards in the Houston vs "
                      "Jacksonville professional football game originally scheduled for Oct 18, 2026."),
    "floor_strike": 64.5, "strike_type": "greater",
    "yes_bid_dollars": "0.2100", "yes_ask_dollars": "0.4000", "yes_ask_size_fp": "53.00",
    "volume_fp": "4597.28", "open_interest_fp": "4147.28",
}


def test_prop_rung_parses_the_live_payload(cfg):
    from nfl.sources.kalshi_props import parse_rung
    r = parse_rung(LAWRENCE, "pass_yds")
    assert r["player_raw"] == "Trevor Lawrence" and r["k"] == 225.0
    assert r["jersey"] == 16 and r["seg_letters"] == "JACTLAWRENCE"
    assert (r["away_raw"], r["home_raw"]) == ("Houston", "Jacksonville")
    assert r["date"] == date(2026, 10, 18)
    assert r["tradeable"]
    # "professional football game" is the other wording Kalshi uses
    r2 = parse_rung(RODRIGUEZ, "rush_yds")
    assert r2["player_raw"] == "Chris Rodriguez Jr." and r2["k"] == 65.0
    assert (r2["away_raw"], r2["home_raw"]) == ("Houston", "Jacksonville")
    # a 19-cent-wide quote is not a market
    assert not r2["tradeable"]


def _roster():
    return pd.DataFrame([
        {"team": "JAX", "full_name": "Trevor Lawrence", "jersey_number": 16, "gsis_id": "TL"},
        {"team": "JAX", "full_name": "Chris Rodriguez Jr.", "jersey_number": 24, "gsis_id": "CR"},
        {"team": "JAX", "full_name": "Josh Smith", "jersey_number": 81, "gsis_id": "JS1"},
        {"team": "JAX", "full_name": "Jaylen Smith", "jersey_number": 82, "gsis_id": "JS2"},
        {"team": "HOU", "full_name": "C.J. Stroud", "jersey_number": 7, "gsis_id": "CJS"},
    ])


def _rungs(*payloads):
    from nfl.sources.kalshi_props import parse_rung
    from nfl.sources.odds import build_matcher
    df = pd.DataFrame([parse_rung(p, m) for p, m in payloads])
    match = build_matcher()
    df["away_team"] = df["away_raw"].map(match)
    df["home_team"] = df["home_raw"].map(match)
    return df


def test_players_match_on_jersey_through_kalshis_team_code(cfg):
    """Kalshi writes Jacksonville as JAC in the ticker; nflverse says JAX. The jersey finds
    the player and the surname confirms him - suffix and all."""
    from nfl.sources.kalshi_props import match_players
    r = match_players(_rungs((LAWRENCE, "pass_yds"), (RODRIGUEZ, "rush_yds")), _roster())
    assert list(r["player_id"]) == ["TL", "CR"]
    assert list(r["team"]) == ["JAX", "JAX"]
    assert set(r["match"]) == {"jersey"}


def test_a_jersey_that_disagrees_with_the_name_falls_back_and_never_guesses(cfg):
    from nfl.sources.kalshi_props import match_players
    wrong = dict(LAWRENCE, ticker="KXNFLREC-26OCT18HOUJAC-JACJSMITH99-4",
                 title="J. Smith: 4+ receptions")
    r = match_players(_rungs((wrong, "receptions")), _roster())
    assert r["player_id"].isna().all(), "two Smiths on the roster: refuse, don't pick one"
    assert r["match"].iloc[0] == "ambiguous"


def test_attach_games_joins_on_date_and_both_clubs(cfg):
    from nfl.sources.kalshi_props import attach_games
    from nfl.sources.odds import build_matcher
    from nfl.sources.kalshi_props import parse_rung
    rungs = pd.DataFrame([parse_rung(LAWRENCE, "pass_yds")])
    games = pd.DataFrame([{"game_id": "2026_06_HOU_JAX", "date": date(2026, 10, 18),
                           "home_team": "JAX", "away_team": "HOU"}])
    out = attach_games(rungs, games, build_matcher())
    assert out["game_id"].iloc[0] == "2026_06_HOU_JAX"


# ---------------------------------------------------------------------------------------
# distributions, pricing, grading
# ---------------------------------------------------------------------------------------
def test_distribution_keeps_the_skew(cfg):
    """Receiving yards: a 60-yard average is not a coin flip at 59.5. The median sits below
    the mean, so the probability of clearing the mean must be under one half."""
    from nfl.props.model import Dist
    rng = np.random.default_rng(1)
    y = rng.gamma(shape=1.6, scale=60 / 1.6, size=20000)
    mu = np.full_like(y, 60.0)
    d = Dist(mu, np.round(y), np.array(["WR"] * len(y), dtype=object))
    assert d.p_at_least(60, 60, "WR") < 0.47
    assert d.quantile(60, 0.5, "WR") < 55
    assert abs(d.mean(60, "WR") - 60) < 1.5


def test_platt_pulls_overconfident_probabilities_in(cfg):
    from nfl.props.model import platt, platt_fit
    rng = np.random.default_rng(2)
    true = rng.uniform(0.3, 0.7, 20000)
    y = (rng.random(20000) < true).astype(float)
    over = 1 / (1 + np.exp(-2.0 * np.log(true / (1 - true))))   # twice as extreme as the truth
    a, b = platt_fit(over, y)
    assert 0.35 < b < 0.65
    assert abs(platt(np.array([0.8]), a, b)[0] - 0.67) < 0.04


def _priced(p_yes, bid, ask, size=200.0):
    from nfl.props.board import price
    from nfl.props.model import Dist
    rungs = pd.DataFrame([{"game_id": "g", "player_id": "p", "market": "anytime_td", "k": 1.0,
                           "ticker": "T", "yes_bid": bid, "yes_ask": ask, "quote_spread": ask - bid,
                           "ask_size": size, "volume": 500.0, "tradeable": (ask - bid) <= 0.06 and size >= 20,
                           "team": "KC", "match": "jersey", "pulled_at": "x"}])
    proj = pd.DataFrame([{"game_id": "g", "player_id": "p", "market": "anytime_td",
                          "player_name": "P", "position": "WR", "opponent_team": "LV",
                          "proj_mean": p_yes, "proj_median": np.nan, "proj_avg": np.nan,
                          "injury": "", "p_play": 1.0}])
    return price(rungs, proj, {}, {"markets": {}}).iloc[0]


def test_value_flag_needs_edge_band_liquidity_and_agreement(cfg):
    # model 50%, Yes asks 40c: cheap, in band, close to the market -> flagged
    assert _priced(0.50, 0.38, 0.40)["flag"] == "yes"
    # same edge on a 19-cent-wide quote is not a market
    assert _priced(0.50, 0.21, 0.40)["flag"] == ""
    # a long shot outside the probability band is never flagged
    assert _priced(0.12, 0.05, 0.06)["flag"] == ""
    # twenty points away from the market: far likelier the model is missing news
    assert _priced(0.62, 0.38, 0.40)["flag"] == ""


def test_a_player_who_sits_loses_the_over(cfg):
    """Kalshi settles 'records N+': no game, no yards. The board multiplies by the chance he
    plays, and the grader scores a missing player as zero."""
    from nfl.props.board import P_PLAY
    assert P_PLAY["out"] == 0.0 and P_PLAY["questionable"] < 1 and P_PLAY[""] < 1

    from nfl.props import grade
    cfg.PROPS_LADDERS.mkdir(parents=True, exist_ok=True)
    cfg.PROPS_PLAYERS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"season": 2026, "week": 6, "season_type": "REG", "game_id": "2026_06_A_B",
                   "team": "B", "opponent_team": "A", "player_id": "played", "player_name": "X",
                   "position": "WR", "receptions": 5, "receiving_yards": 70, "rushing_tds": 0,
                   "receiving_tds": 1, "passing_yards": 0, "rushing_yards": 0}]
                 ).to_csv(cfg.PROPS_PLAYERS / "2026.csv", index=False)
    rows = []
    for pid in ("played", "sat"):
        rows.append({"pulled_at": "2026-10-18T15:00:00+00:00", "game_id": "2026_06_A_B",
                     "player_id": pid, "player_name": pid, "market": "receptions", "k": 3.0,
                     "ticker": f"T-{pid}", "yes_bid": 0.5, "yes_ask": 0.52, "ask_size": 100,
                     "volume": 100, "p_yes": 0.6, "p_play": 1.0, "proj_mean": 4.0, "gap": 0.09,
                     "flag": "yes", "kickoff_utc": "2026-10-18T17:00:00+00:00"})
    pd.DataFrame(rows).to_csv(cfg.PROPS_LADDERS / "2026-10-18.csv.gz", index=False, compression="gzip")
    out = grade.grade().set_index("player_id")
    assert out.loc["played", "hit"] == 1 and out.loc["sat", "hit"] == 0
    assert out.loc["sat", "actual"] == 0


def test_props_workflow_and_tab_are_wired(cfg):
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    wf = (root / ".github" / "workflows" / "nfl-props.yml").read_text()
    assert "python -m nfl.props.run" in wf and "workflow_dispatch" in wf
    assert "concurrency: {group: nfl-data" in wf, "must not race the other NFL jobs' commits"
    nfl_tpl = (root / "nfl" / "templates" / "index.html").read_text()
    assert 'href="props/"' in nfl_tpl, "the NFL page must link to the props tab"
    props_tpl = (root / "nfl" / "templates" / "props.html").read_text()
    assert 'href="../"' in props_tpl and 'href="../../"' in props_tpl
