"""Offline end-to-end test for the college-football pipeline.

No network. Builds synthetic seasons with realistic-looking lines, then runs
train -> predict -> grade -> site and checks the wiring plus the leak-free guarantee.
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
    root = tmp_path_factory.mktemp("dp")
    os.environ["DEGEN_ROOT"] = str(root)
    os.environ["DEGEN_DATA"] = str(root / "data")
    os.environ["DEGEN_DOCS"] = str(root / "docs")
    os.environ["DEGEN_FIRST_SEASON"] = "2022"
    os.environ["CFBD_API_KEY"] = "test"
    import importlib
    from cfb import config
    importlib.reload(config)
    config.ensure_dirs()
    return config


# Seasons are relative to today so the fixture reproduces the real situation: three completed
# seasons in the cache and a Week 1 board for the current season with zero games played.
from cfb import config as _c  # noqa: E402
THIS = _c.season_of(_c.today_et())
SEASONS = (THIS - 3, THIS - 2, THIS - 1)


def synth() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(11)
    teams = [f"Team {i:03d}" for i in range(130)]
    power = {t: rng.normal(0, 9) for t in teams}
    scoring = {t: rng.normal(28, 5) for t in teams}
    games, lines, gid = [], [], 0
    for s in SEASONS:
        first = date(s, 9, 3)
        for wk in range(1, 14):
            day = first + timedelta(days=7 * (wk - 1))
            order = rng.permutation(teams)
            for i in range(0, len(order) - 1, 2):
                h, a = order[i], order[i + 1]
                exp_m = power[h] - power[a] + 2.5
                exp_t = scoring[h] + scoring[a]
                margin = rng.normal(exp_m, 15)
                total = max(20, rng.normal(exp_t, 13))
                hp = int(round((total + margin) / 2))
                ap = int(round((total - margin) / 2))
                games.append(dict(game_id=str(gid), season=s, week=wk, season_type="regular",
                                  date=day, start_time_tbd=False, tip_et=f"Sat Sep {wk}, 3:30 PM",
                                  home_team=h, away_team=a, home_points=float(hp),
                                  away_points=float(ap), total_points=float(hp + ap),
                                  home_margin=float(hp - ap), neutral_site=False,
                                  conference_game=bool(i % 3),
                                  home_conf=("SEC" if i % 4 == 0 else "Mid-American"),
                                  away_conf=("Big Ten" if i % 3 == 0 else "Sun Belt"),
                                  completed=True))
                # a market that knows the truth plus noise, i.e. a hard target
                lines.append(dict(game_id=str(gid), season=s, week=wk, date=day,
                                  home_team=h, away_team=a, provider="Bovada",
                                  spread_home=round(-(exp_m + rng.normal(0, 1.5)) * 2) / 2,
                                  spread_open=round(-(exp_m + rng.normal(0, 2)) * 2) / 2,
                                  total_line=round((exp_t + rng.normal(0, 2)) * 2) / 2,
                                  total_open=round((exp_t + rng.normal(0, 3)) * 2) / 2,
                                  n_providers=4))
                gid += 1
    return pd.DataFrame(games), pd.DataFrame(lines)


def test_leak_free(env):
    from cfb.features import build
    g, ln = synth()
    full, _, _ = build(g, lines=ln)
    half, _, _ = build(g.iloc[: len(g) // 2], lines=ln)
    cols = ["exp_margin", "exp_total", "h_margin", "a_margin", "h_off", "a_def"]
    pd.testing.assert_frame_equal(full.iloc[: len(half)][cols].reset_index(drop=True),
                                  half[cols].reset_index(drop=True), atol=1e-9)


def test_ratings_sane(env):
    """A team that wins every game by 20 should end up rated well above average."""
    from cfb.ratings import RatingBook
    b = RatingBook()
    d = date(2024, 9, 7)
    for i in range(12):
        b.update(2024, "Good", f"Foe{i}", 38, 18, d + timedelta(days=7 * i))
    assert b.get(2024, "Good").margin > 8
    assert b.get(2024, "Foe0").margin < 0


def test_pipeline(env, monkeypatch):
    from cfb import grade, predict, site, train
    from cfb.sources import cfbd, odds

    games, lines = synth()
    env.ensure_dirs()
    games.to_csv(env.GAMES, index=False)
    lines.to_csv(env.LINES, index=False)
    monkeypatch.setattr(cfbd, "update_games", lambda *a, **k: cfbd.load_games())
    monkeypatch.setattr(cfbd, "update_lines", lambda *a, **k: cfbd.load_lines())
    monkeypatch.setattr(cfbd, "update_sp", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(cfbd, "update_returning", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(odds, "snapshot", lambda *a, **k: pd.DataFrame())

    train.main(["--no-fetch"])
    meta = json.loads((env.MODEL_DIR / "meta.json").read_text())
    assert {"total_market", "margin_market"} <= set(meta["models"])
    ev = meta["eval"]["margin_market"]
    # the market in this fixture is deliberately near-perfect, so the model should NOT beat it
    assert ev["mae_market_baseline"] <= ev["mae_model"] * 1.2
    # walk-forward must never hold out the in-progress season
    assert THIS not in ev["test_seasons"]
    assert ev["n_test_total"] > 200
    # shrink is capped and can't be pushed to 1.0 by a lucky slice
    assert 0.0 <= ev["shrink"] <= 0.6
    assert "ats_stderr" in ev and "beats_market" in ev
    buckets = ev["ats_by_disagreement"]
    assert buckets and all(b["n"] >= 50 for b in buckets)
    assert all("cover_pct" in b and "roi_pct" in b for b in buckets)
    assert 50.0 <= ev["break_even_pct"] <= 53.0
    soft = ev["market_softness"]
    assert "by_week" in soft and "by_spread_size" in soft
    for rows in soft.values():
        for r in rows:
            assert r["n"] >= 100 and "vs_break_even_se" in r

    # Week 1 of the CURRENT season: no games played yet, exactly the real week-1 situation
    upcoming = games.iloc[:20].copy()
    upcoming["game_id"] = ["u" + g for g in upcoming["game_id"]]
    upcoming["completed"] = False
    upcoming["season"] = THIS
    upcoming["week"] = 1
    upcoming["date"] = env.today_el() if False else (env.today_et() + timedelta(days=2))
    for c in ("home_points", "away_points", "total_points", "home_margin"):
        upcoming[c] = np.nan
    up_lines = lines.iloc[:20].copy()
    up_lines["game_id"] = list(upcoming["game_id"])
    up_lines["season"] = THIS
    up_lines["week"] = 1
    pd.concat([games, upcoming]).to_csv(env.GAMES, index=False)
    pd.concat([lines, up_lines]).to_csv(env.LINES, index=False)

    out = predict.run()
    assert len(out) == 20
    for c in ("total_pred", "total_edge", "total_disagree", "total_p_win", "total_stake",
              "margin_pred", "margin_edge", "margin_disagree", "spread_pick", "spread_stake"):
        assert c in out.columns
    assert out["total_p_win"].between(0, 1).all()
    assert (out["total_stake"] >= 0).all()
    assert out["spread_pick"].str.contains("Team").all()
    # week 1 => no current-season games played => every pick flagged thin, none staked as play
    assert out["thin_data"].all()
    assert set(out["total_strength"]) <= {"pass", "thin"}

    # finalise those games so grading has scores
    finished = upcoming.copy()
    finished["home_points"] = 31.0
    finished["away_points"] = 24.0
    finished["total_points"] = 55.0
    finished["home_margin"] = 7.0
    finished["completed"] = True
    pd.concat([games, finished]).to_csv(env.GAMES, index=False)

    done = grade.grade()
    assert len(done) == 20
    assert set(done["total_result"]) <= {"win", "loss", "push"}
    assert set(done["spread_result"]) <= {"win", "loss", "push"}
    m = grade.metrics(done)
    assert m["totals"]["all_games"]["n"] == 20
    env.METRICS.write_text(json.dumps(m, default=str))

    site.build()
    html = (env.DOCS / "index.html").read_text()
    assert "College Football" in html and "Team " in html


def test_odds_matcher():
    from cfb.sources.odds import build_matcher
    m = build_matcher(["Ohio State", "Miami", "Miami (OH)", "Ole Miss", "UConn", "Texas A&M"])
    assert m("Ohio State Buckeyes") == "Ohio State"
    assert m("Ole Miss Rebels") == "Ole Miss"
    assert m("Miami (OH) RedHawks") == "Miami (OH)"
    assert m("Connecticut Huskies") == "UConn"
    assert m("Texas A&M Aggies") == "Texas A&M"


def test_matcher_handles_state_abbreviation():
    """The live Kalshi board abbreviates State -> St; all 46 first-run misses contained
    'State'. Every spelling must resolve to the CFBD name."""
    from cfb.sources.odds import build_matcher
    cfbd = ["Ohio State", "Penn State", "San José State", "Appalachian State",
            "Long Island University", "Southeastern Louisiana", "Florida International",
            "North Dakota State", "Tarleton State", "Kent State"]
    m = build_matcher(cfbd)
    cases = {
        "Ohio St": "Ohio State", "Ohio St Buckeyes": "Ohio State",
        "Penn St Nittany Lions": "Penn State",
        "San Jose St": "San José State",                 # accent + abbreviation
        "Appalachian St": "Appalachian State", "App State Mountaineers": "Appalachian State",
        "LIU Sharks": "Long Island University",
        "SE Louisiana": "Southeastern Louisiana",
        "FIU Panthers": "Florida International",
        "North Dakota St Bison": "North Dakota State",
        "Tarleton St": "Tarleton State", "Kent St Golden Flashes": "Kent State",
    }
    for raw, want in cases.items():
        assert m(raw) == want, f"{raw} -> {m(raw)}, expected {want}"
    assert not m.unmatched


def test_kelly():
    from cfb.predict import american_payout, kelly
    assert abs(american_payout(-110) - 0.909) < 0.01
    assert kelly(0.50, 0.909) == 0.0
    assert kelly(0.60, 0.909) > 0


# --- Kalshi -------------------------------------------------------------------------
# Fixtures copied verbatim from a live /markets response (2026-09-04) so the parser is
# tested against the real payload shape rather than my guess at it.
KALSHI_SAMPLE = [
    {"event_ticker": "KXNCAAFGAME-26SEP17SYRPITT", "ticker": "KXNCAAFGAME-26SEP17SYRPITT-SYR",
     "title": "Syracuse wins", "yes_sub_title": "Syracuse", "no_sub_title": "Pittsburgh",
     "market_type": "binary", "yes_bid_dollars": "0.0800", "yes_ask_dollars": "0.8100",
     "yes_ask_size_fp": "131.00", "yes_bid_size_fp": "538.00", "volume_fp": "0.00",
     "open_interest_fp": "0.00", "last_price_dollars": "0.0000",
     "occurrence_datetime": "2026-09-18T02:30:00Z",
     "rules_primary": "If Syracuse wins the Syracuse vs Pittsburgh college football game "
                      "originally scheduled for Sep 17, 2026, then the market resolves to Yes."},
    {"event_ticker": "KXNCAAFGAME-26SEP17SYRPITT", "ticker": "KXNCAAFGAME-26SEP17SYRPITT-PITT",
     "title": "Pittsburgh wins", "yes_sub_title": "Pittsburgh", "no_sub_title": "Syracuse",
     "market_type": "binary", "yes_bid_dollars": "0.1300", "yes_ask_dollars": "0.8200",
     "yes_ask_size_fp": "138.00", "yes_bid_size_fp": "231.00", "volume_fp": "0.00",
     "open_interest_fp": "0.00", "last_price_dollars": "0.0000",
     "occurrence_datetime": "2026-09-18T02:30:00Z",
     "rules_primary": "If Pittsburgh wins the Syracuse vs Pittsburgh college football game "
                      "originally scheduled for Sep 17, 2026, then the market resolves to Yes."},
    {"event_ticker": "KXNCAAFGAME-26SEP05CWUCP", "ticker": "KXNCAAFGAME-26SEP05CWUCP-CP",
     "title": "Cal Poly wins", "yes_sub_title": "Cal Poly", "no_sub_title": "Central Washington Wildcats",
     "market_type": "binary", "yes_bid_dollars": "0.8400", "yes_ask_dollars": "0.8600",
     "yes_ask_size_fp": "318.00", "yes_bid_size_fp": "143.00", "volume_fp": "766.09",
     "open_interest_fp": "696.28", "last_price_dollars": "0.8400",
     "occurrence_datetime": "2026-09-06T03:00:00Z",
     "rules_primary": "If Cal Poly wins the Central Washington Wildcats vs Cal Poly college "
                      "football game originally scheduled for Sep 5, 2026, then the market "
                      "resolves to Yes."},
]


def test_kalshi_parse(env):
    from datetime import date as _date
    from cfb.sources import kalshi
    rows = [kalshi.parse_market(m) for m in KALSHI_SAMPLE]
    assert all(r for r in rows)
    syr = rows[0]
    assert syr["kalshi_team"] == "Syracuse"
    assert syr["home_raw"] == "Pittsburgh" and syr["away_raw"] == "Syracuse"
    assert syr["date"] == _date(2026, 9, 17)
    assert syr["yes_ask"] == 0.81
    # 73c spread, zero volume -> must NOT be considered tradeable
    assert syr["tradeable"] is False
    cp = rows[2]
    assert cp["kalshi_team"] == "Cal Poly"
    assert cp["home_raw"] == "Cal Poly" and cp["away_raw"] == "Central Washington Wildcats"
    assert round(cp["quote_spread"], 2) == 0.02
    assert cp["tradeable"] is True          # 2c spread, real size and volume


def test_kalshi_fee_and_ev(env):
    from cfb.sources import kalshi
    # published taker formula: 0.07 * P * (1-P); peaks at 1.75c on a 50c contract
    assert abs(kalshi.fee(0.50, 0.07) - 0.0175) < 1e-9
    assert kalshi.fee(0.90, 0.07) < kalshi.fee(0.50, 0.07)
    # no edge -> negative EV once the fee is paid
    ev, roi = kalshi.contract_ev(0.50, 0.50)
    assert ev < 0
    # a real 8-point probability edge survives the fee
    ev, roi = kalshi.contract_ev(0.58, 0.50)
    assert ev > 0.05 and roi > 0.1


def test_kalshi_board_matches_names(env, monkeypatch):
    from cfb.sources import kalshi, odds
    monkeypatch.setattr(kalshi, "fetch_markets", lambda *a, **k: KALSHI_SAMPLE)
    matcher = odds.build_matcher(["Pittsburgh", "Syracuse", "Cal Poly"])
    df = kalshi.moneyline_board(matcher)
    assert len(df) == 3
    assert set(df[df.event_ticker.str.endswith("SYRPITT")]["team"]) == {"Syracuse", "Pittsburgh"}
    assert int(df["tradeable"].sum()) == 1


# --- Kalshi spread / total ladders --------------------------------------------------
# Verbatim from a live /markets response (2026-09-04).
KALSHI_TOTAL = [
    {"event_ticker": "KXNCAAFTOTAL-26SEP05SHUMONM", "ticker": "KXNCAAFTOTAL-26SEP05SHUMONM-79",
     "title": "Over 78.5 points scored", "yes_sub_title": "Over 78.5 points scored",
     "floor_strike": 78.5, "strike_type": "greater", "market_type": "binary",
     "yes_bid_dollars": "0.0900", "yes_ask_dollars": "0.1700", "yes_ask_size_fp": "150.00",
     "volume_fp": "0.00", "open_interest_fp": "0.00",
     "rules_primary": "If the teams collectively score more than 78.5 points in the Sacred "
                      "Heart vs Monmouth college football game originally scheduled for "
                      "Sep 5, 2026, then the market resolves to Yes."},
    {"event_ticker": "KXNCAAFTOTAL-26SEP05SHUMONM", "ticker": "KXNCAAFTOTAL-26SEP05SHUMONM-71",
     "title": "Over 70.5 points scored", "yes_sub_title": "Over 70.5 points scored",
     "floor_strike": 70.5, "strike_type": "greater", "market_type": "binary",
     "yes_bid_dollars": "0.1600", "yes_ask_dollars": "0.2900", "yes_ask_size_fp": "150.00",
     "volume_fp": "0.00", "open_interest_fp": "0.00",
     "rules_primary": "If the teams collectively score more than 70.5 points in the Sacred "
                      "Heart vs Monmouth college football game originally scheduled for "
                      "Sep 5, 2026, then the market resolves to Yes."},
]

KALSHI_SPREAD = [
    {"event_ticker": "KXNCAAFSPREAD-26SEP05UCDUSD", "ticker": "KXNCAAFSPREAD-26SEP05UCDUSD-USD9",
     "title": "San Diego wins by over 8.5 points", "floor_strike": 8.5, "strike_type": "greater",
     "market_type": "binary", "yes_bid_dollars": "0.0700", "yes_ask_dollars": "0.9000",
     "yes_ask_size_fp": "701.00", "volume_fp": "0.00", "open_interest_fp": "0.00",
     "rules_primary": "If San Diego wins by more than 8.5 points in the UC Davis vs San Diego "
                      "college football game originally scheduled for Sep 5, 2026, then the "
                      "market resolves to Yes."},
    {"event_ticker": "KXNCAAFSPREAD-26SEP05UCDUSD", "ticker": "KXNCAAFSPREAD-26SEP05UCDUSD-USD5",
     "title": "San Diego wins by over 4.5 points", "floor_strike": 4.5, "strike_type": "greater",
     "market_type": "binary", "yes_bid_dollars": "0.0700", "yes_ask_dollars": "0.1700",
     "yes_ask_size_fp": "150.00", "volume_fp": "0.00", "open_interest_fp": "0.00",
     "rules_primary": "If San Diego wins by more than 4.5 points in the UC Davis vs San Diego "
                      "college football game originally scheduled for Sep 5, 2026, then the "
                      "market resolves to Yes."},
]


def test_kalshi_ladder_parse(env):
    from datetime import date as _date
    from cfb.sources import kalshi
    t = kalshi.parse_ladder(KALSHI_TOTAL[0], "total")
    assert t["strike"] == 78.5 and t["date"] == _date(2026, 9, 5)
    assert t["away_raw"] == "Sacred Heart" and t["home_raw"] == "Monmouth"
    assert t["yes_ask"] == 0.17

    s = kalshi.parse_ladder(KALSHI_SPREAD[0], "spread")
    assert s["strike"] == 8.5
    assert s["team_raw"] == "San Diego"
    # matchup is phrased away-first: "UC Davis vs San Diego" => San Diego hosts
    assert s["away_raw"] == "UC Davis" and s["home_raw"] == "San Diego"


def test_kalshi_monotonicity_detects_incoherent_ladder(env, monkeypatch):
    """P(win by >8.5) cannot exceed P(win by >4.5). The live sample violated this."""
    import pandas as pd
    from cfb.sources import kalshi
    monkeypatch.setattr(kalshi, "fetch_markets", lambda *a, **k: KALSHI_SPREAD)
    lad = kalshi.ladder_board("spread")
    breaks = kalshi.monotonicity_breaks(lad)
    assert len(breaks) == 1
    b = breaks[0]
    assert b["lower_strike"] == 4.5 and b["higher_strike"] == 8.5
    assert b["higher_ask"] > b["lower_ask"]

    # a coherent ladder (totals sample) must produce no breaks
    monkeypatch.setattr(kalshi, "fetch_markets", lambda *a, **k: KALSHI_TOTAL)
    assert kalshi.monotonicity_breaks(kalshi.ladder_board("total")) == []


def test_ladder_probabilities_are_ordered(env):
    """Sanity on the pricing math itself: higher strike => lower probability."""
    from scipy.stats import norm
    p_hi = float(norm.sf(78.5, loc=72.0, scale=16.0))
    p_lo = float(norm.sf(70.5, loc=72.0, scale=16.0))
    assert p_lo > p_hi
    # away-side spread contract mirrors through the margin distribution
    p_home_by_7 = float(norm.sf(7.0, loc=3.0, scale=15.0))
    p_away_by_7 = float(norm.cdf(-7.0, loc=3.0, scale=15.0))
    assert 0 < p_home_by_7 < 1 and 0 < p_away_by_7 < 1
    assert p_home_by_7 > p_away_by_7          # home is favoured by 3


def test_ladder_guards_reject_thin_and_tails(env, monkeypatch):
    """Week 1 (no games played) must produce zero exchange picks, and tail rungs are refused."""
    import numpy as np
    import pandas as pd
    from datetime import date as _date
    from cfb import config as C, predict
    from cfb.sources import kalshi, odds

    day = _date(2026, 9, 5)
    out = pd.DataFrame([{
        "date": day, "home_team": "Monmouth", "away_team": "Sacred Heart",
        "total_pred": 72.0, "total_sigma": 16.0, "total_line": 72.0,
        "margin_pred": 3.0, "margin_sigma": 15.0, "spread_home": -3.0,
        "p_home_win": 0.58, "p_away_win": 0.42,
        "thin_data": True,          # <- week 1
    }])
    ladder = pd.DataFrame([{
        "kind": "total", "ticker": "T-79", "event_ticker": "E", "strike": 78.5,
        "team": None, "home_team": "Monmouth", "away_team": "Sacred Heart", "date": day,
        "yes_bid": 0.09, "yes_ask": 0.17, "quote_spread": 0.02, "ask_size": 500.0,
        "volume": 900.0, "tradeable": True,
    }])
    monkeypatch.setattr(kalshi, "ladder_board",
                        lambda kind, matcher=None: ladder if kind == "total" else ladder.iloc[0:0])
    monkeypatch.setattr(odds, "build_matcher", lambda teams: (lambda n: n))
    monkeypatch.setattr(predict, "fbs_teams", lambda g, s: {"Monmouth", "Sacred Heart"})

    res = predict._price_ladders(out.copy(), pd.DataFrame())
    assert res["kt_pick"].isna().all(), "thin-data games must never produce an exchange pick"

    # Not thin any more, but 78.5 is 6.5 above the book total and deep in the tail:
    # P(total > 78.5) with mu=72, sigma=16 is ~0.34, inside the band, so it survives the band
    # check -- what must stop it is the minimum-EV bar once the fee is paid.
    out2 = out.copy()
    out2.loc[0, "thin_data"] = False
    res2 = predict._price_ladders(out2, pd.DataFrame())
    if res2["kt_pick"].notna().any():
        assert res2["kt_ev"].iloc[0] >= C.KALSHI_MIN_EV
        assert C.KALSHI_PROB_MIN <= res2["kt_prob"].iloc[0] <= C.KALSHI_PROB_MAX
        assert abs(res2["kt_strike"].iloc[0] - 72.0) <= C.KALSHI_MAX_BOOK_GAP

    # A rung far from the book number must be refused outright.
    far = ladder.copy()
    far.loc[0, "strike"] = 95.0
    monkeypatch.setattr(kalshi, "ladder_board",
                        lambda kind, matcher=None: far if kind == "total" else far.iloc[0:0])
    res3 = predict._price_ladders(out2.copy(), pd.DataFrame())
    assert res3["kt_pick"].isna().all()
    assert res3["kt_rungs"].fillna(0).iloc[0] == 0
