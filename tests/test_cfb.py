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


def test_site_hides_empty_kalshi_sections(env):
    """Regression: pandas cannot hold None in a float64 column, so `where(notna, None)` left
    NaN in place. bool(nan) is True, so every Jinja `{% if %}` passed and the page rendered the
    literal string 'nan'. Empty Kalshi sections must not render at all."""
    import json
    import numpy as np
    import pandas as pd
    from cfb import site

    env.ensure_dirs()
    pd.DataFrame([{
        "prediction_date": "2026-09-04", "game_id": "g1", "season": 2026, "week": 1,
        "date": "2026-09-04", "tip_et": "Fri Sep 04, 6:30 PM",
        "home_team": "Eastern Michigan", "away_team": "San José State",
        "neutral_site": False, "total_line": 56.0, "spread_home": -3.0,
        "total_raw": 57.9, "total_pred": 57.9, "total_disagree": 1.9, "total_pick": "Over",
        "total_strength": "pass", "total_p_win": 0.51, "total_ev": -0.023, "total_stake": 0.0,
        "margin_raw": 5.9, "margin_pred": 5.9, "margin_disagree": 2.9,
        "spread_pick": "Eastern Michigan -3.0", "spread_strength": "pass",
        "spread_p_win": 0.51, "spread_ev": -0.024, "spread_stake": 0.0,
        "thin_data": True, "h_games": 0, "a_games": 0,
        # every Kalshi field empty - the guards published nothing
        "kt_pick": np.nan, "kt_ask": np.nan, "kt_prob": np.nan, "kt_ev": np.nan,
        "ks_pick": np.nan, "ks_ask": np.nan, "ks_prob": np.nan, "ks_ev": np.nan,
        "ml_pick": np.nan, "ml_ask": np.nan, "ml_roi": np.nan, "ml_stake": np.nan,
        "ml_model_cents": np.nan, "kalshi_incoherent": False,
    }]).to_csv(env.PICKS, index=False)
    env.METRICS.write_text(json.dumps({"updated": "2026-09-04", "break_even": 51.75}))

    site.build()
    html = (env.DOCS / "index.html").read_text()
    assert "Eastern Michigan" in html
    assert "nan" not in _rendered_text(html), "empty fields must not render as 'nan'"
    for section in ("Kalshi total", "Kalshi spread", "Kalshi moneyline", "Kalshi play"):
        assert section not in html, f"{section} rendered with no pick"


def test_kickoff_sort_key_is_chronological_not_alphabetical():
    """Sorting the printed label sorts by weekday NAME, which is not a time order.

    Reported from the live board: with "Sort by kickoff" on, a Wednesday game went to the
    bottom. The cause was `data-kick="{{ g.tip_et }}"` compared with localeCompare, so
    "Wed Sep 09, 8:20 PM" sorted against "Mon Sep 14, 8:15 PM" as text: Fri < Mon < Sat < Sun
    < Thu < Tue < Wed. Monday came first, Wednesday last. Two more failures rode along - the
    hour compared as text put "10:00 PM" ahead of "12:00 PM" on the same day, and dates
    interleaved across weeks. The ISO timestamp is the only field that orders correctly.
    """
    from cfb.site import _kick_key

    assert _kick_key("2026-09-09T20:20:00-04:00", "Wed Sep 09, 8:20 PM") == \
        "2026-09-09T20:20:00-04:00"

    # an unannounced kickoff sorts last rather than at a placeholder time
    for tip in ("", None, float("nan"), "nan", "   "):
        assert _kick_key("2026-09-09T20:20:00-04:00", tip) == ""
    for iso in ("", None, float("nan"), "NaT", "None"):
        assert _kick_key(iso, "Wed Sep 09, 8:20 PM") == ""

    # the three orderings the old label sort got wrong
    board = [
        ("2026-09-09T20:20:00-04:00", "Wed Sep 09, 8:20 PM"),
        ("2026-09-14T20:15:00-04:00", "Mon Sep 14, 8:15 PM"),
        ("2026-09-12T22:00:00-04:00", "Sat Sep 12, 10:00 PM"),
        ("2026-09-12T12:00:00-04:00", "Sat Sep 12, 12:00 PM"),
        ("2026-09-12T09:30:00-04:00", "Sat Sep 12, 9:30 AM"),
        ("", "Time TBD"),
    ]
    keys = [_kick_key(i, t if t != "Time TBD" else "") for i, t in board]
    labels = [t for _, t in board]
    order = sorted(range(len(keys)), key=lambda i: (keys[i] == "", keys[i]))
    got = [labels[i] for i in order]

    assert got[0] == "Wed Sep 09, 8:20 PM", "Wednesday must lead, not trail"
    assert got.index("Sat Sep 12, 9:30 AM") < got.index("Sat Sep 12, 12:00 PM")
    assert got.index("Sat Sep 12, 12:00 PM") < got.index("Sat Sep 12, 10:00 PM")
    assert got.index("Wed Sep 09, 8:20 PM") < got.index("Mon Sep 14, 8:15 PM")
    assert got[-1] == "Time TBD"

    # and the label sort really would have failed - this is what was shipped
    assert sorted(labels)[0] != "Wed Sep 09, 8:20 PM"


def test_kickoff_sort_uses_a_numeric_comparator_in_the_page():
    """The template must compare timestamps numerically, not with localeCompare."""
    from pathlib import Path
    tpl = (Path(__file__).resolve().parent.parent
           / "cfb" / "templates" / "index.html").read_text()
    assert 'data-kick="{{ g.kick_sort' in tpl, "the page must sort on the ISO key"
    assert "g.tip_et or 'zzz'" not in tpl, "the display label must not be the sort key"
    assert "kickAt" in tpl and "Date.parse" in tpl
    assert "dataset.kick || '').localeCompare" not in tpl


def _rendered_text(html: str) -> str:
    """Page content with <script> and <style> stripped.

    The "no literal nan" guard below greps a lowercased page, so it fires on any identifier
    that happens to contain those three letters - isNaN, financial, tenant. What it is
    actually guarding is a pandas NaN leaking into a displayed field, so it should look at
    the content and not at the code around it.
    """
    import re
    body = re.sub(r"<script\b.*?</script>", "", html, flags=re.S | re.I)
    return re.sub(r"<style\b.*?</style>", "", body, flags=re.S | re.I).lower()


# --------------------------------------------------------------------------------------
# Guardrails for the failures that were live in production, each one written so that it
# fails again if the fix is reverted.
# --------------------------------------------------------------------------------------
def _sp_table(teams, seasons):
    return pd.DataFrame([{"season": s, "team": t, "sp_overall": 5.0, "sp_off": 30.0,
                          "sp_def": 25.0} for s in seasons for t in teams])


def test_fbs_membership_prefers_the_roster_over_whatever_the_feed_returned(env):
    """The CFBD `division`/`classification` filter is silently ignored when misnamed, so the
    games file is not proof of anything. Membership has to come from a roster we hold."""
    from cfb.features import fbs_membership, is_fbs_game

    games = pd.DataFrame([
        {"season": 2024, "home_team": "Team 001", "away_team": "Kenyon",
         "home_classification": "fbs", "away_classification": "iii"},
    ])
    sp = _sp_table(["Team 001", "Team 002"], [2024])
    members = fbs_membership(games, sp)
    assert members[2024] == {"Team 001", "Team 002"}
    assert is_fbs_game(members, 2024, "Team 001", "Team 002")
    assert not is_fbs_game(members, 2024, "Team 001", "Kenyon")
    # a season we hold no roster for must not be silently deleted
    assert is_fbs_game(members, 1999, "Anyone", "Anyone Else")


def test_season_roster_stays_in_a_sane_band(env):
    """FBS has run 129-139 teams and ~800 FBS-vs-FBS games a season. A roster or a game count
    far outside that means the feed changed shape, which is exactly how 699 teams including
    Sewanee and Wayland Baptist got into the training data unnoticed."""
    from cfb.features import fbs_membership, is_fbs_game

    g, _ = synth()
    sp = _sp_table(sorted(set(g.home_team) | set(g.away_team)), SEASONS)
    members = fbs_membership(g, sp)
    for season in SEASONS:
        assert 100 <= len(members[season]) <= 145, f"{season}: {len(members[season])} teams"
        played = g[g.season == season]
        fbs = sum(is_fbs_game(members, season, r.home_team, r.away_team)
                  for r in played.itertuples())
        assert fbs / len(played) > 0.9, f"{season}: only {fbs}/{len(played)} FBS vs FBS"


def test_the_board_drops_games_the_model_cannot_price(env):
    """36 of one week's 85 published games were FBS vs FCS, priced off a rating the visitor
    does not have, on a site that says it covers FBS."""
    from cfb.predict import build_board

    today = env.today_et()
    season = env.season_of(today)
    sched = pd.DataFrame([
        {"game_id": "1", "season": season, "week": 2, "date": today + timedelta(days=1),
         "home_team": "Team 001", "away_team": "Team 002", "completed": False},
        {"game_id": "2", "season": season, "week": 2, "date": today + timedelta(days=1),
         "home_team": "Team 001", "away_team": "Florida A&M", "completed": False},
    ])
    lines = pd.DataFrame([{"game_id": "1", "spread_home": -3.5, "total_line": 55.5},
                          {"game_id": "2", "spread_home": -56.5, "total_line": 62.5}])
    board = build_board(sched, lines, sp=_sp_table(["Team 001", "Team 002"], [season]))
    assert list(board["game_id"]) == ["1"]


def test_nothing_is_staked_on_a_market_we_declined(env):
    from cfb.predict import _stakes

    p_win = [0.99, 0.99, 0.99, 0.99]
    payout = [1.0, 1.0, 1.0, 1.0]
    stakes = _stakes(p_win, payout, ["play", "bold", "pass", "thin"])
    assert stakes[0] > 0 and stakes[1] > 0
    assert stakes[2] == 0.0 and stakes[3] == 0.0, "a pass or a thin game must stake nothing"


def test_clv_moves_when_the_line_moves(env, monkeypatch):
    """CLV was identically 0.00 across all 105 graded games because `grade` compared the
    cached line against itself. This fails if the close is ever read from the same snapshot
    the pick was built from, or if it is measured against the last overwrite."""
    from cfb import grade as G
    from cfb.sources import cfbd

    picks = pd.DataFrame([{
        "game_id": "1", "season": 2024, "week": 3, "date": date(2024, 9, 14),
        "home_team": "Team 001", "away_team": "Team 002",
        "total_line": 55.5, "spread_home": -3.5,
        "first_seen_total": 55.5, "first_seen_spread": -3.5, "first_seen_at": "2024-09-10",
        "total_pick": "Over", "margin_edge": 1.0,
        "total_pred": 57.0, "margin_pred": 5.0,
        "total_payout": 0.91, "spread_payout": 0.91,
        "total_stake": 0.0, "spread_stake": 0.0,
        "total_disagree": 1.0, "margin_disagree": 1.0,
        "total_strength": "pass", "spread_strength": "pass",
    }])
    finals = pd.DataFrame([{"game_id": "1", "season": 2024, "week": 3,
                            "date": date(2024, 9, 14), "completed": True,
                            "home_points": 30.0, "away_points": 24.0,
                            "total_points": 54.0, "home_margin": 6.0}])
    # the close moved two points on the total and a point and a half on the spread
    closed = pd.DataFrame([{"game_id": "1", "spread_home": -5.0, "total_line": 57.5}])

    picks.to_csv(env.PICKS, index=False)
    if env.RESULTS.exists():
        env.RESULTS.unlink()
    monkeypatch.setattr(cfbd, "update_games", lambda *a, **k: finals)
    monkeypatch.setattr(cfbd, "update_lines", lambda *a, **k: closed)
    # if grade ever falls back to the cached file, this poisoned copy makes CLV zero again
    monkeypatch.setattr(cfbd, "load_lines",
                        lambda *a, **k: pd.DataFrame([{"game_id": "1", "spread_home": -3.5,
                                                       "total_line": 55.5}]))
    done = G.grade()
    assert done["total_clv"].iloc[0] == pytest.approx(2.0)    # Over 55.5, closed 57.5
    assert done["spread_clv"].iloc[0] == pytest.approx(-1.5)  # took home -3.5, closed -5.0
    assert done[["total_clv", "spread_clv"]].abs().to_numpy().sum() > 0


def test_first_published_number_survives_the_daily_overwrite(env):
    from cfb.predict import _carry_first_seen

    prev = pd.DataFrame([{"game_id": "1", "first_seen_spread": -3.5,
                          "first_seen_total": 55.5, "first_seen_at": "2024-09-10"}])
    out = pd.DataFrame([{"game_id": "1", "spread_home": -5.0, "total_line": 57.5,
                         "first_seen_spread": -5.0, "first_seen_total": 57.5,
                         "first_seen_at": "2024-09-14"},
                        {"game_id": "2", "spread_home": -7.0, "total_line": 44.5,
                         "first_seen_spread": -7.0, "first_seen_total": 44.5,
                         "first_seen_at": "2024-09-14"}])
    got = _carry_first_seen(prev, out).set_index("game_id")
    assert got.loc["1", "first_seen_spread"] == -3.5      # Tuesday's number, not Saturday's
    assert got.loc["1", "first_seen_at"] == "2024-09-10"
    assert got.loc["1", "spread_home"] == -5.0            # display fields still refresh
    assert got.loc["2", "first_seen_spread"] == -7.0      # a new game keeps today's


def test_missing_opening_line_is_missing_not_zero(env):
    """`spread_move` is absent for every pre-2021 row and present on 96% of live rows.
    Filling the gap with 0.0 teaches the model that six seasons of football never moved."""
    from cfb.features import _market

    row = {"exp_total": 50.0, "exp_margin": 3.0}
    got = _market(dict(row), total_line=55.5, spread_home=-3.5)
    assert np.isnan(got["total_move"]) and np.isnan(got["spread_move"])
    got = _market(dict(row), total_line=55.5, spread_home=-3.5,
                  total_open=53.5, spread_open=-2.5)
    assert got["total_move"] == pytest.approx(2.0)
    assert got["spread_move"] == pytest.approx(-1.0)


def test_a_live_score_is_not_a_final(env, monkeypatch):
    """CFBD returns a `completed` flag. Treating "has points" as final grades a pick and
    permanently moves the rating book on a game that is still being played."""
    from cfb.sources import cfbd

    payload = [
        {"id": 1, "season": 2024, "week": 3, "startDate": "2024-09-14T20:00:00.000Z",
         "homeTeam": "Team 001", "awayTeam": "Team 002",
         "homePoints": 21, "awayPoints": 17, "completed": False},
        {"id": 2, "season": 2024, "week": 3, "startDate": "2024-09-14T20:00:00.000Z",
         "homeTeam": "Team 003", "awayTeam": "Team 004",
         "homePoints": 31, "awayPoints": 10, "completed": True},
    ]
    monkeypatch.setattr(cfbd, "get_json", lambda *a, **k: payload)
    got = cfbd.fetch_games(2024, season_type="regular").set_index("game_id")
    assert not got.loc["1", "completed"], "a live game with a score is not final"
    assert got.loc["2", "completed"]
    assert np.isnan(got.loc["1", "home_margin"])


def test_consensus_resolves_each_market_independently(env):
    """A provider listed with a total but no spread used to win the whole row and hand back
    a NaN spread, dropping a game other books had priced."""
    from cfb.sources import cfbd

    lines = pd.DataFrame([
        {"game_id": "1", "season": 2024, "week": 3, "date": date(2024, 9, 14),
         "home_team": "H", "away_team": "A", "provider": "Bovada",
         "spread_home": np.nan, "spread_open": np.nan,
         "total_line": 55.5, "total_open": 54.5, "home_ml": np.nan, "away_ml": np.nan},
        {"game_id": "1", "season": 2024, "week": 3, "date": date(2024, 9, 14),
         "home_team": "H", "away_team": "A", "provider": "DraftKings",
         "spread_home": -3.5, "spread_open": -3.0,
         "total_line": 56.0, "total_open": 55.0, "home_ml": np.nan, "away_ml": np.nan},
    ])
    got = cfbd.consensus(lines).iloc[0]
    assert got["total_line"] == 55.5      # Bovada is preferred and priced the total
    assert got["spread_home"] == -3.5     # but DraftKings is the only one with a spread


def test_moneyline_probability_belongs_to_the_side_that_won(env, monkeypatch):
    """`prob` and `ev` used to survive from the final loop iteration, so choosing the home
    side wrote the home ask beside the AWAY probability - and then applied the 0.20-0.80
    playability band to the wrong side of the game."""
    from cfb import predict as P
    from cfb.sources import kalshi

    today = env.today_et()
    board = pd.DataFrame([
        {"date": today, "home_team": "Team 001", "away_team": "Team 002", "team": "Team 001",
         "yes_ask": 0.40, "ticker": "H", "tradeable": True, "quote_spread": 0.02},
        {"date": today, "home_team": "Team 001", "away_team": "Team 002", "team": "Team 002",
         "yes_ask": 0.95, "ticker": "A", "tradeable": True, "quote_spread": 0.02},
    ])
    monkeypatch.setattr(kalshi, "moneyline_board", lambda *a, **k: board)
    monkeypatch.setattr(P.odds, "build_matcher", lambda *a, **k: object())

    out = pd.DataFrame([{"date": today, "home_team": "Team 001", "away_team": "Team 002",
                         "p_home_win": 0.70, "p_away_win": 0.30, "thin_data": False}])
    sp = _sp_table(["Team 001", "Team 002"], [env.season_of(today)])
    got = P._attach_kalshi(out, pd.DataFrame(), sp).iloc[0]

    # home is the better buy at 40c against a 70% model, so home must be the side taken
    assert got["ml_ref_side"] == "Team 001"
    assert got["ml_ref_prob"] == pytest.approx(0.70), "the away probability leaked across"
    assert got["ml_ref_ev"] == pytest.approx(got["kalshi_home_ev"])
    assert got["ml_pick"] == "Team 001 to win"


def test_missing_completed_field_falls_back_to_the_score(env, monkeypatch):
    """Guarding against a live score must not turn into "nothing is ever final" if CFBD stops
    sending the flag - that failure is far worse than the one it prevents."""
    from cfb.sources import cfbd

    payload = [{"id": 1, "season": 2024, "week": 3,
                "startDate": "2024-09-14T20:00:00.000Z",
                "homeTeam": "Team 001", "awayTeam": "Team 002",
                "homePoints": 21, "awayPoints": 17}]
    monkeypatch.setattr(cfbd, "get_json", lambda *a, **k: payload)
    got = cfbd.fetch_games(2024, season_type="regular")
    assert bool(got["completed"].iloc[0])


def test_the_record_scores_only_the_games_the_board_still_publishes(env, monkeypatch):
    """The first 105 graded picks included 66 FBS-vs-FCS games, published before `build_board`
    filtered them out. They ran 63.6% on totals against 51.3% for the FBS-vs-FBS picks, so
    averaging the two reported a 59% headline for a board that will never offer another one.
    The record splits on the flag, and the off-board sample is reported rather than dropped."""
    from cfb import grade as G

    sp = _sp_table(["Alpha", "Bravo"], [2026])
    monkeypatch.setattr(G.cfbd, "load_sp", lambda: sp)
    monkeypatch.setattr(G.cfbd, "load_games", lambda: pd.DataFrame(
        columns=["season", "home_team", "away_team"]))

    def row(away, total_result, spread_result):
        return {"game_id": f"g{away}{total_result}", "season": 2026, "week": 1,
                "home_team": "Alpha", "away_team": away,
                "total_result": total_result, "spread_result": spread_result,
                "total_strength": "play", "spread_strength": "play",
                "total_stake": 1.0, "spread_stake": 1.0,
                "total_units": 1.0 if total_result == "win" else -1.0,
                "spread_units": 1.0 if spread_result == "win" else -1.0,
                "total_disagree": 2.0, "margin_disagree": 2.0,
                "total_abs_err": 10.0, "margin_abs_err": 10.0}

    done = pd.DataFrame([
        row("Bravo", "win", "loss"),        # FBS vs FBS
        row("Bravo", "loss", "loss"),       # FBS vs FBS
        row("Citadel", "win", "win"),       # FBS vs FCS - off the board now
        row("Furman", "win", "win"),        # FBS vs FCS - off the board now
    ])
    m = G.metrics(done)

    assert m["graded_games"] == 4 and m["off_board_games"] == 2
    # headline counts the two comparable games only, not the flattering FCS pair
    assert m["totals"]["all_games"]["n"] == 2
    assert m["totals"]["all_games"]["win_pct"] == 50.0
    # the off-board picks are still reported, not quietly discarded
    assert m["totals"]["off_board"]["n"] == 2
    assert m["totals"]["off_board"]["win_pct"] == 100.0
    assert m["spreads"]["all_games"]["win_pct"] == 0.0
    # by_week describes the same population as the headline
    assert sum(w["games"] for w in m["by_week"]) == 2


def test_slates_bucket_a_saturday_by_kickoff(env):
    """A CFB Saturday is ~50 games in one list. The slate chips split it into the windows
    people actually think in, and the boundaries have to land on real kickoffs: 12:45 is noon,
    3:30 and 4:15 are afternoon, 6:00 through 10:15 are night, 10:30+ is west-coast late."""
    from cfb.site import _slate

    def at(label):
        return _slate(None, "Sat Sep 12, " + label)

    assert at("12:00 PM") == "early"
    assert at("12:45 PM") == "early"
    assert at("2:30 PM") == "early"
    assert at("3:30 PM") == "afternoon"
    assert at("4:15 PM") == "afternoon"
    assert at("6:00 PM") == "night", "6pm is an evening kickoff, not an afternoon one"
    assert at("8:00 PM") == "night"
    assert at("10:15 PM") == "night"
    assert at("10:30 PM") == "late"
    assert at("11:59 PM") == "late"
    # midnight and noon are where 12-hour parsing usually breaks
    assert at("12:30 AM") == "early"
    # an exact ISO kickoff wins over the printed label, and carries its own offset
    assert _slate("2026-09-12T15:30:00-04:00", "Sat Sep 12, 9:99 XM") == "afternoon"
    assert _slate("2026-09-13T00:30:00-04:00", "") == "early"
    # no time anywhere is its own bucket, never a silent drop
    assert _slate(None, None) == "tbd"
    assert _slate("nan", "") == "tbd"


def test_slate_tabs_skip_empty_slates_and_put_tbd_last(env):
    """An empty chip invites a click that blanks the board, so only slates with games get one.
    TBD is offered only when something is genuinely unannounced, and never first."""
    from cfb.site import _slate_tabs

    picks = ([{"slate": "night"}] * 3 + [{"slate": "early"}] * 2
             + [{"slate": "tbd"}] + [{"slate": "late"}])
    tabs = _slate_tabs(picks)
    assert [t["key"] for t in tabs] == ["early", "night", "late", "tbd"], "kickoff order"
    assert [t["n"] for t in tabs] == [2, 3, 1, 1]
    assert sum(t["n"] for t in tabs) == len(picks), "counts must account for every game"
    # no afternoon games -> no afternoon chip
    assert all(t["key"] != "afternoon" for t in tabs)
    # and a board with nothing unannounced offers no TBD chip
    assert all(t["key"] != "tbd" for t in _slate_tabs([{"slate": "night"}]))


def test_slate_boundaries_are_configurable(env, monkeypatch):
    """The cuts are a judgement call, so they are overridable - and a typo in the env var must
    fall back to the defaults rather than take down the daily site build."""
    from cfb import config as C

    monkeypatch.setenv("DEGEN_CFB_SLATES", "13,17,21")
    assert [s[2] for s in C.slates()] == [0.0, 13.0, 17.0, 21.0]
    monkeypatch.setenv("DEGEN_CFB_SLATES", "not,a,number")
    assert [s[2] for s in C.slates()] == [0.0, 15.0, 18.0, 22.5]
    monkeypatch.setenv("DEGEN_CFB_SLATES", "15,18")        # wrong arity
    assert [s[2] for s in C.slates()] == [0.0, 15.0, 18.0, 22.5]
