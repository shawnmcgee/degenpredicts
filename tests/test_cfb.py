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


def test_kelly():
    from cfb.predict import american_payout, kelly
    assert abs(american_payout(-110) - 0.909) < 0.01
    assert kelly(0.50, 0.909) == 0.0
    assert kelly(0.60, 0.909) > 0
