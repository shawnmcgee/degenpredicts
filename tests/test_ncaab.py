"""Offline tests for the college basketball board. No network, no key.

Pinned here:

* the board lives in data/ncaab/ and docs/ncaab/ - the first version wrote to the root and would
  have replaced the sport chooser - and the package imports no other sport;
* ESPN's scoreboard parses in both odds shapes, with ranks, neutral sites, conferences, finals
  and the spread in book convention (negative = home favoured);
* the slate never carries a game that has tipped, and the games cache never re-asks for a day it
  already has - while a day that failed to load is asked for again;
* ratings are leak-free: a game's own result never reaches its features;
* the bake-off scores every candidate walk-forward - never on a season it trained on, the stack
  included - and the one-standard-error rule picks the simplest model that is not beaten;
* predict writes the NFL card's columns, never re-prices a started game, keeps the first
  published number, and still works off the ratings before any training run;
* grading is by event id, pushes are pushes, and CLV has the right sign on all four sides;
* the page renders the NFL card - projected score, spread and total tiles, pts off, the model's
  number - with ranks, filters and the bake-off table, and no "nan" anywhere.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ncaab import config, features, grade, models, predict, site, train
from ncaab.sources import espn, lines

ROOT = Path(__file__).resolve().parent.parent
TODAY = date(2026, 11, 4)


# ---------------------------------------------------------------------------------------
# fixtures: a sandboxed data/docs tree, a synthetic league, ESPN-shaped payloads
# ---------------------------------------------------------------------------------------
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    data, docs = tmp_path / "data" / "ncaab", tmp_path / "docs" / "ncaab"
    for name, path in {"DATA": data, "DOCS": docs, "GAMES": data / "games.csv",
                       "DAYS": data / "days.csv", "PICKS": data / "picks.csv",
                       "RESULTS": data / "results.csv", "METRICS": data / "metrics.json",
                       "MODEL_DIR": data / "models", "LINES": data / "lines.csv"}.items():
        monkeypatch.setattr(config, name, path)
    monkeypatch.setattr(config, "ESPN_SLEEP", 0.0)
    monkeypatch.setattr(config, "today_et", lambda: TODAY)
    config.ensure_dirs()
    return tmp_path


def league(seasons=range(2018, 2023), n_teams=80, n_visitors=10, games_per_team=30, seed=0):
    """A synthetic Division I in games.csv shape: strengths that churn between seasons, a home
    edge, a line that is sharper than any rating, and D2 visitors in November."""
    rng = np.random.default_rng(seed)
    ids = [str(100 + i) for i in range(n_teams)]
    vis = [str(9000 + i) for i in range(n_visitors)]
    confs = ["2", "4", "7", "8", "23", "62", "3", "44"]
    conf = {t: confs[i % len(confs)] for i, t in enumerate(ids)}
    strength = {t: rng.normal(0, 9) for t in ids}
    off = {t: rng.normal(0, 4) for t in ids}
    for v in vis:
        strength[v], off[v] = rng.normal(-25, 4), -6.0
    rows, gid = [], 1
    for s in seasons:
        for t in ids:
            strength[t] = 0.65 * strength[t] + rng.normal(0, 5)
            off[t] = 0.6 * off[t] + rng.normal(0, 2.5)
        order = sorted(ids, key=lambda t: -strength[t])
        rank = {t: i + 1 for i, t in enumerate(order[:25])}
        k = int(n_teams * games_per_team / 2 / 110)
        for day in range(110):
            d = date(s, 11, 4) + timedelta(days=day)
            teams = rng.permutation(ids)[: 2 * k]
            for i in range(k):
                h, a = teams[2 * i], teams[2 * i + 1]
                if day < 15 and i % 6 == 0:
                    a = vis[rng.integers(n_visitors)]
                neutral = bool(rng.random() < 0.08)
                form = rng.normal(0, 3)
                exp_m = strength[h] - strength[a] + (0.0 if neutral else 3.0)
                exp_t = 140 + off[h] + off[a]
                m = exp_m + form + rng.normal(0, 10.5)
                tot = exp_t + rng.normal(0, 12)
                hp, ap = int(round((tot + m) / 2)), int(round((tot - m) / 2))
                hp += hp == ap
                lined = a not in vis and rng.random() < 0.85
                rows.append({
                    "game_id": str(400000 + gid), "date": d, "season": s, "season_type": 2,
                    "kickoff_utc": f"{d.isoformat()}T23:00:00+00:00",
                    "home_id": h, "away_id": a, "home_team": f"Team {h}", "away_team": f"Team {a}",
                    "home_abbr": f"T{h}", "away_abbr": f"T{a}",
                    "home_conf": conf.get(h, ""), "away_conf": conf.get(a, ""),
                    "home_rank": rank.get(h, 99), "away_rank": rank.get(a, 99),
                    "neutral_site": neutral, "conf_game": conf.get(h) == conf.get(a), "note": "",
                    "home_points": hp, "away_points": ap, "total_points": hp + ap,
                    "home_margin": hp - ap,
                    "spread_home": -round(2 * (exp_m + 0.5 * form + rng.normal(0, 2))) / 2 if lined else np.nan,
                    "total_line": round(2 * (exp_t + rng.normal(0, 2))) / 2 if lined else np.nan,
                    "provider": "DraftKings"})
                gid += 1
    return pd.DataFrame(rows)


def team(tid, name, abbr, conf="", rank=99):
    return {"id": tid, "team": {"id": tid, "location": name, "displayName": f"{name} Mascots",
                                "shortDisplayName": name, "abbreviation": abbr,
                                "conferenceId": conf},
            "curatedRank": {"current": rank}}


def nested(spread_home, total, sp=(-110, -110), ou=(-105, -115)):
    s = (lambda v: f"{v:+g}" if v else "PK")
    return {"provider": {"name": "DraftKings"},
            "pointSpread": {"home": {"open": {"line": s(spread_home)},
                                     "close": {"line": s(spread_home), "odds": str(sp[0])}},
                            "away": {"close": {"line": s(-spread_home), "odds": str(sp[1])}}},
            "total": {"over": {"open": {"line": f"o{total}"},
                               "close": {"line": f"o{total}", "odds": str(ou[0])}},
                      "under": {"close": {"line": f"u{total}", "odds": str(ou[1])}}}}


def flat(spread_home, total, home_abbr, away_abbr):
    fav = home_abbr if spread_home <= 0 else away_abbr
    o = {"provider": {"name": "ESPN BET"},
         "details": f"{fav} {-abs(spread_home):g}" if spread_home else "EVEN",
         "spread": abs(spread_home),
         "homeTeamOdds": {"favorite": spread_home < 0, "spreadOdds": -110},
         "awayTeamOdds": {"favorite": spread_home > 0, "spreadOdds": -110}}
    if total is not None:
        o.update({"overUnder": total, "overOdds": -110, "underOdds": -110})
    return o


def event(eid, when, home, away, odds=None, state="pre", score=None, neutral=False,
          conf_game=False, note=None, status=None):
    h, a = dict(home, homeAway="home"), dict(away, homeAway="away")
    if score:
        h["score"], a["score"] = str(score[0]), str(score[1])
    name = status or {"pre": "STATUS_SCHEDULED", "in": "STATUS_IN_PROGRESS",
                      "post": "STATUS_FINAL"}[state]
    comp = {"id": eid, "date": when, "neutralSite": neutral, "conferenceCompetition": conf_game,
            "status": {"type": {"state": state, "completed": state == "post", "name": name}},
            "competitors": [h, a], "odds": [odds] if odds else []}
    if note:
        comp["notes"] = [{"type": "event", "headline": note}]
    return {"id": eid, "date": when, "season": {"year": 2027, "type": 2}, "competitions": [comp]}


class FakeESPN:
    """Serves scoreboards by day; records which days were asked for."""

    def __init__(self, book=None):
        self.book, self.asked = dict(book or {}), []

    def __call__(self, day):
        self.asked.append(day)
        payload = self.book.get(day, {"events": []})
        return None if payload is None else espn.parse_scoreboard(payload)


class FakeOdds:
    """ESPN's two per-game odds endpoints, keyed by event id. A game it does not know gets an
    empty core listing - ESPN answered, and had no line."""

    def __init__(self, core=None, summary=None):
        self.core_map, self.summary_map = dict(core or {}), dict(summary or {})
        self.core, self.summary = [], []

    def __call__(self, url, params=None, **kw):
        if params and "event" in params:
            self.summary.append(params["event"])
            return self.summary_map.get(params["event"])
        gid = url.rstrip("/").split("/")[-2]           # .../competitions/{id}/odds
        self.core.append(gid)
        return self.core_map.get(gid, {"count": 0, "items": []})


def core_item(details, spread, total, home_fav, provider="ESPN BET"):
    """One entry of the core API's odds listing, in its flat shape."""
    return {"provider": {"id": "58", "name": provider, "priority": 1}, "details": details,
            "overUnder": total, "spread": spread, "overOdds": -110, "underOdds": -105,
            "homeTeamOdds": {"favorite": home_fav, "spreadOdds": -112, "moneyLine": -300},
            "awayTeamOdds": {"favorite": not home_fav, "spreadOdds": -108, "moneyLine": 240}}


@pytest.fixture(scope="module")
def history():
    return league()


@pytest.fixture(scope="module")
def meta(history, tmp_path_factory):
    """One bake-off on the synthetic league, shared by the tests that need a meta.json."""
    tr, _, _ = features.build(history)
    D = features.d1_rows(tr.dropna(subset=["home_margin", "total_points"]))
    seasons = train.test_seasons(D)
    return {"sport": "ncaab", "trained_at": "2026-11-03T17:00:00+00:00", "select_rule": "1se",
            "jobs": {train.job_name(k, m): train.run_job(D, k, m, seasons) for k, m in train.JOBS}}


def slate(day, ids, names, state="pre", scores=None):
    """Six games on ``day``: nested and flat odds, a missing total, a ranked neutral-site game."""
    evs = []
    for i in range(6):
        h, a = ids[2 * i], ids[2 * i + 1]
        sp, tot = [-4.5, 2.5, -11.0, 0.0, -1.5, 6.5][i], [151.5, 140.0, 133.5, 145.0, 128.0, 149.5][i]
        odds = nested(sp, tot) if i % 2 == 0 else flat(sp, None if i == 3 else tot, f"T{h}", f"T{a}")
        if state == "post":
            odds = None                 # like ESPN: a finished game's scoreboard has no odds
        evs.append(event(f"{day:%m%d}{i}", f"{day.isoformat()}T23:{i:02d}:00Z",
                         team(h, names[h], f"T{h}", "2", 5 if i == 0 else 99),
                         team(a, names[a], f"T{a}", "23", 12 if i == 0 else 99), odds,
                         state=state, score=scores[i] if scores else None,
                         neutral=(i == 0), conf_game=(i % 2 == 1),
                         note="Maui Invitational - Semifinal" if i == 0 else None))
    return {"events": evs}


# ---------------------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------------------
def test_board_lives_in_its_own_folders_not_the_root():
    """The first version wrote data/games.csv and docs/index.html - the chooser's page."""
    assert config.DATA.parts[-2:] == ("data", "ncaab")
    assert config.DOCS.parts[-2:] == ("docs", "ncaab")
    for name in ("GAMES", "PICKS", "RESULTS", "METRICS", "MODEL_DIR", "DAYS", "LINES"):
        assert getattr(config, name).is_relative_to(config.DATA), name


def test_site_never_writes_the_chooser(sandbox):
    site.build()
    assert (config.DOCS / "index.html").exists()
    assert not (sandbox / "docs" / "index.html").exists()


def test_module_does_not_import_other_sports():
    for path in (ROOT / "ncaab").rglob("*.py"):
        for line in path.read_text().splitlines():
            assert not re.match(r"\s*(from|import)\s+(cfb|nfl|epl|nhl|nba|core)\b", line), \
                f"{path.relative_to(ROOT)} imports another sport: {line.strip()}"


def test_knobs_do_not_share_names_with_other_sports():
    """DEGEN_FIRST_SEASON retunes the NFL; it must not retune this board too."""
    src = (ROOT / "ncaab" / "config.py").read_text()
    names = set(re.findall(r'_env(?:_int|_float)?\("(DEGEN_[A-Z_]+)"', src))
    shared_ok = {"DEGEN_HTTP_TIMEOUT", "DEGEN_HTTP_RETRIES", "DEGEN_SITE_TITLE",
                 "DEGEN_SUPPORT_LABEL"}
    assert names and all(n.startswith("DEGEN_NCAAB_") or n in shared_ok for n in names), names


# ---------------------------------------------------------------------------------------
# ESPN
# ---------------------------------------------------------------------------------------
def test_scoreboard_parses_both_odds_shapes_with_context():
    p = {"events": [
        event("1", "2026-11-05T00:00Z", team("150", "Duke", "DUKE", "2", 5),
              team("153", "North Carolina", "UNC", "2", 12), nested(-4.5, 151.5), conf_game=True),
        event("2", "2026-11-05T01:00Z", team("2", "Auburn", "AUB", "23"),
              team("96", "Kentucky", "UK", "23", 9), flat(2.5, 148.0, "AUB", "UK"), neutral=True,
              note="Champions Classic"),
        event("3", "2026-11-04T17:00Z", team("41", "UConn", "CONN", "4", 1),
              team("9001", "Somewhere D2", "SD2"), None, state="post", score=(101, 55)),
    ]}
    df = espn.parse_scoreboard(p).set_index("game_id")
    duke = df.loc["1"]
    assert (duke.spread_home, duke.total_line, duke.over_price, duke.under_price) == (-4.5, 151.5, -105, -115)
    assert (duke.home_rank, duke.away_rank, duke.home_conf, bool(duke.conf_game)) == (5, 12, "2", True)
    assert duke.date == date(2026, 11, 4)          # 7pm ET on the 4th, not midnight UTC on the 5th
    auburn = df.loc["2"]
    assert auburn.spread_home == 2.5               # "UK -2.5": the away side is favoured
    assert bool(auburn.neutral_site) and auburn.note == "Champions Classic"
    assert auburn.home_rank == 99
    final = df.loc["3"]
    assert bool(final.final) and (final.home_points, final.home_margin, final.total_points) == (101, 46, 156)
    assert np.isnan(final.spread_home)


def test_a_moneyline_in_the_spread_slot_is_not_a_handicap():
    o = {"provider": {"name": "x"}, "details": "DUKE -5000", "overUnder": 150.5}
    df = espn.parse_scoreboard({"events": [event("9", "2026-11-05T00:00Z",
                                                 team("150", "Duke", "DUKE"),
                                                 team("9", "Tiny", "TINY"), o)]})
    assert np.isnan(df.loc[0, "spread_home"]) and df.loc[0, "total_line"] == 150.5


@pytest.mark.parametrize("junk", [None, [], "<html>", {"error": 1}, {"events": [{"id": 1}]}])
def test_anything_that_is_not_a_scoreboard_parses_to_nothing(junk):
    assert espn.parse_scoreboard(junk).empty


def test_slate_drops_games_under_way_postponed_or_past_tip(sandbox):
    day = TODAY
    fake = FakeESPN({day: {"events": [
        event("a", f"{day}T23:00:00Z", team("1", "A", "A"), team("2", "B", "B"), nested(-3, 140)),
        event("b", f"{day}T17:00:00Z", team("3", "C", "C"), team("4", "D", "D"), nested(-3, 140), state="in"),
        event("c", f"{day}T23:30:00Z", team("5", "E", "E"), team("6", "F", "F"), nested(-3, 140),
              status="STATUS_POSTPONED"),
        # still marked "pre", but its tip time has passed: ESPN is late, the price is in-play
        event("d", f"{day}T15:00:00Z", team("7", "G", "G"), team("8", "H", "H"), nested(-3, 140)),
    ]}})
    s = espn.slate(0, fetch=fake, now=pd.Timestamp(f"{day}T16:00:00Z"))
    assert list(s["game_id"]) == ["a"]


def test_cache_remembers_empty_days_and_retries_failed_ones(sandbox):
    d0 = date(2026, 11, 1)
    fake = FakeESPN({d0: {"events": [event("x", "2026-11-01T23:00Z", team("1", "A", "A"),
                                           team("2", "B", "B"), state="post", score=(70, 60))]},
                     d0 + timedelta(days=1): None})          # ESPN failed that day
    espn.update(start=d0, end=d0 + timedelta(days=2), recheck_days=0, fetch=fake)
    assert len(espn.load()) == 1
    fake.asked.clear()
    espn.update(start=d0, end=d0 + timedelta(days=2), recheck_days=0, fetch=fake)
    assert fake.asked == [d0 + timedelta(days=1)], "only the failed day is asked for again"


# ---------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------
def test_a_games_own_result_never_reaches_its_features(history):
    g = history[history["season"] <= 2019].copy()
    a, _, _ = features.build(g)
    n = len(g) // 2
    g2 = g.copy()
    g2.loc[g2.index[n], ["home_points", "home_margin", "total_points"]] += 40
    b, _, _ = features.build(g2)
    cols = features.BASE_FEATURES
    pd.testing.assert_frame_equal(a[cols].iloc[: n + 1], b[cols].iloc[: n + 1])
    assert not a[cols].iloc[n + 1:].equals(b[cols].iloc[n + 1:])


def test_spread_sign_is_flipped_exactly_once():
    df = pd.DataFrame({"spread_home": [-6.5, 3.0], "total_line": [140.5, 150.0]})
    assert list(features.market_line(df, "margin")) == [6.5, -3.0]
    assert list(features.market_line(df, "total")) == [140.5, 150.0]


def test_a_newcomer_starts_well_below_average(history):
    _, _, book = features.build(history)
    season = int(history["season"].max())
    d2 = [t for t in book.teams[season] if t.startswith("90")]
    d1 = [t for t in book.teams[season] if not t.startswith("90")]
    assert d2 and np.mean([book.teams[season][t].margin for t in d2]) < \
        np.mean([book.teams[season][t].margin for t in d1]) - 10


def test_d2_visitors_are_never_training_rows(history):
    tr, _, _ = features.build(history)
    D = features.d1_rows(tr)
    assert not D["away_id"].str.startswith("90").any()
    assert D["season"].min() == history["season"].min() + config.WARMUP_SEASONS


# ---------------------------------------------------------------------------------------
# the bake-off
# ---------------------------------------------------------------------------------------
def test_walk_forward_never_trains_on_the_season_it_scores(history, monkeypatch):
    seen = []
    real = models.fit_predict

    def spy(name, Xtr, ytr, Xte, market):
        seen.append((name, len(Xtr), len(Xte)))
        return real(name, Xtr, ytr, Xte, market)

    monkeypatch.setattr(models, "fit_predict", spy)
    monkeypatch.setattr(config, "SKIP_MODELS", {"rf", "mlp", "xgb"})
    tr, _, _ = features.build(history)
    D = features.d1_rows(tr.dropna(subset=["home_margin", "total_points"]))
    oos, used = train.walk_forward(D, "margin", True, train.test_seasons(D))
    lined = D[D["spread_home"].notna()]
    for S in oos["season"].unique():
        assert set(oos.loc[oos["season"] == S, "game_id"]) <= set(lined.loc[lined["season"] == S, "game_id"])
    n_by_season = lined.groupby("season").size()
    for S in sorted(oos["season"].unique()):
        expected_train = int(n_by_season[n_by_season.index < S].sum())
        assert any(n == expected_train for _, n, _ in seen), f"season {S} saw its own games"
    # the stack's first scored season has nothing earlier to learn weights from
    first = str(min(oos["season"]))
    assert used[int(first)] == "blend"


def test_stack_weights_come_only_from_earlier_seasons(history, monkeypatch):
    monkeypatch.setattr(config, "SKIP_MODELS", {"rf", "mlp", "xgb"})
    calls = []
    real = train._stack_weights

    def spy(frames, bases, market):
        calls.append(sorted({int(s) for f in frames for s in f["season"].unique()}))
        return real(frames, bases, market)

    monkeypatch.setattr(train, "_stack_weights", spy)
    tr, _, _ = features.build(history)
    D = features.d1_rows(tr.dropna(subset=["home_margin", "total_points"]))
    oos, _ = train.walk_forward(D, "total", True, train.test_seasons(D))
    scored = sorted(int(s) for s in oos["season"].unique())
    assert calls == [scored[:i] for i in range(1, len(scored))]


def test_every_candidate_is_scored_on_the_same_games(meta):
    for job in meta["jobs"].values():
        board = [r for r in job["leaderboard"] if not r.get("reference")]
        assert len({r["n"] for r in board}) == 1
        names = {r["model"] for r in board}
        assert {"ridge", "huber", "hgb", "rf", "mlp", "blend", "stack"} <= names
        assert ("xgb" in names) == models.available("xgb")
        assert job["chosen"] in names and job["best"] in names


def test_market_models_are_compared_with_the_line_itself(meta):
    for kind in ("margin", "total"):
        job = meta["jobs"][f"{kind}_market"]
        line = [r for r in job["leaderboard"] if r.get("reference")]
        assert line and line[0]["model"] == "line"
        assert 0 <= job["shrink"] <= 1 and job["sigma"] > 5
        assert job["by_gap"] and sum(b["n"] for b in job["by_gap"]) == job["hit_n"]
        # the synthetic line is far sharper than any rating, as a real one is
        nm = meta["jobs"][f"{kind}_nomarket"]
        ratings = next(r for r in nm["leaderboard"] if r["model"] == "ratings")
        assert line[0]["mae"] < ratings["mae"]


def test_one_standard_error_rule_takes_the_simplest_model_not_beaten():
    board = [{"model": "rf", "mae": 8.50, "z": 0.0}, {"model": "stack", "mae": 8.51, "z": 0.4},
             {"model": "ridge", "mae": 8.52, "z": 0.9}, {"model": "ratings", "mae": 9.1, "z": 6.0},
             {"model": "line", "mae": 8.53, "reference": True}]
    assert train.select(board, "1se") == ("ridge", "rf")
    assert train.select(board, "lowest") == ("rf", "rf")
    board[2]["z"] = 1.3                         # ridge is now clearly worse: rf is next in line
    assert train.select(board, "1se") == ("rf", "rf")


def test_nnls_never_bets_against_a_model():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 10, 500)
    P = np.c_[y + rng.normal(0, 3, 500), -y + rng.normal(0, 3, 500), y + rng.normal(0, 8, 500)]
    w = models.nnls_weights(P, y)
    assert (w >= 0).all() and w[1] == 0 and w[0] > w[2]


# ---------------------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------------------
def _seed(history, meta=None):
    espn.save(history.reindex(columns=espn.GAME_COLS))
    if meta is not None:
        (config.MODEL_DIR / "meta.json").write_text(json.dumps(meta))
    names = dict(zip(history["home_id"], history["home_team"]))
    ids = sorted(t for t in names if not t.startswith("90"))
    return ids, names


def test_predict_writes_the_nfl_card_columns(sandbox, history, meta):
    ids, names = _seed(history, meta)
    fake = FakeESPN({TODAY: slate(TODAY, ids, names)})
    out = predict.run(fetch=fake, update_games=False, days=0,
                      now=pd.Timestamp(f"{TODAY}T15:00:00Z"))
    picks = pd.read_csv(config.PICKS, dtype={"game_id": str})
    assert len(picks) == 6 and set(predict.PICK_COLS) <= set(picks.columns)
    for r in out.to_dict("records"):
        # the side always agrees with where the raw number falls against the line
        took_home = r["margin_raw"] > -r["spread_home"]
        assert r["spread_side"] == (r["home_team"] if took_home else r["away_team"])
        assert r["spread_number"] == (r["spread_home"] if took_home else -r["spread_home"])
        assert r["spread_pick"].startswith(r["spread_side"])
        if r["total_line"] == r["total_line"]:
            assert r["total_pick"] == ("Over" if r["total_raw"] > r["total_line"] else "Under")
        assert r["margin_model"].endswith("/market")
    no_total = out[out["total_line"].isna()]
    assert len(no_total) == 1 and no_total["total_pick"].isna().all()
    assert no_total["total_model"].str.endswith("/nomarket").all()   # still projected
    assert out["thin_data"].all()               # nobody has played four games in 2026 yet


def test_a_started_game_is_never_repriced_and_first_number_is_kept(sandbox, history, meta):
    ids, names = _seed(history, meta)
    morning = slate(TODAY, ids, names)
    predict.run(fetch=FakeESPN({TODAY: morning}), update_games=False, days=0,
                now=pd.Timestamp(f"{TODAY}T15:00:00Z"))
    first = pd.read_csv(config.PICKS, dtype={"game_id": str}).set_index("game_id")
    # evening: game 0 has tipped, game 1's line moved
    evening = json.loads(json.dumps(morning))
    evening["events"][0]["competitions"][0]["status"]["type"].update(state="in", name="STATUS_IN_PROGRESS")
    evening["events"][1]["competitions"][0]["odds"] = [flat(4.5, 141.0, f"T{ids[3]}", f"T{ids[2]}")]
    predict.run(fetch=FakeESPN({TODAY: evening}), update_games=False, days=0,
                now=pd.Timestamp(f"{TODAY}T21:30:00Z"))
    later = pd.read_csv(config.PICKS, dtype={"game_id": str}).set_index("game_id")
    g0, g1 = morning["events"][0]["id"], morning["events"][1]["id"]
    pd.testing.assert_series_equal(first.loc[g0], later.loc[g0])
    assert later.loc[g1, "spread_home"] == 4.5
    assert later.loc[g1, "first_seen_spread"] == 2.5
    assert later.loc[g1, "first_seen_at"] == first.loc[g1, "first_seen_at"]


def test_predict_works_off_the_ratings_before_any_training(sandbox, history):
    ids, names = _seed(history)
    out = predict.run(fetch=FakeESPN({TODAY: slate(TODAY, ids, names)}), update_games=False,
                      days=0, now=pd.Timestamp(f"{TODAY}T15:00:00Z"))
    assert len(out) == 6
    assert (out["margin_model"] == "ratings/nomarket").all()
    assert out["spread_pick"].str.len().gt(0).all()


def test_a_missing_model_falls_back_down_the_leaderboard(sandbox, history, meta, monkeypatch):
    m = json.loads(json.dumps(meta))
    job = m["jobs"]["margin_market"]
    job["chosen"] = "xgb"
    monkeypatch.setattr(models, "available", lambda n: n != "xgb")
    ids, names = _seed(history, m)
    out = predict.run(fetch=FakeESPN({TODAY: slate(TODAY, ids, names)}), update_games=False,
                      days=0, now=pd.Timestamp(f"{TODAY}T15:00:00Z"))
    first_other = next(r["model"] for r in job["leaderboard"]
                       if r["model"] != "xgb" and not r.get("reference"))
    assert (out["margin_model"] == f"{first_other}/market").all()


# ---------------------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------------------
def _row(**kw):
    base = {"home_team": "Duke", "away_team": "UNC", "spread_home": -4.5, "total_line": 150.5,
            "spread_side": "Duke", "total_pick": "Over", "first_seen_spread": -3.5,
            "first_seen_total": 148.5, "close_spread": -6.0, "close_total": 152.0,
            "home_margin": 7, "total_points": 151, "total_raw": 153.0, "margin_raw": 6.0}
    return {**base, **kw}


def test_results_and_clv_on_every_side():
    df = pd.DataFrame([
        _row(),                                                     # home -4.5 wins by 7, over
        _row(spread_side="UNC", total_pick="Under"),                # the other sides
        _row(home_margin=4, spread_home=-4.0, total_points=150, total_line=150.0),  # pushes
        _row(spread_home=np.nan, total_line=np.nan),               # no lines
    ])
    r = grade.results_for(df)
    assert list(r["spread_result"]) == ["win", "loss", "push", "none"]
    assert list(r["total_result"]) == ["win", "loss", "push", "none"]
    # home pick at -3.5 that closed -6: we got 2.5 points of value; the away pick lost it
    assert r.loc[0, "spread_clv"] == 2.5 and r.loc[1, "spread_clv"] == -2.5
    # over at 148.5 that closed 152: +3.5; the under: -3.5
    assert r.loc[0, "total_clv"] == 3.5 and r.loc[1, "total_clv"] == -3.5
    assert np.isnan(r.loc[3, "spread_clv"]) and np.isnan(r.loc[3, "total_clv"])


def test_grade_matches_by_event_id_and_voids_what_never_finishes(sandbox, history, meta, monkeypatch):
    ids, names = _seed(history, meta)
    predict.run(fetch=FakeESPN({TODAY: slate(TODAY, ids, names)}), update_games=False, days=0,
                now=pd.Timestamp(f"{TODAY}T15:00:00Z"))
    scores = [(80, 70), (60, 66), (70, 59), (75, 75 - 1), (64, 63), (90, 81)]
    finals = slate(TODAY, ids, names, state="post", scores=scores)
    finals["events"].pop(5)                     # never finalised: postponed
    fake = FakeESPN({TODAY: finals})
    odds = FakeOdds()                           # ESPN has no closing line on any of them
    monkeypatch.setattr(config, "today_et", lambda: TODAY + timedelta(days=1))
    done = grade.grade(fetch=True, fetch_day=fake, get_odds=odds)
    assert len(done) == 5 and set(done["spread_result"]) <= {"win", "loss", "push"}
    assert len(odds.core) == 5, "last night's finals are asked for their closing line"
    monkeypatch.setattr(config, "today_et", lambda: TODAY + timedelta(days=9))
    done = grade.grade(fetch=True, fetch_day=fake, get_odds=odds)
    assert len(done) == 6 and (done["spread_result"] == "void").sum() == 1
    m = grade.metrics(done, today=TODAY + timedelta(days=9))
    assert m["spreads"]["all_games"]["n"] == 5 and m["spreads"]["season"]["n"] == 0
    assert m["by_week"][0]["label"].startswith("Nov 2")


def test_metrics_feed_the_chooser_as_an_unstaked_record(sandbox):
    from core import landing          # the test may import core; the package may not
    m = {"updated": "2026-11-05", "spreads": {"season": {"n": 0, "units": 0.0},
                                              "all_games": {"n": 10, "wins": 6, "losses": 4,
                                                            "clv": 0.4}}}
    rec = landing._record(m, "spreads")
    assert rec["n"] == 0 and rec["model_wins"] == 6 and rec["clv"] == 0.4


# ---------------------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------------------
def test_page_renders_the_nfl_card(sandbox, history, meta):
    ids, names = _seed(history, meta)
    predict.run(fetch=FakeESPN({TODAY: slate(TODAY, ids, names)}), update_games=False, days=0,
                now=pd.Timestamp(f"{TODAY}T15:00:00Z"))
    site.build()
    html = (config.DOCS / "index.html").read_text()
    assert html.count('class="game"') == 6
    for needle in ("Projected score", "pts off", "Model: ", "Model total", 'class="meter"',
                   "#5</span>", "#12</span>", "Maui Invitational", "No line yet",
                   'data-f="top"', 'data-f="power"', "How the model was picked",
                   "The line itself", 'class="badge"', site._bakeoff(meta)["span"]):
        assert needle in html, needle
    text = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    assert not re.search(r"\bnan\b|\bNone\b", text), "a missing value leaked onto the page"
    assert json.loads((config.DOCS / "published.json").read_text())["published"] == str(TODAY)
    assert (config.DOCS / "picks.csv").exists()


def test_card_markup_matches_the_nfl_board():
    """Same classes in the same nesting, so the two boards read as one site."""
    nfl = (ROOT / "nfl" / "templates" / "index.html").read_text()
    ncaab = (ROOT / "ncaab" / "templates" / "index.html").read_text()
    for cls in ("game", "ghead", "matchup", "proj", "ctx", "picks", "pick", "meter"):
        assert f'class="{cls}"' in nfl and f'class="{cls}"' in ncaab, cls
    css = lambda s: re.search(r"\.picks\{[^}]*\}", s).group(0)
    assert css(nfl) == css(ncaab)


def test_every_board_links_to_college_basketball():
    for tpl, up in (("cfb/templates/index.html", "../"), ("cfb/templates/bigcard.html", "../../"),
                    ("nfl/templates/index.html", "../"), ("nfl/templates/props.html", "../../"),
                    ("epl/templates/index.html", "../"), ("nhl/templates/index.html", "../"),
                    ("nba/templates/index.html", "../"), ("nba/templates/props.html", "../../")):
        nav = re.search(r'<nav class="sports".*?</nav>', (ROOT / tpl).read_text(), re.S).group(0)
        assert f'href="{up}ncaab/"' in nav, tpl


def test_workflows_publish_and_rebuild_the_chooser():
    wf = ROOT / ".github" / "workflows"
    for name, cmd in (("ncaab-predict", "python -m ncaab.predict"),
                      ("ncaab-grade", "python -m ncaab.grade"),
                      ("ncaab-train", "python -m ncaab.train")):
        text = (wf / f"{name}.yml").read_text()
        assert cmd in text and "python -m core.landing" in text, name
        assert "concurrency: {group: ncaab-data" in text, name
    assert "ncaab" in (wf / "test.yml").read_text()
    text = (wf / "ncaab-train.yml").read_text()
    assert "python -m ncaab.sources.lines" in text and "if: always()" in text


# ---------------------------------------------------------------------------------------
# past lines: the scoreboard drops them once a game is final
# ---------------------------------------------------------------------------------------
GAME = {"game_id": "401", "season": 2025, "home_abbr": "DUKE", "away_abbr": "UNC"}


def test_core_listing_parses_and_skips_live_in_game_odds():
    live = core_item("UNC -21.5", 21.5, 171.5, False, provider="ESPN Bet - Live Odds")
    row = lines.fetch_one(GAME, FakeOdds(core={"401": {"count": 2, "items": [
        live, core_item("DUKE -7.5", -7.5, 145.5, True)]}}))
    assert row["status"] == "ok" and row["source"] == "core"
    assert (row["spread_home"], row["total_line"], row["provider"]) == (-7.5, 145.5, "ESPN BET")
    assert (row["over_price"], row["under_price"], row["spread_home_price"]) == (-110, -105, -112)
    away_fav = lines.fetch_one(GAME, FakeOdds(core={"401": {"items": [
        core_item("UNC -3", 3.0, 150.0, False)]}}))
    assert away_fav["spread_home"] == 3.0, "the away side favoured means a positive home line"


def test_a_total_no_college_game_could_have_is_refused():
    row = lines.fetch_one(GAME, FakeOdds(core={"401": {"items": [
        core_item("DUKE -7.5", -7.5, 14.5, True)]}}))
    assert row["spread_home"] == -7.5 and np.isnan(row["total_line"])


def test_a_listing_of_links_falls_back_to_the_summary_page():
    summary = {"header": {}, "pickcenter": [nested(-2.5, 138.5)]}
    fake = FakeOdds(core={"401": {"count": 1, "items": [{"$ref": "http://x/odds/58"}]}},
                    summary={"401": summary})
    row = lines.fetch_one(GAME, fake)
    assert fake.summary == ["401"] and row["source"] == "summary"
    assert (row["spread_home"], row["total_line"]) == (-2.5, 138.5)
    # and the source knob can skip the core API altogether
    fake = FakeOdds(summary={"401": summary})
    assert lines.fetch_one(GAME, fake, source="summary")["spread_home"] == -2.5 and not fake.core


def test_no_line_is_remembered_and_a_failed_game_is_retried(sandbox):
    games = pd.DataFrame([{**GAME, "game_id": gid, "date": TODAY, "kickoff_utc": ""}
                          for gid in ("1", "2", "3")])
    fake = FakeOdds(core={"1": {"items": [core_item("DUKE -4", -4.0, 140.0, True)]},
                          "3": "<html>not json</html>"})        # 2: empty listing
    lines.backfill(budget_min=5, only=games, get=fake)
    have = lines.load().set_index("game_id")
    assert have.loc["1", "status"] == "ok" and have.loc["2", "status"] == "none"
    assert "3" not in have.index, "a game neither endpoint answered is not recorded"
    fake.core.clear()
    lines.backfill(budget_min=5, only=games, get=fake)
    assert fake.core == ["3"], "only the failed game is asked for again"


def test_backfill_goes_newest_first_d1_only_and_respects_its_budget(sandbox, history):
    want = lines.todo(history, lines.load())
    assert want["season"].min() == history["season"].min() + config.WARMUP_SEASONS
    assert not want["away_id"].str.startswith("90").any(), "D2 visitors are never asked about"
    assert list(want["date"]) == sorted(want["date"], reverse=True)
    fake = FakeOdds()
    lines.backfill(history, budget_min=0, get=fake)
    assert fake.core == [], "no budget, no requests"
    recent = history[history["season"] >= 2021]
    lines.backfill(recent, budget_min=5, get=fake)
    assert set(lines.load()["season"]) == {2022}        # 2021 is that frame's warm-up season


def test_attach_prefers_the_scoreboard_then_espn_then_our_own_pick():
    g = pd.DataFrame({"game_id": ["a", "b", "c", "d"], "spread_home": [-1.0, np.nan, np.nan, np.nan],
                      "total_line": [140.0, np.nan, np.nan, np.nan]})
    L = pd.DataFrame({"game_id": ["a", "b", "c"], "status": ["ok", "ok", "none"],
                      "spread_home": [-9.0, -2.5, np.nan], "total_line": [150.0, 133.0, np.nan]})
    P = pd.DataFrame({"game_id": ["b", "c", "c"], "spread_home": [-8.0, 4.0, 5.5],
                      "total_line": [131.0, 120.0, 121.5]})
    out = lines.attach(g, L, P).set_index("game_id")
    assert list(out["spread_home"].fillna(99)) == [-1.0, -2.5, 5.5, 99]
    assert list(out["total_line"].fillna(99)) == [140.0, 133.0, 121.5, 99]


def test_training_rows_carry_the_back_filled_lines(sandbox, history):
    bare = history.assign(spread_home=np.nan, total_line=np.nan)
    espn.save(bare.reindex(columns=espn.GAME_COLS))
    lined = history[history["spread_home"].notna() & (history["season"] == 2021)]
    lines.save(lined.assign(status="ok")[["game_id", "season", "status", "spread_home", "total_line"]])
    D = train.assemble(fetch=False)
    got = D[D["spread_home"].notna()]
    assert len(got) and set(got["season"]) == {2021}
    assert set(got["game_id"]) <= set(lined["game_id"])


def test_grade_measures_clv_against_espns_close(sandbox, history, meta, monkeypatch):
    ids, names = _seed(history, meta)
    morning = slate(TODAY, ids, names)
    predict.run(fetch=FakeESPN({TODAY: morning}), update_games=False, days=0,
                now=pd.Timestamp(f"{TODAY}T15:00:00Z"))
    pick = pd.read_csv(config.PICKS, dtype={"game_id": str}).iloc[0]
    assert pick["first_seen_spread"] == -4.5
    scores = [(80, 70), (60, 66), (70, 59), (75, 74), (64, 63), (90, 81)]
    home = f"T{ids[0]}"
    odds = FakeOdds(core={pick["game_id"]: {"items": [core_item(f"{home} -6.5", -6.5, 155.0, True)]}})
    monkeypatch.setattr(config, "today_et", lambda: TODAY + timedelta(days=1))
    done = grade.grade(fetch=True, fetch_day=FakeESPN({TODAY: slate(TODAY, ids, names, "post", scores)}),
                       get_odds=odds)
    r = done.set_index("game_id").loc[pick["game_id"]]
    assert (r["close_spread"], r["close_total"]) == (-6.5, 155.0)
    took_home = pick["spread_side"] == pick["home_team"]
    assert r["spread_clv"] == (2.0 if took_home else -2.0)
    assert r["total_clv"] == (155.0 - 151.5 if pick["total_pick"] == "Over" else 151.5 - 155.0)


def test_page_shows_the_projection_bakeoff_until_lines_arrive(sandbox, history, meta):
    m = json.loads(json.dumps(meta))
    for k in ("margin_market", "total_market"):
        m["jobs"][k] = {"skipped": True, "reason": "not enough seasons with data"}
    ids, names = _seed(history, m)
    predict.run(fetch=FakeESPN({TODAY: slate(TODAY, ids, names)}), update_games=False, days=0,
                now=pd.Timestamp(f"{TODAY}T15:00:00Z"))
    site.build()
    html = (config.DOCS / "index.html").read_text()
    for needle in ("How the model was picked", "projections only", "Scored without the betting line",
                   "Ratings only (no ML)", "projects from team ratings and form alone"):
        assert needle in html, needle
    assert "The line itself" not in html

