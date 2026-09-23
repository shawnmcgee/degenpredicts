"""Offline tests for the NHL pipeline. No network, no API key.

Structured like ``tests/test_nfl.py`` and ``tests/test_epl.py``: thresholds are asserted as
ABSOLUTE values rather than against another sport's config, so retuning another pipeline cannot
fail this suite. CI runs one job per sport, so a red check names the sport that broke.

The guardrails below each pin something that went wrong, or nearly did, while this was built:

* the uploaded predecessor averaged "recent goals" AFTER appending the game being predicted, so
  its features contained the answer - the leak-free replay is pinned directly;
* the first backtest took the median of American odds across books, and -115 with +105 has a
  median of -5, a "20x payout" - so prices are aggregated in decimal and a test says so;
* a Poisson grid misprices the puck line by twelve points of one-goal margins, so the late-game
  chase is pinned to move them the way real games do.
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


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Sandbox every path before any nhl module is re-read.

    The modules read ``config.NAME`` at call time precisely so this works; a module that bound
    its paths at import would quietly read and write the real repo data instead.
    """
    monkeypatch.setenv("DEGEN_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("DEGEN_DOCS", str(tmp_path / "docs"))
    monkeypatch.setenv("DEGEN_ROOT", str(ROOT))
    monkeypatch.delenv("DEGEN_NHL_DOCS", raising=False)
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    import importlib
    import sys
    from nhl import config as cfg
    importlib.reload(cfg)
    for mod in ("nhl.teams", "nhl.odds_math", "nhl.scoreline", "nhl.ratings", "nhl.features",
                "nhl.glm", "nhl.sources.nhle", "nhl.sources.odds", "nhl.sources.history",
                "nhl.sources.starters", "nhl.train", "nhl.predict", "nhl.grade", "nhl.site"):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    # Offline: the starting-goalie page is "down" unless a test hands it a page to read.
    from nhl.sources import starters
    monkeypatch.setattr(starters, "fetch", lambda day: None)
    cfg.ensure_dirs()
    yield cfg


def _rendered_text(html: str) -> str:
    body = re.sub(r"<script\b.*?</script>", "", html, flags=re.S | re.I)
    return re.sub(r"<style\b.*?</style>", "", body, flags=re.S | re.I).lower()


# ---------------------------------------------------------------------------------
# Teams: the guardrail that matters most
# ---------------------------------------------------------------------------------
def test_canonical_map_resolves_and_refuses():
    from nhl.teams import TEAMS, UnknownTeam, canon, from_name, is_known

    assert len(TEAMS) == 32
    for raw, want in [("ATL", "WPG"), ("PHX", "UTA"), ("ARI", "UTA"), ("LA", "LAK"),
                      ("NJ", "NJD"), ("SJ", "SJS"), ("TB", "TBL"), (" tbl ", "TBL"),
                      ("WAS", "WSH"), ("VEG", "VGK"), ("MON", "MTL")]:
        assert canon(raw) == want, f"{raw} -> {canon(raw)}"
    with pytest.raises(UnknownTeam):
        canon("QUE")
    with pytest.raises(UnknownTeam):
        canon(None)
    assert not is_known("HFD")
    # the feed spells these three ways between them
    for name, want in [("Montréal Canadiens", "MTL"), ("Montreal Canadiens", "MTL"),
                       ("St Louis Blues", "STL"), ("St. Louis Blues", "STL"),
                       ("Utah Mammoth", "UTA"), ("Utah Hockey Club", "UTA"),
                       ("Arizona Coyotes", "UTA"), ("Vegas Golden Knights", "VGK")]:
        assert from_name(name) == want
    # a bare city that names two clubs is refused, not guessed
    with pytest.raises(UnknownTeam):
        from_name("New York")


def test_every_team_in_the_committed_data_is_mapped():
    """A silently wrong code splits or merges a franchise's rating history."""
    from nhl.teams import TEAMS
    for name in ("games.csv", "lines.csv"):
        path = ROOT / "data" / "nhl" / name
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["home_team", "away_team"])
        unknown = (set(df["home_team"]) | set(df["away_team"])) - set(TEAMS)
        assert not unknown, f"{name} holds unmapped teams: {unknown}"


def test_relocated_clubs_are_measured_from_the_building_they_played_in():
    from nhl.teams import TEAMS, haversine_km, venue
    assert venue("WPG", 2010)[0] < 40 < venue("WPG", 2011)[0]       # Atlanta, then Winnipeg
    assert venue("UTA", 2023)[0] < 34 < venue("UTA", 2024)[0]       # Tempe, then Salt Lake
    for t in TEAMS:
        for s in (2005, 2015, 2026):
            assert venue(t, s) is not None, f"{t} has no arena in {s}"
    assert 3500 < haversine_km(venue("BOS", 2025), venue("VAN", 2025)) < 4200


# ---------------------------------------------------------------------------------
# Prices: decimal in, decimal out
# ---------------------------------------------------------------------------------
def test_american_and_decimal_convert_both_ways():
    from nhl.odds_math import american_to_decimal, decimal_to_american, devig_two, ev
    assert american_to_decimal(-150) == pytest.approx(1.6667, abs=1e-4)
    assert american_to_decimal(130) == pytest.approx(2.30)
    assert decimal_to_american(2.05) == pytest.approx(105)
    assert decimal_to_american(1.8) == pytest.approx(-125)
    assert np.isnan(american_to_decimal(50))            # |a| < 100 is not a price
    assert devig_two(1.91, 1.91) == pytest.approx(0.5)
    # a whole-number total that pushes returns the stake, it does not lose it
    assert ev(0.45, 0.10, 2.0) == pytest.approx(0.45 - 0.45)


def test_books_are_aggregated_in_decimal_not_american():
    """-115 and +105 have an American median of -5, which reads as a 20x payout. That is exactly
    what the first backtest did, and it reported a +2,000% ROI on totals."""
    from nhl.sources.odds import parse_event

    def book(title, over, under, line=6.0):
        return {"title": title, "markets": [{"key": "totals", "outcomes": [
            {"name": "Over", "price": over, "point": line},
            {"name": "Under", "price": under, "point": line}]}]}

    ev = {"id": "x", "commence_time": "2026-10-08T23:00:00Z", "home_team": "Boston Bruins",
          "away_team": "Toronto Maple Leafs",
          "bookmakers": [book("A", 1.87, 2.05), book("B", 2.05, 1.87), book("C", 1.95, 1.95, 6.5)]}
    r = parse_event(ev)
    assert r["total_line"] == 6.0                        # the modal line, not the odd one out
    assert 1.8 < r["over_dec"] < 2.1 and 1.8 < r["under_dec"] < 2.1
    assert r["p_over"] == pytest.approx(0.5, abs=0.01)   # de-vigged per book, then averaged
    assert parse_event({**ev, "home_team": "Hartford Whalers"}) is None


# ---------------------------------------------------------------------------------
# The scoreline model
# ---------------------------------------------------------------------------------
def test_grid_is_a_distribution_with_no_ties_left():
    from nhl.scoreline import grids, moneyline
    F, R = grids([3.2, 2.5, 4.0], [2.8, 3.1, 2.0])
    assert np.allclose(F.sum(axis=(1, 2)), 1.0)
    assert (F >= 0).all() and (R >= 0).all()
    assert np.allclose(np.diagonal(F, axis1=1, axis2=2), 0.0), "overtime must settle every tie"
    assert moneyline(F[0]) > 0.5 > moneyline(F[1])


def test_the_late_game_chase_moves_one_goal_games_like_real_ones():
    """Plain Poisson puts 30% of regulations on a one-goal margin; the NHL's is 18%."""
    from nhl.scoreline import DEFAULT_THETA, grids
    no_chase = {**DEFAULT_THETA, "m6": 1.0, "en": 0.0, "tie": 0.0}
    _, R0 = grids([3.0], [2.8], no_chase)
    _, R1 = grids([3.0], [2.8])
    k = np.arange(R0.shape[1])
    d = np.abs(k[:, None] - k[None, :])
    one0, one1 = R0[0][d == 1].sum(), R1[0][d == 1].sum()
    tie0, tie1 = R0[0][d == 0].sum(), R1[0][d == 0].sum()
    assert one1 < one0 - 0.05, (one0, one1)
    assert tie1 > tie0, "tied games tighten up late and late equalisers happen"
    assert R1[0][d >= 2].sum() > R0[0][d >= 2].sum(), "empty-netters turn 1-goal games into 2"


def test_whole_number_totals_push_and_half_lines_cannot():
    from nhl.scoreline import grids, over_under
    F, _ = grids([3.1], [2.9])
    o, p, u = over_under(F[0], 6.0)
    assert p > 0.08 and o + p + u == pytest.approx(1.0)
    o, p, u = over_under(F[0], 5.5)
    assert p == 0.0 and o + u == pytest.approx(1.0)


def test_puck_line_sides_mirror_each_other():
    from nhl.scoreline import cover, grids
    F, _ = grids([3.3], [2.7])
    hc, push, ac = cover(F[0], -1.5)
    assert push == 0.0 and hc + ac == pytest.approx(1.0)
    hc2, _, ac2 = cover(F[0], 1.5)              # home +1.5 loses only if the away side wins by 2+
    k = np.arange(F.shape[1])
    assert ac2 == pytest.approx(F[0][(k[:, None] - k[None, :]) <= -2].sum())
    assert hc2 > 0.6 > hc


def test_table_matches_exact_grids_and_inverts_the_market():
    from nhl.scoreline import Table, cover, grids, moneyline, over_under
    t = Table()
    lh, la = np.array([3.4, 2.6, 3.0]), np.array([2.7, 3.3, 3.0])
    F, _ = grids(lh, la)
    for i in range(3):
        assert t.p_home(lh[i:i + 1], la[i:i + 1])[0] == pytest.approx(moneyline(F[i]), abs=0.003)
        o, pu = t.over(lh[i:i + 1], la[i:i + 1], [6.0])
        assert o[0] == pytest.approx(over_under(F[i], 6.0)[0], abs=0.003)
        h, _ = t.cover(lh[i:i + 1], la[i:i + 1], [-1.5])
        assert h[0] == pytest.approx(cover(F[i], -1.5)[0], abs=0.003)
    # round trip: market prices -> rates -> the same prices
    ph = np.array([moneyline(F[i]) for i in range(3)])
    po = np.array([over_under(F[i], 6.0)[0] / (1 - over_under(F[i], 6.0)[1]) for i in range(3)])
    rh, ra = t.invert(ph, np.full(3, 6.0), po)
    assert np.allclose(rh, lh, atol=0.05) and np.allclose(ra, la, atol=0.05)


# ---------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------
TEAMS8 = ["BOS", "TOR", "MTL", "OTT", "NYR", "PHI", "PIT", "WSH"]


def _league(seasons=(2021, 2022), teams=TEAMS8, per_pair=4, seed=3, first=date(2021, 10, 12)):
    """A small league with goalies, shots and overtime - enough to exercise the replay."""
    rng = np.random.default_rng(seed)
    strength = {t: rng.normal(0, 0.12) for t in teams}
    rows, gid = [], 0
    for s in seasons:
        day = date(s, 10, 12)
        for _ in range(per_pair):
            order = list(rng.permutation(teams))
            for i in range(0, len(order), 2):
                h, a = order[i], order[i + 1]
                lh = 3.0 * np.exp(0.05 + strength[h] - strength[a])
                la = 2.9 * np.exp(strength[a] - strength[h])
                hs, as_ = rng.poisson(31), rng.poisson(29)
                hg, ag = rng.poisson(lh), rng.poisson(la)
                dec = "REG"
                if hg == ag:
                    dec = "OT" if rng.random() < 0.6 else "SO"
                    if rng.random() < 0.5:
                        hg += 1
                    else:
                        ag += 1
                so_h = dec == "SO" and hg > ag
                so_a = dec == "SO" and ag > hg
                gid += 1
                rows.append({"game_id": f"{s}02{gid:04d}", "season": s, "game_type": "R",
                             "date": str(day), "start_et": f"{day}T19:00:00",
                             "home_team": h, "away_team": a, "home_goals": float(hg),
                             "away_goals": float(ag), "decided_in": dec, "completed": True,
                             "home_goalie_id": 1000 + TEAMS8.index(h) * 2 + int(rng.random() < 0.35),
                             "home_goalie": f"{h} G", "away_goalie": f"{a} G",
                             "away_goalie_id": 1000 + TEAMS8.index(a) * 2 + int(rng.random() < 0.35),
                             "home_shots": float(hs), "away_shots": float(as_),
                             # a goalie's goals-against include an overtime winner but
                             # never the shootout goal
                             "home_goalie_ga": float(ag - so_a), "away_goalie_ga": float(hg - so_h),
                             "source": "test"})
            day += timedelta(days=int(rng.integers(1, 3)))
    return pd.DataFrame(rows)


def test_leak_free(env):
    """A row for game N is identical whether or not games after N exist."""
    from nhl.features import build
    g = _league()
    full, _, _ = build(g, first_train_season=2022, today=date(2030, 1, 1))
    cut = g.iloc[: len(g) // 2 + 7]
    part, _, _ = build(cut, first_train_season=2022, today=date(2030, 1, 1))
    cols = [c for c in part.columns if c not in ("game_type",)]
    merged = part[cols].merge(full[cols], on="game_id", suffixes=("", "_full"))
    assert len(merged) == len(part) > 0
    for c in ("lam_h", "lam_a", "L_h", "g_h_exp", "h_rest", "a_b2b", "h_games"):
        assert np.allclose(merged[c].astype(float), merged[f"{c}_full"].astype(float),
                           equal_nan=True), c


def test_the_uploaded_scripts_leak_is_not_reproducible_here(env):
    """The predecessor's "recent goals" included the game being predicted. Nothing in a
    training row may move when only that game's score changes."""
    from nhl.features import build
    g = _league()
    base, _, _ = build(g, first_train_season=2022, today=date(2030, 1, 1))
    g2 = g.copy()
    last = g2.index[-1]
    g2.loc[last, ["home_goals", "away_goals", "away_goalie_ga"]] = [9.0, 0.0, 9.0]
    moved, _, _ = build(g2, first_train_season=2022, today=date(2030, 1, 1))
    feats = [c for c in base.select_dtypes("number").columns
             if c.startswith(("lam_", "L_", "S_", "p_", "g_", "h_", "a_")) and not c.endswith("_gg")]
    a = base[base.game_id == g.loc[last, "game_id"]][feats].astype(float).values
    b = moved[moved.game_id == g.loc[last, "game_id"]][feats].astype(float).values
    assert np.allclose(a, b, equal_nan=True)


def test_goals_on_goalie_exclude_overtime_winners_and_survive_missing_logs():
    from nhl.features import goalie_goals
    g = pd.DataFrame({"home_goals": [4.0, 3.0, 2.0], "away_goals": [3.0, 2.0, 1.0],
                      "decided_in": ["OT", "SO", "REG"],
                      "home_goalie_ga": [3.0, 2.0, np.nan], "away_goalie_ga": [4.0, 2.0, np.nan]})
    h, a = goalie_goals(g)
    assert list(h) == [3.0, 2.0, 2.0]    # OT winner removed; SO goal never was a goalie goal
    assert list(a) == [3.0, 2.0, 1.0]    # no goalie log: regulation goals, not NaN


def test_back_to_back_moves_the_expected_goalie_to_the_backup():
    """In 2021-25 the previous night's goalie started the second game of a back-to-back 9% of
    the time. The expectation has to follow that, or every back-to-back is priced with the
    starter in net."""
    from nhl.ratings import RatingBook
    b = RatingBook()
    for i in range(12):
        gid = 1 if i % 4 else 2           # goalie 1 starts three games in four
        b.update("BOS", "TOR", gid, 9, 30, 30, 3, 3, date(2025, 10, 1) + timedelta(days=2 * i))
    b.gk[1], b.gk[2] = -0.05, 0.08        # a good starter, a weak backup
    rested, p_rested = b.expected_goalie("BOS", b2b=False)
    tired, p_tired = b.expected_goalie("BOS", b2b=True)
    assert p_rested[1] > 0.5 > p_tired[1]
    assert tired > rested, "the backup is worse, so a back-to-back must look worse in net"


def test_unplayed_past_games_do_not_tire_anyone():
    from nhl.features import schedule_features
    g = pd.DataFrame({"game_id": ["1", "2", "3"], "season": 2025,
                      "date": ["2025-11-01", "2025-11-02", "2025-11-03"],
                      "home_team": ["BOS", "BOS", "BOS"], "away_team": ["TOR", "MTL", "OTT"],
                      "completed": [True, False, True]})   # game 2 was postponed
    s = schedule_features(g, today=date(2025, 12, 1)).set_index("game_id")
    assert "2" not in s.index
    assert s.loc["3", "h_rest"] == 2 and s.loc["3", "h_b2b"] == 0


# ---------------------------------------------------------------------------------
# Starting goalies
# ---------------------------------------------------------------------------------
def _dfo_page(games: list[dict]) -> str:
    payload = {"props": {"pageProps": {"data": games}}, "page": "/starting-goalies/[date]"}
    return ('<html><body><div id="__next"></div><script id="__NEXT_DATA__" '
            f'type="application/json">{json.dumps(payload)}</script></body></html>')


def test_starting_goalie_pages_parse_by_pattern_not_by_exact_name():
    """The page's field names could not be checked from where this was written, so they are
    matched by pattern: the flat names it is known to use and a nested shape must both come
    through - and an unconfirmed starter must never be read as a confirmed one."""
    from nhl.sources import starters as st
    html = _dfo_page([
        {"homeTeamName": "Boston Bruins", "homeTeamSlug": "boston-bruins",
         "homeGoalieName": "Jeremy Swayman", "homeNewsStrengthName": "Confirmed",
         "homeGoalieSavePercentage": 0.915, "awayTeamName": "Toronto Maple Leafs",
         "awayGoalieName": "Anthony Stolarz", "awayNewsStrengthName": "Likely"},
        {"homeTeam": {"name": "Utah Mammoth"}, "homeStatus": "Unconfirmed",
         "homeGoalie": {"firstName": "Karel", "lastName": "Vejmelka"},
         "awayTeam": {"abbreviation": "VGK"}, "awayGoalie": {"name": "Adin Hill"},
         "awayStatus": "Expected"},
        {"homeTeamName": "Bruins", "homeGoalieName": "A", "awayTeamName": "Hartford Whalers",
         "awayGoalieName": "B"},                           # a team that maps to nothing: skipped
    ])
    got = st.parse(st.next_data(html))
    assert [(g["away_team"], g["home_team"]) for g in got] == [("TOR", "BOS"), ("VGK", "UTA")]
    bos, uta = got
    assert (bos["h_goalie"], bos["h_status"], bos["a_status"]) == ("Jeremy Swayman", "confirmed",
                                                                   "likely")
    assert (uta["h_goalie"], uta["h_status"], uta["a_goalie"], uta["a_status"]) == (
        "Karel Vejmelka", "projected", "Adin Hill", "likely")
    assert st.next_data("<html>no data here</html>") is None
    assert st.status("Unconfirmed") == "projected" and st.status("CONFIRMED") == "confirmed"
    assert st.team_code("Bruins") == "BOS" and st.team_code("maple-leafs") == "TOR"


def test_goalie_names_resolve_to_nhl_ids_through_accents_nicknames_and_trades():
    from nhl.sources import starters as st
    g = pd.DataFrame([
        {"game_id": "1", "season": 2025, "date": "2026-01-10", "completed": True,
         "home_team": "CGY", "away_team": "MTL", "home_goalie_id": 11,
         "home_goalie": "Jacob Markström", "away_goalie_id": 22,
         "away_goalie": "Samuel Montembeault"},
        {"game_id": "2", "season": 2025, "date": "2026-02-10", "completed": True,
         "home_team": "NJD", "away_team": "BOS", "home_goalie_id": 11,
         "home_goalie": "Jacob Markström", "away_goalie_id": 33, "away_goalie": "Jeremy Swayman"},
    ])
    idx = st.goalie_index(g)
    assert st.resolve("Jacob Markstrom", "NJD", idx) == (11, "name")
    assert st.resolve("Sam Montembeault", "MTL", idx) == (22, "surname")
    assert st.resolve("Jeremy Swayman", "TOR", idx) == (33, "name, other team")   # traded
    gid, how = st.resolve("Brand New Kid", "BOS", idx)
    assert how == "new" and gid < 0, "a debut is rated as a new goalie, never as someone else"
    assert gid == st.resolve("Brand New Kid", "SEA", idx)[0]


def test_a_confirmed_backup_replaces_the_guess_and_a_likely_one_mostly_does():
    from types import SimpleNamespace

    from nhl.features import _row
    from nhl.ratings import RatingBook
    b = RatingBook()
    b.share["BOS"] = b.share_long["BOS"] = {1: 0.8, 2: 0.2}
    b.gk[1], b.gk[2] = -0.05, 0.05                 # a good starter, a weak backup
    b.goalie_name[1] = "Starter One"
    r = SimpleNamespace(home_team="BOS", away_team="TOR", game_id="g")
    guess = _row(b, r, {})
    conf = _row(b, r, {}, known={"h": {"id": 2, "weight": 1.0, "status": "confirmed",
                                       "name": "Backup Two"}})
    likely = _row(b, r, {}, known={"h": {"id": 2, "weight": 0.85, "status": "likely"}})
    assert (guess["g_h_top"], guess["g_h_status"], guess["g_h_backup"]) == ("Starter One", "", 0)
    assert conf["g_h_exp"] == pytest.approx(0.05) and conf["g_h_conf"] == 1.0
    assert (conf["g_h_top"], conf["g_h_status"], conf["g_h_backup"]) == ("Backup Two",
                                                                         "confirmed", 1)
    assert likely["g_h_exp"] == pytest.approx(0.85 * 0.05 + 0.15 * (0.8 * -0.05 + 0.2 * 0.05))
    # the away side shoots at the weaker goalie, and more surely so the firmer the news
    assert conf["lam_a"] > likely["lam_a"] > guess["lam_a"]


def test_training_rows_use_the_starter_each_game_actually_had(env):
    from nhl.features import build
    g = _league()
    actual, _, _ = build(g, first_train_season=2022, today=date(2030, 1, 1), starters="actual")
    guess, _, _ = build(g, first_train_season=2022, today=date(2030, 1, 1), starters="expected")
    m = actual.merge(g[["game_id", "home_goalie_id"]], on="game_id")
    assert (m["g_h_id"] == m["home_goalie_id"]).all() and (m["g_h_conf"] == 1.0).all()
    assert (guess["g_h_conf"] < 1.0).any(), "the guess is a guess"


def test_picks_wait_until_both_starters_are_confirmed_or_likely(env, monkeypatch):
    from nhl import predict
    from nhl.scoreline import grids
    from nhl.sources import starters as st
    d = date(2026, 11, 7)
    board = pd.DataFrame({"game_id": ["A", "B", "C", "D"], "date": [d, d, d, d + timedelta(days=1)]})
    rows = pd.DataFrame([{"game_id": "A", "side": "h", "status": "confirmed"},
                         {"game_id": "A", "side": "a", "status": "likely"},
                         {"game_id": "B", "side": "h", "status": "confirmed"},
                         {"game_id": "B", "side": "a", "status": "projected"}])
    # C: the feed answered for its date without listing it; D: the feed never answered for its date
    assert list(st.pending(board, rows, covered={d})) == [False, True, True, False]
    monkeypatch.setattr(env, "REQUIRE_STARTERS", False)
    assert not st.pending(board, rows, covered={d}).any()

    F, R = grids([3.6], [2.4])
    g = {"home_team": "BOS", "away_team": "TOR", "thin_data": False, "pl_home": -1.5,
         "pl_home_dec": 3.2, "pl_away_dec": 1.45, "p_pl_home": 0.30, "total_line": 6.0,
         "over_dec": 1.91, "under_dec": 1.91, "p_over": 0.5}
    ready = predict.price_game(F[0], R[0], {**g, "goalies_pending": False})
    waiting = predict.price_game(F[0], R[0], {**g, "goalies_pending": True})
    assert ready["spread_strength"] in ("play", "bold") and ready["spread_stake"] > 0
    assert waiting["spread_strength"] == "wait" and waiting["spread_stake"] == 0
    assert waiting["spread_pick"] == ready["spread_pick"], "the pick is still published"


def test_a_failed_evening_pull_keeps_the_mornings_news(env):
    from nhl.sources import starters as st
    morning = pd.DataFrame([{"pulled_at": "2026-11-07T14:15:00+00:00", "date": "2026-11-07",
                             "game_id": "A", "side": "h", "team": "BOS",
                             "goalie": "Jeremy Swayman", "goalie_id": 33, "status": "likely",
                             "matched_by": "name"}])
    morning.to_csv(env.STARTERS, index=False)
    evening = morning.assign(pulled_at="2026-11-07T21:45:00+00:00", status="confirmed")
    assert st.latest(["A"])["status"].tolist() == ["likely"]
    assert st.latest(["A"], fresh=evening)["status"].tolist() == ["confirmed"]
    assert st.known(st.latest(["A"], fresh=evening))["A"]["h"] == {
        "id": 33, "status": "confirmed", "name": "Jeremy Swayman", "weight": 1.0}


# ---------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------
def test_offset_model_is_the_market_when_the_market_is_right():
    """With y drawn from the market's own rate, every coefficient should be ~0 - the model
    defaults to the market, which is the right default against an efficient one."""
    from nhl.glm import PoissonGLM
    rng = np.random.default_rng(0)
    n = 20000
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                       "log_mkt": np.log(rng.uniform(2.0, 4.0, n))})
    df["y"] = rng.poisson(np.exp(df["log_mkt"]))
    m = PoissonGLM(["x1", "x2"], offset="log_mkt").fit(df)
    assert abs(m.intercept) < 0.02 and np.abs(m.coef).max() < 0.02
    m2 = PoissonGLM.from_dict(json.loads(json.dumps(m.to_dict())))
    assert np.allclose(m2.predict(df.head(50)), m.predict(df.head(50)))


def test_betting_knobs_are_conservative(env):
    """Pinned to absolute floors, never to another sport's config."""
    assert env.SPREAD_EV_MIN >= 0.03
    assert env.TOTAL_EV_MIN >= 0.05
    assert env.KELLY_FRACTION <= 0.125
    assert env.MIN_GAMES >= 5
    assert env.SHRINK_CAP <= 0.8
    assert env.DEFAULT_TOTAL_SHRINK <= 0.3, "the totals market was the one the model never beat"


def test_empty_env_vars_fall_back_to_defaults(monkeypatch):
    """An unset GitHub Actions repo variable arrives as an empty string, not as absent."""
    import importlib
    from nhl import config
    for var in ("DEGEN_NHL_SPREAD_EV", "DEGEN_NHL_TOTAL_EV", "DEGEN_KELLY", "DEGEN_MIN_GAMES",
                "DEGEN_BOARD_DAYS", "DEGEN_FIRST_SEASON"):
        monkeypatch.setenv(var, "")
    importlib.reload(config)
    assert config.SPREAD_EV_MIN == 0.03 and config.KELLY_FRACTION == 0.125
    assert config.MIN_GAMES == 5 and config.FIRST_SEASON == 2007


# ---------------------------------------------------------------------------------
# Sources, against payloads shaped like the real ones
# ---------------------------------------------------------------------------------
TEAMS_JSON = {1: "NJD", 6: "BOS", 10: "TOR", 11: "ATL", 53: "ARI", 59: "UTA"}


def _game(gid, season, gtype, state, home, away, hs=None, vs=None, period=3, day="2025-10-10"):
    return {"id": gid, "season": season, "gameType": gtype, "gameStateId": state,
            "homeTeamId": home, "visitingTeamId": away, "homeScore": hs, "visitingScore": vs,
            "period": period, "gameDate": day, "easternStartTime": f"{day}T19:00:00"}


def test_stats_api_games_parse():
    from nhl.sources.nhle import parse_games
    rows = [_game(2025020001, 20252026, 2, 7, 6, 10, 4, 3, 4),      # OT
            _game(2025020002, 20252026, 2, 7, 10, 6, 2, 3, 5),      # shootout
            _game(2025020003, 20252026, 2, 6, 59, 1, 1, 5, 3),      # final (state 6), Utah
            _game(2025020004, 20252026, 2, 5, 6, 1, 2, 2, 3),       # "game over" - not final yet
            _game(2025010001, 20252026, 1, 7, 6, 1, 1, 0, 3),       # preseason: dropped
            _game(2010020001, 20102011, 2, 7, 11, 6, 3, 2, 3),      # Atlanta -> WPG
            _game(2025020005, 20252026, 2, 1, 6, 999, None, None),  # unknown team: skipped
            _game(2025030111, 20252026, 3, 7, 6, 10, 3, 2, 6)]      # playoff double OT
    g = parse_games(rows, TEAMS_JSON).set_index("game_id")
    assert "2025010001" not in g.index and "2025020005" not in g.index
    assert g.loc["2025020001", "decided_in"] == "OT"
    assert g.loc["2025020002", "decided_in"] == "SO"
    assert g.loc["2025030111", "decided_in"] == "OT" and g.loc["2025030111", "game_type"] == "P"
    assert bool(g.loc["2025020003", "completed"]) and g.loc["2025020003", "home_team"] == "UTA"
    assert not bool(g.loc["2025020004", "completed"])
    assert np.isnan(g.loc["2025020004", "home_goals"])
    assert g.loc["2010020001", "home_team"] == "WPG" and g.loc["2010020001", "season"] == 2010


def test_goalie_logs_give_starters_and_shots_for_the_other_side():
    from nhl.sources.nhle import attach_goalies, goalie_table, parse_games
    games = parse_games([_game(2025020001, 20252026, 2, 7, 6, 10, 4, 3, 3)], TEAMS_JSON)
    logs = [{"gameId": 2025020001, "playerId": 1, "teamAbbrev": "BOS", "goalieFullName": "A",
             "gamesStarted": 1, "timeOnIce": 3000, "shotsAgainst": 25, "goalsAgainst": 3},
            {"gameId": 2025020001, "playerId": 2, "teamAbbrev": "BOS", "goalieFullName": "B",
             "gamesStarted": 0, "timeOnIce": 600, "shotsAgainst": 5, "goalsAgainst": 0},
            {"gameId": 2025020001, "playerId": 3, "teamAbbrev": "TOR", "goalieFullName": "C",
             "gamesStarted": 1, "timeOnIce": 3600, "shotsAgainst": 35, "goalsAgainst": 4}]
    g = attach_goalies(games, goalie_table(logs)).iloc[0]
    assert g["home_goalie_id"] == 1 and g["away_goalie_id"] == 3
    assert g["away_shots"] == 30 and g["home_shots"] == 35     # shots FOR are the other side's SA
    assert g["home_goalie_ga"] == 3 and g["away_goalie_ga"] == 4


def test_a_bad_refresh_never_erases_history():
    from nhl.sources.nhle import _merge, parse_games
    old = parse_games([_game(2025020001, 20252026, 2, 7, 6, 10, 4, 3, 3)], TEAMS_JSON)
    old["home_goalie_id"], old["home_shots"] = 1, 30.0
    new = parse_games([_game(2025020001, 20252026, 2, 1, 6, 10)], TEAMS_JSON)   # hiccup: FUT
    m = _merge(old, new).iloc[0]
    assert bool(m["completed"]) and m["home_goals"] == 4 and m["home_shots"] == 30


def test_sbr_history_keeps_only_real_puck_lines():
    from nhl.sources.history import sbr_lines
    panel = pd.DataFrame([
        {"game_id": 1, "date": "2015-10-10", "season": 2015, "team": "BOS", "opponent": "TOR",
         "home_away": "H", "moneyline": -150, "total_line": 5.5, "line": -1.5},
        {"game_id": 1, "date": "2015-10-10", "season": 2015, "team": "TOR", "opponent": "BOS",
         "home_away": "A", "moneyline": 130, "total_line": 5.5, "line": 1.5},
        {"game_id": 2, "date": "2015-10-11", "season": 2015, "team": "ARI", "opponent": "BOS",
         "home_away": "H", "moneyline": 120, "total_line": 5.0, "line": 2.5},
        {"game_id": 2, "date": "2015-10-11", "season": 2015, "team": "BOS", "opponent": "ARI",
         "home_away": "A", "moneyline": -140, "total_line": 5.0, "line": -2.5}])
    ln = sbr_lines(panel).set_index("game_id")
    assert ln.loc["1", "p_home"] == pytest.approx(0.5798, abs=1e-3)
    assert ln.loc["1", "pl_home"] == -1.5 and np.isnan(ln.loc["2", "pl_home"])
    assert ln.loc["2", "home_team"] == "UTA"        # the Coyotes' rating follows the roster
    assert set(ln["source"]) == {"sbr_close"} and ln["p_over"].isna().all()


# ---------------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------------
LEAGUE = ["ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET", "EDM", "FLA",
          "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
          "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH"]


def _synth(this: int):
    """Five full seasons of a 32-team league whose market is deliberately near-perfect.

    Scores are drawn from the scoreline model itself, so the late-game shape is the real one;
    the last two seasons carry full prices, like the noon archive, and the rest only a
    moneyline and a total, like the closing archive.
    """
    from nhl.scoreline import grids
    rng = np.random.default_rng(11)
    att = {t: rng.normal(0, 0.12) for t in LEAGUE}
    dfn = {t: rng.normal(0, 0.10) for t in LEAGUE}
    fx = []
    for s in range(this - 5, this):
        day = date(s, 10, 8)
        for _ in range(82):
            order = list(rng.permutation(LEAGUE))
            fx += [(s, day, order[i], order[i + 1]) for i in range(0, 32, 2)]
            day += timedelta(days=2)
    lh = np.array([2.85 * np.exp(0.04 + att[h] + dfn[a]) for _, _, h, a in fx])
    la = np.array([2.75 * np.exp(att[a] + dfn[h]) for _, _, h, a in fx])
    F, _ = grids(lh, la)
    K = F.shape[1]
    k = np.arange(K)
    d, t = k[:, None] - k[None, :], k[:, None] + k[None, :]
    p_home = F[:, d > 0].sum(1)
    p_h2 = F[:, d >= 2].sum(1)
    o6, p6 = F[:, t > 6].sum(1), F[:, t == 6].sum(1)
    games, lines = [], []
    for n, (s, day, h, a) in enumerate(fx):
        flat = F[n].ravel()
        hg, ag = divmod(int(rng.choice(len(flat), p=flat / flat.sum())), K)
        dec = "REG" if abs(hg - ag) != 1 or rng.random() < 0.55 else str(rng.choice(["OT", "SO"]))
        so_h, so_a = dec == "SO" and hg > ag, dec == "SO" and ag > hg
        gid = f"{s}02{n:05d}"
        games.append({"game_id": gid, "season": s, "game_type": "R", "date": str(day),
                      "start_et": f"{day}T19:00:00", "home_team": h, "away_team": a,
                      "home_goals": float(hg), "away_goals": float(ag), "decided_in": dec,
                      "completed": True,
                      "home_goalie_id": 1000 + 2 * LEAGUE.index(h) + int(rng.random() < 0.3),
                      "home_goalie": f"{h} Starter", "away_goalie": f"{a} Starter",
                      "away_goalie_id": 1000 + 2 * LEAGUE.index(a) + int(rng.random() < 0.3),
                      "home_shots": float(rng.poisson(30 * np.exp(att[h] + dfn[a]))),
                      "away_shots": float(rng.poisson(29 * np.exp(att[a] + dfn[h]))),
                      "home_goalie_ga": float(ag - so_a), "away_goalie_ga": float(hg - so_h),
                      "source": "test"})
        ph = float(np.clip(p_home[n] + rng.normal(0, 0.01), 0.05, 0.95))
        po = o6[n] / (1 - p6[n])
        priced = s >= this - 2
        lines.append({"game_id": gid, "season": s, "date": str(day), "home_team": h,
                      "away_team": a, "source": "oddsapi_noon" if priced else "sbr_close",
                      "is_closing": not priced, "home_dec": 1 / (ph * 1.02),
                      "away_dec": 1 / ((1 - ph) * 1.02), "p_home": ph, "total_line": 6.0,
                      "over_dec": 1 / (po * 1.02) if priced else np.nan,
                      "under_dec": 1 / ((1 - po) * 1.02) if priced else np.nan,
                      "p_over": po if priced else np.nan, "pl_home": -1.5,
                      "pl_home_dec": 1 / (p_h2[n] * 1.03) if priced else np.nan,
                      "pl_away_dec": 1 / ((1 - p_h2[n]) * 1.03) if priced else np.nan,
                      "p_pl_home": p_h2[n] if priced else np.nan, "n_books": 2})
    return pd.DataFrame(games), pd.DataFrame(lines)


def test_pipeline(env, monkeypatch):
    from nhl import grade, predict, site, train
    from nhl.scoreline import DEFAULT_THETA
    from nhl.sources import nhle, odds

    this = env.season_of(env.today_et())
    games, lines = _synth(this)
    games.to_csv(env.GAMES, index=False)
    lines.to_csv(env.LINES, index=False)
    # the late-game fit is a couple of minutes of optimisation; the defaults are its fixed point
    monkeypatch.setattr(train, "fit_theta", lambda *a, **k: {
        **DEFAULT_THETA, "c_home_fit": 1.0, "c_away_fit": 1.0, "nll": 0.0, "n_games": 0})
    monkeypatch.setattr(env, "FIRST_SEASON", this - 5)
    monkeypatch.setattr(env, "LOAD_FROM_SEASON", this - 5)

    train.main(["--no-fetch"])
    meta = json.loads((env.MODEL_DIR / "meta.json").read_text())
    assert set(meta["models"]) == {"market", "nomarket"}
    assert meta["sport"] == "nhl"
    ev = meta["eval"]
    assert this not in ev["test_seasons"], "the in-progress season must never be the holdout"
    ml = ev["moneyline"]
    # the fixture's market knows the truth, so the model must not beat it by much, if at all
    assert ml["log_loss_edge"] < 0.01
    assert 0 <= meta["shrink"]["split"] <= env.SHRINK_CAP
    sp = ev["spread"]
    assert sp["roi_by_ev"] and all("roi_pct" in b and "candidate" in b for b in sp["roi_by_ev"])
    assert "market_expected_rate" in sp, "a bare puck-line cover rate means nothing"
    assert not any(r["significant"] and not r.get("candidate", True) for r in sp["roi_by_ev"]), \
        "declined bets are not candidates and must not count as findings"
    assert "eval_strict" in meta
    for r in meta["scoreline_calibration"]:
        assert "predicted_pct" in r

    # an upcoming slate: a matinee, tonight and tomorrow, with one team on a back-to-back
    today = env.today_et()
    monkeypatch.setattr(env, "now_et", lambda: datetime.combine(today, time(10, 15), env.ET))
    up = []
    for i in range(0, 12, 2):
        d = today if i < 6 else today + timedelta(days=1)
        up.append({"game_id": f"{this}02{9000 + i}", "season": this, "game_type": "R",
                   "date": str(d), "start_et": f"{d}T{'13' if i == 0 else '19'}:00:00",
                   "home_team": LEAGUE[i], "away_team": LEAGUE[i + 1], "completed": False})
    up.append({"game_id": f"{this}029999", "season": this, "game_type": "R",
               "date": str(today + timedelta(days=1)), "start_et": "",
               "home_team": LEAGUE[0], "away_team": LEAGUE[20], "completed": False})
    allg = pd.concat([games, pd.DataFrame(up)], ignore_index=True)
    monkeypatch.setattr(nhle, "update_games", lambda *a, **k: allg)
    snap = pd.DataFrame([{
        "pulled_at": f"{today}T14:00:00+00:00", "event_id": u["game_id"],
        "commence_time": f"{u['date']}T{'17' if u is up[0] else '23'}:00:00+00:00",
        "date": pd.Timestamp(u["date"]).date(),
        "home_team": u["home_team"], "away_team": u["away_team"], "home_dec": 1.70,
        "away_dec": 2.25, "p_home": 0.57, "n_ml": 4, "total_line": 6.0, "over_dec": 1.91,
        "under_dec": 1.91, "p_over": 0.5, "n_total": 4, "pl_home": -1.5, "pl_home_dec": 2.6,
        "pl_away_dec": 1.55, "p_pl_home": 0.37, "n_pl": 4} for u in up[:-1]])
    monkeypatch.setattr(odds, "snapshot", lambda *a, **k: snap)
    # the day's news: the matinee's starters are confirmed, one evening game is half-known, and
    # nothing is posted for tomorrow yet
    from nhl.sources import starters

    def page(day):
        if day != today:
            return []
        return [{"home_team": LEAGUE[0], "away_team": LEAGUE[1], "h_goalie": f"{LEAGUE[0]} Starter",
                 "h_status": "confirmed", "a_goalie": f"{LEAGUE[1]} Starter", "a_status": "confirmed"},
                {"home_team": LEAGUE[2], "away_team": LEAGUE[3], "h_goalie": "Brand New Kid",
                 "h_status": "confirmed", "a_goalie": f"{LEAGUE[3]} Starter",
                 "a_status": "projected"}]
    monkeypatch.setattr(starters, "fetch", page)

    out = predict.run()
    assert len(out) == 7
    for c in ("spread_pick", "spread_ev", "spread_p_win", "spread_stake", "total_pick", "total_ev",
              "total_p_win", "p_home_win", "p_reg_home", "p_ot", "p_reg_away", "first_m_lh",
              "g_h_top", "puck_drop"):
        assert c in out.columns, c
    probs = out[["p_reg_home", "p_ot", "p_reg_away"]].sum(axis=1)
    assert np.allclose(probs, 1.0, atol=5e-4)      # each is stored to four decimals
    # a new season: nobody has played, so nothing may be staked
    assert out["thin_data"].all()
    assert set(out["spread_strength"]) <= {"pass", "thin"}
    assert (out["spread_stake"] == 0).all() and (out["total_stake"] == 0).all()
    # the unpriced game is still predicted, never staked, and has no invented price
    tbd = out[out["game_id"] == f"{this}029999"].iloc[0]
    assert tbd["spread_strength"] == "pass" and not tbd["spread_price"]
    # the back-to-back is seen from tonight's game, which has not been played yet
    assert int(tbd["h_b2b"]) == 1
    # who is in net: the news where there is some, the model's guess where there is none
    by = out.set_index("game_id")
    assert (by.loc[up[0]["game_id"], "g_h_status"], by.loc[up[0]["game_id"], "g_a_status"]) == (
        "confirmed", "confirmed")
    half = by.loc[up[1]["game_id"]]
    assert (half["g_h_top"], half["g_h_status"], half["g_a_status"]) == ("Brand New Kid",
                                                                         "confirmed", "projected")
    assert not by.loc[up[0]["game_id"], "goalies_pending"] and half["goalies_pending"]
    assert by.loc[up[2]["game_id"], "goalies_pending"], "the feed answered for today without it"
    assert not by.loc[up[3]["game_id"], "goalies_pending"], "no news for tomorrow: the guess stands"
    assert by.loc[up[3]["game_id"], "g_h_status"] == ""
    logged = pd.read_csv(env.STARTERS, dtype={"game_id": str})
    assert set(logged["game_id"]) == {up[0]["game_id"], up[1]["game_id"]}
    assert logged.loc[logged["goalie"] == "Brand New Kid", "goalie_id"].iloc[0] < 0

    # the evening run rewrites the rows; the number first published must survive it
    monkeypatch.setattr(env, "now_et", lambda: datetime.combine(today, time(17, 45), env.ET))
    snap2 = snap.assign(pl_home_dec=2.4, pl_away_dec=1.62, pulled_at=f"{today}T21:45:00+00:00")
    monkeypatch.setattr(odds, "snapshot", lambda *a, **k: snap2)
    out2 = predict.run()
    picks = pd.read_csv(env.PICKS, dtype={"game_id": str})
    assert len(picks) == 7
    row = picks[picks["game_id"] == up[2]["game_id"]].iloc[0]
    first = out[out["game_id"] == up[2]["game_id"]].iloc[0]
    assert row["first_spread_dec"] == pytest.approx(first["spread_dec"])
    assert row["first_seen_at"] == first["first_seen_at"]
    # the matinee is under way by evening: its pre-game row stands, it is not re-priced in-play
    assert len(out2) == 6 and up[0]["game_id"] not in set(out2["game_id"])
    mat = picks[picks["game_id"] == up[0]["game_id"]].iloc[0]
    was = out[out["game_id"] == up[0]["game_id"]].iloc[0]
    assert mat["spread_dec"] == pytest.approx(was["spread_dec"])
    assert mat["spread_pick"] == was["spread_pick"]

    site.build()
    html = (env.DOCS / "index.html").read_text()
    assert "NHL" in html and 'href="../"' in html
    assert "In net:" in html and "Brand New Kid" in html and ">confirmed<" in html
    assert "nan" not in _rendered_text(html), "empty fields must not render as 'nan'"
    assert html.count('class="game ') + html.count('class="game"') == 7

    # finish the slate and grade it
    fin = allg.copy()
    m = fin["game_id"].isin([u["game_id"] for u in up])
    fin.loc[m, ["home_goals", "away_goals", "decided_in", "completed"]] = [3.0, 1.0, "REG", True]
    monkeypatch.setattr(nhle, "update_games", lambda *a, **k: fin)
    done = grade.grade()
    assert len(done) == 7
    assert set(done["spread_result"]) <= {"win", "loss", "push"}
    home = done["spread_side"] == "home"
    side_margin = np.where(home, done["home_margin"], -done["home_margin"])
    expect = np.where(side_margin + done["spread_line"] > 0, "win", "loss")
    assert (done["spread_result"] == expect).all(), "grading must settle the side we published"
    assert (done["total_result"] == np.where(done["total_line_used"] == 4.0, "push",
                                             np.where((done["total_side"] == "over")
                                                      == (4 > done["total_line_used"]),
                                                      "win", "loss"))).all()
    met = grade.metrics(done)
    assert met["sport"] == "nhl" and met["spreads"]["all_games"]["n"] == 7
    assert met["spreads"]["season"]["n"] == 0, "nothing was staked, so there is no staked record"
    env.METRICS.write_text(json.dumps(met, default=str))
    site.build()
    assert "nan" not in _rendered_text((env.DOCS / "index.html").read_text())


def test_games_under_way_are_never_priced_in_play(env, monkeypatch):
    """The odds feed lists games already under way, at in-play prices, and on a weekend the
    matinees are mid-game when the evening run fires. Neither the feed nor the board may treat
    those as pre-game numbers."""
    from nhl import predict
    from nhl.sources import odds

    now = datetime(2026, 11, 7, 17, 45, tzinfo=env.ET)       # a Saturday, 22:45 UTC

    def event(eid, start):
        return {"id": eid, "commence_time": start, "home_team": "Boston Bruins",
                "away_team": "Toronto Maple Leafs", "bookmakers": [{"title": "A", "markets": [
                    {"key": "h2h", "outcomes": [{"name": "Boston Bruins", "price": 1.8},
                                                {"name": "Toronto Maple Leafs", "price": 2.1}]}]}]}

    class Response:
        status_code, headers = 200, {}

        def json(self):
            return [event("matinee", "2026-11-07T18:00:00Z"), event("tonight", "2026-11-08T00:00:00Z")]

    monkeypatch.setattr(env, "ODDS_API_KEY", "test")
    monkeypatch.setattr(env, "now_et", lambda: now)
    monkeypatch.setattr(odds, "get", lambda *a, **k: Response())
    assert list(odds.snapshot()["event_id"]) == ["tonight"]

    board = pd.DataFrame([
        {"commence_time": np.nan, "start_et": "2026-11-07T13:00:00"},      # NHL time: started
        {"commence_time": "2026-11-08T00:00:00+00:00", "start_et": ""},    # feed time: tonight
        {"commence_time": np.nan, "start_et": ""}])                        # unknown: kept
    assert predict.started(board, now).tolist() == [True, False, False]


# ---------------------------------------------------------------------------------
# The published site
# ---------------------------------------------------------------------------------
def test_the_nhl_page_is_reachable_and_links_everywhere():
    tpl = (ROOT / "nhl" / "templates" / "index.html").read_text()
    for href in ('href="../"', 'href="../cfb/"', 'href="../nfl/"', 'href="../epl/"'):
        assert href in tpl, href
    for other in ("cfb", "nfl", "epl"):
        t = (ROOT / other / "templates" / "index.html").read_text()
        assert 'href="../nhl/"' in t, f"{other} does not link to the NHL board"


def test_landing_page_lists_the_nhl():
    from core import landing
    assert any(s["slug"] == "nhl" for s in landing.SPORTS)


def test_predict_runs_morning_and_evening_and_forwards_the_knobs():
    wf = (ROOT / ".github" / "workflows" / "nhl-predict.yml").read_text()
    assert len(re.findall(r"- cron: '", wf)) == 2, "the evening run is the closing line"
    for var in ("DEGEN_NHL_SPREAD_EV", "DEGEN_NHL_TOTAL_EV", "DEGEN_KELLY"):
        assert f"{var}: ${{{{ vars.{var} }}}}" in wf, f"{var} is not forwarded"
    assert "ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}" in wf
    test = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    assert "nhl" in test


def test_puck_drop_sort_key_is_chronological():
    from nhl.predict import puck_drop
    label, iso = puck_drop({"commence_time": "2026-10-08T23:00:00Z"})
    assert iso.startswith("2026-10-08T23:00") and "7:00 PM" in label
    label2, iso2 = puck_drop({"commence_time": None, "start_et": "2026-10-09T22:00:00"})
    assert iso2 > iso and "10:00 PM" in label2
    assert puck_drop({"commence_time": None, "start_et": ""}) == ("", "")


def test_nhl_module_does_not_import_other_sports():
    """nhl/ must not reach into another sport or the shared core - blast radius of one."""
    pkg = ROOT / "nhl"
    offenders = []
    for path in sorted(pkg.rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.match(r"\s*(from|import)\s+(cfb|nfl|epl|ncaab|core)\b", line):
                offenders.append(f"{path.relative_to(pkg.parent)}:{i}: {line.strip()}")
    assert not offenders, "nhl/ imports another sport or the shared core: " + "; ".join(offenders)


def test_list_endpoints_are_called_without_paging_first(monkeypatch):
    """/game and /team are called the way the working reference code calls them. If a paged call
    were refused, the refresh would quietly keep the cache and the board would stay empty."""
    from nhl.sources import nhle
    calls = []

    def fake(url, params=None, **kw):
        calls.append((url.rsplit("/", 1)[-1], dict(params or {})))
        if "limit" in (params or {}):
            return None                                   # the endpoint refuses paging
        if url.endswith("/team"):
            return {"data": [{"id": 6, "triCode": "BOS"}, {"id": 10, "triCode": "TOR"}], "total": 2}
        return {"data": [_game(2024020001, 20242025, 2, 7, 6, 10, 3, 2),
                         _game(2019020001, 20192020, 2, 7, 6, 10, 1, 0)], "total": 2}

    monkeypatch.setattr(nhle, "get_json", fake)
    assert nhle.fetch_teams() == {6: "BOS", 10: "TOR"}
    rows = nhle.fetch_games(2024)
    assert [r["id"] for r in rows] == [2024020001], "an ignored filter is applied locally"
    assert calls[0] == ("team", {})
