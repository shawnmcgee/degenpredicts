"""Leak-free pre-game features for every NHL game since 2010-11.

    python -m poc.nhl_totals.features          # writes <cache>/built/features.parquet

One chronological pass. For each date, every game on that date is described from the state as it
stood the night before, and only then is the state updated with that date's results - so nothing
a feature knows comes from the game it describes or anything after it.

What the state tracks, and the feature families it produces:

  team       exponentially weighted per-game rates (fast and slow) of xG, goals, shot attempts,
             shots, power-play xG, penalties taken/drawn and empty-net goals, for and against,
             each relative to a league rate that drifts with the scoring environment
  goalie     the starter's goals allowed against xG faced (GSAx as a ratio), decayed, plus how
             established he is as the #1 and how rested he is
  lineup     every skater carries his own decayed TOI, individual xG, points and 5v5 on-ice xG
             for/against. Tonight's dressed lineup is valued against the team's usual lineup at
             the same ratings, so an injury, a scratch or a call-up shows up as a delta
  referees   each referee's penalties, goals and over/under margin against the closing line,
             all relative to what was expected, shrunk toward neutral
  schedule   rest, back-to-backs, 3-in-4, travel distance and time-zone shift
  market     the closing total, its price where known, the moneyline favourite

xG note: the season files were scored by different xG model versions (a game summed to ~7.4 xG
before 2023-24 and ~6.1 after, against ~6 real goals throughout), so every xG number is converted
to goals with a running, within-season goals/xG ratio before it touches a rating.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from nhl.teams import haversine_km, venue

from .fetch import cache_dir

REPO = Path(__file__).resolve().parents[2]

# team rate vector layout (per team-game)
Q = ["xgf", "xga", "gf", "ga", "cf", "ca", "sf", "sa", "xgf_pp", "xga_pk", "pen_t", "pen_d",
     "en_f", "en_a", "xgf5", "xga5"]
QI = {k: i for i, k in enumerate(Q)}
# offence/defence pairs: own attack x opponent's defence / league
PAIRS = [("xgf", "xga"), ("gf", "ga"), ("cf", "ca"), ("sf", "sa"), ("xgf_pp", "xga_pk"),
         ("pen_d", "pen_t"), ("en_f", "en_a"), ("xgf5", "xga5")]

HL_FAST, HL_SLOW = 10, 40          # team half-lives, in games
SEASON_CARRY = 0.6                 # share of a team's deviation from league kept over a summer
LEAGUE_ALPHA = 1 / 600             # league rates, per team-game
C_PRIOR_K = 150.0                  # goals of prior weight on last season's goals/xG ratio
GOALIE_HL, GOALIE_K = 50, 25.0     # goalie: half-life in games, prior in goals
PLAYER_HL, PLAYER_K = 60, 400.0    # skater: half-life in games, prior in minutes
SHARE_ALPHA = 0.2                  # how fast a team's "usual lineup" and starter share move
REF_HL = 150                       # referee half-life, in games worked
REF_K_PEN, REF_K_GOALS, REF_K_OU = 150.0, 300.0, 40.0
OU_ALPHA = 0.1                     # team over/under margin vs the line


def _decay(hl):
    return 0.5 ** (1.0 / hl)


def _alpha(hl):
    return 1.0 - _decay(hl)


def load_lines() -> pd.DataFrame:
    """Closing (or latest pre-game) total per game: lines.csv, then the live season's snapshots."""
    l = pd.read_csv(REPO / "data/nhl/lines.csv")
    l = l.sort_values("is_closing").drop_duplicates("game_id", keep="last")
    keep = ["game_id", "source", "total_line", "over_dec", "under_dec", "p_over", "p_home", "home_dec",
            "away_dec"]
    l = l[keep]
    snap_p = REPO / "data/nhl/snapshots.csv"
    if snap_p.exists():
        s = pd.read_csv(snap_p)
        s = s[pd.to_datetime(s["pulled_at"], utc=True) < pd.to_datetime(s["commence_time"], utc=True)]
        s = s.sort_values("pulled_at").drop_duplicates("game_id", keep="last")
        s = s[~s["game_id"].isin(l["game_id"])].assign(source="snapshot_last")
        l = pd.concat([l, s.reindex(columns=keep)], ignore_index=True)
    return l.rename(columns={"source": "line_source"})


def _load_built():
    b = cache_dir() / "built"
    return (pd.read_parquet(b / "team_games.parquet"), pd.read_parquet(b / "goalie_games.parquet"),
            pd.read_parquet(b / "player_games.parquet"), pd.read_parquet(b / "refs.parquet"))


def build_features() -> pd.DataFrame:
    games = pd.read_csv(REPO / "data/nhl/games.csv")
    games = games[(games["season"] >= 2010) & games["game_type"].isin(["R", "P"])].copy()
    games["date"] = pd.to_datetime(games["date"])
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    lines = load_lines()
    games = games.merge(lines, on="game_id", how="left")

    tg, gg, pg, refs = _load_built()
    tg_by = {(r.game_id, r.side): r for r in tg.itertuples(index=False)}
    gg_by = defaultdict(list)
    for r in gg.itertuples(index=False):
        gg_by[r.game_id].append(r)
    pg_by = defaultdict(list)
    for r in pg.itertuples(index=False):
        pg_by[(r.game_id, r.team)].append(r)
    ref_by = {r.game_id: (r.ref1, r.ref2) for r in refs.itertuples(index=False)}

    nq = len(Q)
    # --- state ---------------------------------------------------------------------------
    # league per-team-game rates, started near 2010-11 levels (that season is warm-up only)
    league = np.array([2.65, 2.65, 2.65, 2.65, 55, 55, 30, 30, 0.5, 0.5, 4.0, 4.0, 0.12, 0.12, 1.8, 1.8])
    league_tot = 5.6                       # final goals per game, incl. EN and the shootout goal
    team_fast, team_slow, team_season = {}, {}, {}
    c_state = {"season": None, "sg": 0.0, "sx": 0.0, "prev": 0.75}

    goalie = {}                            # gid -> [S_ga, S_xg, n, last_date]
    starter_share = defaultdict(dict)      # team -> gid -> share
    player = {}                            # pid -> dict
    pos_tot = {"F": np.array([1e4, 1e4 * 0.008, 1e4 * 0.03, 1e4 * 0.04, 1e4 * 0.04]),
               "D": np.array([1e4, 1e4 * 0.003, 1e4 * 0.02, 1e4 * 0.04, 1e4 * 0.04])}
    # pos_tot: [toi, ixg, pts, oixgf, oixga] decayed league sums per position
    lineup_share = defaultdict(dict)       # team -> pid -> share
    player_team = {}
    ref = {}                               # name -> [S_n, S_pen, S_pen_exp, S_goals, S_goals_exp, S_ou, S_ou_n]
    team_ou = {}
    team_gp = defaultdict(int)             # (team, season) -> games played
    sched = {}                             # team -> (last_date, (lat, lon, tz), [recent dates])

    def c_now():
        return (c_state["sg"] + C_PRIOR_K * c_state["prev"]) / (c_state["sx"] + C_PRIOR_K / c_state["prev"]) \
            if c_state["sx"] > 0 else c_state["prev"]

    def team_vec(store, t):
        return store.get(t, league.copy())

    def player_rates(pid, pos):
        p = player.get(pid)
        L = pos_tot.get(pos, pos_tot["F"])
        lr = L[1:] / L[0]                  # league rates per minute
        if p is None:
            return lr, (11.0 if pos != "D" else 17.0), 0
        s = p["s"]
        r = (s[1:] + PLAYER_K * lr) / (s[0] + PLAYER_K)
        return r, p["toi_ew"], p["n"]

    def lineup_value(entries):
        """entries: [(pid, pos, weight)] -> TOI-weighted [ixg, pts, oixgf, oixga] per game."""
        if not entries:
            return np.full(4, np.nan)
        rs, ws = [], []
        for pid, pos, w in entries:
            r, toi, _ = player_rates(pid, pos)
            rs.append(r)
            ws.append(w * toi)
        ws = np.array(ws)
        ws = ws / ws.sum() * 300.0         # 5 skaters x 60 minutes
        v = (np.array(rs) * ws[:, None]).sum(0)
        v[2:] /= 5.0                       # on-ice rates are shared by the five on the ice
        return v

    rows = []
    gk_d = _decay(GOALIE_HL)
    pl_d = _decay(PLAYER_HL)
    ref_d = _decay(REF_HL)
    a_fast, a_slow = _alpha(HL_FAST), _alpha(HL_SLOW)

    for date, day in games.groupby("date", sort=True):
        day_rows = []
        for g in day.itertuples(index=False):
            sdv_season = g.season + 1
            if c_state["season"] != sdv_season:     # new season: roll the goals/xG ratio
                if c_state["season"] is not None and c_state["sx"] > 0:
                    c_state["prev"] = c_state["sg"] / c_state["sx"]
                c_state.update(season=sdv_season, sg=0.0, sx=0.0)
            h, a = g.home_team, g.away_team
            f = {"game_id": g.game_id, "season": g.season, "date": date, "game_type": g.game_type,
                 "home_team": h, "away_team": a}
            for t in (h, a):                       # summer regression on a team's first game
                if team_season.get(t) != g.season:
                    if t in team_fast:
                        team_fast[t] = league + SEASON_CARRY * (team_fast[t] - league)
                        team_slow[t] = league + SEASON_CARRY * (team_slow[t] - league)
                    team_season[t] = g.season
                    lineup_share[t] = {k: v * 0.7 for k, v in lineup_share[t].items()}
            # --- team ratings -------------------------------------------------------------
            for tag, store in (("f", team_fast), ("s", team_slow)):
                vh, va = team_vec(store, h), team_vec(store, a)
                for o, d in PAIRS:
                    lo = league[QI[o]]
                    f[f"{tag}_{o}_h"] = vh[QI[o]] * va[QI[d]] / lo     # home attack vs away def
                    f[f"{tag}_{o}_a"] = va[QI[o]] * vh[QI[d]] / lo
                    f[f"{tag}_{o}_tot"] = f[f"{tag}_{o}_h"] + f[f"{tag}_{o}_a"]
                # finishing: goals per xG, own and allowed
                f[f"{tag}_fin_h"] = vh[QI["gf"]] / max(vh[QI["xgf"]], 1e-6)
                f[f"{tag}_fin_a"] = va[QI["gf"]] / max(va[QI["xgf"]], 1e-6)
                f[f"{tag}_sv_h"] = vh[QI["ga"]] / max(vh[QI["xga"]], 1e-6)
                f[f"{tag}_sv_a"] = va[QI["ga"]] / max(va[QI["xga"]], 1e-6)
            for k, i in QI.items():
                f[f"lg_{k}"] = league[i]
            f["lg_tot"] = league_tot
            f["c_xg"] = c_now()

            # --- goalies: tonight's starters ------------------------------------------------
            starters = {r.team: r for r in gg_by.get(g.game_id, []) if r.starter}
            for side, t in (("h", h), ("a", a)):
                st = starters.get(t)
                gid = None if st is None else st.goalie_id
                s = goalie.get(gid)
                if s is None:
                    f[f"gk_{side}"] = 1.0
                    f[f"gk_n_{side}"] = 0
                    f[f"gk_rest_{side}"] = 10
                else:
                    f[f"gk_{side}"] = (s[0] + GOALIE_K) / (s[1] + GOALIE_K)
                    f[f"gk_n_{side}"] = s[2]
                    f[f"gk_rest_{side}"] = min((date - s[3]).days, 10)
                sh = starter_share[t]
                top = max(sh.values()) if sh else 0.0
                f[f"gk_share_{side}"] = sh.get(gid, 0.0)
                f[f"gk_backup_{side}"] = int(sh.get(gid, 0.0) < 0.5 * top) if sh else 0
                f[f"gk_id_{side}"] = gid
                # the guess available before any news: the usual starter, except that the
                # previous night's goalie rarely starts the second half of a back-to-back
                order = sorted(sh, key=sh.get, reverse=True)
                proj = order[0] if order else None
                last = sched.get(t)
                if proj is not None and last is not None and (date - last[0]).days == 1 \
                        and goalie.get(proj, [0, 0, 0, None])[3] == last[0] and len(order) > 1:
                    proj = order[1]
                ps = goalie.get(proj)
                f[f"gkp_{side}"] = 1.0 if ps is None else (ps[0] + GOALIE_K) / (ps[1] + GOALIE_K)
                f[f"gkp_hit_{side}"] = int(proj == gid)

            # --- lineups: tonight vs the usual -------------------------------------------
            for side, t in (("h", h), ("a", a)):
                dressed = [(r.player_id, r.position if r.position == "D" else "F", 1.0)
                           for r in pg_by.get((g.game_id, t), [])]
                sh = lineup_share[t]
                usual = [(pid, player.get(pid, {}).get("pos", "F"), s) for pid, s in sh.items() if s > 0.05]
                tv, uv = lineup_value(dressed), lineup_value(usual)
                for j, k in enumerate(["ixg", "pts", "oixgf", "oixga"]):
                    f[f"lu_{k}_{side}"] = tv[j]
                    f[f"lu_d{k}_{side}"] = tv[j] - uv[j] if usual else 0.0
                ids = {pid for pid, _, _ in dressed}
                f[f"lu_miss_{side}"] = sum(1 for pid, s in sh.items() if s >= 0.6 and pid not in ids)
                f[f"lu_new_{side}"] = sum(1 for pid in ids if sh.get(pid, 0.0) < 0.2)
                f[f"lu_n_{side}"] = len(dressed)

            # --- referees -----------------------------------------------------------------
            rp = ref_by.get(g.game_id, (None, None))
            pen_idx, goal_idx, ou_idx, ref_n = [], [], [], []
            for name in rp:
                s = ref.get(name) if isinstance(name, str) else None
                if s is None:
                    pen_idx.append(1.0); goal_idx.append(1.0); ou_idx.append(0.0); ref_n.append(0)
                else:
                    pen_idx.append((s[1] + REF_K_PEN) / (s[2] + REF_K_PEN))
                    goal_idx.append((s[3] + REF_K_GOALS) / (s[4] + REF_K_GOALS))
                    ou_idx.append(s[5] / (s[6] + REF_K_OU))
                    ref_n.append(s[0])
            f["ref_pen"] = float(np.mean(pen_idx))
            f["ref_goals"] = float(np.mean(goal_idx))
            f["ref_ou"] = float(np.mean(ou_idx))
            f["ref_n"] = float(np.min(ref_n))

            # --- schedule -----------------------------------------------------------------
            here = venue(h, g.season)
            for side, t in (("h", h), ("a", a)):
                last = sched.get(t)
                if last is None or (date - last[0]).days > 30:
                    f[f"rest_{side}"], f[f"b2b_{side}"], f[f"3in4_{side}"] = 5, 0, 0
                    f[f"km_{side}"], f[f"tz_{side}"] = 0.0, 0.0
                else:
                    rest = (date - last[0]).days
                    f[f"rest_{side}"] = min(rest, 5)
                    f[f"b2b_{side}"] = int(rest == 1)
                    f[f"3in4_{side}"] = int(sum((date - d).days <= 3 for d in last[2]) >= 2)
                    f[f"km_{side}"] = haversine_km(last[1], here) if last[1] and here else 0.0
                    f[f"tz_{side}"] = (here[2] - last[1][2]) if last[1] and here else 0.0
                f[f"ou_{side}"] = team_ou.get(t, 0.0)
                f[f"gp_{side}"] = team_gp.get((t, g.season), 0)
            day_rows.append(f)

        # ---------------- update state with the day's results ---------------------------
        for g, f in zip(day.itertuples(index=False), day_rows):
            if not bool(g.completed) or (g.game_id, "home") not in tg_by:
                continue
            h, a = g.home_team, g.away_team
            th, ta = tg_by[(g.game_id, "home")], tg_by[(g.game_id, "away")]
            c = c_now()
            gf_ne_h, gf_ne_a = th.gf_all - th.gf_en, ta.gf_all - ta.gf_en
            c_state["sg"] += gf_ne_h + gf_ne_a
            c_state["sx"] += th.xgf + ta.xgf

            def vec(o, d):
                return np.array([o.xgf * c, d.xgf * c, o.gf_all - o.gf_en, d.gf_all - d.gf_en, o.cf, d.cf,
                                 o.sf, d.sf, o.xgf_pp * c, d.xgf_pp * c, o.pen_taken, d.pen_taken,
                                 o.gf_en, d.gf_en, o.xgf_5v5 * c, d.xgf_5v5 * c], dtype=float)

            for t, x in ((h, vec(th, ta)), (a, vec(ta, th))):
                team_fast[t] = team_vec(team_fast, t) + a_fast * (x - team_vec(team_fast, t))
                team_slow[t] = team_vec(team_slow, t) + a_slow * (x - team_vec(team_slow, t))
                league[:] = league + LEAGUE_ALPHA * (x - league)
            total = g.home_goals + g.away_goals
            league_tot += (LEAGUE_ALPHA * 2) * (total - league_tot)

            # goalies: everyone who played, on what he faced
            for r in gg_by.get(g.game_id, []):
                if r.toi <= 0:
                    continue
                s = goalie.setdefault(r.goalie_id, [0.0, 0.0, 0, date])
                s[0] = s[0] * gk_d + r.ga
                s[1] = s[1] * gk_d + r.xga * c
                s[2] += 1
                s[3] = date
            starts = {r.team: r.goalie_id for r in gg_by.get(g.game_id, []) if r.starter}
            for t in (h, a):
                sh = starter_share[t]
                for k in list(sh):
                    sh[k] *= (1 - SHARE_ALPHA)
                if t in starts:
                    sh[starts[t]] = sh.get(starts[t], 0.0) + SHARE_ALPHA

            # skaters
            for t in (h, a):
                team_gp[(t, g.season)] += 1
                sh = lineup_share[t]
                for k in list(sh):
                    sh[k] *= (1 - SHARE_ALPHA)
                    if sh[k] < 0.01:
                        del sh[k]
                for r in pg_by.get((g.game_id, t), []):
                    pos = "D" if r.position == "D" else "F"
                    old = player_team.get(r.player_id)
                    if old is not None and old != t:          # traded: leaves the old lineup
                        lineup_share[old].pop(r.player_id, None)
                    player_team[r.player_id] = t
                    sh[r.player_id] = sh.get(r.player_id, 0.0) + SHARE_ALPHA
                    x = np.array([r.toi, r.ixg * c, r.goals + r.assists, r.oi5_xgf * c, r.oi5_xga * c])
                    p = player.get(r.player_id)
                    if p is None:
                        p = player[r.player_id] = {"s": np.zeros(5), "toi_ew": r.toi, "n": 0, "pos": pos}
                    p["s"] = p["s"] * pl_d + x
                    p["toi_ew"] += 0.25 * (r.toi - p["toi_ew"])
                    p["n"] += 1
                    pos_tot[pos] = pos_tot[pos] * 0.99995 + x

            # referees
            pen_tot = th.pen_taken + ta.pen_taken
            pen_exp = f["lg_pen_t"] * 2
            ou = (total - g.total_line) if pd.notna(g.total_line) else None
            for name in ref_by.get(g.game_id, (None, None)):
                if not isinstance(name, str):
                    continue
                s = ref.setdefault(name, [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                s[0] += 1
                s[1] = s[1] * ref_d + pen_tot
                s[2] = s[2] * ref_d + pen_exp
                s[3] = s[3] * ref_d + total
                s[4] = s[4] * ref_d + f["lg_tot"]
                if ou is not None:
                    s[5] = s[5] * ref_d + ou
                    s[6] = s[6] * ref_d + 1

            # team over/under margin vs the line
            if ou is not None:
                for t in (h, a):
                    team_ou[t] = team_ou.get(t, 0.0) + OU_ALPHA * (ou - team_ou.get(t, 0.0))

        # schedule is known in advance, so it updates for every game, played or not
        for g in day.itertuples(index=False):
            here = venue(g.home_team, g.season)
            for t in (g.home_team, g.away_team):
                prev = sched.get(t)
                recent = [d for d in (prev[2] if prev else []) if (date - d).days <= 3] + [date]
                sched[t] = (date, here, recent)

        for g, f in zip(day.itertuples(index=False), day_rows):
            f.update(completed=bool(g.completed), home_goals=g.home_goals, away_goals=g.away_goals,
                     decided_in=g.decided_in, line_source=g.line_source, total_line=g.total_line,
                     over_dec=g.over_dec, under_dec=g.under_dec, p_over_mkt=g.p_over, p_home_mkt=g.p_home,
                     home_dec=g.home_dec, away_dec=g.away_dec)
            rows.append(f)

    out = pd.DataFrame(rows)
    out["total"] = out["home_goals"] + out["away_goals"]
    out["game_no"] = out.groupby("season").cumcount()
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = build_features()
    p = cache_dir() / "built" / "features.parquet"
    df.to_parquet(p, index=False)
    print(f"{len(df)} games, {df.shape[1]} columns -> {p} in {time.time() - t0:.0f}s")
