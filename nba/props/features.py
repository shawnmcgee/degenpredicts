"""Leak-free player-game features for the NBA props models.

Built by replaying every game in date order, as :func:`nba.features.build` does for teams: the
row for a player's game N is computed from state that has only ever seen games 1..N-1, and the
state advances after the row is recorded. ``tests/test_nba_props.py`` rebuilds from a truncated
schedule and asserts the earlier rows are identical.

The decomposition is the one the rest of the repo landed on - NHL goals are shots x shooting
percentage, NFL yards are targets x yards per target - and it is the right one here for the same
reason: **a stat line is minutes times a rate**, minutes are persistent and learn fast, and the
rate is noisier and learns slower.

* **Minutes** - exponentially weighted (half-life eight games, halved again over the summer),
  last game's minutes and how often he starts. In the NBA minutes are most of the prop: a
  player who plays 36 instead of 30 scores a fifth more at the same rate.
* **Rates per minute** - points, rebounds, assists, threes, shots, free throws and usage, both
  recent and long-run, the long-run ones shrunk toward his position's league rate until he has
  minutes behind them. Three-point percentage is shrunk hard: 40 attempts at 45% is mostly a
  36% shooter on a run.
* **Who else is out** - the points, rebounds, assists, usage and minutes of core teammates
  (twenty minutes a night or more) who are not playing, among those who played in the team's
  last ten games. When a star sits, his shots go somewhere. Twenty minutes is the game board's
  core threshold, for the game board's reason: a 13-minute player's absence is often decided by
  how the game is going, and nothing about tonight may move a pre-game number. Long absences
  are left out on purpose - after ten games without him, everyone's own numbers already show it.
* **The game** - the expected team total and margin (the closing line where there is one, the
  ratings' number where there is not; the board's published number on the day), both clubs'
  pace, the opponent's defence, and what the opponent allows **to his position group** per 48
  minutes, because "bad defence" is not one thing for a centre and a point guard.
* **Rest** - days since his last game, and a back-to-back flag.

Training rows are games the player played. Whether he plays at all is the injury report's
question, and the board multiplies it in (:mod:`nba.props.board`).
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger(__name__)

PLAYER_HL = 8.0         # games
LONG_HL = 40.0          # games - the long-run rate
TEAM_HL = 12.0          # team-games
OFFSEASON = 0.5         # extra decay across a season boundary
CORE_MIN = 20.0         # a core teammate's usual minutes
RECENT_GAMES = 10       # a core teammate counts as missing only if he played in this many
K_MIN = 400.0           # minutes of league-rate prior on the long-run rates
K_3PA = 60.0            # attempts of league-rate prior on three-point percentage
REST_CAP = 7
GROUPS = ("G", "F", "C")
COUNTED = ("min", "start", "pts", "reb", "ast", "fg3m", "fg3a", "fga", "fta", "tov", "oreb",
           "dreb", "usg")

FEATURES = [
    # who he is and how much is known
    "pos_g", "pos_f", "pos_c", "n_eff", "games_season", "rest", "b2b",
    # minutes
    "ew_min", "l_min", "last_min", "start_rate",
    # per game, recent
    "ew_pts", "ew_reb", "ew_ast", "ew_fg3m", "ew_fg3a",
    # per minute, recent and long-run
    "pm_pts", "pm_reb", "pm_ast", "pm_fg3m", "pm_fg3a", "pm_fga", "pm_fta", "pm_usg", "pm_oreb",
    "lpm_pts", "lpm_reb", "lpm_ast", "lpm_fg3m", "lpm_fg3a", "lpm_usg", "fg3_pct",
    # teammates out
    "miss_min", "miss_pts", "miss_reb", "miss_ast", "miss_usg", "n_miss",
    # the game
    "is_home", "playoff", "exp_total", "exp_margin", "team_total", "abs_margin",
    "team_pace", "opp_pace", "team_ortg", "opp_drtg",
    "opp_pts_pos", "opp_reb_pos", "opp_ast_pos", "opp_fg3m_pos",
]

# What each Kalshi market settles on, as a column of the player files.
TARGETS = {"points": "pts", "rebounds": "reb", "assists": "ast", "threes": "fg3m"}


def eligible(row: dict, market: str) -> bool:
    """Which players a market is modelled for, from PRE-GAME state only - deciding it from the
    box score would select on the outcome. Kalshi lists props for the regulars, so this does."""
    if row["n_eff"] < 3.0 or row["ew_min"] < 15.0:
        return False
    if market == "threes":
        return row["ew_fg3a"] >= 1.0
    return True


class _EW:
    """Exponentially weighted per-game sums, short and long. `roll` applies the summer."""

    __slots__ = ("w", "s", "wl", "sl", "season", "games_season", "last", "last_date")

    def __init__(self):
        self.w = self.wl = 0.0
        self.s: dict[str, float] = defaultdict(float)
        self.sl: dict[str, float] = defaultdict(float)
        self.season = None
        self.games_season = 0
        self.last: dict[str, float] = {}
        self.last_date = None

    def roll(self, season: int) -> None:
        if self.season is not None and season != self.season:
            self.w *= OFFSEASON
            self.wl *= OFFSEASON
            for d in (self.s, self.sl):
                for k in d:
                    d[k] *= OFFSEASON
            self.games_season = 0
        self.season = season

    def add(self, vals: dict, hl: float = PLAYER_HL, hl_long: float = LONG_HL) -> None:
        a, al = 0.5 ** (1.0 / hl), 0.5 ** (1.0 / hl_long)
        self.w = self.w * a + 1.0
        self.wl = self.wl * al + 1.0
        for k, v in vals.items():
            self.s[k] = self.s[k] * a + v
            self.sl[k] = self.sl[k] * al + v
        for k in set(self.s) - set(vals):
            self.s[k] *= a
            self.sl[k] *= al
        self.last = dict(vals)
        self.games_season += 1

    def mean(self, k: str, default=np.nan) -> float:
        return self.s[k] / self.w if self.w > 0 else default


def _group(pos) -> str:
    return pos if pos in GROUPS else "F"


def _line(rec: dict) -> dict:
    f = lambda k: float(rec.get(k) or 0.0) if rec.get(k) == rec.get(k) else 0.0
    out = {k: f(k) for k in ("pts", "fg3m", "fg3a", "fga", "fta", "tov", "oreb", "dreb", "ast")}
    out["min"] = f("minutes")
    out["start"] = f("starter")
    out["reb"] = out["oreb"] + out["dreb"]
    out["usg"] = out["fga"] + 0.44 * out["fta"] + out["tov"]
    return out


class Book:
    """All the replayed state: players, clubs, what clubs allow by position, the league."""

    def __init__(self, positions: dict | None = None):
        self.players: dict[int, _EW] = defaultdict(_EW)
        self.pos = dict(positions or {})
        self.team_of: dict[int, str] = {}
        self.members: dict[str, set] = defaultdict(set)
        self.last_idx: dict[int, int] = {}            # his team's game count when he last played
        self.team_games: dict[str, int] = defaultdict(int)
        self.team: dict[str, _EW] = defaultdict(_EW)  # pace, offence, defence
        self.allow: dict[str, _EW] = defaultdict(_EW)  # what a defence allows, by group
        self.lg = defaultdict(lambda: 1e-9)           # league sums by group, slowly forgotten

    # ---- teammates -----------------------------------------------------------------------
    def core_recent(self, team: str) -> list[int]:
        n = self.team_games[team]
        out = []
        for q in self.members.get(team, ()):
            st = self.players.get(q)
            if st is None or st.w <= 0 or st.mean("min", 0.0) < CORE_MIN:
                continue
            if n - self.last_idx.get(q, -10 ** 9) <= RECENT_GAMES:
                out.append(q)
        return out

    def missing(self, team: str, p_out: dict, exclude=None) -> dict:
        """Expected production of core teammates who will not play. ``p_out`` maps player id ->
        chance he sits; in training it is 1 for everyone who did not play."""
        m = {"miss_min": 0.0, "miss_pts": 0.0, "miss_reb": 0.0, "miss_ast": 0.0,
             "miss_usg": 0.0, "n_miss": 0.0}
        for q in self.core_recent(team):
            if q == exclude:
                continue
            w = float(p_out.get(q, 0.0))
            if w <= 0:
                continue
            st = self.players[q]
            m["miss_min"] += w * st.mean("min", 0.0)
            m["miss_pts"] += w * st.mean("pts", 0.0)
            m["miss_reb"] += w * st.mean("reb", 0.0)
            m["miss_ast"] += w * st.mean("ast", 0.0)
            m["miss_usg"] += w * st.mean("usg", 0.0)
            m["n_miss"] += w
        return m

    # ---- one row ---------------------------------------------------------------------------
    def _lg_rate(self, group: str, stat: str) -> float:
        return self.lg[(group, stat)] / self.lg[(group, "min")]

    def row(self, pid: int, team: str, opp: str, gdate, season: int, home: bool,
            playoff: bool, ctx: dict, miss: dict) -> dict:
        st = self.players[pid]
        st.roll(season)
        for e in (self.team[team], self.team[opp], self.allow[opp]):
            e.roll(season)
        g = _group(self.pos.get(pid))
        s, sl = st.s, st.sl
        smin, lmin = max(s["min"], 1e-9), sl["min"]
        r = {"pos_g": int(g == "G"), "pos_f": int(g == "F"), "pos_c": int(g == "C"),
             "n_eff": st.w, "games_season": st.games_season,
             "rest": float(min((gdate - st.last_date).days, REST_CAP)) if st.last_date else float(REST_CAP),
             "ew_min": st.mean("min"), "l_min": sl["min"] / st.wl if st.wl > 0 else np.nan,
             "last_min": st.last.get("min", np.nan), "start_rate": st.mean("start")}
        r["b2b"] = int(r["rest"] == 1.0)
        for k in ("pts", "reb", "ast", "fg3m", "fg3a"):
            r[f"ew_{k}"] = st.mean(k)
        for k in ("pts", "reb", "ast", "fg3m", "fg3a", "fga", "fta", "usg", "oreb"):
            r[f"pm_{k}"] = s[k] / smin if st.w > 0 else np.nan
        for k in ("pts", "reb", "ast", "fg3m", "fg3a", "usg"):
            r[f"lpm_{k}"] = (sl[k] + K_MIN * self._lg_rate(g, k)) / (lmin + K_MIN)
        lg3 = self.lg[(g, "fg3m")] / max(self.lg[(g, "fg3a")], 1e-9)
        r["fg3_pct"] = (sl["fg3m"] + K_3PA * lg3) / (sl["fg3a"] + K_3PA)
        r.update(miss)
        t, o, a = self.team[team], self.team[opp], self.allow[opp]
        r.update({"team_pace": t.mean("pace"), "opp_pace": o.mean("pace"),
                  "team_ortg": t.mean("ortg"), "opp_drtg": o.mean("drtg")})
        for k in ("pts", "reb", "ast", "fg3m"):
            m = a.s.get(f"{g}_min", 0.0)
            r[f"opp_{k}_pos"] = 48 * a.s.get(f"{g}_{k}", 0.0) / m if m > 0 else np.nan
        margin, total = ctx.get("margin", np.nan), ctx.get("total", np.nan)
        mt = margin if home else (-margin if margin == margin else np.nan)
        r.update({"is_home": int(home), "playoff": int(playoff), "exp_total": total,
                  "exp_margin": mt,
                  "team_total": (total + mt) / 2 if total == total and mt == mt else np.nan,
                  "abs_margin": abs(mt) if mt == mt else np.nan})
        return r

    # ---- learning ----------------------------------------------------------------------------
    def update_player(self, pid: int, team: str, line: dict, season: int, gdate) -> None:
        st = self.players[pid]
        st.roll(season)
        st.add(line)
        st.last_date = gdate
        old = self.team_of.get(pid)
        if old is not None and old != team:
            self.members[old].discard(pid)
        self.team_of[pid] = team
        self.members[team].add(pid)
        self.last_idx[pid] = self.team_games[team]
        g = _group(self.pos.get(pid))
        for k in ("min", "pts", "reb", "ast", "fg3m", "fg3a", "usg"):
            self.lg[(g, k)] += line[k]

    def update_game(self, home: str, away: str, lines: dict, poss, minutes: float,
                    pts_h, pts_a, season: int) -> None:
        """Clubs and defences after one game. `lines` maps team -> [(group, line), ...]."""
        for t in (home, away):
            self.team_games[t] += 1
        for t, o, pf, pa in ((home, away, pts_h, pts_a), (away, home, pts_a, pts_h)):
            self.team[t].roll(season)
            vals = {}
            if poss == poss and poss and poss > 50:
                vals["pace"] = float(poss) * 48 / minutes
                vals["ortg"] = 100 * float(pf) / float(poss)
                vals["drtg"] = 100 * float(pa) / float(poss)
            if vals:
                self.team[t].add(vals, TEAM_HL, TEAM_HL)
            # what `t`'s defence allowed, to each of the opponent's position groups
            allowed = {}
            for g, ln in lines.get(o, []):
                for k in ("min", "pts", "reb", "ast", "fg3m"):
                    allowed[f"{g}_{k}"] = allowed.get(f"{g}_{k}", 0.0) + ln[k]
            if allowed:
                self.allow[t].roll(season)
                self.allow[t].add(allowed, TEAM_HL, TEAM_HL)
        for k in list(self.lg):
            self.lg[k] *= 0.9997


def game_context(train_rows: pd.DataFrame) -> dict:
    """{game_id: {"margin", "total"}} - the pre-game expected home margin and total, from the
    game board's leak-free rows: the closing line where there is one, the ratings' number
    otherwise."""
    if train_rows is None or train_rows.empty:
        return {}
    d = train_rows
    mm = pd.to_numeric(d.get("mkt_margin"), errors="coerce") if "mkt_margin" in d else None
    mt = pd.to_numeric(d.get("mkt_total"), errors="coerce") if "mkt_total" in d else None
    margin = d["e_margin"] if mm is None else mm.where(mm.notna(), d["e_margin"])
    total = d["e_total"] if mt is None else mt.where(mt.notna(), d["e_total"])
    return {str(g): {"margin": float(m), "total": float(t)}
            for g, m, t in zip(d["game_id"], margin, total)}


def build(players: pd.DataFrame, games: pd.DataFrame, ctx: dict | None = None,
          positions: dict | None = None, first_row_season: int | None = None,
          upcoming: pd.DataFrame | None = None, news: dict | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame, Book]:
    """Replay every completed game; emit one row per player-game from `first_row_season` on.

    ``upcoming`` is a frame of (game_id, athlete_id, team) for games not yet played; their rows
    are built from the final state, with teammates' absences from ``news`` - {team: {athlete_id:
    chance he sits}} - and the game's expected margin and total from ``ctx``.
    """
    first = config.PROPS_FIRST_SEASON if first_row_season is None else first_row_season
    ctx = ctx or {}
    g = games.copy()
    g["game_id"] = g["game_id"].astype(str)
    g["date"] = pd.to_datetime(g["date"]).dt.date
    done = g[g["completed"].astype(bool)].sort_values(["date", "game_id"])
    p = players.copy()
    p["game_id"] = p["game_id"].astype(str)
    p = p[pd.to_numeric(p["minutes"], errors="coerce").fillna(0) > 0]
    by_game = {gid: grp.to_dict("records") for gid, grp in p.groupby("game_id", sort=False)}

    book, rows = Book(positions), []
    for r in done.itertuples(index=False):
        recs = by_game.get(r.game_id)
        if not recs:
            continue
        season, gdate = int(r.season), r.date
        playoff = str(r.game_type) in ("P", "I")
        c = ctx.get(r.game_id, {})
        played = defaultdict(set)
        for rec in recs:
            played[rec["team"]].add(int(rec["athlete_id"]))
        if season >= first:
            for team, opp, home in ((r.home_team, r.away_team, True), (r.away_team, r.home_team, False)):
                absent = {q: 1.0 for q in book.core_recent(team) if q not in played[team]}
                for rec in recs:
                    if rec["team"] != team:
                        continue
                    pid = int(rec["athlete_id"])
                    miss = book.missing(team, absent, exclude=pid)
                    row = book.row(pid, team, opp, gdate, season, home, playoff, c, miss)
                    row.update({"game_id": r.game_id, "season": season, "date": gdate,
                                "team": team, "opponent": opp, "athlete_id": pid,
                                "position": _group(book.pos.get(pid)),
                                "minutes": float(rec["minutes"]),
                                **{t: float(rec.get(t) or 0.0) for t in ("pts", "fg3m", "ast")},
                                "reb": float(rec.get("oreb") or 0.0) + float(rec.get("dreb") or 0.0)})
                    rows.append(row)
        # State advances only AFTER every row of this game is recorded - the leak-free guarantee.
        lines = defaultdict(list)
        for rec in recs:
            pid = int(rec["athlete_id"])
            ln = _line(rec)
            book.update_player(pid, rec["team"], ln, season, gdate)
            lines[rec["team"]].append((_group(book.pos.get(pid)), ln))
        periods = r.periods if r.periods == r.periods and r.periods else 4
        minutes = 48 + 5 * max(0, int(periods) - 4)
        book.update_game(r.home_team, r.away_team, lines, r.poss, minutes, r.home_points,
                         r.away_points, season)
    train = pd.DataFrame(rows)

    up_rows = []
    if upcoming is not None and len(upcoming):
        sched = g.set_index("game_id")
        news = news or {}
        for rec in upcoming.to_dict("records"):
            gid = str(rec["game_id"])
            if gid not in sched.index:
                continue
            s = sched.loc[gid]
            team, pid = rec["team"], int(rec["athlete_id"])
            home = team == s["home_team"]
            opp = s["away_team"] if home else s["home_team"]
            miss = book.missing(team, news.get(team, {}), exclude=pid)
            row = book.row(pid, team, opp, s["date"], int(s["season"]), home,
                           str(s["game_type"]) in ("P", "I"), ctx.get(gid, {}), miss)
            row.update({"game_id": gid, "season": int(s["season"]), "date": s["date"],
                        "team": team, "opponent": opp, "athlete_id": pid,
                        "position": _group(book.pos.get(pid))})
            up_rows.append(row)
    return train, pd.DataFrame(up_rows), book
