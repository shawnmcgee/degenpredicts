"""Leak-free player-game features for the props models.

Built by replaying every game in date order, exactly like ``nfl.features.build``: the row for
a player's game N is computed from state that has only ever seen games 1..N-1, and the state
advances after the row is recorded. ``tests/test_nfl_props.py`` rebuilds from a truncated
history and asserts the earlier rows are identical.

The decomposition is the one the other sports in this repo landed on - NHL goals are shots x
shooting percentage, NBA points are minutes x value - and for the same reason: the volume half
is persistent and learns fast, the efficiency half is mostly noise and learns slowly.

**Opportunity** - snap share, targets, carries, dropbacks, each an exponentially weighted
per-game average (half-life six games, halved again over the offseason). A receiver's target
share is about as stable as anything in football; his yards per target is not.

**Efficiency** - yards per target, catch rate, yards per carry, yards per attempt and
touchdowns per touch, each shrunk toward its position's league rate until the player has
enough volume behind it. A back with 40 carries at 6.1 a carry is mostly a 4.3 back who has
been lucky.

**Team volume and game script** - the team's own dropbacks and carries per game, and the
market's view of this game: the spread and total turned into the team's implied points. A
team expected to trail throws more, and the books' number is the best estimate of that there
is. Historical rows use nflverse's closing line; the board uses the live one.

**Opponent** - what this defence has allowed, split by position group (receiving yards to
wide receivers, to tight ends, to backs), because "bad pass defence" is not one thing.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger(__name__)

PLAYER_HL = 6.0        # games
TEAM_HL = 8.0          # games
OFFSEASON = 0.5        # extra decay across a season boundary

# Shrinkage: (stat + k * prior) / (opportunities + k). Priors are league rates by position,
# rough on purpose - they only matter while a player has little volume of his own.
PRIOR_YPT = {"WR": 8.3, "TE": 7.3, "RB": 5.9, "FB": 4.5, "QB": 6.0}
PRIOR_CATCH = {"WR": 0.63, "TE": 0.70, "RB": 0.77, "FB": 0.70, "QB": 0.5}
PRIOR_YPC = {"RB": 4.2, "QB": 4.8, "WR": 6.5, "TE": 3.5, "FB": 3.0}
PRIOR_YPA, PRIOR_TD = 6.6, 0.035
K_TARGETS, K_CARRIES, K_ATTEMPTS, K_TOUCHES = 25.0, 40.0, 120.0, 60.0

POSITIONS = ("QB", "RB", "WR", "TE")

FEATURES = [
    # who he is and how much we know
    "pos_qb", "pos_rb", "pos_wr", "pos_te", "n_eff", "games_season", "is_starter_qb",
    # opportunity (per game, exponentially weighted)
    "ew_snap_pct", "ew_targets", "ew_receptions", "ew_carries", "ew_attempts", "ew_dropbacks",
    "ew_target_share", "ew_air_share", "ew_carry_share", "ew_air_yards", "last_snap_pct",
    "last_targets", "last_carries",
    # production (per game)
    "ew_rec_yds", "ew_rush_yds", "ew_pass_yds", "ew_tds", "ew_pass_tds",
    # efficiency, shrunk
    "ypt", "catch_rate", "ypc", "ypa", "td_rate",
    # team volume
    "team_dropbacks", "team_carries", "team_plays", "team_pass_yds", "team_rush_yds",
    # game script from the market
    "is_home", "spread_team", "total_line", "implied_team_total",
    # venue
    "is_indoor", "wind", "temp",
    # opponent defence
    "opp_pass_yds", "opp_rush_yds", "opp_dropbacks", "opp_carries",
    "opp_rec_yds_pos", "opp_receptions_pos", "opp_rush_yds_pos", "opp_tds_allowed",
]

# What each prop is, measured on a player-game row.
TARGETS = {
    "pass_yds": "passing_yards",
    "rush_yds": "rushing_yards",
    "rec_yds": "receiving_yards",
    "receptions": "receptions",
    "anytime_td": "any_td",
}


def norm_name(name) -> str:
    """Lower-case, ASCII, no punctuation, no generational suffix. 'Chris Rodriguez Jr.' and
    'Chris Rodriguez' are the same key; 'D.K. Metcalf' and 'DK Metcalf' are too."""
    n = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z ]+", "", n.replace(".", "").replace("'", "").replace("-", " "))
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def eligible(row: dict, market: str) -> bool:
    """Which players a market is modelled for, decided from PRE-GAME state only - deciding it
    from the box score would select on the outcome."""
    pos, n = row["position"], row["n_eff"]
    if n < 0.5:
        return False
    if market == "pass_yds":
        return pos == "QB" and bool(row["is_starter_qb"])
    if market == "rush_yds":
        return (pos == "QB" and bool(row["is_starter_qb"])) or \
               (pos in ("RB", "WR") and row["ew_carries"] >= 1.0)
    if market in ("rec_yds", "receptions"):
        return pos in ("WR", "TE", "RB") and (row["ew_targets"] >= 1.0 or row["ew_snap_pct"] >= 0.3)
    if market == "anytime_td":
        return (pos == "QB" and bool(row["is_starter_qb"])) or \
               (pos in ("WR", "TE", "RB") and (row["ew_targets"] + row["ew_carries"]) >= 1.5)
    return False


class _EW:
    """Exponentially weighted per-game averages. `decay` is applied before each new game."""

    __slots__ = ("w", "s", "last", "season", "games_season")

    def __init__(self):
        self.w = 0.0
        self.s: dict[str, float] = defaultdict(float)
        self.last: dict[str, float] = {}
        self.season = None
        self.games_season = 0

    def roll(self, season: int, hl: float):
        if self.season is not None and season != self.season:
            self.w *= OFFSEASON
            for k in self.s:
                self.s[k] *= OFFSEASON
            self.games_season = 0
        self.season = season

    def add(self, vals: dict, hl: float):
        a = 0.5 ** (1.0 / hl)
        self.w = self.w * a + 1.0
        for k in set(self.s) | set(vals):
            self.s[k] = self.s[k] * a + float(vals.get(k, 0.0) or 0.0)
        self.last = dict(vals)
        self.games_season += 1

    def mean(self, k: str, default=np.nan) -> float:
        return self.s.get(k, 0.0) / self.w if self.w > 0 else default


def _team_tables(players: pd.DataFrame) -> pd.DataFrame:
    """Per team-game totals summed from the player rows (all positions in the file)."""
    p = players
    g = p.groupby(["game_id", "team"], as_index=False).agg(
        opponent_team=("opponent_team", "first"),
        attempts=("attempts", "sum"), sacks=("sacks_suffered", "sum"),
        carries=("carries", "sum"), passing_yards=("passing_yards", "sum"),
        rushing_yards=("rushing_yards", "sum"), targets=("targets", "sum"),
        tds=("any_td_n", "sum"))
    g["dropbacks"] = g["attempts"] + g["sacks"]
    g["plays"] = g["dropbacks"] + g["carries"]
    by_pos = p.groupby(["game_id", "team", "pos_group"], as_index=False).agg(
        rec_yds=("receiving_yards", "sum"), receptions=("receptions", "sum"),
        rush_yds=("rushing_yards", "sum"))
    wide = by_pos.pivot_table(index=["game_id", "team"], columns="pos_group",
                              values=["rec_yds", "receptions", "rush_yds"], fill_value=0.0)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return g.merge(wide.reset_index(), on=["game_id", "team"], how="left").fillna(0.0)


def prepare(players: pd.DataFrame) -> pd.DataFrame:
    p = players.copy()
    p["position"] = p["position"].replace({"FB": "RB"})
    p["pos_group"] = p["position"].where(p["position"].isin(POSITIONS), "RB")
    p["any_td_n"] = p["rushing_tds"].fillna(0) + p["receiving_tds"].fillna(0)
    p["any_td"] = (p["any_td_n"] > 0).astype(int)
    p["snap_pct"] = p["snap_pct"].fillna(np.nan)
    return p


def _game_context(games: pd.DataFrame, lines: pd.DataFrame) -> dict[str, dict]:
    g = games.copy()
    if lines is not None and len(lines):
        g = g.merge(lines[["game_id", "spread_home", "total_line"]], on="game_id", how="left",
                    suffixes=("", "_l"))
        for c in ("spread_home", "total_line"):
            if f"{c}_l" in g:
                g[c] = g[c].combine_first(g[f"{c}_l"]) if c in g else g[f"{c}_l"]
    ctx = {}
    for r in g.to_dict("records"):
        roof = str(r.get("roof") or "").lower()
        indoor = roof in ("dome", "closed")
        temp, wind = r.get("temp", np.nan), r.get("wind", np.nan)
        ctx[str(r["game_id"])] = {
            "date": r.get("date"), "season": int(r["season"]), "week": int(r["week"]),
            "home_team": r["home_team"], "away_team": r["away_team"],
            "home_qb": norm_name(r.get("home_qb")), "away_qb": norm_name(r.get("away_qb")),
            "spread_home": r.get("spread_home", np.nan), "total_line": r.get("total_line", np.nan),
            "is_indoor": int(indoor),
            "temp": 68.0 if indoor and temp != temp else temp,
            "wind": 0.0 if indoor and wind != wind else wind,
        }
    return ctx


def _context_feats(c: dict, team: str) -> dict:
    home = team == c["home_team"]
    sh, tl = c.get("spread_home", np.nan), c.get("total_line", np.nan)
    sh = float(sh) if sh is not None and sh == sh else np.nan
    tl = float(tl) if tl is not None and tl == tl else np.nan
    spread_team = sh if home else (-sh if sh == sh else np.nan)
    return {"is_home": int(home), "spread_team": spread_team, "total_line": tl,
            "implied_team_total": (tl - spread_team) / 2 if tl == tl and spread_team == spread_team
            else np.nan,
            "is_indoor": c["is_indoor"], "wind": c["wind"], "temp": c["temp"]}


class Book:
    """All the replayed state: players, teams on offence, teams on defence."""

    def __init__(self):
        self.players: dict[str, _EW] = defaultdict(_EW)
        self.pos: dict[str, str] = {}
        self.name: dict[str, str] = {}
        self.off: dict[str, _EW] = defaultdict(_EW)
        self.dfn: dict[str, _EW] = defaultdict(_EW)

    def row(self, pid: str, position: str, team: str, opp: str, ctx: dict,
            is_starter_qb: bool) -> dict:
        st = self.players[pid]
        st.roll(ctx["season"], PLAYER_HL)
        o, d = self.off[team], self.dfn[opp]
        o.roll(ctx["season"], TEAM_HL)
        d.roll(ctx["season"], TEAM_HL)
        pos = position if position in POSITIONS else "RB"
        m = st.mean
        tg, rec, ry = st.s.get("targets", 0.0), st.s.get("receptions", 0.0), st.s.get("rec_yds", 0.0)
        ca, rsh = st.s.get("carries", 0.0), st.s.get("rush_yds", 0.0)
        at, py = st.s.get("attempts", 0.0), st.s.get("pass_yds", 0.0)
        touches, tds = tg + ca, st.s.get("tds", 0.0)
        r = {
            "pos_qb": int(pos == "QB"), "pos_rb": int(pos == "RB"), "pos_wr": int(pos == "WR"),
            "pos_te": int(pos == "TE"), "n_eff": st.w, "games_season": st.games_season,
            "is_starter_qb": int(bool(is_starter_qb)),
            "ew_snap_pct": m("snap_pct"), "ew_targets": m("targets", 0.0),
            "ew_receptions": m("receptions", 0.0), "ew_carries": m("carries", 0.0),
            "ew_attempts": m("attempts", 0.0), "ew_dropbacks": m("dropbacks", 0.0),
            "ew_target_share": m("target_share"), "ew_air_share": m("air_share"),
            "ew_carry_share": m("carry_share"), "ew_air_yards": m("air_yards", 0.0),
            "last_snap_pct": st.last.get("snap_pct", np.nan),
            "last_targets": st.last.get("targets", np.nan),
            "last_carries": st.last.get("carries", np.nan),
            "ew_rec_yds": m("rec_yds", 0.0), "ew_rush_yds": m("rush_yds", 0.0),
            "ew_pass_yds": m("pass_yds", 0.0), "ew_tds": m("tds", 0.0),
            "ew_pass_tds": m("pass_tds", 0.0),
            "ypt": (ry + K_TARGETS * PRIOR_YPT.get(pos, 7.0)) / (tg + K_TARGETS),
            "catch_rate": (rec + K_TARGETS * PRIOR_CATCH.get(pos, 0.65)) / (tg + K_TARGETS),
            "ypc": (rsh + K_CARRIES * PRIOR_YPC.get(pos, 4.2)) / (ca + K_CARRIES),
            "ypa": (py + K_ATTEMPTS * PRIOR_YPA) / (at + K_ATTEMPTS),
            "td_rate": (tds + K_TOUCHES * PRIOR_TD) / (touches + K_TOUCHES),
            "team_dropbacks": o.mean("dropbacks"), "team_carries": o.mean("carries"),
            "team_plays": o.mean("plays"), "team_pass_yds": o.mean("passing_yards"),
            "team_rush_yds": o.mean("rushing_yards"),
            "opp_pass_yds": d.mean("passing_yards"), "opp_rush_yds": d.mean("rushing_yards"),
            "opp_dropbacks": d.mean("dropbacks"), "opp_carries": d.mean("carries"),
            "opp_rec_yds_pos": d.mean(f"rec_yds_{pos}"),
            "opp_receptions_pos": d.mean(f"receptions_{pos}"),
            "opp_rush_yds_pos": d.mean(f"rush_yds_{pos}"),
            "opp_tds_allowed": d.mean("tds"),
        }
        r.update(_context_feats(ctx, team))
        return r

    def update_player(self, pid: str, rec: dict, team_carries: float):
        st = self.players[pid]
        car = float(rec.get("carries", 0) or 0)
        self.players[pid].add({
            "snap_pct": rec.get("snap_pct") if rec.get("snap_pct") == rec.get("snap_pct") else 0.0,
            "targets": rec.get("targets", 0), "receptions": rec.get("receptions", 0),
            "rec_yds": rec.get("receiving_yards", 0), "carries": car,
            "rush_yds": rec.get("rushing_yards", 0), "attempts": rec.get("attempts", 0),
            "dropbacks": float(rec.get("attempts", 0) or 0) + float(rec.get("sacks_suffered", 0) or 0),
            "pass_yds": rec.get("passing_yards", 0), "pass_tds": rec.get("passing_tds", 0),
            "tds": rec.get("any_td_n", 0), "target_share": rec.get("target_share", 0),
            "air_share": rec.get("air_yards_share", 0), "air_yards": rec.get("receiving_air_yards", 0),
            "carry_share": car / team_carries if team_carries > 0 else 0.0,
        }, PLAYER_HL)
        self.pos[pid] = rec.get("position")
        self.name[pid] = rec.get("player_name")

    def update_team(self, t: dict):
        self.off[t["team"]].add({k: t[k] for k in ("dropbacks", "carries", "plays",
                                                    "passing_yards", "rushing_yards")}, TEAM_HL)
        allowed = {k: t[k] for k in ("passing_yards", "rushing_yards", "dropbacks", "carries", "tds")}
        for pos in POSITIONS:
            for k in ("rec_yds", "receptions", "rush_yds"):
                allowed[f"{k}_{pos}"] = t.get(f"{k}_{pos}", 0.0)
        self.dfn[t["opponent_team"]].add(allowed, TEAM_HL)


def build(players: pd.DataFrame, games: pd.DataFrame, lines: pd.DataFrame | None = None,
          first_row_season: int | None = None, upcoming: pd.DataFrame | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame, Book]:
    """Replay every played game; emit one row per player-game from `first_row_season` on.

    ``upcoming`` is a frame of (game_id, team, opponent_team, player_id, position,
    player_name) for games not yet played; their rows are built from the final state.
    """
    first = (config.PROPS_FIRST_SEASON + 1) if first_row_season is None else first_row_season
    p = prepare(players)
    ctx = _game_context(games, lines)
    teams = _team_tables(p).set_index(["game_id", "team"])
    p["date"] = p["game_id"].map(lambda g: (ctx.get(str(g)) or {}).get("date"))
    p = p.dropna(subset=["date"])
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["date", "game_id", "team", "player_id"])

    book, rows = Book(), []
    for gid, gp in p.groupby("game_id", sort=False):
        c = ctx[str(gid)]
        recs = gp.to_dict("records")
        if c["season"] >= first:
            for rec in recs:
                team = rec["team"]
                starter = c["home_qb"] if team == c["home_team"] else c["away_qb"]
                is_st = rec["position"] == "QB" and bool(starter) and \
                    norm_name(rec["player_name"]) == starter
                r = book.row(rec["player_id"], rec["position"], team, rec["opponent_team"], c, is_st)
                r.update({"game_id": gid, "season": c["season"], "week": c["week"],
                          "date": rec["date"], "team": team, "opponent_team": rec["opponent_team"],
                          "player_id": rec["player_id"], "player_name": rec["player_name"],
                          "position": rec["position"] if rec["position"] in POSITIONS else "RB",
                          "season_type": rec["season_type"], "zero_filled": rec.get("zero_filled", 0),
                          "passing_yards": rec["passing_yards"], "rushing_yards": rec["rushing_yards"],
                          "receiving_yards": rec["receiving_yards"], "receptions": rec["receptions"],
                          "any_td": rec["any_td"]})
                rows.append(r)
        # state advances only AFTER every row of this game is recorded - the leak-free guarantee
        for rec in recs:
            tc = teams.loc[(gid, rec["team"]), "carries"] if (gid, rec["team"]) in teams.index else 0.0
            book.players[rec["player_id"]].roll(c["season"], PLAYER_HL)
            book.update_player(rec["player_id"], rec, float(tc))
        for team in gp["team"].unique():
            if (gid, team) in teams.index:
                t = teams.loc[(gid, team)].to_dict()
                t["team"] = team
                book.off[team].roll(c["season"], TEAM_HL)
                book.dfn[t["opponent_team"]].roll(c["season"], TEAM_HL)
                book.update_team(t)

    train = pd.DataFrame(rows)
    up_rows = []
    if upcoming is not None and len(upcoming):
        for rec in upcoming.to_dict("records"):
            c = ctx.get(str(rec["game_id"]))
            if c is None:
                continue
            team = rec["team"]
            starter = c["home_qb"] if team == c["home_team"] else c["away_qb"]
            is_st = rec["position"] == "QB" and bool(starter) and norm_name(rec["player_name"]) == starter
            r = book.row(rec["player_id"], rec["position"], team, rec["opponent_team"], c, is_st)
            r.update({k: rec[k] for k in rec})
            r["position"] = rec["position"] if rec["position"] in POSITIONS else "RB"
            up_rows.append(r)
    return train, pd.DataFrame(up_rows), book
