"""Who is playing, and what they are worth - the player layer under the team ratings.

One player can be worth five points of spread, and the books move within minutes of an injury
report. A team rating built only from results cannot see that: it learns a missing star two
weeks late and then forgets him two weeks after he is back. So every game's expectation carries
an **availability delta** - today's minutes distribution against the one the team's rating was
built on:

    delta(team) = sum over rotation players of (target_i - typical_i) x (value_i - replacement)

* ``typical_i`` - the player's share of the team's minutes over its recent games (minutes/48,
  exponentially weighted, zeros for the games he missed). This is the roster the team rating
  reflects.
* ``target_i`` - his expected share TODAY: his usual minutes when he plays, or zero if he is
  out. In training that is who actually played; on the board it is the injury report, the
  current roster and the trade wire.
* ``value_i`` - box-score production per 48 minutes above the league (Hollinger's game score,
  shrunk toward a below-average prior until he has minutes behind him), with an offensive-only
  version for totals.
* ``replacement`` - what the minutes are worth in the hands of whoever takes them.

**Why game score and not plus-minus.** Plus-minus is the direct measure of impact and exists
from 2014; walk-forward, availability built on it cut margin error by 0.14 points against 0.21
for game score, and blending the two did no better than game score alone. A single game's
plus-minus is mostly lineup noise; a box score is mostly the player.

**The leak this layout exists to avoid.** The obvious version - "available means he logged
minutes" - leaks the result. Garbage time hands minutes to the end of the bench only in
blowouts, so who got into the game says how the game went; boosted trees found that and looked
0.15 points better than they were. Here only ROTATION players count (usual minutes of 12 or
more when they play) and their target is their usual minutes, never tonight's - so nothing about
how tonight went can move a pre-game number.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class PlayerConfig:
    min_hl: float = 20.0        # games: usual minutes when he plays
    value_hl: float = 600.0     # minutes played: how fast a value forgets
    prior_min: float = 400.0    # minutes of prior before his own numbers take over
    prior_frac: float = 0.75    # the prior is a below-average player
    typical_hl: float = 15.0    # team games: the roster the team rating reflects
    replacement: float = -2.0   # game score per 48 above league of whoever takes the minutes
    # A rotation player plays at least this much when he plays. Eight scored 0.03 points better
    # in tuning and 0.016 on the holdout - but a player averaging 8-12 minutes is exactly the one
    # who sits in a close game and plays in a blowout, and the holdout gain did not survive
    # against the closing line. A gain that could be the game script leaking in does not go in.
    rotation_min: float = 12.0
    kappa: float = 0.35         # points per 100 possessions per unit of delta (fitted)


def game_score(df: pd.DataFrame) -> pd.Series:
    """Hollinger's game score: points, less the possessions it cost, plus everything else."""
    f = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return (f("points") + 0.4 * f("fgm") - 0.7 * f("fga") - 0.4 * (f("fta") - f("ftm"))
            + 0.7 * f("oreb") + 0.3 * f("dreb") + f("stl") + 0.7 * f("ast") + 0.7 * f("blk")
            - 0.4 * f("pf") - f("tov"))


def offence_score(df: pd.DataFrame) -> pd.Series:
    """The offensive half: points created net of the possessions spent creating them."""
    f = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return (f("points") + 0.4 * f("fgm") - 0.7 * f("fga") - 0.4 * (f("fta") - f("ftm"))
            + 0.7 * f("oreb") + 0.7 * f("ast") - f("tov"))


class PlayerBook:
    def __init__(self, cfg: PlayerConfig | None = None):
        self.cfg = cfg or PlayerConfig()
        c = self.cfg
        self._a_min = 0.5 ** (1 / c.min_hl)
        self._a_val = 0.5 ** (1 / c.value_hl)
        self._a_typ = 0.5 ** (1 / c.typical_hl)
        self.exp_min: dict[int, float] = {}
        self.gs_num: dict[int, float] = defaultdict(float)
        self.os_num: dict[int, float] = defaultdict(float)
        self.den: dict[int, float] = defaultdict(float)
        self.lg_gs = self.lg_os = self.lg_den = 1e-9
        self.typical: dict[str, dict[int, float]] = defaultdict(dict)
        self.team_of: dict[int, str] = {}           # the team he last played for
        self.last_date: dict[int, object] = {}
        self.missed: dict[int, int] = defaultdict(int)   # his team's games in a row he sat out
        self.names: dict[int, str] = {}

    def rollover(self, season: int) -> None:
        """A new season: nobody has missed a game of it yet. Without this, stars rested for the
        last week of April would read as absent on opening night."""
        if getattr(self, "season", None) != season:
            self.missed = defaultdict(int)
            self.season = season

    # ---- values --------------------------------------------------------------------
    def value(self, pid) -> float:
        """Game score per 48 minutes above the league, shrunk toward a below-average prior."""
        lg = self.lg_gs / self.lg_den
        c = self.cfg
        v = (self.gs_num[pid] + c.prior_min * c.prior_frac * lg) / (self.den[pid] + c.prior_min)
        return (v - lg) * 48

    def ovalue(self, pid) -> float:
        lg = self.lg_os / self.lg_den
        c = self.cfg
        v = (self.os_num[pid] + c.prior_min * c.prior_frac * lg) / (self.den[pid] + c.prior_min)
        return (v - lg) * 48

    def is_rotation(self, pid) -> bool:
        m = self.exp_min.get(pid)
        return m is not None and m >= self.cfg.rotation_min

    def impact(self, pid, share=None) -> float:
        """Points a game this player is worth over his replacement at `share` of the minutes -
        his usual share by default - on the RATINGS' scale. The board replaces it with what the
        fitted no-market model charges for him (:func:`nba.predict.price_notes`), which is the
        number that actually moves a prediction; this one stands in where no model is loaded."""
        s = (self.exp_min.get(pid, 0.0) / 48) if share is None else share
        return self.cfg.kappa * s * (self.value(pid) - self.cfg.replacement)

    # ---- the delta -------------------------------------------------------------------
    def delta(self, team: str, target: dict) -> dict:
        """Availability delta for `team` given each rotation player's expected share today.

        ``target`` maps player id -> expected minutes/48 today. Rotation players in the team's
        typical distribution who are absent from `target` count as out. Returns the net and
        offensive deltas (raw units; times ``kappa`` for points per 100) and the largest single
        absence.
        """
        c = self.cfg
        typ = self.typical.get(team, {})
        net = off = top = 0.0
        for pid in set(typ) | set(target):
            if not self.is_rotation(pid):
                continue
            gap = target.get(pid, 0.0) - typ.get(pid, 0.0)
            v, ov = self.value(pid) - c.replacement, self.ovalue(pid) - c.replacement / 2
            net += gap * v
            off += gap * ov
            if gap < 0:
                top = max(top, -gap * v)
        return {"net": net, "off": off, "top": top}

    def adjustment(self, d_home: dict, d_away: dict) -> tuple:
        """(home off, home def, away off, away def) in points per 100 possessions for
        :meth:`nba.ratings.RatingBook.expect`. Defence is the non-offensive part of the delta,
        signed like the ratings' ``def`` (positive = worse)."""
        k = self.cfg.kappa
        return (k * d_home["off"], -k * (d_home["net"] - d_home["off"]),
                k * d_away["off"], -k * (d_away["net"] - d_away["off"]))

    def played_target(self, pids) -> dict:
        """Training: who actually played, at his USUAL minutes - never tonight's."""
        return {int(p): self.exp_min[int(p)] / 48 for p in pids if self.is_rotation(int(p))}

    # ---- update ----------------------------------------------------------------------
    def update(self, team: str, rows, gdate=None) -> None:
        """Learn from one team-game. `rows`: (player id, minutes, game score, offence score)
        for everyone who logged minutes."""
        typ = self.typical[team]
        played = set()
        for pid in list(typ):
            typ[pid] *= self._a_typ
            if typ[pid] < 1e-3:
                del typ[pid]
        for pid, m, gs, os_ in rows:
            pid = int(pid)
            if not m or m != m or m <= 0:
                continue
            played.add(pid)
            typ[pid] = typ.get(pid, 0.0) + (1 - self._a_typ) * m / 48
            e = self.exp_min.get(pid)
            self.exp_min[pid] = m if e is None else self._a_min * e + (1 - self._a_min) * m
            dec = self._a_val ** m
            gs = 0.0 if gs != gs else float(gs)
            os_ = 0.0 if os_ != os_ else float(os_)
            self.gs_num[pid] = self.gs_num[pid] * dec + gs
            self.os_num[pid] = self.os_num[pid] * dec + os_
            self.den[pid] = self.den[pid] * dec + m
            self.lg_gs += gs
            self.lg_os += os_
            self.lg_den += m
            self.team_of[pid] = team
            self.last_date[pid] = gdate
            self.missed[pid] = 0
        for pid in typ:
            if pid not in played and self.team_of.get(pid) == team:
                self.missed[pid] += 1
        # the league reference forgets slowly, so it follows the era rather than 2003
        self.lg_gs *= 0.9995
        self.lg_os *= 0.9995
        self.lg_den *= 0.9995

    # ---- the board ---------------------------------------------------------------------
    def board_target(self, team: str, report: dict | None = None,
                     roster: set | None = None) -> tuple[dict, list[dict]]:
        """Expected shares for `team` tonight, from the report, the roster and the trade wire.

        ``report`` maps player id -> status ("out", "questionable", ...) for this team.
        ``roster`` is the set of player ids on the team's current roster, where known.
        Returns the target shares and one line per player the board is unsure about or
        missing, for the card.
        """
        report = report or {}
        miss = config.MISS_PROB
        target, notes = {}, []
        candidates = set(self.typical.get(team, {}))
        if roster:
            candidates |= {p for p in roster if self.is_rotation(p)}
        for pid in candidates:
            if not self.is_rotation(pid):
                continue
            share = self.exp_min[pid] / 48
            status = report.get(pid)
            if self.team_of.get(pid) not in (None, team) and not (roster and pid in roster):
                p_out, why = 1.0, "left"               # has since played for someone else
            elif roster and pid not in roster and self.team_of.get(pid) == team:
                p_out, why = 1.0, "left"               # no longer on the roster
            elif status is not None:
                p_out, why = miss.get(status, 0.5), status
            elif self.missed.get(pid, 0) >= 3:
                # sat out his team's last three games and is not on the report: a G League
                # assignment, a personal absence, an injury the report has not caught up with
                p_out, why = 0.75, "absent"
            else:
                p_out, why = 0.0, ""
            if p_out < 1.0:
                target[pid] = share * (1.0 - p_out)
            if why and why != "left":
                c = self.cfg
                notes.append({"id": pid, "name": self.names.get(pid, str(pid)), "status": why,
                              "p_out": p_out, "impact": round(self.impact(pid), 1),
                              # what his minutes add to the delta, net and offensive, so the
                              # board can price him with the fitted model's own coefficients
                              "d_net": share * (self.value(pid) - c.replacement),
                              "d_off": share * (self.ovalue(pid) - c.replacement / 2)})
        notes.sort(key=lambda n: -n["impact"] * n["p_out"])
        return target, notes

    def pending_points(self, notes: list[dict]) -> float:
        """How many points of this team's number still hang on unresolved news: the largest
        impact among players who are neither confirmed in nor confirmed out."""
        return max([n["impact"] for n in notes if 0.0 < n["p_out"] < 1.0], default=0.0)
