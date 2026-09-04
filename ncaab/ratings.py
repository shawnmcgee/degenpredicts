"""Online power ratings computed by replaying the game log.

These are the backbone of the model and they're leak-free by construction: a rating used for
game N only ever saw games 1..N-1.

Three ratings per team, all updated after every game:

* ``margin``  - Elo-style rating in points. home_margin ~ (margin_h - margin_a) + HCA
* ``off``/``def`` - points scored/allowed per 100 possessions, opponent-adjusted by gradient
  step. total ~ off_h + off_a adjusted by the opposing defences.
* ``pace``    - possessions per game.

Ratings carry over between seasons with regression to the mean (rosters turn over hard in
college, so the carry-over is deliberately aggressive).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Tuned to be sane defaults, not fitted to death. train.py can override via RatingConfig.
HCA_POINTS = 3.0          # typical D1 home-court advantage in points
MARGIN_K = 0.10           # step size for the margin rating
EFF_K = 0.06              # step size for off/def ratings
PACE_K = 0.10
SEASON_CARRY = 0.60       # fraction of last season's rating retained
LEAGUE_PPP = 1.02         # points per possession, D1 average
LEAGUE_PACE = 68.0


@dataclass
class RatingConfig:
    hca: float = HCA_POINTS
    margin_k: float = MARGIN_K
    eff_k: float = EFF_K
    pace_k: float = PACE_K
    carry: float = SEASON_CARRY


@dataclass
class Team:
    margin: float = 0.0
    off: float = LEAGUE_PPP * 100
    deff: float = LEAGUE_PPP * 100
    pace: float = LEAGUE_PACE
    games: int = 0
    last_date: object = None
    recent_margin: list = field(default_factory=list)   # actual - expected, last 5
    recent_total: list = field(default_factory=list)

    def copy_regressed(self, carry: float) -> "Team":
        return Team(margin=self.margin * carry,
                    off=LEAGUE_PPP * 100 + (self.off - LEAGUE_PPP * 100) * carry,
                    deff=LEAGUE_PPP * 100 + (self.deff - LEAGUE_PPP * 100) * carry,
                    pace=LEAGUE_PACE + (self.pace - LEAGUE_PACE) * carry,
                    games=0, last_date=None)


def est_possessions(points_h: int, points_a: int, pace_h: float, pace_a: float) -> float:
    """We don't get box scores from the scoreboard feed, so possessions are estimated from the
    two teams' pace ratings. Crude, but it keeps efficiency and pace on separate axes."""
    return max(50.0, (pace_h + pace_a) / 2)


class RatingBook:
    """Holds every team's rating and knows how to predict + update."""

    def __init__(self, cfg: RatingConfig | None = None):
        self.cfg = cfg or RatingConfig()
        self.teams: dict[int, dict[str, Team]] = {}

    def season(self, season: int) -> dict[str, Team]:
        if season not in self.teams:
            prev = self.teams.get(season - 1)
            self.teams[season] = ({t: r.copy_regressed(self.cfg.carry) for t, r in prev.items()}
                                  if prev else {})
        return self.teams[season]

    def get(self, season: int, team: str) -> Team:
        s = self.season(season)
        if team not in s:
            s[team] = Team()
        return s[team]

    # --- prediction ---------------------------------------------------------------
    def expect(self, season: int, home: str, away: str, neutral: bool = False) -> dict:
        h, a = self.get(season, home), self.get(season, away)
        hca = 0.0 if neutral else self.cfg.hca
        exp_margin = (h.margin - a.margin) + hca
        pace = (h.pace + a.pace) / 2
        # each side's efficiency vs the other's defence, expressed per 100 poss
        off_h = h.off + (a.deff - LEAGUE_PPP * 100)
        off_a = a.off + (h.deff - LEAGUE_PPP * 100)
        exp_total = (off_h + off_a) * pace / 100.0
        return {"exp_margin": exp_margin, "exp_total": exp_total, "exp_pace": pace,
                "off_h": off_h, "off_a": off_a}

    # --- update -------------------------------------------------------------------
    def update(self, season: int, home: str, away: str, hp: int, ap: int, gdate,
               neutral: bool = False) -> None:
        h, a = self.get(season, home), self.get(season, away)
        e = self.expect(season, home, away, neutral)
        k = self.cfg

        margin_err = (hp - ap) - e["exp_margin"]
        h.margin += k.margin_k * margin_err
        a.margin -= k.margin_k * margin_err

        poss = est_possessions(hp, ap, h.pace, a.pace)
        eff_h, eff_a = 100 * hp / poss, 100 * ap / poss
        h.off += k.eff_k * (eff_h - e["off_h"])
        a.deff += k.eff_k * (eff_h - e["off_h"])
        a.off += k.eff_k * (eff_a - e["off_a"])
        h.deff += k.eff_k * (eff_a - e["off_a"])

        total_err = (hp + ap) - e["exp_total"]
        pace_adj = k.pace_k * total_err / 2.0 / (LEAGUE_PPP * 2)
        h.pace += pace_adj
        a.pace += pace_adj

        for t, err_m, err_t in ((h, margin_err, total_err), (a, -margin_err, total_err)):
            t.recent_margin.append(err_m)
            t.recent_total.append(err_t)
            del t.recent_margin[:-5]
            del t.recent_total[:-5]
            t.games += 1
            t.last_date = gdate
