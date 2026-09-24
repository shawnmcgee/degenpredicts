"""Online team ratings for the NBA, replayed game by game, per possession.

Points are the product of two things a basketball team controls separately - how many
possessions the game has, and how many points each one is worth - so the ratings keep them
apart:

    possessions  = P  x exp(pace_home + pace_away)
    home per 100 = L + off_home + def_away + H/2 + availability
    away per 100 = L + off_away + def_home - H/2 + availability

``off`` is points per 100 possessions a team scores above the league; ``def`` is points per 100
it ALLOWS above the league, so a good defence is negative. ``pace`` is a log multiplier on
possessions. A total needs all four; a margin mostly needs off and def. Keeping pace out of the
efficiency ratings is what lets a fast bad team and a slow good one both be priced correctly.

**League levels drift and are tracked, not fixed.** Scoring rose from about 190 points a game in
2003 to 230 in 2025, and possessions from 91 to 101. ``L`` and ``P`` follow the league at a slow
rate and the team ratings are re-centred on them at each season boundary, so a rising league
never shows up as every team getting better at once.

**Home court is worth about three points per 100 possessions and shrinking** - home margin ran
+3.3 to +3.9 through 2008 and +1.8 to +2.6 since 2021. ``H`` drifts too, slowly, and is switched
off entirely for neutral-site games and the 2020 Orlando bubble, whose 172 games had no home
crowd at all whatever the schedule called the home side.

**Who played is taken out before a result is learned from.** A team that loses without its best
player has not become a worse team. Each game's expectation includes the availability
adjustment from :mod:`nba.players`, and the rating moves only on the error left after it - so
a star's night off does not drag his team's rating down for the fortnight after he is back.

Blowouts are capped at 45 points per 100 possessions of surprise: the last eight minutes of a
40-point game are played by the end of both benches and say little about either team.

Every learning rate was tuned jointly with the player layer by walk-forward error on 2012-2018,
and nothing was tuned on 2019-2026, the seasons the report scores against the close.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RatingConfig:
    k_off: float = 0.03        # offence, per point per 100 of surprise
    k_def: float = 0.03        # defence
    k_pace: float = 0.10       # pace, per unit of log surprise: it is learned fastest
    k_league: float = 0.005    # league points per 100
    k_league_pace: float = 0.002
    k_hca: float = 0.0005      # home court drifts, slowly
    carry: float = 0.70        # season-to-season carry for off/def
    carry_pace: float = 0.75   # pace is a coach's system as much as a roster, and it carries over
    hca: float = 3.5           # points per 100 possessions at the start of the replay (2003)
    cap: float = 45.0          # per-100 surprise cap: the end of a blowout is not evidence
    league0: float = 100.0     # 2003-04 levels; the warm-up seasons walk them to the present
    pace0: float = 92.0


def _clip(x: float, cap: float) -> float:
    return max(-cap, min(cap, x))


class RatingBook:
    def __init__(self, cfg: RatingConfig | None = None):
        self.cfg = cfg or RatingConfig()
        self.off: dict[str, float] = {}
        self.dfn: dict[str, float] = {}
        self.pace: dict[str, float] = {}
        self.L = self.cfg.league0
        self.P = self.cfg.pace0
        self.H = self.cfg.hca
        self.season: int | None = None
        self.games: dict[str, int] = {}

    # ---- season boundary -----------------------------------------------------------
    def rollover(self, season: int) -> None:
        if self.season is None:
            self.season = season
            return
        if season == self.season:
            return
        c = self.cfg
        for d, carry in ((self.off, c.carry), (self.dfn, c.carry), (self.pace, c.carry_pace)):
            if d:
                m = sum(d.values()) / len(d)
                for t in d:
                    d[t] = (d[t] - m) * carry
        self.games = {}
        self.season = season

    # ---- expectations ---------------------------------------------------------------
    def expect(self, home: str, away: str, neutral: bool = False,
               adj: tuple = (0.0, 0.0, 0.0, 0.0)) -> dict:
        """Expected possessions and points per 100 for both sides.

        ``adj`` = (home offence, home defence, away offence, away defence) availability
        adjustments in points per 100 - defence signed like ``def``, so positive is worse.
        """
        H = 0.0 if neutral else self.H
        poss = self.P * math.exp(self.pace.get(home, 0.0) + self.pace.get(away, 0.0))
        oh = self.L + self.off.get(home, 0.0) + self.dfn.get(away, 0.0) + H / 2 + adj[0] + adj[3]
        oa = self.L + self.off.get(away, 0.0) + self.dfn.get(home, 0.0) - H / 2 + adj[2] + adj[1]
        pts_h, pts_a = poss * oh / 100, poss * oa / 100
        return {"poss": poss, "ortg_h": oh, "ortg_a": oa, "pts_h": pts_h, "pts_a": pts_a,
                "margin": pts_h - pts_a, "total": pts_h + pts_a}

    # ---- update ----------------------------------------------------------------------
    def update(self, home: str, away: str, home_pts, away_pts, poss, periods,
               neutral: bool = False, adj: tuple = (0.0, 0.0, 0.0, 0.0)) -> None:
        """Advance every rating with one completed game.

        ``poss`` is the game's possessions (the two sides' estimates averaged); overtime is
        taken out of pace - five extra minutes are more possessions, not a faster team - but
        kept in efficiency, which is per possession already.
        """
        c = self.cfg
        e = self.expect(home, away, neutral, adj)
        minutes = 48 + 5 * max(0, int(periods or 4) - 4) if periods == periods else 48
        if poss is not None and poss == poss and poss > 50:
            ep = math.log((poss * 48 / minutes) / e["poss"])
            for t in (home, away):
                self.pace[t] = self.pace.get(t, 0.0) + c.k_pace * ep
            self.P *= math.exp(c.k_league_pace * ep)
            played = poss
        else:
            played = e["poss"] * minutes / 48
        eh = _clip(100 * home_pts / played - e["ortg_h"], c.cap)
        ea = _clip(100 * away_pts / played - e["ortg_a"], c.cap)
        self.off[home] = self.off.get(home, 0.0) + c.k_off * eh
        self.dfn[away] = self.dfn.get(away, 0.0) + c.k_def * eh
        self.off[away] = self.off.get(away, 0.0) + c.k_off * ea
        self.dfn[home] = self.dfn.get(home, 0.0) + c.k_def * ea
        self.L += c.k_league * (eh + ea) / 2
        if not neutral:
            self.H += c.k_hca * (eh - ea)
        for t in (home, away):
            self.games[t] = self.games.get(t, 0) + 1
