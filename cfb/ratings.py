"""Online power ratings for college football, replayed game by game.

Football differs from basketball in ways that matter here:

* ~12 regular-season games per team, not ~31. Every game is a much larger share of the
  evidence, so the update step is bigger and the prior-season carry-over matters far more.
* Scoring margins are heavy-tailed (blowouts). Margin is capped before it updates the rating
  so a 63-0 result doesn't overstate the winner.
* Home-field advantage is ~2.5 points and is zero at neutral sites.
* FBS-vs-FCS games exist. FCS opponents aren't in our data at all, so those games are
  excluded from training and rating updates rather than being treated as normal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

HFA = 2.5
MARGIN_CAP = 28.0          # diminishing returns past four scores
MARGIN_K = 0.22
SCORE_K = 0.16
SEASON_CARRY = 0.72        # CFB rosters are stickier than CBB, but coaching/portal churn is real
LEAGUE_PPG = 28.0          # points per team per game, FBS


@dataclass
class RatingConfig:
    hfa: float = HFA
    margin_k: float = MARGIN_K
    score_k: float = SCORE_K
    carry: float = SEASON_CARRY
    cap: float = MARGIN_CAP


def squash(x: float, cap: float) -> float:
    """Soft-cap a margin: linear near zero, saturating past `cap`."""
    if x >= 0:
        return min(x, cap + (x - cap) * 0.25) if x > cap else x
    return -squash(-x, cap)


@dataclass
class Team:
    margin: float = 0.0            # points better than an average FBS team, neutral field
    off: float = LEAGUE_PPG        # points scored vs average defence
    deff: float = LEAGUE_PPG       # points allowed vs average offence
    games: int = 0
    last_date: object = None
    recent_margin: list = field(default_factory=list)
    recent_total: list = field(default_factory=list)

    def regressed(self, carry: float) -> "Team":
        return Team(margin=self.margin * carry,
                    off=LEAGUE_PPG + (self.off - LEAGUE_PPG) * carry,
                    deff=LEAGUE_PPG + (self.deff - LEAGUE_PPG) * carry)


class RatingBook:
    def __init__(self, cfg: RatingConfig | None = None):
        self.cfg = cfg or RatingConfig()
        self.seasons: dict[int, dict[str, Team]] = {}

    def season(self, season: int) -> dict[str, Team]:
        if season not in self.seasons:
            prev = self.seasons.get(season - 1)
            self.seasons[season] = ({t: r.regressed(self.cfg.carry) for t, r in prev.items()}
                                    if prev else {})
        return self.seasons[season]

    def get(self, season: int, team: str) -> Team:
        s = self.season(season)
        if team not in s:
            s[team] = Team()
        return s[team]

    def expect(self, season: int, home: str, away: str, neutral: bool = False) -> dict:
        h, a = self.get(season, home), self.get(season, away)
        hfa = 0.0 if neutral else self.cfg.hfa
        exp_margin = (h.margin - a.margin) + hfa
        # each side's scoring vs the other's defence
        pts_h = h.off + (a.deff - LEAGUE_PPG) + (hfa / 2)
        pts_a = a.off + (h.deff - LEAGUE_PPG) - (hfa / 2)
        return {"exp_margin": exp_margin, "exp_total": max(20.0, pts_h + pts_a),
                "exp_home_points": pts_h, "exp_away_points": pts_a}

    def update(self, season: int, home: str, away: str, hp: float, ap: float, gdate,
               neutral: bool = False) -> None:
        h, a = self.get(season, home), self.get(season, away)
        e = self.expect(season, home, away, neutral)
        k = self.cfg

        err = squash((hp - ap) - e["exp_margin"], k.cap)
        h.margin += k.margin_k * err
        a.margin -= k.margin_k * err

        eh = hp - e["exp_home_points"]
        ea = ap - e["exp_away_points"]
        h.off += k.score_k * eh
        a.deff += k.score_k * eh
        a.off += k.score_k * ea
        h.deff += k.score_k * ea

        total_err = (hp + ap) - e["exp_total"]
        for t, m in ((h, err), (a, -err)):
            t.recent_margin.append(m)
            t.recent_total.append(total_err)
            del t.recent_margin[:-4]
            del t.recent_total[:-4]
            t.games += 1
            t.last_date = gdate
