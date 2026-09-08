"""Online power ratings for the NFL, replayed game by game.

Same shape as ``cfb/ratings.py`` - an Elo-style margin rating plus scoring offence and defence
ratings, updated in schedule order - with four constants moved for reasons specific to this
league:

* **HFA is 1.7, not 2.5.** College home-field is worth roughly two and a half to three points.
  The NFL's has been eroding for a decade (crowd noise rules, better travel, no student
  section) and has sat near 1.5-2 since 2020. Overstating it biases every home side.
* **Carry-over is 0.55, not 0.72.** NFL rosters turn over harder than college ones: free
  agency, the cap, and a 53-man roster mean a good team can lose a third of its snaps in one
  offseason. Ratings regress most of the way to the mean between seasons, and the preseason
  signal is meant to come from prior EPA and roster continuity, not from stale ratings.
* **The update step is larger.** 17 games, not 12, but far more importantly there are only 272
  games in a season league-wide. Each result is a big share of the evidence.
* **League scoring is ~22.5 points per team per game**, against college's 28.

The margin cap matters here too, for the same reason as college: a 59-0 result is not four
times the evidence of a 15-0 one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

HFA = 1.7                  # modern NFL home-field, in points
MARGIN_CAP = 24.0          # diminishing returns past three-and-a-half scores
MARGIN_K = 0.20
SCORE_K = 0.14
SEASON_CARRY = 0.55        # NFL rosters churn hard; regress most of the way to the mean
LEAGUE_PPG = 22.5          # points per team per game


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
    margin: float = 0.0            # points better than an average team, neutral field
    off: float = LEAGUE_PPG        # points scored vs an average defence
    deff: float = LEAGUE_PPG       # points allowed vs an average offence
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

        for t, m in ((h, err), (a, -err)):
            t.recent_margin.append(m)
            t.recent_total.append((hp + ap) - e["exp_total"])
            # Four games is a quarter of an NFL season - a longer form window than college's
            # four out of twelve would be, but shortening it makes the feature pure noise.
            del t.recent_margin[:-4]
            del t.recent_total[:-4]
            t.games += 1
            t.last_date = gdate
