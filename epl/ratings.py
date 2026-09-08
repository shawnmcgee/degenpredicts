"""Online attack and defence ratings, replayed match by match.

The other two sports rate a team by a *margin* in points and update it with the error against
an expected margin. That is the right shape for a sport whose scores are large and roughly
symmetric. It is the wrong shape here, so this engine is built differently in three ways.

**Ratings are multiplicative, in log-goals.** A team is an attack rating and a defence rating,
both log deviations from the league average of ~1.4 goals per side per match:

    lambda_home = LEAGUE_GPG * exp(att_home + def_away + HFA_LOG)
    lambda_away = LEAGUE_GPG * exp(att_away + def_home - HFA_LOG)

Additive goal ratings would let a bad enough defence produce a negative expected goal count,
and would treat "concedes 0.4 more than average" as the same quantity against Manchester City
and against Luton. It is not. Scoring is multiplicative, which is also exactly the form the
Dixon-Coles scoreline model in :mod:`epl.poisson` consumes, so no translation step is needed.

**The update is the Poisson score.** The gradient of the Poisson log-likelihood with respect to
a log-rate is simply ``goals - lambda``, so the update is that error, scaled. This is not a
hand-tuned Elo analogue - it is online gradient ascent on the actual likelihood of the actual
result, which is what makes the ratings feed a probabilistic model honestly.

**Promoted clubs are seeded, not defaulted.** Three of twenty clubs are replaced every season.
A newly promoted club with no top-flight history would otherwise enter rated exactly league
average, which is badly wrong in a knowable direction: promoted sides have historically been
around 0.4 goals a match worse than the division they join. Seeding them at that prior is the
single largest early-season correction in this pipeline, and it has no analogue in the NFL,
where the same 32 franchises come back every year.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# League scoring, per side per match. Measured at 1.415 across 2015-16 to 2024-25; the
# Premier League has run in the 1.35-1.45 band for two decades. This is only the ANCHOR - the
# book carries an adaptive league level on top of it (see RatingBook.league_log), so a
# high-scoring era corrects itself instead of inflating every club's ratings.
LEAGUE_GPG = 1.40
# Home advantage as a log multiplier, applied +to the home rate and -to the away rate, so the
# home/away scoring ratio is exp(2 * HFA_LOG) = 1.22 - about 0.28 goals of supremacy.
# Measured over 2015-16 to 2024-25 the league ran +0.274 goals; dropping the two
# behind-closed-doors seasons it ran +0.302, and those two alone ran +0.161. Home advantage in
# English football has been eroding for thirty years and collapsed almost entirely with the
# crowds, which is why those matches suppress it rather than averaging it away.
HFA_LOG = 0.10
# Learning rates for the two ratings. Defence moves slower than attack because a clean sheet
# is weaker evidence than a hat-trick: goals conceded is the noisier of the two series.
ATT_K = 0.055
DEF_K = 0.045
# How far a rating is allowed from league average, in log space. exp(0.9) is 2.5x the league
# scoring rate, which is beyond anything a real club sustains, and the clamp stops a freak run
# of results from producing a rating the Poisson grid cannot handle.
RATING_CAP = 0.9
# How fast the league-wide scoring level tracks. Deliberately an order of magnitude slower
# than a club's own ratings: this is an era-level property (rule changes, VAR, the 2023-24
# spike to 3.28 goals a match) and it must not chase a run of high-scoring weekends.
LEAGUE_K = 0.004
# Season-to-season carry-over. Higher than the NFL's 0.55: there is no salary cap, no draft and
# no hard roster limit in English football, so the top of the table is far more persistent
# year to year. Lower than 1.0 because summer transfer windows are real.
SEASON_CARRY = 0.80
# What a newly promoted club is worth, relative to the division it is joining. Promoted sides
# score less and concede more; both halves are seeded because a single "overall" prior would
# make them look like a solid defensive side that cannot score, which is not the pattern.
PROMOTED_ATT = -0.22
PROMOTED_DEF = 0.20
# Form window. Five matches is the number the sport itself talks in, and unlike the NFL's
# four-of-seventeen it is a small enough share of a 38-match season to be genuinely recent.
FORM_WINDOW = 5


@dataclass
class RatingConfig:
    hfa: float = HFA_LOG
    att_k: float = ATT_K
    def_k: float = DEF_K
    carry: float = SEASON_CARRY
    cap: float = RATING_CAP
    promoted_att: float = PROMOTED_ATT
    promoted_def: float = PROMOTED_DEF


@dataclass
class Team:
    att: float = 0.0               # log multiplier on goals scored
    deff: float = 0.0              # log multiplier on goals conceded; POSITIVE = leaky
    games: int = 0
    last_date: object = None
    recent_sup: list = field(default_factory=list)     # supremacy vs expectation
    recent_total: list = field(default_factory=list)   # match goals vs expectation

    def regressed(self, carry: float) -> "Team":
        return Team(att=self.att * carry, deff=self.deff * carry)

    @property
    def strength(self) -> float:
        """One number for display and sorting: how many goals a match better than average.

        Attack above average plus defence below it, converted from log space at the league
        rate. Never a model input - the model sees att and deff separately - but it is what a
        reader means by "how good are they".
        """
        return LEAGUE_GPG * (math.exp(self.att) - 1.0) + LEAGUE_GPG * (1.0 - math.exp(self.deff))


class RatingBook:
    def __init__(self, cfg: RatingConfig | None = None):
        self.cfg = cfg or RatingConfig()
        self.seasons: dict[int, dict[str, Team]] = {}
        self._seen: set[str] = set()      # every club ever rated, across all seasons
        # League-wide scoring level, in log space, on top of LEAGUE_GPG. Without it the model
        # is unanchored: nothing forces the mean attack and mean defence rating to stay at
        # zero, so in a season scoring above the anchor EVERY club's ratings drift up together
        # to absorb the difference. That is not merely untidy. It breaks two things -
        # "defence 0 means average" stops being true, so the feature means something different
        # in 2016 than in 2024; and the season rollover regresses ratings toward zero, which
        # is no longer the league mean, so every August the model quietly predicts fewer goals
        # than the league actually scores. Holding the level here and centring the clubs
        # around it fixes both, and is an exact reparameterisation: shifting a constant out of
        # every att and def and into the level leaves every predicted rate unchanged.
        self.league_log = 0.0

    def season(self, season: int) -> dict[str, Team]:
        if season not in self.seasons:
            prev = self.seasons.get(season - 1)
            self.seasons[season] = ({t: r.regressed(self.cfg.carry) for t, r in
                                     self._centred(prev).items()} if prev else {})
        return self.seasons[season]

    def _centred(self, teams: dict[str, "Team"]) -> dict[str, "Team"]:
        """Move the league mean out of the club ratings and into the level.

        Done at the season boundary, which is the only place it matters: it is what makes the
        subsequent regression-to-the-mean regress toward the actual mean. Clubs that never
        played are left alone - a promoted club seeded and then immediately relegated without
        a match should not drag the centring.
        """
        played = [t for t in teams.values() if t.games]
        if not played:
            return teams
        m_att = sum(t.att for t in played) / len(played)
        m_def = sum(t.deff for t in played) / len(played)
        self.league_log += m_att + m_def
        return {name: Team(att=t.att - m_att, deff=t.deff - m_def, games=t.games,
                           last_date=t.last_date, recent_sup=list(t.recent_sup),
                           recent_total=list(t.recent_total))
                for name, t in teams.items()}

    def get(self, season: int, team: str) -> Team:
        """Fetch a club's rating, seeding a newcomer at the promoted-club prior.

        The distinction that matters: a club we have never seen while the book is still empty
        is simply the start of history and belongs at league average. A club we have never seen
        once the league is already populated has been *promoted into* it, and starting it at
        average would hand it roughly a third of a goal a match it has not earned. That is a
        two-point swing on a handicap - larger than any feature in the model.
        """
        s = self.season(season)
        if team not in s:
            if self._seen:
                s[team] = Team(att=self.cfg.promoted_att, deff=self.cfg.promoted_def)
            else:
                s[team] = Team()
        return s[team]

    def is_new(self, season: int, team: str) -> bool:
        """True if this club has never been rated - i.e. it was promoted into the division."""
        return team not in self._seen

    def expect(self, season: int, home: str, away: str, no_hfa: bool = False) -> dict:
        h, a = self.get(season, home), self.get(season, away)
        hfa = 0.0 if no_hfa else self.cfg.hfa
        lvl = self.league_log
        lh = LEAGUE_GPG * math.exp(lvl + h.att + a.deff + hfa)
        la = LEAGUE_GPG * math.exp(lvl + a.att + h.deff - hfa)
        return {"exp_home_goals": lh, "exp_away_goals": la,
                "exp_sup": lh - la, "exp_total": lh + la}

    def update(self, season: int, home: str, away: str, hg: float, ag: float, gdate,
               no_hfa: bool = False) -> None:
        h, a = self.get(season, home), self.get(season, away)
        e = self.expect(season, home, away, no_hfa)
        k, cap = self.cfg, self.cfg.cap
        lh, la = e["exp_home_goals"], e["exp_away_goals"]

        # Poisson score function: d/d(log rate) of the log-likelihood is (goals - rate).
        # Normalised by the league rate so the step size means the same thing regardless of
        # how high-scoring the fixture was expected to be.
        eh = (hg - lh) / LEAGUE_GPG
        ea = (ag - la) / LEAGUE_GPG
        h.att = _clamp(h.att + k.att_k * eh, cap)
        a.deff = _clamp(a.deff + k.def_k * eh, cap)
        a.att = _clamp(a.att + k.att_k * ea, cap)
        h.deff = _clamp(h.deff + k.def_k * ea, cap)

        sup_err = (hg - ag) - e["exp_sup"]
        tot_err = (hg + ag) - e["exp_total"]
        # Track the league's scoring level from the pooled total error, slowly. This is the
        # term that absorbs an era shift so the club ratings do not have to.
        self.league_log += LEAGUE_K * tot_err / (2 * LEAGUE_GPG)
        for t, s in ((h, sup_err), (a, -sup_err)):
            t.recent_sup.append(s)
            t.recent_total.append(tot_err)
            del t.recent_sup[:-FORM_WINDOW]
            del t.recent_total[:-FORM_WINDOW]
            t.games += 1
            t.last_date = gdate
        self._seen.add(home)
        self._seen.add(away)


def _clamp(x: float, cap: float) -> float:
    return max(-cap, min(cap, x))
