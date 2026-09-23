"""Online team and goalie ratings, replayed game by game.

Hockey scores are too rare to rate a team on goals alone - a 3-2 game carries very little
evidence about who was better. So scoring is decomposed the way the sport produces it:

    goals on a goalie  =  shots on goal  x  shooting percentage

    shots_home = S0 * exp(ls + hs + sa_home + sd_away)           shot volume: stable, informative
    sh%_home   = P0 * exp(lsh + fin_home + sq_away + gk_away)    finishing and GOALTENDING: noisy

Shot volume is the most persistent thing a hockey team does, so it learns fastest. Finishing
and save support are mostly noise and learn slowly. Goaltending is rated **per goalie**, not
per team, and follows the goalie through trades and signings - a team's save percentage is
largely a property of who is in net, and that changes night to night.

A parallel goals-only rating (att/def) is kept because it catches what shot counts miss -
power-play quality, shot quality - and the models downstream weigh the two.

Every learning rate below was tuned by walk-forward likelihood on 2008-2018 and checked on
2019-2025. The gains are small in absolute terms (log-likelihood per game up ~0.03 on a season
average) because hockey outcomes are mostly noise; the shot model beat the goals-only model on
both halves.

**League levels drift and are tracked, not fixed.** Shots per game fell from 62.5 to 55.3
between 2021 and 2025 and save percentage from .907 to .896; empty-net goals doubled. Every
component is centred at each season boundary and a slow league term absorbs the era.

**Who starts.** The models are fitted on the starter each game actually had, and the board
takes the starter from the day's news (:mod:`nhl.sources.starters`). Before there is any news,
this book's own guess stands in: each team carries an exponentially weighted share of recent
starts (half-life 8 games), and the back-to-back rule is applied on top - in 2021-2025 the
previous night's goalie started the second game of a back-to-back just 9% of the time. That
guess names the actual starter about two times in three, and its confidence is roughly
calibrated, so an uncertain tandem stays uncertain rather than being rounded to its #1.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

S0 = 30.0          # shots on goal per team per game (anchor; the level term tracks the era)
P0 = 0.090         # shooting percentage on goal (anchor)
G0 = S0 * P0       # goals on a goalie per team per game (anchor)


@dataclass
class RatingConfig:
    k_sa: float = 0.045        # shot generation
    k_sd: float = 0.034        # shot suppression
    carry_s: float = 0.63      # season-to-season carry for shot ratings
    k_fin: float = 0.0075      # team finishing
    k_sq: float = 0.0056       # team save support (shot quality allowed)
    carry_f: float = 0.60
    k_g: float = 0.0048        # goalie
    carry_g: float = 0.80      # goaltending persists more than skating does
    new_goalie: float = 0.009  # a goalie never seen before starts slightly below average
    k_att: float = 0.0089      # goals-only model
    k_def: float = 0.0089
    carry_goal: float = 0.76
    k_lvl: float = 0.0015      # league level drift
    hs: float = 0.047          # home shot advantage, log
    hg: float = 0.05           # home goal advantage for the goals-only model, log
    cap: float = 0.6
    share_hl: float = 8.0      # starts half-life, games
    regular_hl: float = 30.0   # a slower one, for who a team's #1 is - not who is hot this week
    b2b_mult: float = 0.12     # weight on the previous night's starter in a back-to-back


def _c(x: float, cap: float) -> float:
    return max(-cap, min(cap, x))


class RatingBook:
    def __init__(self, cfg: RatingConfig | None = None):
        self.cfg = cfg or RatingConfig()
        self.sa = defaultdict(float)
        self.sd = defaultdict(float)
        self.fin = defaultdict(float)
        self.sq = defaultdict(float)
        self.att = defaultdict(float)
        self.dfn = defaultdict(float)
        self.gk: dict[int, float] = {}
        self.goalie_name: dict[int, str] = {}
        self.ls = self.lsh = self.lg = 0.0
        self.season: int | None = None
        self.share: dict[str, dict[int, float]] = defaultdict(dict)
        self.share_long: dict[str, dict[int, float]] = defaultdict(dict)
        self.last_start: dict[str, int] = {}
        self.last_date: dict[str, object] = {}
        self.games: dict[str, int] = defaultdict(int)

    # ---- season boundary -----------------------------------------------------------
    def rollover(self, season: int) -> None:
        if self.season is None or season == self.season:
            self.season = season if self.season is None else self.season
            return
        c = self.cfg
        for d, carry in ((self.sa, c.carry_s), (self.sd, c.carry_s), (self.fin, c.carry_f),
                         (self.sq, c.carry_f), (self.att, c.carry_goal), (self.dfn, c.carry_goal)):
            if d:
                m = sum(d.values()) / len(d)
                for t in d:
                    d[t] = (d[t] - m) * carry
        if self.gk:
            m = sum(self.gk.values()) / len(self.gk)
            for k in self.gk:
                self.gk[k] = (self.gk[k] - m) * c.carry_g
        self.games = defaultdict(int)
        self.season = season

    # ---- goalies ------------------------------------------------------------------
    def goalie(self, gid) -> float:
        if gid is None or gid != gid:
            return 0.0
        return self.gk.get(int(gid), self.cfg.new_goalie)

    def starter_probs(self, team: str, b2b: bool, last=None) -> dict[int, float]:
        """P(each goalie starts) from the recent-start shares and the back-to-back rule."""
        w = dict(self.share.get(team) or {})
        if not w:
            return {}
        last = self.last_start.get(team) if last is None else last
        if b2b and last in w and len(w) > 1:
            w[last] *= self.cfg.b2b_mult
        tot = sum(w.values())
        return {k: v / tot for k, v in w.items()}

    def expected_goalie(self, team: str, b2b: bool, pending_today: bool = False) -> tuple[float, dict]:
        """Expected goalie rating for `team`'s next game.

        ``pending_today`` means the team plays today too and tomorrow's starter depends on who
        starts tonight - unknown this morning. The expectation is taken over tonight's likely
        starters, each followed by the back-to-back rule for tomorrow.
        """
        if not pending_today:
            probs = self.starter_probs(team, b2b)
        else:
            tonight = self.starter_probs(team, b2b=False)
            probs = defaultdict(float)
            for k, pk in tonight.items():
                for g, pg in self.starter_probs(team, b2b=True, last=k).items():
                    probs[g] += pk * pg
            probs = dict(probs)
        return self.rating(probs), probs

    def rating(self, probs: dict) -> float:
        """The expected goalie rating under a distribution over who starts."""
        if not probs:
            return self.cfg.new_goalie
        return sum(p * self.goalie(k) for k, p in probs.items())

    def regular(self, team: str):
        """The team's usual #1: the largest share of its starts over the last couple of months,
        so a starter back from a fortnight's injury is still the #1, not the backup."""
        w = self.share_long.get(team) or {}
        return max(w, key=w.get) if w else None

    def is_backup(self, team: str, gid) -> bool:
        """True when `gid` is not the team's clear #1 - one with at least 55% of its starts.
        A genuine tandem has no backup, so a 50/50 split never raises the flag."""
        w = self.share_long.get(team) or {}
        top = self.regular(team)
        if gid is None or gid != gid or top is None or int(gid) == top:
            return False
        return w[top] / sum(w.values()) >= 0.55

    # ---- expectations ---------------------------------------------------------------
    def expect(self, home: str, away: str, g_vs_home: float, g_vs_away: float) -> dict:
        """Expected shots, shooting percentages and goals-on-goalie for both sides.

        ``g_vs_home`` is the rating of the goalie the HOME shooters face - the away starter.
        """
        c = self.cfg
        S_h = S0 * math.exp(self.ls + c.hs + self.sa[home] + self.sd[away])
        S_a = S0 * math.exp(self.ls - c.hs + self.sa[away] + self.sd[home])
        p_h = P0 * math.exp(self.lsh + self.fin[home] + self.sq[away] + g_vs_home)
        p_a = P0 * math.exp(self.lsh + self.fin[away] + self.sq[home] + g_vs_away)
        L_h = G0 * math.exp(self.lg + c.hg + self.att[home] + self.dfn[away])
        L_a = G0 * math.exp(self.lg - c.hg + self.att[away] + self.dfn[home])
        return {"S_h": S_h, "S_a": S_a, "p_h": p_h, "p_a": p_a,
                "lam_h": S_h * p_h, "lam_a": S_a * p_a, "L_h": L_h, "L_a": L_a}

    # ---- update ----------------------------------------------------------------------
    def update(self, home: str, away: str, home_goalie, away_goalie, home_shots, away_shots,
               home_gg, away_gg, gdate) -> None:
        """Advance every rating with one completed game.

        ``home_gg``/``away_gg`` are goals scored against a goalie in regulation - empty-net,
        overtime and shootout goals are the scoreline model's business, not the ratings'.
        """
        c, cap = self.cfg, self.cfg.cap
        e = self.expect(home, away, self.goalie(away_goalie), self.goalie(home_goalie))
        if home_shots == home_shots and away_shots == away_shots and home_shots is not None:
            eh = (home_shots - e["S_h"]) / S0
            ea = (away_shots - e["S_a"]) / S0
            self.sa[home] = _c(self.sa[home] + c.k_sa * eh, cap)
            self.sd[away] = _c(self.sd[away] + c.k_sd * eh, cap)
            self.sa[away] = _c(self.sa[away] + c.k_sa * ea, cap)
            self.sd[home] = _c(self.sd[home] + c.k_sd * ea, cap)
            self.ls += c.k_lvl * (eh + ea) / 2
            fh = (home_gg - home_shots * e["p_h"]) / G0
            fa = (away_gg - away_shots * e["p_a"]) / G0
            self.fin[home] = _c(self.fin[home] + c.k_fin * fh, cap)
            self.sq[away] = _c(self.sq[away] + c.k_sq * fh, cap)
            self.fin[away] = _c(self.fin[away] + c.k_fin * fa, cap)
            self.sq[home] = _c(self.sq[home] + c.k_sq * fa, cap)
            self.lsh += c.k_lvl * (fh + fa) / 2
            for gid, f in ((away_goalie, fh), (home_goalie, fa)):
                if gid is not None and gid == gid:
                    gid = int(gid)
                    self.gk[gid] = _c(self.gk.get(gid, c.new_goalie) + c.k_g * f, cap)
        dh = (home_gg - e["L_h"]) / G0
        da = (away_gg - e["L_a"]) / G0
        self.att[home] = _c(self.att[home] + c.k_att * dh, cap)
        self.dfn[away] = _c(self.dfn[away] + c.k_def * dh, cap)
        self.att[away] = _c(self.att[away] + c.k_att * da, cap)
        self.dfn[home] = _c(self.dfn[home] + c.k_def * da, cap)
        self.lg += c.k_lvl * (dh + da) / 2

        for team, gid in ((home, home_goalie), (away, away_goalie)):
            for sh, hl in ((self.share[team], c.share_hl), (self.share_long[team], c.regular_hl)):
                decay = 0.5 ** (1 / hl)
                for k in list(sh):
                    sh[k] *= decay
                    if sh[k] < 1e-3:
                        del sh[k]
                if gid is not None and gid == gid:
                    sh[int(gid)] = sh.get(int(gid), 0.0) + (1 - decay)
            if gid is not None and gid == gid:
                self.last_start[team] = int(gid)
            self.last_date[team] = gdate
            self.games[team] += 1
