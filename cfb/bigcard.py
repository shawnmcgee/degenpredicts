"""Big Game Cards: the bet you were going to make anyway, cheaper and better informed.

Not a pick sheet. On the games people watch - an SEC Saturday, a top-25 matchup - the side and
the total are the sharpest numbers in college football, and this model does not beat them: on
Power-4 games in 2026 its spread side went 65-50 with a closing-line value of -0.02, which is
luck rather than edge. So the card shows only what is true whether or not the model is right:

1. **Where each side is cheapest** across the sportsbooks and the exchanges, with exchange fees
   included, so a Kalshi 56c reads as the -137 it actually costs.
2. **How far the line has moved** since our first pull.
3. **What the model thinks**, next to how much that has been worth on games like this one.
4. **Fair value for every rung of Kalshi's ladders**, priced from the sportsbook's own number
   plus how college games have finished against similar lines since 2015. That is a
   hypothesis, not a claim. ``core.ladders`` logs every rung so it can be graded.

How "fair" is built
-------------------
The book's main line pins the middle; history fills in the steps between rungs. For a game
where the favourite gives L points, let H(y) be the share of past games with a similar line
(within a few points, weighted toward L) that the favourite won by more than y. Then

    fair(y) = logistic(logit(H(y)) + d),   d chosen so that fair(L) = p_market

where p_market is the de-vigged price of the favourite at L. Anchoring to the market means the
card never claims to know the main line better than the book; it only says what the line
implies about every other number. The shift is applied on the log-odds scale so a correction
at the main number moves a 20% tail rung by less than it moves a coin flip - an additive
shift overstated the tails and painted every long alt line green. Football margins pile up on
3, 7, 10 and 14, so the steps are lumpy, which is the point: a normal curve would smear them
out and misprice exactly the rungs around those numbers.

A price is only called cheap when it beats fair by more than twice the sampling error of the
history behind it. A couple of thousand past games pin a 50% number to about +/-1.2 points;
that is the honest resolution of this method, and most rungs sit inside it.

Totals work the same way with total points.

Run: ``python -m cfb.bigcard`` (after ``cfb.predict``). Writes ``docs/cfb/big/index.html``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from core import ladders as ladder_log

from . import config
from .sources import odds as odds_src

log = logging.getLogger(__name__)
TEMPLATES = Path(__file__).parent / "templates"

# Which games get a card: both teams in one of these conferences, plus any game involving a
# team named in DEGEN_BIG_TEAMS (for the Ohio State - Michigan week).
BIG_CONFS = {c.strip() for c in os.environ.get("DEGEN_BIG_CONFS", "SEC").split(",") if c.strip()}
BIG_TEAMS = {t.strip() for t in os.environ.get("DEGEN_BIG_TEAMS", "").split(",") if t.strip()}
# Games involving these teams go to the top of the page (the week's marquee game); the rest
# follow in kickoff order.
FEATURE = {t.strip() for t in os.environ.get("DEGEN_BIG_FEATURE", "").split(",") if t.strip()}
P4 = {"SEC", "Big Ten", "Big 12", "ACC"}

# History window: past games whose line was within this many points of today's line, weighted
# toward it, widened until the weighted sample is at least MIN_EFF games.
SPREAD_WINDOW = 3.0
TOTAL_WINDOW = 4.0
MIN_EFF = 400.0
# Alt-line menu: rungs whose fair chance sits in this band are shown by default, the rest go
# behind "show all". Out in the tails both the history and the quotes get thin.
MENU_BAND = (0.25, 0.75)
# How close (in points) an exchange rung must sit to the main number to be offered as another
# way of making the main bet.
NEAR_MAIN = 1.5
# Below this value a price is "fair" rather than cheap - history is an estimate, not a fact.
CHEAP_EV = 0.02

MINUS = "−"


# --------------------------------------------------------------------------------------
# odds arithmetic
# --------------------------------------------------------------------------------------
def american_to_prob(a) -> float:
    if a is None or a != a:
        return float("nan")
    a = float(a)
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def prob_to_american(p) -> float:
    if p is None or p != p or p <= 0 or p >= 1:
        return float("nan")
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def fmt_american(a) -> str:
    if a is None or a != a:
        return "—"
    a = int(round(a))
    return f"+{a}" if a > 0 else f"{MINUS}{abs(a)}"


def fmt_point(x) -> str:
    """-8.5 -> '−8.5', 8.5 -> '+8.5', 0 -> 'PK'."""
    if x is None or x != x:
        return ""
    if abs(x) < 1e-9:
        return "PK"
    s = f"{abs(x):g}"
    return f"+{s}" if x > 0 else f"{MINUS}{s}"


def exchange_fee(price: float, key: str) -> float:
    coef = config.EXCHANGE_FEES.get(key, 0.0)
    return coef * price * (1 - price) if price == price else float("nan")


def cost_of(key: str, american=None, contract=None) -> float:
    """What one dollar of payout costs, as a probability: the break-even win rate.

    A sportsbook's margin is already inside its American price. An exchange sells a 0/1
    contract at ``contract`` dollars and charges its fee on top; when the exchange is quoted
    through a feed as an American price, the price is its contract price before fees.
    """
    if contract is None:
        contract = american_to_prob(american)
    if contract != contract:
        return float("nan")
    return contract + (exchange_fee(contract, key) if key in config.EXCHANGE_FEES else 0.0)


def expected_value(p_win: float, p_push: float, cost: float) -> float:
    """Expected profit per dollar risked when a win pays (1 - cost) / cost."""
    if any(v != v for v in (p_win, p_push, cost)) or not 0 < cost < 1:
        return float("nan")
    return p_win * (1 - cost) / cost - (1 - p_win - p_push)


# --------------------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------------------
@dataclass
class Dist:
    """A weighted sample of integer outcomes (favourite margin, or total points)."""
    values: np.ndarray
    weights: np.ndarray

    @property
    def n_eff(self) -> float:
        """Kish effective sample size of the weighted sample."""
        w = self.weights
        return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0

    @property
    def n_games(self) -> int:
        return int(len(self.values))

    def p_gt(self, y: float) -> float:
        return float((self.weights * (self.values > y)).sum() / self.weights.sum())

    def p_eq(self, y: float) -> float:
        if abs(y - round(y)) > 1e-9:
            return 0.0
        return float((self.weights * (self.values == round(y))).sum() / self.weights.sum())


@dataclass
class Anchored:
    """History's shape, pinned to the market's price at the market's own number."""
    dist: Dist
    line: float          # favourite gives `line` (spreads) / the total (totals)
    p_line: float        # de-vigged market probability of "more than line"

    def __post_init__(self):
        self.shift = _logit(self.p_line) - _logit(self.dist.p_gt(self.line))

    def p_gt(self, y: float) -> float:
        return _logistic(_logit(self.dist.p_gt(y)) + self.shift)

    def p_eq(self, y: float) -> float:
        return self.dist.p_eq(y)

    def se(self, p: float) -> float:
        """Sampling error of a probability read off this history."""
        n = self.dist.n_eff
        return math.sqrt(max(p * (1 - p), 1e-6) / n) if n > 0 else float("nan")


def _logit(p: float) -> float:
    p = min(max(p, 0.002), 0.998)
    return math.log(p / (1 - p))


def _logistic(z: float) -> float:
    return 1 / (1 + math.exp(-z))


class History:
    """Every completed game with a closing line: favourite margin and total points."""

    def __init__(self, spread: pd.DataFrame, total: pd.DataFrame):
        self.spread = spread            # columns L (favourite's line, >= 0), M (fav margin)
        self.total = total              # columns L (total line), T (total points)

    @classmethod
    def build(cls, games: pd.DataFrame, lines: pd.DataFrame) -> "History":
        rank = {p: i for i, p in enumerate(config.PREFERRED_PROVIDERS)}
        ln = lines.assign(_r=lines["provider"].map(rank).fillna(len(rank)))
        ln = ln.sort_values("_r").drop_duplicates("game_id")[["game_id", "spread_home", "total_line"]]
        g = games[games["completed"].astype(str).str.lower().isin(["true", "1"])]
        d = g.merge(ln, on="game_id", how="inner")
        s = d.dropna(subset=["spread_home", "home_margin"])
        fav_home = s["spread_home"] <= 0
        spread = pd.DataFrame({"L": s["spread_home"].abs().to_numpy(),
                               "M": np.where(fav_home, s["home_margin"], -s["home_margin"])})
        t = d.dropna(subset=["total_line", "total_points"])
        total = pd.DataFrame({"L": t["total_line"].to_numpy(), "T": t["total_points"].to_numpy()})
        return cls(spread, total)

    @classmethod
    def load(cls) -> "History":
        games = pd.read_csv(config.GAMES, dtype={"game_id": str}, low_memory=False)
        lines = pd.read_csv(config.LINES, dtype={"game_id": str}, low_memory=False)
        return cls.build(games, lines)

    @staticmethod
    def _dist(df: pd.DataFrame, col: str, target: float, window: float) -> Dist:
        w = window
        while True:
            wt = np.clip(1 - np.abs(df["L"].to_numpy() - target) / (w + 0.5), 0, None)
            if wt.sum() >= MIN_EFF or w >= window * 4:
                break
            w += 1
        keep = wt > 0
        return Dist(df[col].to_numpy()[keep], wt[keep])

    def spread_dist(self, fav_line: float) -> Dist:
        return self._dist(self.spread, "M", abs(fav_line), SPREAD_WINDOW)

    def total_dist(self, total: float) -> Dist:
        return self._dist(self.total, "T", total, TOTAL_WINDOW)


# --------------------------------------------------------------------------------------
# outcome probabilities in sportsbook language
# --------------------------------------------------------------------------------------
def fav_cover(a: Anchored, point: float) -> tuple[float, float]:
    """Favourite at `point` (-8.5 = gives 8.5; +3 = gets 3) -> (P win, P push)."""
    y = -point
    return a.p_gt(y), a.p_eq(y)


def dog_cover(a: Anchored, point: float) -> tuple[float, float]:
    """Underdog at `point` (+8.5 = gets 8.5; -6.5 = wins by 7+) -> (P win, P push)."""
    push = a.p_eq(point)
    return max(1 - a.p_gt(point) - push, 0.0), push


def over(a: Anchored, x: float) -> tuple[float, float]:
    return a.p_gt(x), a.p_eq(x)


def under(a: Anchored, x: float) -> tuple[float, float]:
    push = a.p_eq(x)
    return max(1 - a.p_gt(x) - push, 0.0), push


# --------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------
def select_games(board: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Games that get a card: both teams in BIG_CONFS, or either team in BIG_TEAMS."""
    if board.empty:
        return board
    conf = games[["game_id", "home_conf", "away_conf"]].drop_duplicates("game_id")
    b = board.merge(conf, on="game_id", how="left")
    both = b["home_conf"].isin(BIG_CONFS) & b["away_conf"].isin(BIG_CONFS)
    named = b["home_team"].isin(BIG_TEAMS) | b["away_team"].isin(BIG_TEAMS)
    out = b[(both | named) & b["spread_home"].notna()].copy()
    out["_feature"] = ~(out["home_team"].isin(FEATURE) | out["away_team"].isin(FEATURE))
    sort = ["_feature"] + [c for c in ("kickoff_utc", "tip_et", "home_team") if c in out]
    return out.sort_values(sort).drop(columns="_feature").reset_index(drop=True)


def current_board(picks: pd.DataFrame, today) -> pd.DataFrame:
    """The newest published row for every game that has not been played yet."""
    if picks.empty:
        return picks
    p = picks.copy()
    p["_d"] = pd.to_datetime(p["date"]).dt.date
    p = p[p["_d"] >= today]
    if p.empty:
        return p.drop(columns="_d")
    p = p.sort_values("prediction_date").drop_duplicates("game_id", keep="last")
    return p.drop(columns="_d").reset_index(drop=True)


def _quotes_for(frame: pd.DataFrame, g) -> list[dict]:
    """The latest per-venue quotes our odds pull recorded for this game, if any.

    ``frame`` is the quotes log (``odds.quotes_history``) or anything else carrying the same
    ``pulled_at / date / home_team / away_team / quotes`` columns.
    """
    if frame is None or frame.empty or "quotes" not in frame:
        return []
    s = frame[(frame["home_team"] == g["home_team"]) & (frame["away_team"] == g["away_team"])
              & (frame["date"].astype(str).str[:10] == str(g["date"])[:10])]
    s = s.dropna(subset=["quotes"])
    if s.empty:
        return []
    try:
        q = json.loads(s.sort_values("pulled_at")["quotes"].iloc[-1])
    except (TypeError, ValueError):
        return []
    return [x for x in q if isinstance(x, dict) and {"m", "side", "point", "price"} <= set(x)]


def _movement(snaps: pd.DataFrame, g, fav_home: bool) -> dict | None:
    if snaps is None or snaps.empty:
        return None
    s = snaps[(snaps["home_team"] == g["home_team"]) & (snaps["away_team"] == g["away_team"])
              & (snaps["date"].astype(str).str[:10] == str(g["date"])[:10])]
    s = s.dropna(subset=["live_spread_home"]).sort_values("pulled_at")
    if s.empty:
        return None
    first, last = s.iloc[0], s.iloc[-1]
    since = pd.Timestamp(first["pulled_at"]).strftime("%b %-d")
    fav = g["home_team"] if fav_home else g["away_team"]
    sign = 1 if fav_home else -1
    out = {"since": since, "fav": fav,
           "spread_from": sign * float(first["live_spread_home"]),
           "spread_to": sign * float(last["live_spread_home"]),
           "total_from": float(first["live_total"]) if first["live_total"] == first["live_total"] else None,
           "total_to": float(last["live_total"]) if last["live_total"] == last["live_total"] else None}
    out["spread_moved"] = round(out["spread_to"] - out["spread_from"], 1)
    out["total_moved"] = (round(out["total_to"] - out["total_from"], 1)
                          if out["total_from"] is not None and out["total_to"] is not None else 0)
    return out


def _model_record(results: pd.DataFrame, games: pd.DataFrame, season: int) -> dict | None:
    """The model's spread record and CLV on Power-4 v Power-4 games this season."""
    if results is None or results.empty or "spread_result" not in results:
        return None
    conf = games[["game_id", "home_conf", "away_conf"]].drop_duplicates("game_id")
    r = results.merge(conf, on="game_id", how="left")
    r = r[(r["season"] == season) & r["home_conf"].isin(P4) & r["away_conf"].isin(P4)]
    s = r[r["spread_result"].isin(["win", "loss"])]
    if s.empty:
        return None
    clv = s["spread_clv"].dropna() if "spread_clv" in s else pd.Series(dtype=float)
    return {"wins": int((s["spread_result"] == "win").sum()),
            "losses": int((s["spread_result"] == "loss").sum()),
            "clv": round(float(clv.mean()), 2) if len(clv) else None, "clv_n": int(len(clv))}


# --------------------------------------------------------------------------------------
# one card
# --------------------------------------------------------------------------------------
def _row(venue: str, key: str, label: str, p_win: float, p_push: float, cost: float,
         price_text: str, liquid: bool = True, note: str = "", se: float = 0.0) -> dict:
    ev = expected_value(p_win, p_push, cost)
    # the cost at which this bet breaks even: p_win * (1 - c) / c = 1 - p_win - p_push
    fair = p_win / (1 - p_push) if p_push < 1 else float("nan")
    # Cheap means beating fair by more than the history behind "fair" can resolve.
    cheap = (ev == ev and ev >= CHEAP_EV and liquid and fair - cost > 2 * (se or 0.0))
    return {"venue": venue, "key": key, "label": label, "price": price_text,
            "costs_like": fmt_american(prob_to_american(cost)), "cost": cost,
            "fair": fair, "fair_like": fmt_american(prob_to_american(fair)),
            "ev": ev, "liquid": liquid, "note": note,
            "verdict": "cheap" if cheap else "fair" if ev == ev and ev > -0.02 else "pricey"}


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _kalshi_price(contract: float) -> str:
    return f"{int(round(contract * 100))}¢"


def build_card(g, hist: History, rungs: pd.DataFrame, quotes: list[dict],
               snaps: pd.DataFrame | None, model_rec: dict | None) -> dict:
    home, away = g["home_team"], g["away_team"]
    spread_home = float(g["spread_home"])
    fav_home = spread_home <= 0
    fav, dog = (home, away) if fav_home else (away, home)
    L = abs(spread_home)

    # Market's own price at its own number, vig removed.
    p_fav_side = american_to_prob(g.get("spread_home_price") if fav_home else g.get("spread_away_price"))
    p_dog_side = american_to_prob(g.get("spread_away_price") if fav_home else g.get("spread_home_price"))
    p_fav = p_fav_side / (p_fav_side + p_dog_side) if p_fav_side == p_fav_side and p_dog_side == p_dog_side else 0.5
    sp = Anchored(hist.spread_dist(L), L, p_fav)

    total = g.get("total_line")
    tot = None
    if total == total and total is not None:
        po, pu = american_to_prob(g.get("over_price")), american_to_prob(g.get("under_price"))
        p_over = po / (po + pu) if po == po and pu == pu else 0.5
        tot = Anchored(hist.total_dist(float(total)), float(total), p_over)

    book = g.get("spread_book") if isinstance(g.get("spread_book"), str) else "Sportsbook"
    tbook = g.get("total_book") if isinstance(g.get("total_book"), str) else book

    # ---- where to bet each side ------------------------------------------------------
    sides = {"fav": [], "dog": [], "over": [], "under": []}
    have_ladder = rungs is not None and len(rungs) > 0
    if quotes:
        for q in quotes:
            key, venue = q.get("key", ""), q.get("book", q.get("key", ""))
            if key == "kalshi" and have_ladder:
                continue          # the ladder below prices Kalshi rung by rung, YES and NO
            point, price = float(q["point"]), float(q["price"])
            cost = cost_of(key, american=price)
            ptext = fmt_american(price) + (" before fee" if config.EXCHANGE_FEES.get(key) else "")
            if q["m"] == "spread":
                is_fav = (q["side"] == "home") == fav_home
                team = fav if is_fav else dog
                pw, pp = (fav_cover if is_fav else dog_cover)(sp, point)
                sides["fav" if is_fav else "dog"].append(
                    _row(venue, key, f"{team} {fmt_point(point)}", pw, pp, cost, ptext,
                         se=sp.se(pw)))
            elif q["m"] == "total" and tot is not None:
                pw, pp = (over if q["side"] == "over" else under)(tot, point)
                sides[q["side"]].append(
                    _row(venue, key, f"{q['side'].title()} {point:g}", pw, pp, cost, ptext,
                         se=tot.se(pw)))
    else:
        # Before per-venue quotes existed: the one book the board priced from.
        for is_fav, price in ((True, g.get("spread_home_price") if fav_home else g.get("spread_away_price")),
                              (False, g.get("spread_away_price") if fav_home else g.get("spread_home_price"))):
            point = -L if is_fav else L
            pw, pp = (fav_cover if is_fav else dog_cover)(sp, point)
            sides["fav" if is_fav else "dog"].append(
                _row(book, "book", f"{fav if is_fav else dog} {fmt_point(point)}", pw, pp,
                     cost_of("book", american=price if price == price else -110),
                     fmt_american(price if price == price else -110), se=sp.se(pw)))
        if tot is not None:
            for side, price in (("over", g.get("over_price")), ("under", g.get("under_price"))):
                pw, pp = (over if side == "over" else under)(tot, float(total))
                sides[side].append(_row(tbook, "book", f"{side.title()} {float(total):g}", pw, pp,
                                        cost_of("book", american=price if price == price else -110),
                                        fmt_american(price if price == price else -110),
                                        se=tot.se(pw)))

    # Kalshi rungs near the main number are another way of making the same bet.
    menu, menu_tot = [], []
    if rungs is not None and len(rungs):
        for r in rungs.itertuples():
            bid = float(r.yes_bid) if r.yes_bid == r.yes_bid else float("nan")
            ask = float(r.yes_ask) if r.yes_ask == r.yes_ask else float("nan")
            liquid = bool(r.tradeable)
            if r.kind == "spread":
                x = float(r.strike)
                fav_ladder = r.team == fav
                if not fav_ladder and r.team != dog:
                    continue
                # YES on "<fav> wins by over x" is fav -x; its NO is dog +x.
                # YES on "<dog> wins by over x" is dog -x; its NO is fav +x.
                yes = ("fav" if fav_ladder else "dog", -x, ask, "Yes")
                no = ("dog" if fav_ladder else "fav", x, 1 - bid, "No")
                legs = {}
                for side, pt, contract, how in (yes, no):
                    if contract != contract or not 0 < contract < 1:
                        legs[side] = None
                        continue
                    team = fav if side == "fav" else dog
                    pw, pp = (fav_cover if side == "fav" else dog_cover)(sp, pt)
                    row = _row("Kalshi", "kalshi", f"{team} {fmt_point(pt)}", pw, pp,
                               cost_of("kalshi", contract=contract), _kalshi_price(contract),
                               liquid, note=f"buy {how} on “{r.team} by {int(x + 0.5)}+”",
                               se=sp.se(pw))
                    legs[side] = row
                    main_pt = -L if side == "fav" else L
                    if abs(pt - main_pt) <= NEAR_MAIN:
                        sides[side].append(row)
                y = x if fav_ladder else -x          # favourite-margin threshold, for ordering
                menu.append({"y": y, "fav": legs.get("fav"), "dog": legs.get("dog"),
                             "core": MENU_BAND[0] <= sp.p_gt(y) <= MENU_BAND[1]})
            elif r.kind == "total" and tot is not None:
                x = float(r.strike)
                legs = []
                for side, contract in (("over", ask), ("under", 1 - bid)):
                    if contract != contract or not 0 < contract < 1:
                        legs.append(None)
                        continue
                    pw, pp = (over if side == "over" else under)(tot, x)
                    row = _row("Kalshi", "kalshi", f"{side.title()} {x:g}", pw, pp,
                               cost_of("kalshi", contract=contract), _kalshi_price(contract), liquid,
                               note=f"buy {'Yes' if side == 'over' else 'No'} on “Over {x:g}”",
                               se=tot.se(pw))
                    legs.append(row)
                    if abs(x - float(total)) <= NEAR_MAIN:
                        sides[side].append(row)
                p_mid = tot.p_gt(x)
                menu_tot.append({"y": x, "over": legs[0], "under": legs[1],
                                 "core": MENU_BAND[0] <= p_mid <= MENU_BAND[1]})
    menu.sort(key=lambda m: -m["y"])
    menu_tot.sort(key=lambda m: m["y"])

    def rank(rows):
        # liquid quotes first, then best value; a thin exchange quote is shown, never ranked top
        return sorted(rows, key=lambda r: (not r["liquid"], -(r["ev"] if r["ev"] == r["ev"] else -9)))

    for k in sides:
        sides[k] = rank(sides[k])

    # ---- moneyline -------------------------------------------------------------------
    ml = []
    for team, ask in ((home, g.get("kalshi_home_ask")), (away, g.get("kalshi_away_ask"))):
        if ask is None or ask != ask:
            continue
        pw = sp.p_gt(0) if team == fav else 1 - sp.p_gt(0)
        ml.append(_row("Kalshi", "kalshi", f"{team} to win", pw, 0.0,
                       cost_of("kalshi", contract=float(ask)), _kalshi_price(float(ask)),
                       bool(g.get("kalshi_tradeable", True)), se=sp.se(pw)))

    # ---- model -----------------------------------------------------------------------
    model = None
    mp, tp = _num(g.get("margin_pred")), _num(g.get("total_pred"))
    if mp == mp:
        fav_margin_model = mp if fav_home else -mp
        diff = round(fav_margin_model - L, 1)
        total_diff = round(tp - float(total), 1) if tot is not None and tp == tp else None
        model = {"fav_margin": round(fav_margin_model, 1), "diff": diff,
                 "total": tp if tp == tp else None, "total_diff": total_diff,
                 "agrees": abs(diff) < 1.5 and (total_diff is None or abs(total_diff) < 1.5),
                 "record": model_rec}

    return {
        "game_id": str(g["game_id"]), "home": home, "away": away, "fav": fav, "dog": dog,
        "neutral": bool(g.get("neutral_site", False)), "kick": g.get("tip_et", ""),
        "line": fmt_point(-L), "total": float(total) if tot is not None else None,
        "conf": g.get("home_conf") if g.get("home_conf") == g.get("away_conf") else "",
        "sides": sides, "ml": ml, "menu": menu, "menu_tot": menu_tot,
        "has_quotes": bool(quotes), "has_ladder": bool(menu or menu_tot),
        "move": _movement(snaps, g, fav_home), "model": model,
        "hist_n": sp.dist.n_games, "hist_n_total": tot.dist.n_games if tot else 0,
    }


# --------------------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------------------
def _pct(v) -> str:
    if v is None or v != v:
        return "—"
    return f"{'+' if v > 0 else MINUS if v < 0 else ''}{abs(v) * 100:.1f}%"


def build(ladder_path: Path | None = None, out_dir: Path | None = None) -> Path | None:
    today = config.today_et()
    if not config.PICKS.exists():
        log.info("bigcard: no picks yet")
        return None
    picks = pd.read_csv(config.PICKS, dtype={"game_id": str}, low_memory=False)
    games = pd.read_csv(config.GAMES, dtype={"game_id": str}, low_memory=False)
    board = select_games(current_board(picks, today), games)
    snaps = pd.read_csv(config.SNAPSHOTS, low_memory=False) if config.SNAPSHOTS.exists() else None
    results = (pd.read_csv(config.RESULTS, dtype={"game_id": str}, low_memory=False)
               if config.RESULTS.exists() else None)
    if ladder_path is not None:
        rungs_all = pd.read_csv(ladder_path, dtype={"game_id": str})
    else:
        rungs_all = ladder_log.latest(config.LADDERS)
    hist = History.load()
    rec = _model_record(results, games, config.season_of(today))
    quotes = odds_src.quotes_history()

    cards = []
    for _, g in board.iterrows():
        r = (rungs_all[rungs_all["game_id"].astype(str) == str(g["game_id"])]
             if len(rungs_all) else rungs_all)
        try:
            cards.append(build_card(g, hist, r, _quotes_for(quotes, g), snaps, rec))
        except Exception as e:                    # one bad game must not blank the page
            log.warning("bigcard: skipped %s @ %s (%s)", g["away_team"], g["home_team"], e)

    week = int(board["week"].iloc[0]) if len(board) and "week" in board else None
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    env.filters["pct"] = _pct
    env.filters["american"] = fmt_american
    env.filters["point"] = fmt_point
    env.filters["cents"] = lambda p: "—" if p is None or p != p else f"{p * 100:.0f}¢"
    html = env.get_template("bigcard.html").render(
        title=config.SITE_TITLE, cards=cards, week=week, updated=str(today),
        confs=", ".join(sorted(BIG_CONFS)), cheap_ev=CHEAP_EV,
        fees=config.EXCHANGE_FEES, support_url=config.SUPPORT_URL,
        support_label=config.SUPPORT_LABEL,
        generated=datetime.now(config.ET).strftime("%a %b %-d, %-I:%M %p ET"))
    out_dir = Path(out_dir) if out_dir else config.DOCS / "big"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html)
    log.info("bigcard: %d cards -> %s", len(cards), out_dir / "index.html")
    return out_dir / "index.html"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ladders", type=Path, default=None,
                    help="a ladder CSV to use instead of the newest logged pull")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default docs/cfb/big)")
    a = ap.parse_args(argv)
    build(a.ladders, a.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
