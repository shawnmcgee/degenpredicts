"""Feature construction for the NHL, replayed in date order.

A feature on game N only ever sees games 1..N-1. That guarantee is what the project rests on,
and it is verified in CI by rebuilding from a truncated schedule and asserting identical rows.
It is also exactly what the uploaded predecessor of this pipeline got wrong: its "last five
games" averages were taken AFTER appending the game being predicted, so the score leaked into
its own features. Its own evaluation reported a 1.58-goal error on totals; on leak-free
features the same model managed 2.14, worse than predicting the league average every night.

**The targets are goals scored against a goalie in regulation**, one per side. Empty-net goals,
overtime and the shootout are added back by :mod:`nhl.scoreline`, which is where they belong:
they are driven by the score late in the game, not by how good either team is.

Each game becomes two "side" rows - the home side's scoring and the away side's - with the same
features seen from each side. That is what lets one Poisson model learn home ice, rest and
travel without a separate model per venue.

The features, in the order they earn their keep:

* **Ratings** - expected shots, shooting percentage (with the expected opposing goalie) and the
  goals-only rating, in logs because scoring is multiplicative.
* **The expected goalie** - see :mod:`nhl.ratings`. The confidence of the top pick is a feature
  too: a coin-flip tandem is worth knowing about.
* **Back-to-backs, rest and three-in-four** - the schedule effects hockey is known for, and the
  largest non-rating coefficients the models fit. A team on the second night of a back-to-back
  concedes more (tired skaters, usually its backup in net).
* **Travel and time zones**, measured from the arena each team actually played in that season.
* **The market** - for the market-aware model, the market's implied goals for each side, as an
  OFFSET: the model learns how the market errs, rather than relearning hockey from scratch.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .ratings import RatingBook, RatingConfig
from .teams import haversine_km, venue

log = logging.getLogger(__name__)

REST_CAP = 7
EARLY_GAMES = 10

# Side features for the model used when there is no market number yet.
BASE_FEATURES = ["log_lam", "log_L", "log_S", "log_p", "opp_goalie", "own_goalie_conf",
                 "is_home", "playoff", "own_b2b", "opp_b2b", "own_rest", "opp_rest",
                 "own_km", "opp_km", "own_tz_abs", "opp_tz_abs", "own_3in4", "opp_3in4", "early"]
# Side features for the market-aware model. The market's own number is not among them: it is
# the model's OFFSET (log_mkt), so these explain only where the market goes wrong.
SCHEDULE = ["opp_b2b", "own_b2b", "own_rest", "opp_rest", "own_3in4", "opp_3in4", "playoff",
            "early", "is_home", "own_km", "own_tz_abs"]
MARKET_FEATURES = ["dev_lam", "dev_L", "log_S", "opp_goalie"] + SCHEDULE


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if f == f else np.nan


def schedule_features(games: pd.DataFrame, today=None) -> pd.DataFrame:
    """Rest, back-to-backs, three-in-four, travel and time-zone shift for every game.

    Built over the schedule, not just completed games, so tomorrow's game knows tonight's is a
    back-to-back. Past games that were never completed (postponed, cancelled, the "if necessary"
    playoff games) are skipped - they never happened, so they cannot tire anyone.
    """
    today = today or config.today_et()
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    done = g["completed"].astype(bool)
    g = g[done | (g["date"] >= today)].sort_values(["date", "game_id"])
    last: dict[str, tuple] = {}
    rows = []
    for r in g.itertuples(index=False):
        here = venue(r.home_team, int(r.season))
        out = {"game_id": r.game_id}
        for side, team in (("h", r.home_team), ("a", r.away_team)):
            prev = last.get(team)
            if prev is None or (r.date - prev[0]).days > 30:
                out.update({f"{side}_rest": float(REST_CAP), f"{side}_b2b": 0, f"{side}_3in4": 0,
                            f"{side}_km": 0.0, f"{side}_tz": 0.0})
                continue
            rest = (r.date - prev[0]).days
            out[f"{side}_rest"] = float(min(rest, REST_CAP))
            out[f"{side}_b2b"] = int(rest == 1)
            out[f"{side}_3in4"] = int(sum(1 for d in prev[2] if (r.date - d).days <= 3) >= 2)
            out[f"{side}_km"] = haversine_km(prev[1], here) if prev[1] and here else 0.0
            out[f"{side}_tz"] = float(here[2] - prev[1][2]) if prev[1] and here else 0.0
        rows.append(out)
        for team in (r.home_team, r.away_team):
            prev = last.get(team)
            recent = [d for d in (prev[2] if prev else []) if (r.date - d).days <= 6] + [r.date]
            last[team] = (r.date, here, recent)
    return pd.DataFrame(rows)


def _row(book: RatingBook, r, sched: dict, pending_today: set | None = None) -> dict:
    h, a = r.home_team, r.away_team
    s = sched.get(r.game_id, {})
    h_b2b, a_b2b = bool(s.get("h_b2b", 0)), bool(s.get("a_b2b", 0))
    pend = pending_today or set()
    g_h, probs_h = book.expected_goalie(h, h_b2b, pending_today=h in pend)
    g_a, probs_a = book.expected_goalie(a, a_b2b, pending_today=a in pend)
    e = book.expect(h, a, g_vs_home=g_a, g_vs_away=g_h)
    top_h = max(probs_h, key=probs_h.get) if probs_h else None
    top_a = max(probs_a, key=probs_a.get) if probs_a else None
    return {
        "S_h": e["S_h"], "S_a": e["S_a"], "p_h": e["p_h"], "p_a": e["p_a"],
        "lam_h": e["lam_h"], "lam_a": e["lam_a"], "L_h": e["L_h"], "L_a": e["L_a"],
        "g_h_exp": g_h, "g_a_exp": g_a,
        "g_h_conf": probs_h[top_h] if top_h is not None else 0.0,
        "g_a_conf": probs_a[top_a] if top_a is not None else 0.0,
        "g_h_top": book.goalie_name.get(top_h, "") if top_h is not None else "",
        "g_a_top": book.goalie_name.get(top_a, "") if top_a is not None else "",
        "h_games": book.games[h], "a_games": book.games[a],
        **{k: s.get(k, np.nan) for k in ("h_rest", "a_rest", "h_b2b", "a_b2b", "h_3in4",
                                          "a_3in4", "h_km", "a_km", "h_tz", "a_tz")},
    }


def goalie_goals(g: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Goals each side scored against a goalie in regulation.

    The goalie logs record every goal a goalie allowed, which includes an overtime winner and
    excludes empty-net and shootout goals. So regulation goals-on-goalie are the opponents'
    goalie goals-against, less one if that side won in overtime.

    A game with no goalie log falls back to regulation goals, empty-netters included. That is a
    slight overcount, but a NaN here would propagate through every rating it touched and
    silently poison the whole replay from that game on.
    """
    dec = g["decided_in"].astype(str)
    ot, extra = dec == "OT", dec.isin(["OT", "SO"])
    hg = pd.to_numeric(g["home_goals"], errors="coerce")
    ag = pd.to_numeric(g["away_goals"], errors="coerce")
    home_won = hg > ag
    reg_h = hg - (extra & home_won).astype(int)
    reg_a = ag - (extra & ~home_won).astype(int)
    h = pd.to_numeric(g.get("away_goalie_ga"), errors="coerce") - (ot & home_won).astype(int)
    a = pd.to_numeric(g.get("home_goalie_ga"), errors="coerce") - (ot & ~home_won).astype(int)
    h = h.where(h.notna(), reg_h).clip(lower=0)
    a = a.where(a.notna(), reg_a).clip(lower=0)
    return h.fillna(0.0), a.fillna(0.0)


def build(games: pd.DataFrame, lines: pd.DataFrame | None = None,
          upcoming: pd.DataFrame | None = None, table=None,
          cfg: RatingConfig | None = None, first_train_season: int | None = None,
          today=None) -> tuple[pd.DataFrame, pd.DataFrame, RatingBook]:
    """Replay every completed game; emit one pre-game row per game from the training window.

    Seasons before `first_train_season` are warm-up: they advance the ratings and the goalie
    tracker but emit no rows. ``upcoming`` rows (the board) are priced off the final state.
    """
    first = config.FIRST_SEASON if first_train_season is None else first_train_season
    today = today or config.today_et()
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    g["completed"] = g["completed"].astype(bool)
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    sched_df = schedule_features(g, today=today)
    sched = {r["game_id"]: r for r in sched_df.to_dict("records")}

    done = g[g["completed"]].copy()
    done["h_gg"], done["a_gg"] = goalie_goals(done)

    book = RatingBook(cfg)
    rows = []
    for r in done.itertuples(index=False):
        book.rollover(int(r.season))
        for gid, name in ((r.home_goalie_id, getattr(r, "home_goalie", "")),
                          (r.away_goalie_id, getattr(r, "away_goalie", ""))):
            if gid == gid and gid is not None and name == name and name:
                book.goalie_name[int(gid)] = str(name)
        if int(r.season) >= first:
            row = _row(book, r, sched)
            row.update({"game_id": r.game_id, "season": int(r.season), "date": r.date,
                        "game_type": r.game_type, "home_team": r.home_team,
                        "away_team": r.away_team, "home_goals": r.home_goals,
                        "away_goals": r.away_goals, "decided_in": r.decided_in,
                        "h_gg": r.h_gg, "a_gg": r.a_gg})
            rows.append(row)
        # State advances only AFTER the row is recorded. This ordering IS the leak-free
        # guarantee, and test_leak_free rebuilds from a truncated schedule to prove it.
        book.update(r.home_team, r.away_team, r.home_goalie_id, r.away_goalie_id,
                    r.home_shots, r.away_shots, r.h_gg, r.a_gg, r.date)
    train = pd.DataFrame(rows)

    up_rows = []
    if upcoming is not None and len(upcoming):
        up = upcoming.copy()
        up["date"] = pd.to_datetime(up["date"]).dt.date
        up = up.sort_values(["date", "game_id"])
        # a team with an earlier game still to play has an unknown starter for that one
        first_date = {}
        for r in g[~g["completed"] & (g["date"] >= today)].itertuples(index=False):
            for t in (r.home_team, r.away_team):
                first_date.setdefault(t, r.date)
        season_now = int(up["season"].max()) if "season" in up else config.season_of(today)
        book.rollover(season_now)
        for r in up.itertuples(index=False):
            pend = {t for t in (r.home_team, r.away_team) if first_date.get(t, r.date) < r.date}
            row = _row(book, r, sched, pending_today=pend)
            row.update({k: v for k, v in r._asdict().items() if k not in row})
            up_rows.append(row)
    board = pd.DataFrame(up_rows)

    for frame in (train, board):
        if len(frame):
            frame["playoff"] = (frame["game_type"].astype(str) == "P").astype(int)
    if lines is not None and len(lines) and len(train):
        train = attach_market(train, lines, table)
    if len(board):
        board = market_view(board, table)
    return train, board, book


LINE_COLS = ["p_home", "total_line", "p_over", "pl_home", "home_dec", "away_dec", "over_dec",
             "under_dec", "pl_home_dec", "pl_away_dec", "p_pl_home", "source", "is_closing"]


def attach_market(train: pd.DataFrame, lines: pd.DataFrame, table=None) -> pd.DataFrame:
    keep = ["game_id"] + [c for c in LINE_COLS if c in lines.columns]
    ln = lines[keep].drop_duplicates("game_id", keep="last")
    out = train.merge(ln, on="game_id", how="left")
    return market_view(out, table)


def market_view(df: pd.DataFrame, table=None) -> pd.DataFrame:
    """The market's implied goals-on-goalie for each side, from its moneyline and total.

    Where the total comes without an over price (the SBR closing archive), the price is read
    as the 2023-26 average rather than dropped, and `p_over_assumed` records that it was.
    """
    df = df.copy()
    for c in LINE_COLS:
        if c not in df:
            df[c] = np.nan
    po = pd.to_numeric(df["p_over"], errors="coerce")
    tl = pd.to_numeric(df["total_line"], errors="coerce")
    df["p_over_assumed"] = po.isna() & tl.notna()
    po = po.where(po.notna(), np.where(tl.notna(), config.P_OVER_DEFAULT, np.nan))
    if table is None:
        from .scoreline import Table
        table = Table()
    mh, ma = table.invert(pd.to_numeric(df["p_home"], errors="coerce").values, tl.values,
                          np.asarray(po, float))
    df["m_lh"], df["m_la"] = mh, ma
    return df


def sides(d: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game: the home side's scoring and the away side's."""
    out = []
    for side, opp, is_home in (("h", "a", 1), ("a", "h", 0)):
        x = pd.DataFrame({
            "game_id": d["game_id"].values, "season": d["season"].values, "side": side,
            "log_lam": np.log(d[f"lam_{side}"].astype(float).values),
            "log_L": np.log(d[f"L_{side}"].astype(float).values),
            "log_S": np.log(d[f"S_{side}"].astype(float).values),
            "log_p": np.log(d[f"p_{side}"].astype(float).values),
            "opp_goalie": d[f"g_{opp}_exp"].astype(float).values,
            "own_goalie_conf": d[f"g_{opp}_conf"].astype(float).values,
            "is_home": is_home, "playoff": d["playoff"].astype(int).values,
            "own_b2b": d[f"{side}_b2b"].astype(float).values,
            "opp_b2b": d[f"{opp}_b2b"].astype(float).values,
            "own_rest": d[f"{side}_rest"].astype(float).values,
            "opp_rest": d[f"{opp}_rest"].astype(float).values,
            "own_km": d[f"{side}_km"].astype(float).values / 1000,
            "opp_km": d[f"{opp}_km"].astype(float).values / 1000,
            "own_tz_abs": np.abs(d[f"{side}_tz"].astype(float).values),
            "opp_tz_abs": np.abs(d[f"{opp}_tz"].astype(float).values),
            "own_3in4": d[f"{side}_3in4"].astype(float).values,
            "opp_3in4": d[f"{opp}_3in4"].astype(float).values,
            "early": (np.minimum(d["h_games"].values, d["a_games"].values) < EARLY_GAMES).astype(int),
        })
        m = d[f"m_l{side}"].astype(float).values if f"m_l{side}" in d else np.full(len(d), np.nan)
        x["log_mkt"] = np.log(m)
        x["dev_lam"] = x["log_lam"] - x["log_mkt"]
        x["dev_L"] = x["log_L"] - x["log_mkt"]
        x["y"] = d[f"{side}_gg"].astype(float).values if f"{side}_gg" in d else np.nan
        out.append(x)
    s = pd.concat(out, ignore_index=True)
    for c in BASE_FEATURES + MARKET_FEATURES:
        s[c] = s[c].fillna(0.0) if c not in ("log_lam", "log_L", "log_S", "log_p") else s[c]
    return s


def unstack(sides_df: pd.DataFrame, values: np.ndarray, game_ids) -> tuple[np.ndarray, np.ndarray]:
    """Per-side predictions back to (home, away) arrays in `game_ids` order."""
    v = pd.Series(np.asarray(values, float))
    key = sides_df["game_id"].astype(str).values
    h = pd.Series(v[sides_df["side"].values == "h"].values, index=key[sides_df["side"].values == "h"])
    a = pd.Series(v[sides_df["side"].values == "a"].values, index=key[sides_df["side"].values == "a"])
    ids = pd.Index(pd.Series(game_ids).astype(str))
    return h.reindex(ids).values, a.reindex(ids).values
