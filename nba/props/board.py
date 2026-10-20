"""Price every Kalshi NBA player-prop rung for today's and tomorrow's games.

    python -m nba.props.board              # fetch Kalshi, project, write board.csv + ladder log
    python -m nba.props.board --dry-run    # print, write nothing

Nothing here is staked. Each rung gets the model's probability, the all-in cost of each side at
Kalshi (ask plus the taker fee), and a flag when one side is cheap by more than
``DEGEN_NBA_PROPS_MIN_EDGE`` *and* the probability sits where the distribution has data behind
it. Every rung of every pull is logged, so the flag can be graded against what happened.

**The published number leans on Kalshi**, as the NFL props board's does and for its reason: the
model's ladders are flatter than the market's at the tails, and a quarter of a liquid market
flagged as "value" is a shape error, not a quarter of a market mispriced. So each rung's
published probability is Kalshi's mid moved ``DEGEN_NBA_PROPS_SHRINK`` (0.25) of the way toward
the model, in log-odds; the card still shows the model's own number.

**A player who sits loses the over.** Kalshi settles "records N+", so a scratch records nothing.
The model's distribution is for a player who plays, so the board multiplies by the chance he
does, from the day's injury report on the game board's own scale (``MISS_PROB``): Out never
plays, Doubtful a quarter of the time, Questionable half, Day-To-Day 60%, Probable 90%, and a
player not on the report 98%.

**Who else is out moves the projection too.** The same report sets each core teammate's chance
of sitting, and his expected points, rebounds, assists and usage go into the features as
missing production - the board's version of what the training rows learned from who played.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .. import config
from ..sources import espn, hoopr, kalshi
from . import model
from .features import FEATURES, build, eligible

log = logging.getLogger("nba.props.board")

P_PLAY_UNLISTED = 0.98
MARKET_LABEL = {"points": "Points", "rebounds": "Rebounds", "assists": "Assists",
                "threes": "Threes"}
BOARD_COLS = ["game_id", "date", "tip_utc", "tip_et", "away_team", "home_team", "team",
              "opponent", "athlete_id", "player_name", "position", "market", "k", "ticker",
              "proj_mean", "proj_median", "proj_avg", "p_cond", "p_play", "injury", "p_model",
              "mid", "p_yes", "yes_bid", "yes_ask", "quote_spread", "ask_size", "volume",
              "tradeable", "cost_yes", "cost_no", "edge_yes", "edge_no", "gap", "flag", "match",
              "pulled_at"]


def p_play(status) -> float:
    """The chance a player on the report plays, on the game board's scale."""
    if not isinstance(status, str) or not status:
        return P_PLAY_UNLISTED
    return round(1.0 - config.MISS_PROB.get(status, 0.5), 3)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def publish(p_model, mid, shrink: float | None = None):
    """Kalshi's mid moved `shrink` of the way toward the model, in log-odds. With no quote to
    anchor to, the model's own number stands."""
    w = config.PROPS_SHRINK if shrink is None else shrink
    p_model, mid = np.asarray(p_model, float), np.asarray(mid, float)
    out = 1 / (1 + np.exp(-(_logit(mid) + w * (_logit(p_model) - _logit(mid)))))
    return np.where(np.isfinite(mid) & (mid > 0) & (mid < 1), out, p_model)


def upcoming_games(games: pd.DataFrame, now=None) -> pd.DataFrame:
    """Today's and tomorrow's games that have not tipped."""
    today = config.today_et()
    g = games[~games["completed"].astype(bool)].copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    g = g[(g["date"] >= today) & (g["date"] <= today + timedelta(days=config.BOARD_DAYS))]
    tip = pd.to_datetime(g["tip_utc"], utc=True, errors="coerce")
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    return g[tip.isna() | (tip > now)]


def board_context() -> dict:
    """{game_id: {"margin", "total"}} from the game board's published numbers - the line where
    there is one, moved by the fitted shrink - so props and games read the same game."""
    if not config.PICKS.exists():
        return {}
    p = pd.read_csv(config.PICKS, dtype={"game_id": str}, low_memory=False)
    p = p.drop_duplicates("game_id", keep="last")
    out = {}
    for r in p.itertuples(index=False):
        m, t = getattr(r, "pub_margin", np.nan), getattr(r, "pub_total", np.nan)
        out[str(r.game_id)] = {"margin": float(m) if m == m else np.nan,
                               "total": float(t) if t == t else np.nan}
    return out


def news_from(report: pd.DataFrame) -> tuple[dict, dict]:
    """({team: {athlete_id: chance he sits}}, {athlete_id: status}) from the injury report."""
    by_team, status = {}, {}
    for r in report.itertuples(index=False):
        pid = int(r.athlete_id)
        by_team.setdefault(r.team, {})[pid] = config.MISS_PROB.get(r.status, 0.5)
        status[pid] = r.status
    return by_team, status


def project(rows: pd.DataFrame, models: dict, dists: dict) -> pd.DataFrame:
    """Projection per (game, player, market) for the board's feature rows."""
    out = []
    for market, m in models.items():
        cand = rows[[eligible(r, market) for r in rows[["n_eff", "ew_min", "ew_fg3a"]]
                     .to_dict("records")]] if len(rows) else rows
        if cand.empty:
            continue
        mu = model._predict(m, cand[FEATURES])
        d = dists.get(market)
        med = [d.quantile(float(v), 0.5, pos) if d else np.nan for v, pos in zip(mu, cand["position"])]
        avg = [round(d.mean(float(v), pos), 1) if d else float(v) for v, pos in zip(mu, cand["position"])]
        out.append(pd.DataFrame({"game_id": cand["game_id"].values,
                                 "athlete_id": cand["athlete_id"].values,
                                 "position": cand["position"].values,
                                 "opponent": cand["opponent"].values, "market": market,
                                 "proj_mean": np.round(mu, 2), "proj_median": med,
                                 "proj_avg": avg}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def price(rungs: pd.DataFrame, proj: pd.DataFrame, dists: dict) -> pd.DataFrame:
    """Join each rung to its player's projection and price both sides.

    Three probabilities per rung, kept apart on purpose: ``p_model`` (the model, times the
    chance he plays - shown on the card), ``mid`` (Kalshi's), and ``p_yes`` (the published one:
    the mid moved PROPS_SHRINK toward the model - the flag, EV and fair price come from it).
    """
    if rungs.empty or proj.empty:
        return pd.DataFrame(columns=BOARD_COLS)
    r = rungs.merge(proj, on=["game_id", "athlete_id", "market"], how="inner")
    p_cond = []
    for row in r.itertuples():
        d = dists.get(row.market)
        p_cond.append(d.p_at_least(row.proj_mean, row.k, row.position) if d else np.nan)
    r["p_cond"] = np.round(p_cond, 4)
    r["p_model"] = np.round(r["p_cond"] * r["p_play"], 4)
    r["mid"] = np.round((r["yes_bid"] + r["yes_ask"]) / 2, 4)
    r["p_yes"] = np.round(publish(r["p_model"], r["mid"]), 4)
    r = r[r["p_model"].notna()].copy()
    r["cost_yes"] = r["yes_ask"].map(kalshi.cost)
    r["cost_no"] = (1 - r["yes_bid"]).map(kalshi.cost)
    r["edge_yes"] = np.round(r["p_yes"] / r["cost_yes"] - 1, 4)
    r["edge_no"] = np.round((1 - r["p_yes"]) / r["cost_no"] - 1, 4)
    r["gap"] = np.round(r["p_model"] - r["mid"], 4)
    band = r["p_yes"].between(config.PROPS_PROB_MIN, config.PROPS_PROB_MAX) & \
        (r["gap"].abs() <= config.PROPS_MAX_GAP)
    tradeable = r["tradeable"].astype(bool)
    yes = band & tradeable & (r["edge_yes"] >= config.PROPS_MIN_EDGE)
    no = band & tradeable & (r["edge_no"] >= config.PROPS_MIN_EDGE)
    r["flag"] = np.select([yes, no], ["yes", "no"], default="")
    for c in BOARD_COLS:
        if c not in r:
            r[c] = np.nan
    return r[BOARD_COLS]


def roster() -> pd.DataFrame:
    """Today's rosters with jersey numbers; yesterday's saved copy if hoopR does not answer."""
    t = hoopr.roster_table()
    if len(t):
        config.PROPS.mkdir(parents=True, exist_ok=True)
        t.to_csv(config.PROPS_ROSTER, index=False)
        return t
    if config.PROPS_ROSTER.exists():
        log.warning("rosters unavailable - matching on the last saved copy")
        return pd.read_csv(config.PROPS_ROSTER)
    return pd.DataFrame(columns=hoopr.ROSTER_COLS)


def run(dry_run: bool = False) -> pd.DataFrame:
    models, dists, _ = model.load()
    games = hoopr.load_games()
    games["game_id"] = games["game_id"].astype(str)
    up = upcoming_games(games)
    if up.empty:
        log.info("no NBA games ahead")
        return pd.DataFrame(columns=BOARD_COLS)
    rungs = kalshi.fetch()
    if rungs.empty:
        log.warning("no Kalshi prop markets returned - the board will be empty")
        return pd.DataFrame(columns=BOARD_COLS)
    rungs = kalshi.attach_games(rungs, up).dropna(subset=["game_id"])
    ros = roster()
    rungs = kalshi.match_players(rungs, ros)
    matched = rungs.dropna(subset=["athlete_id"]).copy()
    matched["athlete_id"] = matched["athlete_id"].astype(int)
    log.info("kalshi props: %d rungs on %d games, %d players matched (%d by jersey)",
             len(rungs), rungs["game_id"].nunique(), matched["athlete_id"].nunique(),
             matched.loc[matched["match"] == "jersey", "athlete_id"].nunique())
    if matched.empty:
        return pd.DataFrame(columns=BOARD_COLS)

    fresh = espn.injuries(write=not dry_run)
    report = espn.latest_injuries(fresh)
    news, status = news_from(report)
    wanted = matched[["game_id", "athlete_id", "team"]].drop_duplicates(["game_id", "athlete_id"])
    first = config.PROPS_FIRST_SEASON - 2
    players = hoopr.load_players(seasons=range(first, config.season_of(config.today_et()) + 1))
    _, rows, _ = build(players, games[games["season"] >= first], board_context(),
                       hoopr.load_positions(), upcoming=wanted, news=news)
    proj = project(rows, models, dists)
    if proj.empty:
        log.warning("no listed player has the history to project")
        return pd.DataFrame(columns=BOARD_COLS)
    proj["injury"] = proj["athlete_id"].map(status).fillna("")
    proj["p_play"] = proj["injury"].map(p_play)
    board = price(matched, proj, dists)
    names = dict(zip(ros["athlete_id"], ros["name"])) if len(ros) else {}
    names = {**hoopr.load_names(), **names}
    g = up.set_index("game_id")
    board["player_name"] = board["athlete_id"].map(names)
    for c in ("tip_utc", "away_team", "home_team", "date"):
        board[c] = board["game_id"].map(g[c])
    tip = pd.to_datetime(board["tip_utc"], utc=True, errors="coerce")
    board["tip_et"] = [t.tz_convert(config.ET).strftime("%a %b %-d, %-I:%M %p") if t == t else ""
                       for t in tip]
    log.info("props board: %d rungs, %d players, %d flagged (unstaked)", len(board),
             board["athlete_id"].nunique(), int((board["flag"] != "").sum()))
    if dry_run:
        cols = ["player_name", "market", "k", "proj_mean", "p_model", "mid", "p_yes", "edge_yes",
                "edge_no", "flag"]
        print(board.sort_values(["game_id", "player_name", "market", "k"])[cols].to_string(index=False))
        return board
    config.PROPS.mkdir(parents=True, exist_ok=True)
    board.to_csv(config.PROPS_BOARD, index=False)
    log_rungs(board)
    return board


def log_rungs(board: pd.DataFrame) -> None:
    """Append this pull to today's ladder log. Never fatal - the board is already written."""
    if board.empty:
        return
    try:
        config.PROPS_LADDERS.mkdir(parents=True, exist_ok=True)
        path = config.PROPS_LADDERS / f"{config.today_et()}.csv.gz"
        cols = ["pulled_at", "game_id", "athlete_id", "player_name", "market", "k", "ticker",
                "yes_bid", "yes_ask", "ask_size", "volume", "p_model", "p_yes", "p_play",
                "proj_mean", "gap", "flag", "tip_utc"]
        new = board[cols]
        if path.exists():
            new = pd.concat([pd.read_csv(path, dtype={"game_id": str}), new], ignore_index=True)
        new.to_csv(path, index=False, compression="gzip")
    except Exception as e:                      # pragma: no cover - logging only
        log.warning("could not log prop rungs: %s", e)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    run(dry_run=a.dry_run)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
