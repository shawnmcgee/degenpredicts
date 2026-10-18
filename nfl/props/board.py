"""Price every Kalshi player-prop rung for the upcoming games.

    python -m nfl.props.board              # fetch Kalshi, project, write board.csv + ladder log
    python -m nfl.props.board --dry-run    # print, write nothing

Nothing here is staked. Each rung gets the model's probability, the all-in cost of each side
at Kalshi (ask plus the taker fee), and a flag when one side is cheap by more than
``DEGEN_PROPS_MIN_EDGE`` *and* the probability sits inside the band where the distribution has
data behind it. Every rung of every pull is logged, so the flag can be graded against what
happened - that, not the backtest, is what will say whether any of this is an edge.

**A player who sits loses the over.** Kalshi's rules read "If <player> records 65+ rushing
yards", so a player who does not play records nothing and the YES side loses. The model's
distribution is for a player who plays, so the board multiplies by the chance he plays: the
injury report's Out is 0, Doubtful 0.25, Questionable 0.80, no designation 0.98. Those are
round numbers, not fitted ones, and `injury` is shown on the card so the reader can judge it.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config
from ..sources import kalshi, kalshi_props, nflverse, odds
from . import data, model
from .features import FEATURES, build, eligible

log = logging.getLogger("nfl.props.board")

P_PLAY = {"out": 0.0, "doubtful": 0.25, "questionable": 0.80, "": 0.98}
MARKET_LABEL = {"pass_yds": "Passing yards", "rush_yds": "Rushing yards",
                "rec_yds": "Receiving yards", "receptions": "Receptions",
                "anytime_td": "Anytime TD"}
BOARD_COLS = ["game_id", "date", "kickoff_utc", "tip_et", "away_team", "home_team", "team",
              "opponent_team", "player_id", "player_name", "position", "market", "k", "ticker",
              "proj_mean", "proj_median", "proj_avg", "p_cond", "p_play", "injury", "p_yes",
              "yes_bid", "yes_ask", "quote_spread", "ask_size", "volume", "tradeable",
              "cost_yes", "cost_no", "edge_yes", "edge_no", "gap", "flag", "match", "pulled_at"]


def _cost(price: float) -> float:
    if price != price or not (0 < price < 1):
        return float("nan")
    return price + kalshi.fee(price)


def _upcoming_games(games: pd.DataFrame, now) -> pd.DataFrame:
    today = config.today_et()
    g = games[(~games["completed"].astype(bool))].copy()
    g = g[(g["date"] >= today) & (g["date"] <= today + pd.Timedelta(days=config.BOARD_DAYS))]
    ko = pd.to_datetime(g["kickoff_utc"], errors="coerce", utc=True)
    return g[ko.isna() | (ko > pd.Timestamp(now))]


def _board_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """nflverse's lines, with the game board's live number laid over them where it has one."""
    if not config.PICKS.exists():
        return lines
    picks = pd.read_csv(config.PICKS, dtype={"game_id": str})
    live = picks[["game_id", "spread_home", "total_line"]].dropna(how="all", subset=["spread_home", "total_line"])
    out = lines.set_index("game_id")
    live = live.drop_duplicates("game_id", keep="last").set_index("game_id")
    for c in ("spread_home", "total_line"):
        out.loc[out.index.intersection(live.index), c] = \
            live.loc[out.index.intersection(live.index), c].combine_first(
                out.loc[out.index.intersection(live.index), c])
    return out.reset_index()


def _injury_status(inj: pd.DataFrame, season: int, week: int) -> dict[str, str]:
    if inj is None or inj.empty:
        return {}
    w = inj[(inj["season"] == season) & (inj["week"] == week)]
    st = w.dropna(subset=["gsis_id"]).set_index("gsis_id")["report_status"].fillna("")
    return {k: str(v).strip().lower() for k, v in st.items()}


def price(rungs: pd.DataFrame, proj: pd.DataFrame, dists: dict, meta: dict) -> pd.DataFrame:
    """Join each rung to its player's projection and price both sides."""
    if rungs.empty or proj.empty:
        return pd.DataFrame(columns=BOARD_COLS)
    r = rungs.merge(proj, on=["game_id", "player_id", "market"], how="inner")
    p_cond, med = [], []
    for row in r.itertuples():
        if row.market == "anytime_td":
            p_cond.append(row.proj_mean)
            med.append(np.nan)
            continue
        d = dists.get(row.market)
        raw = d.p_at_least(row.proj_mean, row.k, row.position) if d else np.nan
        # The walk-forward's calibration: the raw distribution runs too extreme against a line
        # that sits where the player's history puts it, which is where Kalshi's ladders sit too.
        cal = (meta["markets"].get(row.market) or {}).get("platt")
        p_cond.append(float(model.platt(raw, cal["a"], cal["b"])) if cal and raw == raw else raw)
        med.append(row.proj_median)
    r["p_cond"] = np.round(p_cond, 4)
    r["p_yes"] = np.round(r["p_cond"] * r["p_play"], 4)
    r["cost_yes"] = r["yes_ask"].map(_cost)
    r["cost_no"] = (1 - r["yes_bid"]).map(_cost)
    r["edge_yes"] = np.round(r["p_yes"] / r["cost_yes"] - 1, 4)
    r["edge_no"] = np.round((1 - r["p_yes"]) / r["cost_no"] - 1, 4)
    r["gap"] = np.round(r["p_yes"] - (r["yes_bid"] + r["yes_ask"]) / 2, 4)
    band = r["p_yes"].between(config.PROPS_PROB_MIN, config.PROPS_PROB_MAX) & \
        (r["gap"].abs() <= config.PROPS_MAX_GAP)
    yes = band & r["tradeable"] & (r["edge_yes"] >= config.PROPS_MIN_EDGE)
    no = band & r["tradeable"] & (r["edge_no"] >= config.PROPS_MIN_EDGE)
    r["flag"] = np.select([yes, no], ["yes", "no"], default="")
    for c in BOARD_COLS:
        if c not in r:
            r[c] = np.nan
    return r[BOARD_COLS]


def project(players: pd.DataFrame, games: pd.DataFrame, lines: pd.DataFrame,
            wanted: pd.DataFrame, models: dict, dists: dict, meta: dict,
            injuries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Projection per (game, player, market) for the players Kalshi lists."""
    if wanted.empty:
        return pd.DataFrame()
    last = players.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
    info = last.set_index("player_id")[["player_name", "position"]]
    gw = games.set_index("game_id")
    up = wanted.drop_duplicates(["game_id", "player_id"]).copy()
    up = up[up["player_id"].isin(info.index)]
    if up.empty:
        return pd.DataFrame()
    up["player_name"] = up["player_id"].map(info["player_name"])
    up["position"] = up["player_id"].map(info["position"]).replace({"FB": "RB"})
    up["opponent_team"] = [gw.loc[g, "away_team"] if t == gw.loc[g, "home_team"] else gw.loc[g, "home_team"]
                           for g, t in zip(up["game_id"], up["team"])]
    _, rows, _ = build(players, games, lines, upcoming=up)
    if rows.empty:
        return pd.DataFrame()
    out = []
    for market, m in models.items():
        cand = rows[[eligible(r, market) or market == "anytime_td" and r["n_eff"] >= 0.5
                     for r in rows[["position", "n_eff", "is_starter_qb", "ew_carries",
                                    "ew_targets", "ew_snap_pct"]].to_dict("records")]]
        if cand.empty:
            continue
        mu = model._predict(m, cand[FEATURES], market)
        if market == "anytime_td":
            table = (meta["markets"].get(market) or {}).get("recalibration") or []
            mu = np.array([model.recalibrate(float(p), table) for p in mu])
        med, avg = [], []
        d = dists.get(market)
        for v, pos in zip(mu, cand["position"]):
            med.append(d.quantile(float(v), 0.5, pos) if d is not None else np.nan)
            avg.append(round(d.mean(float(v), pos), 1) if d is not None else float(v))
        out.append(pd.DataFrame({
            "game_id": cand["game_id"].values, "player_id": cand["player_id"].values,
            "player_name": cand["player_name"].values, "position": cand["position"].values,
            "opponent_team": cand["opponent_team"].values,
            "market": market, "proj_mean": np.round(mu, 3), "proj_median": med,
            "proj_avg": avg}))
    proj = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if proj.empty:
        return proj
    season = config.season_of(config.today_et())
    week_of = gw["week"].to_dict()
    status = {}
    for wk in set(int(week_of.get(g, 0)) for g in proj["game_id"]):
        status.update({(wk, k): v for k, v in _injury_status(injuries, season, wk).items()})
    proj["injury"] = [status.get((int(week_of.get(g, 0)), pid), "")
                      for g, pid in zip(proj["game_id"], proj["player_id"])]
    proj["p_play"] = proj["injury"].map(lambda s: P_PLAY.get(s, P_PLAY[""]))
    return proj


def run(dry_run: bool = False) -> pd.DataFrame:
    models, dists, meta = model.load()
    games = nflverse.load_games()
    games["game_id"] = games["game_id"].astype(str)
    now = datetime.now(timezone.utc)
    up_games = _upcoming_games(games, now)
    if up_games.empty:
        log.info("no upcoming games")
        return pd.DataFrame(columns=BOARD_COLS)

    matcher = odds.build_matcher()
    rungs = kalshi_props.fetch()
    if rungs.empty:
        log.warning("no Kalshi prop markets returned - the board will be empty")
        return pd.DataFrame(columns=BOARD_COLS)
    rungs = kalshi_props.attach_games(rungs, up_games, matcher)
    rungs = rungs.dropna(subset=["game_id"])
    rungs = kalshi_props.match_players(rungs, data.load_roster())
    matched = rungs.dropna(subset=["player_id"])
    log.info("kalshi props: %d rungs on %d games, %d players matched (%d by jersey)",
             len(rungs), rungs["game_id"].nunique(), matched["player_id"].nunique(),
             matched.loc[matched["match"] == "jersey", "player_id"].nunique())

    players = data.load()
    lines = _board_lines(nflverse.load_lines())
    proj = project(players, games, lines, matched[["game_id", "player_id", "team"]],
                   models, dists, meta, data.load_injuries())
    board = price(matched, proj, dists, meta)
    g = up_games.set_index("game_id")
    board["kickoff_utc"] = board["game_id"].map(g["kickoff_utc"])
    board["tip_et"] = board["game_id"].map(g["tip_et"])
    board["away_team"] = board["game_id"].map(g["away_team"])
    board["home_team"] = board["game_id"].map(g["home_team"])
    board["date"] = board["game_id"].map(g["date"])
    log.info("props board: %d rungs, %d players, %d flagged (unstaked)", len(board),
             board["player_id"].nunique(), int((board["flag"] != "").sum()))
    if dry_run:
        cols = ["player_name", "market", "k", "proj_mean", "p_yes", "yes_ask", "edge_yes",
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
        cols = ["pulled_at", "game_id", "player_id", "player_name", "market", "k", "ticker",
                "yes_bid", "yes_ask", "ask_size", "volume", "p_yes", "p_play", "proj_mean", "gap", "flag",
                "kickoff_utc"]
        new = board[cols]
        if path.exists():
            new = pd.concat([pd.read_csv(path, dtype={"game_id": str, "player_id": str}), new],
                            ignore_index=True)
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
