"""Generate the week's college-football picks.

    python -m cfb.predict              # normal run
    python -m cfb.predict --dry-run    # print, write nothing
    python -m cfb.predict --week 3     # force a specific week

The board is every FBS game kicking off in the next BOARD_DAYS days that has a line. Numbers
come from CFBD; if ODDS_API_KEY is set we also pull live prices from a book you can actually
bet, which is what makes EV and Kelly meaningful (otherwise -110 is assumed).
"""
from __future__ import annotations

import argparse
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config
from .features import BASE_FEATURES, MARKET_FEATURES, build, fbs_teams
from .sources import cfbd, odds
from .train import load_models

log = logging.getLogger("cfb.predict")


def american_payout(price) -> float:
    if price is None or price != price:
        return 100 / 110
    p = float(price)
    return p / 100 if p > 0 else 100 / -p


def kelly(p_win, payout) -> float:
    if p_win is None or p_win != p_win or payout <= 0:
        return 0.0
    f = (p_win * (payout + 1) - 1) / payout
    return max(0.0, f) * config.KELLY_FRACTION


def _strength(edge, minimum, thin) -> str:
    e = abs(edge) if edge == edge else 0.0
    if e < minimum:
        return "pass"
    if thin:
        return "thin"
    return "bold" if e >= minimum * config.BOLD_MULT else "play"


def build_board(games: pd.DataFrame, lines: pd.DataFrame, week: int | None = None) -> pd.DataFrame:
    today = config.today_et()
    season = config.season_of(today)
    sched = games[(games["season"] == season) & (~games["completed"])].copy()
    if week is not None:
        sched = sched[sched["week"] == week]
    else:
        sched = sched[(sched["date"] >= today) &
                      (sched["date"] <= today + timedelta(days=config.BOARD_DAYS))]
    if sched.empty:
        return sched
    cols = ["game_id", "spread_home", "spread_open", "total_line", "total_open",
            "provider", "n_providers"]
    have = [c for c in cols if c in lines.columns]
    board = sched.merge(lines[have], on="game_id", how="left")
    return board


def run(dry_run: bool = False, week: int | None = None) -> pd.DataFrame:
    today = config.today_et()
    models, meta = load_models()
    if not models:
        raise SystemExit("no trained models - run python -m cfb.train")

    games = cfbd.update_games()
    lines = cfbd.update_lines()
    cfbd.update_sp()
    cfbd.update_returning()

    board = build_board(games, lines, week)
    if board.empty:
        log.info("no upcoming games on the board")
        return board

    # optional live prices, joined on team names
    live = odds.snapshot(odds.build_matcher(sorted(fbs_teams(games, config.season_of(today)))))
    if not dry_run:
        odds.append_snapshot(live)
    if len(live):
        board = board.merge(
            live[["date", "home_team", "away_team", "live_total", "over_price", "under_price",
                  "total_book", "live_spread_home", "spread_home_price", "spread_away_price",
                  "spread_book"]],
            on=["date", "home_team", "away_team"], how="left")
        # prefer the live number when we have it; it's fresher than CFBD's
        board["total_line"] = board["live_total"].combine_first(board["total_line"])
        board["spread_home"] = board["live_spread_home"].combine_first(board["spread_home"])
    for c in ("over_price", "under_price", "spread_home_price", "spread_away_price",
              "total_book", "spread_book"):
        if c not in board:
            board[c] = np.nan

    priced = board["total_line"].notna() | board["spread_home"].notna()
    if not priced.any():
        log.warning("no lines posted yet for these games")
        return board.iloc[0:0]
    board = board[priced].copy()

    _, up, _ = build(games, board, lines=lines, sp=cfbd.load_sp(),
                     returning=cfbd.load_returning())

    keep = ["game_id", "season", "week", "date", "tip_et", "home_team", "away_team",
            "neutral_site", "conference_game", "total_line", "spread_home", "provider",
            "over_price", "under_price", "spread_home_price", "spread_away_price",
            "total_book", "spread_book", "h_games", "a_games", "exp_total", "exp_margin"]
    out = up[[c for c in keep if c in up.columns]].copy()

    for kind, line_col, sign in (("total", "total_line", 1), ("margin", "spread_home", -1)):
        market_name, base_name = f"{kind}_market", f"{kind}_nomarket"
        has_line = up[line_col].notna()
        use_market = market_name in models and bool(has_line.any())
        name = market_name if use_market else base_name
        if name not in models:
            log.warning("no model for %s - skipping", kind)
            continue
        feats = BASE_FEATURES + (MARKET_FEATURES if use_market else [])
        usable = has_line if use_market else pd.Series(True, index=up.index)

        raw = np.full(len(up), np.nan)
        if usable.any():
            raw[usable.values] = models[name].predict(up.loc[usable, feats])
        if use_market and (~usable).any() and base_name in models:
            raw[(~usable).values] = models[base_name].predict(up.loc[~usable, BASE_FEATURES])

        line = up[line_col] * sign          # spread_home -6.5 -> expected home_margin +6.5
        shrink = meta["shrink"].get(name, config.DEFAULT_SHRINK)
        sigma = meta["sigma"].get(name, 16.0)
        blended = np.where(line.notna(), line.fillna(0) + shrink * (raw - line.fillna(0)), raw)

        out[f"{kind}_pred"] = np.round(blended, 1)
        out[f"{kind}_raw"] = np.round(raw, 1)
        out[f"{kind}_edge"] = np.round(blended - line, 1)
        out[f"{kind}_sigma"] = sigma
        out[f"{kind}_model"] = name
        out[f"{kind}_p_over"] = np.round(norm.cdf((blended - line) / sigma), 4)

    out["total_pick"] = np.where(out["total_edge"] > 0, "Over", "Under")
    out["total_p_win"] = np.where(out["total_edge"] > 0, out["total_p_over"], 1 - out["total_p_over"])
    out["total_price"] = np.where(out["total_edge"] > 0, out["over_price"], out["under_price"])
    out["total_payout"] = out["total_price"].apply(american_payout)
    out["total_ev"] = (out["total_p_win"] * out["total_payout"] - (1 - out["total_p_win"])).round(3)
    out["total_stake"] = [round(kelly(p, b) * config.BANKROLL_UNITS, 2)
                          for p, b in zip(out["total_p_win"], out["total_payout"])]

    took_home = out["margin_edge"] > 0
    out["spread_side"] = np.where(took_home, out["home_team"], out["away_team"])
    out["spread_number"] = np.where(took_home, out["spread_home"], -out["spread_home"])
    out["spread_pick"] = out["spread_side"] + " " + out["spread_number"].map(
        lambda x: f"{x:+.1f}" if x == x else "")
    out["spread_p_win"] = np.where(took_home, out["margin_p_over"], 1 - out["margin_p_over"])
    out["spread_price"] = np.where(took_home, out["spread_home_price"], out["spread_away_price"])
    out["spread_payout"] = out["spread_price"].apply(american_payout)
    out["spread_ev"] = (out["spread_p_win"] * out["spread_payout"] - (1 - out["spread_p_win"])).round(3)
    out["spread_stake"] = [round(kelly(p, b) * config.BANKROLL_UNITS, 2)
                           for p, b in zip(out["spread_p_win"], out["spread_payout"])]

    thin = (out["h_games"] < config.MIN_GAMES) | (out["a_games"] < config.MIN_GAMES)
    out["thin_data"] = thin
    out["total_strength"] = [_strength(e, config.TOTAL_EDGE_MIN, t)
                             for e, t in zip(out["total_edge"], thin)]
    out["spread_strength"] = [_strength(e, config.SPREAD_EDGE_MIN, t)
                              for e, t in zip(out["margin_edge"], thin)]
    out["prediction_date"] = str(today)
    out = out.sort_values(["date", "tip_et"]).reset_index(drop=True)

    log.info("week %s: %d games | totals %s | spreads %s",
             out["week"].iloc[0] if len(out) else "-", len(out),
             out.total_strength.value_counts().to_dict(),
             out.spread_strength.value_counts().to_dict())

    if dry_run:
        cols = ["tip_et", "away_team", "home_team", "total_line", "total_pred", "total_edge",
                "total_strength", "spread_home", "spread_pick", "margin_edge", "spread_strength"]
        print(out[[c for c in cols if c in out]].to_string(index=False))
        return out

    config.ensure_dirs()
    if config.PICKS.exists():
        old = pd.read_csv(config.PICKS, dtype={"game_id": str})
        old = old[~old["game_id"].isin(out["game_id"])]
        pd.concat([old, out], ignore_index=True).to_csv(config.PICKS, index=False)
    else:
        out.to_csv(config.PICKS, index=False)
    log.info("wrote %d picks to %s", len(out), config.PICKS)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--week", type=int, default=None)
    a = ap.parse_args(argv)
    run(dry_run=a.dry_run, week=a.week)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
