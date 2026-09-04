"""Generate today's picks.

    python -m ncaab.predict              # normal daily run
    python -m ncaab.predict --dry-run    # print, write nothing

For every game we produce, for both the total and the spread:
    prediction  -> shrunk toward the market by the fitted `shrink`
    edge        -> prediction minus the line
    probability -> from the fitted residual sigma (normal approximation)
    EV          -> against the book's de-vigged price
    stake       -> fractional Kelly in units, capped

A pick is only published when the edge clears the minimum AND both teams have enough games.
"""
from __future__ import annotations

import argparse
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config
from .features import BASE_FEATURES, MARKET_FEATURES, build, known_teams
from .sources import ncaa, odds, torvik
from .team_names import report_unmapped
from .train import load_models

log = logging.getLogger("ncaab.predict")


def american_payout(price) -> float:
    """Profit per 1 unit staked."""
    if price is None or price != price:
        return 100 / 110  # assume -110
    p = float(price)
    return p / 100 if p > 0 else 100 / -p


def kelly(p_win: float, payout: float) -> float:
    if p_win is None or p_win != p_win:
        return 0.0
    b = payout
    f = (p_win * (b + 1) - 1) / b
    return max(0.0, f) * config.KELLY_FRACTION


def run(dry_run: bool = False) -> pd.DataFrame:
    today = config.today_et()
    models, meta = load_models()
    if not models:
        raise SystemExit("no trained models - run python -m ncaab.train")

    snap = odds.snapshot()
    if snap.empty:
        log.warning("no lines returned")
        return snap
    if not dry_run:
        odds.append_snapshot(snap)          # keep the full history no matter what we pick
    board = snap[snap["date"] <= today + timedelta(days=1)].copy()
    if board.empty:
        log.info("no games in the next 36h")
        return board

    games = ncaa.update(recheck_days=3)
    torvik.update()
    report_unmapped(log)

    # judge "known" against the season each game belongs to, not today's calendar season
    seasons = {config.season_of(d) for d in board["date"]}
    known = set().union(*(known_teams(games, s) for s in seasons)) if seasons else set()
    unknown = board[~board["home_team"].isin(known) | ~board["away_team"].isin(known)]
    if len(unknown):
        cols = ["home_team", "away_team"]
        log.warning("%d games skipped, team not in game history:\n%s", len(unknown),
                    unknown[cols].to_string(index=False))
    board = board.drop(unknown.index)
    if board.empty:
        return board

    _, up, _ = build(games, board, torvik_cache=torvik.load(), lines=odds.line_history())

    out = up[["event_id", "date", "tip_et", "home_team", "away_team",
              "total_line", "over_price", "under_price", "total_book",
              "spread_home", "spread_home_price", "spread_away_price", "spread_book",
              "p_over", "p_under", "p_home_cover", "p_away_cover",
              "h_games", "a_games", "exp_total", "exp_margin"]].copy()

    for kind, line_col, sign in (("total", "total_line", 1), ("margin", "spread_home", -1)):
        market_name, base_name = f"{kind}_market", f"{kind}_nomarket"
        has_line = up[line_col].notna()
        use_market = market_name in models and bool(has_line.any())
        name = market_name if use_market else base_name
        if name not in models:
            log.warning("no model available for %s - skipping", kind)
            continue
        feats = BASE_FEATURES + (MARKET_FEATURES if use_market else [])
        usable = has_line if use_market else pd.Series(True, index=up.index)

        raw = np.full(len(up), np.nan)
        if usable.any():
            raw[usable.values] = models[name].predict(up.loc[usable, feats])
        # games without a line still get a market-free prediction if that model exists
        if use_market and (~usable).any() and base_name in models:
            raw[(~usable).values] = models[base_name].predict(up.loc[~usable, BASE_FEATURES])

        line = up[line_col] * sign  # spread_home -6.5 -> market expects home_margin +6.5
        shrink = meta["shrink"].get(name, config.DEFAULT_SHRINK)
        sigma = meta["sigma"].get(name, 11.0)
        blended = np.where(line.notna(), line.fillna(0) + shrink * (raw - line.fillna(0)), raw)

        out[f"{kind}_pred"] = np.round(blended, 1)
        out[f"{kind}_raw"] = np.round(raw, 1)
        out[f"{kind}_edge"] = np.round(blended - line, 1)
        out[f"{kind}_sigma"] = sigma
        out[f"{kind}_model"] = name
        z = (blended - line) / sigma
        out[f"{kind}_p_over"] = np.round(norm.cdf(z), 4)

    # --- totals pick ---------------------------------------------------------------
    out["total_pick"] = np.where(out["total_edge"] > 0, "Over", "Under")
    out["total_p_win"] = np.where(out["total_edge"] > 0, out["total_p_over"], 1 - out["total_p_over"])
    out["total_price"] = np.where(out["total_edge"] > 0, out["over_price"], out["under_price"])
    out["total_payout"] = out["total_price"].apply(american_payout)
    out["total_ev"] = (out["total_p_win"] * out["total_payout"] - (1 - out["total_p_win"])).round(3)
    out["total_stake"] = [round(kelly(p, b) * config.BANKROLL_UNITS, 2)
                          for p, b in zip(out["total_p_win"], out["total_payout"])]

    # --- spread pick ---------------------------------------------------------------
    # margin_edge > 0 means we think home beats its number
    out["spread_pick"] = np.where(out["margin_edge"] > 0,
                                  out["home_team"] + " " + out["spread_home"].map(_fmt),
                                  out["away_team"] + " " + (-out["spread_home"]).map(_fmt))
    out["spread_p_win"] = np.where(out["margin_edge"] > 0, out["margin_p_over"], 1 - out["margin_p_over"])
    out["spread_price"] = np.where(out["margin_edge"] > 0, out["spread_home_price"], out["spread_away_price"])
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
    out = out.sort_values("tip_et").reset_index(drop=True)

    played = out[(out.total_strength != "pass") | (out.spread_strength != "pass")]
    log.info("%d games, %d playable (totals %s, spreads %s)", len(out), len(played),
             out.total_strength.value_counts().to_dict(), out.spread_strength.value_counts().to_dict())

    if dry_run:
        cols = ["tip_et", "home_team", "away_team", "total_line", "total_pred", "total_edge",
                "total_strength", "spread_home", "margin_edge", "spread_pick", "spread_strength"]
        print(out[cols].to_string(index=False))
        return out

    config.ensure_dirs()
    if config.PICKS.exists():
        old = pd.read_csv(config.PICKS)
        old = old[old["prediction_date"] != str(today)]
        pd.concat([old, out], ignore_index=True).to_csv(config.PICKS, index=False)
    else:
        out.to_csv(config.PICKS, index=False)
    log.info("wrote %d rows to %s", len(out), config.PICKS)
    return out


def _fmt(x) -> str:
    if x != x:
        return ""
    return f"{x:+.1f}"


def _strength(edge, minimum, thin) -> str:
    e = abs(edge) if edge == edge else 0
    if e < minimum:
        return "pass"
    if thin:
        return "thin"
    return "bold" if e >= minimum * config.BOLD_MULT else "play"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    run(dry_run=ap.parse_args(argv).dry_run)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
