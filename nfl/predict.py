"""Generate the week's NFL picks.

    python -m nfl.predict              # normal run
    python -m nfl.predict --dry-run    # print, write nothing
    python -m nfl.predict --week 3     # force a specific week

The board is every game kicking off in the next BOARD_DAYS days that has a line. Numbers come
from nflverse; if ODDS_API_KEY is set we also pull live prices from a book you can actually
bet, which is what makes EV and Kelly meaningful (otherwise -110 is assumed).

Two deliberate differences from the college version, both downstream of the same fact - the
NFL close is the sharpest number in sports:

* the published number sits much closer to the line, because the fitted shrink is small;
* stakes are eighth-Kelly, not quarter. Kelly assumes you know your edge. Against this market
  you don't, and overestimating it is the expensive direction to be wrong in.
"""
from __future__ import annotations

import argparse
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config
from .features import BASE_FEATURES, MARKET_FEATURES, build, nfl_teams
from .sources import kalshi, nflverse, odds
from .train import load_models

log = logging.getLogger("nfl.predict")


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
    sched = games[(games["season"] == season) & (~games["completed"].astype(bool))].copy()
    if week is not None:
        sched = sched[sched["week"] == week]
    else:
        sched = sched[(sched["date"] >= today) &
                      (sched["date"] <= today + timedelta(days=config.BOARD_DAYS))]
    if sched.empty:
        return sched
    cols = ["game_id", "spread_home", "spread_open", "total_line", "total_open",
            "home_ml", "away_ml", "provider", "n_providers"]
    have = [c for c in cols if c in lines.columns]
    return sched.merge(lines[have], on="game_id", how="left")


def run(dry_run: bool = False, week: int | None = None) -> pd.DataFrame:
    today = config.today_et()
    models, meta = load_models()
    if not models:
        raise SystemExit("no trained models - run python -m nfl.train")

    games = nflverse.update_games()
    lines = nflverse.load_lines()
    # EPA and continuity are READ here, never refreshed. Both are season-static - prior-season
    # EPA is joined from season-1 and cannot change mid-season, and roster continuity is an
    # offseason measure - so refreshing them daily bought nothing and re-downloaded a 7 MB
    # player crosswalk every morning. They are rebuilt by the Tuesday retrain job, which is
    # where deriving training features belongs.
    epa, continuity = nflverse.load_epa(), nflverse.load_continuity()
    if epa.empty or continuity.empty:
        log.warning("EPA (%d rows) or continuity (%d rows) missing - preseason features will "
                    "be empty. Run `python -m nfl.train` to build them.",
                    len(epa), len(continuity))

    board = build_board(games, lines, week)
    if board.empty:
        log.info("no upcoming games on the board")
        return board

    live = odds.snapshot(odds.build_matcher(sorted(nfl_teams(games, config.season_of(today)))))
    if not dry_run:
        odds.append_snapshot(live)
    if len(live):
        board = board.merge(
            live[["date", "home_team", "away_team", "live_total", "over_price", "under_price",
                  "total_book", "live_spread_home", "spread_home_price", "spread_away_price",
                  "spread_book"]],
            on=["date", "home_team", "away_team"], how="left")
        # prefer the live number when we have it; it is fresher than the nflverse snapshot
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

    _, up, _ = build(games, board, lines=lines, epa=epa, continuity=continuity)

    keep = ["game_id", "season", "week", "season_type", "playoff_round", "date", "tip_et",
            "kickoff_utc", "start_time_tbd", "weekday", "home_team", "away_team",
            "home_div", "away_div", "neutral_site", "div_game", "is_indoor", "wind", "temp",
            "home_qb", "away_qb", "h_qb_new", "a_qb_new", "h_rest", "a_rest", "rest_diff",
            "a_travel_km", "a_tz_shift", "total_line", "spread_home", "provider",
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
        sigma = meta["sigma"].get(name, 13.0)
        blended = np.where(line.notna(), line.fillna(0) + shrink * (raw - line.fillna(0)), raw)

        out[f"{kind}_pred"] = np.round(blended, 1)
        out[f"{kind}_raw"] = np.round(raw, 1)
        out[f"{kind}_edge"] = np.round(blended - line, 1)
        # Which side we take is decided from the UNROUNDED difference. `*_edge` is rounded to
        # a tenth for display, and with the NFL's small fitted shrink (~0.2) a raw
        # disagreement under a quarter-point shrinks to under 0.05 and rounds to exactly 0.0.
        # `0.0 > 0` is False, so the rounded column would silently take Under (or the away
        # side) on every near-tie - a real directional bias, against the model's own sign, on
        # precisely the games where the two sides are closest.
        out[f"{kind}_side_val"] = blended - line
        # Selection uses the model's RAW disagreement with the line. The shrunk `edge` is the
        # honest expected difference and is tiny by construction (shrink fits near 0.2 here),
        # so thresholding on it would produce an empty board. Disagreement is what the
        # ats_by_disagreement table in meta.json is bucketed on, so thresholds set from that
        # table apply to the same quantity.
        out[f"{kind}_disagree"] = np.round(raw - line, 1)
        out[f"{kind}_sigma"] = sigma
        out[f"{kind}_model"] = name
        out[f"{kind}_p_over"] = np.round(norm.cdf((blended - line) / sigma), 4)

    took_over = out["total_side_val"] > 0
    out["total_pick"] = np.where(took_over, "Over", "Under")
    out["total_p_win"] = np.where(took_over, out["total_p_over"], 1 - out["total_p_over"])
    out["total_price"] = np.where(took_over, out["over_price"], out["under_price"])
    out["total_payout"] = out["total_price"].apply(american_payout)
    out["total_ev"] = (out["total_p_win"] * out["total_payout"] - (1 - out["total_p_win"])).round(3)
    out["total_stake"] = [round(kelly(p, b) * config.BANKROLL_UNITS, 2)
                          for p, b in zip(out["total_p_win"], out["total_payout"])]

    took_home = out["margin_side_val"] > 0
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

    # The thin-data flag must exist BEFORE the exchange pricing runs: the Kalshi path refuses
    # to publish a pick on a game the model itself considers unplayable.
    thin = (out["h_games"] < config.MIN_GAMES) | (out["a_games"] < config.MIN_GAMES)
    out["thin_data"] = thin

    if "margin_pred" in out:
        sig = out["margin_sigma"].replace(0, np.nan)
        out["p_home_win"] = np.round(norm.cdf(out["margin_pred"] / sig), 4)
        out["p_away_win"] = (1 - out["p_home_win"]).round(4)
        out = _attach_kalshi(out, games)
        out = _price_ladders(out, games)

    out["total_strength"] = [_strength(d, config.TOTAL_EDGE_MIN, t)
                             for d, t in zip(out["total_disagree"], thin)]
    out["spread_strength"] = [_strength(d, config.SPREAD_EDGE_MIN, t)
                              for d, t in zip(out["margin_disagree"], thin)]
    out["prediction_date"] = str(today)
    out = out.sort_values(["date", "tip_et"]).reset_index(drop=True)

    log.info("week %s: %d games | totals %s | spreads %s",
             out["week"].iloc[0] if len(out) else "-", len(out),
             out.total_strength.value_counts().to_dict(),
             out.spread_strength.value_counts().to_dict())

    if dry_run:
        cols = ["tip_et", "away_team", "home_team", "total_line", "total_raw", "total_disagree",
                "total_pick", "total_strength", "spread_home", "margin_raw", "margin_disagree",
                "spread_pick", "spread_strength", "p_home_win", "kalshi_side",
                "kalshi_home_ask", "kalshi_home_ev", "kalshi_tradeable"]
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


def _attach_kalshi(out: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Join live Kalshi moneyline quotes and price our probability against the actual ask.

    We compare against the **ask**, never the mid: a game listed early can quote absurdly wide
    with no volume behind it, and an edge measured off that mid is fiction.
    ``kalshi_tradeable`` folds together quote width, resting size and traded volume.
    """
    for c in ("kalshi_home_ask", "kalshi_away_ask", "kalshi_home_ev", "kalshi_away_ev"):
        out[c] = np.nan
    for c in ("kalshi_side", "kalshi_ticker"):
        out[c] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["kalshi_tradeable"] = False
    out["ml_pick"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    for c in ("ml_ask", "ml_p_home", "ml_ev", "ml_roi", "ml_stake", "ml_model_cents",
              "ml_ref_ask", "ml_ref_prob", "ml_ref_ev"):
        out[c] = np.nan
    out["ml_ref_side"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    try:
        matcher = odds.build_matcher(
            sorted(nfl_teams(games, config.season_of(config.today_et()))))
        board = kalshi.moneyline_board(matcher)
    except Exception as e:                    # a third-party outage must not kill the run
        log.warning("kalshi unavailable (%s) - continuing without exchange prices", e)
        return out
    if board.empty:
        return out

    idx = {(r.date, r.home_team, r.away_team, r.team): r for r in board.itertuples()}
    hits = 0
    for i, g in out.iterrows():
        key = (g["date"], g["home_team"], g["away_team"])
        h = idx.get((*key, g["home_team"]))
        a = idx.get((*key, g["away_team"]))
        if h is None and a is None:
            continue
        hits += 1
        best_ev, best = -np.inf, None
        for side, rec, prob in (("home", h, g.get("p_home_win")),
                                ("away", a, g.get("p_away_win"))):
            if rec is None:
                continue
            out.at[i, f"kalshi_{side}_ask"] = rec.yes_ask
            ev, _ = kalshi.contract_ev(prob, rec.yes_ask)
            if ev == ev:
                out.at[i, f"kalshi_{side}_ev"] = round(ev, 4)
                if ev > best_ev:
                    best_ev, best = ev, (side, rec, prob)
        if best:
            side, rec, prob = best
            team = g[f"{side}_team"]
            ev, roi = kalshi.contract_ev(prob, rec.yes_ask)
            out.at[i, "kalshi_side"] = team
            out.at[i, "kalshi_ticker"] = rec.ticker
            out.at[i, "kalshi_tradeable"] = bool(rec.tradeable)
            out.at[i, "ml_p_home"] = g.get("p_home_win")
            out.at[i, "ml_ref_ask"] = rec.yes_ask
            out.at[i, "ml_ref_prob"] = round(prob, 4) if prob == prob else np.nan
            out.at[i, "ml_ref_ev"] = round(ev, 4) if ev == ev else np.nan
            out.at[i, "ml_ref_side"] = team
            # Only surface a playable moneyline where the book is genuinely tradeable and the
            # edge survives the fee. Everything else stays visible in the raw CSV but off the
            # site, because an edge against an untraded ask is not an edge.
            if (rec.tradeable and ev == ev and ev >= config.KALSHI_MIN_EV
                    and not bool(g.get("thin_data", False))
                    and prob == prob
                    and config.KALSHI_PROB_MIN <= prob <= config.KALSHI_PROB_MAX):
                cost = rec.yes_ask + kalshi.fee(rec.yes_ask)
                payout = (1 - cost) / cost if 0 < cost < 1 else 0.0
                out.at[i, "ml_pick"] = f"{team} to win"
                out.at[i, "ml_ask"] = rec.yes_ask
                out.at[i, "ml_ev"] = round(ev, 4)
                out.at[i, "ml_roi"] = round(roi, 4)
                out.at[i, "ml_stake"] = round(kelly(prob, payout) * config.BANKROLL_UNITS, 2)
                out.at[i, "ml_model_cents"] = int(round(prob * 100))
    log.info("kalshi: matched %d/%d board games", hits, len(out))
    if hits < len(out):
        stuck = sorted(getattr(matcher, "unmatched", set()))
        if stuck:
            log.warning("kalshi names the matcher could not resolve (%d): %s", len(stuck), stuck[:40])
    return out


def _multiplier(ask, prob):
    """What the contract pays per unit risked, and what it *should* pay."""
    if ask is None or ask != ask or ask <= 0:
        return np.nan, np.nan, np.nan
    cost = ask + kalshi.fee(ask)
    pays = 1.0 / cost if cost > 0 else np.nan
    fair = 1.0 / prob if prob and prob == prob and prob > 0 else np.nan
    edge = (pays / fair - 1.0) * 100 if pays == pays and fair == fair else np.nan
    return round(pays, 3), round(fair, 3), round(edge, 1)


def _price_ladders(out: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Price every rung of Kalshi's spread and total ladders against the model distribution.

    Both series expose ``floor_strike`` with ``strike_type: "greater"``, so each contract pays
    when the quantity exceeds the strike:

        total  "Over X points"          -> P(total  > X)  = sf(X, total_pred,  total_sigma)
        spread "TEAM wins by over X"    -> P(margin > X)  if TEAM is home
                                           P(margin < -X) if TEAM is away

    Because it is a ladder, every rung is scored and the best tradeable positive-EV one kept
    rather than assuming a single line. The guards are stricter than the college version's
    for the reason stated in config: picking the best of ~20 noisy estimates returns a
    positive number almost every time even from a model with no edge, and this model's own
    walk-forward says it does not beat the NFL close.
    """
    text_cols = ["kt_pick", "kt_ticker", "ks_pick", "ks_ticker", "ks_ref_side"]
    num_cols = ["kt_strike", "kt_ask", "kt_prob", "kt_ev", "kt_book_gap", "kt_rungs",
                "ks_strike", "ks_ask", "ks_prob", "ks_ev", "ks_book_gap", "ks_rungs",
                "kt_ref_strike", "kt_ref_ask", "kt_ref_prob", "kt_ref_ev", "kt_ref_spread_c",
                "ks_ref_strike", "ks_ref_ask", "ks_ref_prob", "ks_ref_ev", "ks_ref_spread_c"]
    for c in text_cols:
        out[c] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["kt_ref_tradeable"] = False
    out["ks_ref_tradeable"] = False
    for c in num_cols:
        out[c] = np.nan
    out["kalshi_incoherent"] = False

    try:
        matcher = odds.build_matcher(
            sorted(nfl_teams(games, config.season_of(config.today_et()))))
        tot = kalshi.ladder_board("total", matcher)
        spr = kalshi.ladder_board("spread", matcher)
    except Exception as e:
        log.warning("kalshi ladders unavailable (%s)", e)
        return out

    breaks = set()
    for lad in (tot, spr):
        if len(lad):
            breaks |= {b["event_ticker"] for b in kalshi.monotonicity_breaks(lad)}

    for i, g in out.iterrows():
        key = (g["date"], g["home_team"], g["away_team"])
        thin_row = bool(g.get("thin_data", False))
        book_total = g.get("total_line", np.nan)

        # ---- totals ladder -------------------------------------------------------
        if len(tot) and g.get("total_pred") == g.get("total_pred"):
            rungs = tot[(tot["date"] == key[0]) & (tot["home_team"] == key[1])
                        & (tot["away_team"] == key[2])]
            if len(rungs) and book_total == book_total:
                ref = rungs.iloc[(rungs["strike"] - book_total).abs().argsort().iloc[0]]
                p_ref = float(norm.sf(ref["strike"], loc=g["total_pred"], scale=g["total_sigma"]))
                ev_ref, _ = kalshi.contract_ev(p_ref, ref["yes_ask"])
                out.at[i, "kt_ref_strike"] = ref["strike"]
                out.at[i, "kt_ref_ask"] = ref["yes_ask"]
                out.at[i, "kt_ref_prob"] = round(p_ref, 4)
                out.at[i, "kt_ref_ev"] = round(ev_ref, 4) if ev_ref == ev_ref else np.nan
                out.at[i, "kt_ref_tradeable"] = bool(ref["tradeable"])
                if ref["quote_spread"] == ref["quote_spread"]:
                    out.at[i, "kt_ref_spread_c"] = round(ref["quote_spread"] * 100, 1)
            best, considered = None, 0
            for r in rungs.itertuples():
                if thin_row or not r.tradeable:
                    continue
                if book_total == book_total and abs(r.strike - book_total) > config.KALSHI_MAX_BOOK_GAP:
                    continue
                p_over = float(norm.sf(r.strike, loc=g["total_pred"], scale=g["total_sigma"]))
                for prob, ask, label in ((p_over, r.yes_ask, f"Over {r.strike}"),
                                         (1 - p_over, 1 - r.yes_bid if r.yes_bid == r.yes_bid
                                          else np.nan, f"Under {r.strike}")):
                    if not (config.KALSHI_PROB_MIN <= prob <= config.KALSHI_PROB_MAX):
                        continue
                    considered += 1
                    ev, _ = kalshi.contract_ev(prob, ask)
                    if ev == ev and (best is None or ev > best[0]):
                        best = (ev, r, prob, ask, label)
            out.at[i, "kt_rungs"] = considered
            if best and best[0] >= config.KALSHI_MIN_EV:
                ev, r, prob, ask, label = best
                out.at[i, "kt_pick"] = label
                out.at[i, "kt_strike"] = r.strike
                out.at[i, "kt_ask"] = round(ask, 2)
                out.at[i, "kt_prob"] = round(prob, 4)
                out.at[i, "kt_ev"] = round(ev, 4)
                out.at[i, "kt_ticker"] = r.ticker
                out.at[i, "kt_book_gap"] = round(r.strike - book_total, 1) \
                    if book_total == book_total else np.nan
                if r.event_ticker in breaks:
                    out.at[i, "kalshi_incoherent"] = True

        # ---- spread ladder -------------------------------------------------------
        if len(spr) and g.get("margin_pred") == g.get("margin_pred"):
            rungs = spr[(spr["date"] == key[0]) & (spr["home_team"] == key[1])
                        & (spr["away_team"] == key[2])]
            if len(rungs) and g.get("spread_home") == g.get("spread_home"):
                fav_margin = abs(g["spread_home"])
                ref = rungs.iloc[(rungs["strike"] - fav_margin).abs().argsort().iloc[0]]
                if ref["team"] == g["home_team"]:
                    p_ref = float(norm.sf(ref["strike"], loc=g["margin_pred"],
                                          scale=g["margin_sigma"]))
                else:
                    p_ref = float(norm.cdf(-ref["strike"], loc=g["margin_pred"],
                                           scale=g["margin_sigma"]))
                ev_ref, _ = kalshi.contract_ev(p_ref, ref["yes_ask"])
                out.at[i, "ks_ref_side"] = ref["team"]
                out.at[i, "ks_ref_strike"] = ref["strike"]
                out.at[i, "ks_ref_ask"] = ref["yes_ask"]
                out.at[i, "ks_ref_prob"] = round(p_ref, 4)
                out.at[i, "ks_ref_ev"] = round(ev_ref, 4) if ev_ref == ev_ref else np.nan
                out.at[i, "ks_ref_tradeable"] = bool(ref["tradeable"])
                if ref["quote_spread"] == ref["quote_spread"]:
                    out.at[i, "ks_ref_spread_c"] = round(ref["quote_spread"] * 100, 1)
            best, considered = None, 0
            for r in rungs.itertuples():
                if thin_row or not r.tradeable:
                    continue
                if r.team == g["home_team"]:
                    prob = float(norm.sf(r.strike, loc=g["margin_pred"], scale=g["margin_sigma"]))
                    book_fav = -g["spread_home"] if g.get("spread_home") == g.get("spread_home") else np.nan
                elif r.team == g["away_team"]:
                    prob = float(norm.cdf(-r.strike, loc=g["margin_pred"], scale=g["margin_sigma"]))
                    book_fav = g["spread_home"] if g.get("spread_home") == g.get("spread_home") else np.nan
                else:
                    continue
                if book_fav == book_fav and abs(r.strike - book_fav) > config.KALSHI_MAX_BOOK_GAP:
                    continue
                if not (config.KALSHI_PROB_MIN <= prob <= config.KALSHI_PROB_MAX):
                    continue
                considered += 1
                ev, _ = kalshi.contract_ev(prob, r.yes_ask)
                if ev == ev and (best is None or ev > best[0]):
                    best = (ev, r, prob)
            out.at[i, "ks_rungs"] = considered
            if best and best[0] >= config.KALSHI_MIN_EV:
                ev, r, prob = best
                out.at[i, "ks_pick"] = f"{r.team} by over {r.strike}"
                out.at[i, "ks_strike"] = r.strike
                out.at[i, "ks_ask"] = r.yes_ask
                out.at[i, "ks_prob"] = round(prob, 4)
                out.at[i, "ks_ev"] = round(ev, 4)
                out.at[i, "ks_ticker"] = r.ticker
                # How far Kalshi's strike sits from the sportsbook number. A stale exchange
                # strike is a likelier source of edge than this model outsmarting the close.
                if g.get("spread_home") == g.get("spread_home"):
                    book_fav_margin = -g["spread_home"] if r.team == g["home_team"] else g["spread_home"]
                    out.at[i, "ks_book_gap"] = round(r.strike - book_fav_margin, 1)
                if r.event_ticker in breaks:
                    out.at[i, "kalshi_incoherent"] = True

    for pre in ("kt", "ks", "ml"):
        ask_col, prob_col = f"{pre}_ref_ask", f"{pre}_ref_prob"
        if ask_col not in out or prob_col not in out:
            continue
        trio = [_multiplier(a, p) for a, p in zip(out[ask_col], out[prob_col])]
        out[f"{pre}_pays"] = [t[0] for t in trio]
        out[f"{pre}_fair"] = [t[1] for t in trio]
        out[f"{pre}_edge_pct"] = [t[2] for t in trio]

    log.info("kalshi ladders: %d total picks, %d spread picks (from %d eligible rungs after "
             "guards; min EV %.0fc, prob band %.2f-%.2f)",
             int(out["kt_pick"].notna().sum()), int(out["ks_pick"].notna().sum()),
             int(out["kt_rungs"].fillna(0).sum() + out["ks_rungs"].fillna(0).sum()),
             config.KALSHI_MIN_EV * 100, config.KALSHI_PROB_MIN, config.KALSHI_PROB_MAX)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--week", type=int, default=None)
    a = ap.parse_args(argv)
    run(dry_run=a.dry_run, week=a.week)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
