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
from .features import (BASE_FEATURES, MARKET_FEATURES, build, fbs_membership,
                       fbs_teams, is_fbs_game)
from .sources import cfbd, kalshi, odds
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


PLAYABLE = ("play", "bold")


def _stakes(p_win, payout, strength):
    """Kelly stake, but only on the markets we actually said we were playing.

    ``_strength`` returning "pass" or "thin" is the system declining the bet. Sizing one
    anyway - which is what happened for every graded game so far - makes the units and ROI on
    the front page describe wagers that were never supposed to exist.
    """
    return [round(kelly(p, b) * config.BANKROLL_UNITS, 2) if st in PLAYABLE else 0.0
            for p, b, st in zip(p_win, payout, strength)]


def _strength(edge, minimum, thin) -> str:
    """How hard we back a market, from the model's raw disagreement with the line.

    Classification is deliberately NOT an input here. An early blanket ban on FBS-vs-FCS games
    was measured and withdrawn: judged on the population that actually gets staked - games that
    clear these thresholds - FCS totals cover 56.8% against a 51.75% break-even, above every
    seed, which is the best segment in the table rather than the worst. The unconditioned 50.7%
    that motivated the ban describes games this function already declines. The thresholds are
    the gate; a game is priced on its features, not on its opponent's division.
    """
    e = abs(edge) if edge == edge else 0.0
    if e < minimum:
        return "pass"
    if thin:
        return "thin"
    return "bold" if e >= minimum * config.BOLD_MULT else "play"


def build_board(games: pd.DataFrame, lines: pd.DataFrame, week: int | None = None,
                sp: pd.DataFrame | None = None) -> pd.DataFrame:
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
    # The schedule feed carries every classification, including the D-II and D-III rows the
    # broken CFBD filter drags in. Two different decisions here, and only one of them is about
    # what the model can price:
    #
    #   * a game with NEITHER side FBS is always dropped. Nothing rates those teams, and
    #     publishing a number on one is inventing it.
    #   * an FBS-vs-FCS game is kept when `config.BOARD_FCS`, and carries `fbs=False`. The
    #     flag is for labelling and for the per-class record split in `grade.metrics` - it is
    #     NOT a staking veto. Judged on the games that clear the edge thresholds, FCS totals
    #     cover 56.8% against a 51.75% break-even, so blocking them would have removed the
    #     best-covering segment measured. The thresholds decide, the same as anywhere else.
    members = fbs_membership(games, sp)
    sched["fbs"] = [is_fbs_game(members, r.season, r.home_team, r.away_team)
                    for r in sched.itertuples()]
    rated = fbs_teams(games, season, sp)
    one_side = [(r.home_team in rated) or (r.away_team in rated) for r in sched.itertuples()]
    keep = [f or (config.BOARD_FCS and o) for f, o in zip(sched["fbs"], one_side)]
    if not all(keep):
        log.info("board: dropped %d unrateable games, %d remain",
                 len(keep) - sum(keep), sum(keep))
    sched = sched[keep]
    if sched.empty:
        return sched
    if not sched["fbs"].all():
        log.info("board: %d of %d games are FBS vs FCS (shown, not staked)",
                 int((~sched["fbs"]).sum()), len(sched))
    cols = ["game_id", "spread_home", "spread_open", "total_line", "total_open",
            "provider", "n_providers"]
    have = [c for c in cols if c in lines.columns]
    board = sched.merge(lines[have], on="game_id", how="left")
    return board


def board_teams(board: pd.DataFrame, games: pd.DataFrame, season: int,
                sp: pd.DataFrame | None = None) -> set[str]:
    """Name universe for the odds and Kalshi matchers.

    The FBS roster, plus the FCS teams actually on this week's board. It has to be *plus this
    week's names only*: handing the matcher every name in the games file put 699 teams in the
    2025 table, and `difflib` at cutoff 0.80 mis-resolves names onto the extra collision
    targets. Thirty-odd real opponents is a rounding error on a 135-team roster; five hundred
    is a different failure mode.
    """
    names = set(fbs_teams(games, season, sp))
    if board is not None and len(board):
        names |= set(board["home_team"]) | set(board["away_team"])
    return names


def run(dry_run: bool = False, week: int | None = None) -> pd.DataFrame:
    today = config.today_et()
    season = config.season_of(today)
    models, meta = load_models()
    if not models:
        raise SystemExit("no trained models - run python -m cfb.train")

    games = cfbd.update_games()
    lines = cfbd.update_lines()
    cfbd.update_sp()
    cfbd.update_returning()
    sp = cfbd.load_sp()

    board = build_board(games, lines, week, sp)
    if board.empty:
        log.info("no upcoming games on the board")
        return board

    # optional live prices, joined on team names
    live = odds.snapshot(odds.build_matcher(sorted(board_teams(board, games, season, sp))))
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

    _, up, _ = build(games, board, lines=lines, sp=sp, returning=cfbd.load_returning())

    keep = ["game_id", "season", "week", "date", "tip_et", "kickoff_utc", "start_time_tbd",
            "home_team", "away_team",
            "neutral_site", "conference_game", "total_line", "spread_home", "provider",
            "over_price", "under_price", "spread_home_price", "spread_away_price",
            "total_book", "spread_book", "h_games", "a_games", "exp_total", "exp_margin",
            # `fbs` is load-bearing, not decoration: it is what makes an FBS-vs-FCS game
            # unstakeable further down. Dropping it here silently re-enables betting on them.
            "fbs"]
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
        # Selection uses the model's RAW disagreement with the line. The shrunk `edge` is the
        # honest expected difference and is tiny by construction (shrink is usually < 0.3), so
        # thresholding on it would produce an empty board. Disagreement is what the
        # ats_by_disagreement table in meta.json is bucketed on, so the thresholds you set
        # from that table apply to the same quantity.
        out[f"{kind}_disagree"] = np.round(raw - line, 1)
        out[f"{kind}_sigma"] = sigma
        out[f"{kind}_model"] = name
        out[f"{kind}_p_over"] = np.round(norm.cdf((blended - line) / sigma), 4)

    # Strength is decided BEFORE anything is staked, because it is what decides whether
    # anything is staked at all. It used to be computed at the very end, after the Kelly
    # numbers were already in the frame, which is how a week where all 105 games graded
    # `pass` or `thin` still carried 13.01 units of totals stake and produced an 86% ROI on
    # the front page.
    thin = (out["h_games"] < config.MIN_GAMES) | (out["a_games"] < config.MIN_GAMES)
    out["thin_data"] = thin
    out["total_strength"] = [_strength(d, config.TOTAL_EDGE_MIN, t)
                             for d, t in zip(out["total_disagree"], thin)]
    out["spread_strength"] = [_strength(d, config.SPREAD_EDGE_MIN, t)
                              for d, t in zip(out["margin_disagree"], thin)]

    out["total_pick"] = np.where(out["total_edge"] > 0, "Over", "Under")
    # No line, no pick label. `np.where(NaN > 0, ...)` is False, so an unpriced total used to
    # publish the word "Under" against a market that has no number.
    out.loc[out["total_line"].isna(), "total_pick"] = ""
    out["total_p_win"] = np.where(out["total_edge"] > 0, out["total_p_over"], 1 - out["total_p_over"])
    out["total_price"] = np.where(out["total_edge"] > 0, out["over_price"], out["under_price"])
    out["total_payout"] = out["total_price"].apply(american_payout)
    out["total_ev"] = (out["total_p_win"] * out["total_payout"] - (1 - out["total_p_win"])).round(3)
    out["total_stake"] = _stakes(out["total_p_win"], out["total_payout"], out["total_strength"])

    took_home = out["margin_edge"] > 0
    out["spread_side"] = np.where(took_home, out["home_team"], out["away_team"])
    out["spread_number"] = np.where(took_home, out["spread_home"], -out["spread_home"])
    out["spread_pick"] = out["spread_side"] + " " + out["spread_number"].map(
        lambda x: f"{x:+.1f}" if x == x else "")
    out.loc[out["spread_home"].isna(), "spread_pick"] = ""
    out["spread_p_win"] = np.where(took_home, out["margin_p_over"], 1 - out["margin_p_over"])
    out["spread_price"] = np.where(took_home, out["spread_home_price"], out["spread_away_price"])
    out["spread_payout"] = out["spread_price"].apply(american_payout)
    out["spread_ev"] = (out["spread_p_win"] * out["spread_payout"] - (1 - out["spread_p_win"])).round(3)
    out["spread_stake"] = _stakes(out["spread_p_win"], out["spread_payout"], out["spread_strength"])

    # --- moneyline: margin distribution -> P(home wins), compared to Kalshi's ask ----------
    # A margin prediction plus its fitted residual sigma is a full distribution, so
    # P(home wins) = P(margin > 0). Kalshi's binary contract prices exactly that event, which
    # makes it directly comparable in a way the spread and total series are not.
    # `thin_data` is already set above, which it must be: _price_ladders refuses to publish a
    # pick on a game the model itself considers unplayable.
    if "margin_pred" in out:
        sig = out["margin_sigma"].replace(0, np.nan)
        out["p_home_win"] = np.round(norm.cdf(out["margin_pred"] / sig), 4)
        out["p_away_win"] = (1 - out["p_home_win"]).round(4)
        out = _attach_kalshi(out, games, sp)
        out = _price_ladders(out, games, sp)

    out["prediction_date"] = str(today)
    # Which retrain produced this row. Without it a good or bad stretch cannot be attributed
    # to a specific model.
    out["model_trained_at"] = meta.get("trained_at", "")
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
    out["first_seen_spread"] = out["spread_home"]
    out["first_seen_total"] = out["total_line"]
    out["first_seen_at"] = str(today)
    if config.PICKS.exists():
        prev = pd.read_csv(config.PICKS, dtype={"game_id": str})
        out = _carry_first_seen(prev, out)
        prev = prev[~prev["game_id"].isin(out["game_id"])]
        pd.concat([prev, out], ignore_index=True).to_csv(config.PICKS, index=False)
    else:
        out.to_csv(config.PICKS, index=False)
    log.info("wrote %d picks to %s", len(out), config.PICKS)
    return out


FIRST_SEEN = ("first_seen_spread", "first_seen_total", "first_seen_at")


def _carry_first_seen(prev: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Keep the number we FIRST published for a game, across every later overwrite.

    A game's row is rewritten every morning it stays on the board, so without this the only
    surviving record is the last one - and closing-line value measured against Saturday 9am
    answers a question nobody asked. The display fields still refresh; only these three are
    sticky, and only where the earlier run actually had a number.
    """
    if not len(prev) or "first_seen_at" not in prev.columns:
        return out
    seen = (prev.dropna(subset=["first_seen_at"])
                .drop_duplicates("game_id", keep="first")
                .set_index("game_id"))
    for col in FIRST_SEEN:
        if col not in seen.columns:
            continue
        earlier = out["game_id"].map(seen[col])
        out[col] = earlier.combine_first(out[col]) if col != "first_seen_at" else \
            earlier.fillna(out[col])
    return out


def _attach_kalshi(out: pd.DataFrame, games: pd.DataFrame,
                   sp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join live Kalshi moneyline quotes and price our probability against the actual ask.

    We compare against the **ask**, never the mid. Games listed early can quote absurdly wide
    (0.08 x 0.81 two weeks out, zero volume); an edge measured off that mid is fiction.
    `kalshi_tradeable` folds together quote width, resting size and traded volume.
    """
    # numeric and text columns need different dtypes up front, or assigning a team name into a
    # float64 column raises on pandas >= 2.2
    for c in ("kalshi_home_ask", "kalshi_away_ask", "kalshi_home_ev", "kalshi_away_ev",
              "kalshi_spread_c"):
        out[c] = np.nan
    for c in ("kalshi_side", "kalshi_ticker"):
        out[c] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["kalshi_tradeable"] = False
    # fields the site template renders
    out["ml_pick"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    for c in ("ml_ask", "ml_p_home", "ml_ev", "ml_roi", "ml_stake", "ml_model_cents",
              "ml_ref_ask", "ml_ref_prob", "ml_ref_ev"):
        out[c] = np.nan
    out["ml_ref_side"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    try:
        matcher = odds.build_matcher(sorted(board_teams(out, games, config.season_of(config.today_et()), sp)))
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
                    # `prob` travels WITH the side. Capturing only (side, rec) left `prob` and
                    # `ev` holding whatever the final loop iteration set them to, so picking
                    # home wrote the home ask beside the AWAY probability - and then applied
                    # the 0.20-0.80 playability band to the wrong side of the game.
                    best_ev, best = ev, (side, rec, prob)
        if best:
            side, rec, prob = best
            team = g[f"{side}_team"]
            ev, roi = kalshi.contract_ev(prob, rec.yes_ask)
            out.at[i, "kalshi_side"] = team
            out.at[i, "kalshi_ticker"] = rec.ticker
            out.at[i, "kalshi_tradeable"] = bool(rec.tradeable)
            if rec.quote_spread == rec.quote_spread:
                out.at[i, "kalshi_spread_c"] = round(rec.quote_spread * 100, 1)
            out.at[i, "ml_p_home"] = g.get("p_home_win")
            out.at[i, "ml_ref_ask"] = rec.yes_ask
            out.at[i, "ml_ref_prob"] = round(prob, 4) if prob == prob else np.nan
            out.at[i, "ml_ref_ev"] = round(ev, 4) if ev == ev else np.nan
            out.at[i, "ml_ref_side"] = team
            # Only surface a playable moneyline where the book is genuinely tradeable and the
            # edge survives the fee. Everything else stays visible in the raw CSV but off the
            # site, because an edge against an untraded 81c ask is not an edge.
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
        # Report the names the matcher could not resolve, i.e. the raw feed spellings - those
        # are what need a mapping rule, not our own canonical names.
        stuck = sorted(getattr(matcher, "unmatched", set()))
        if stuck:
            log.warning("kalshi names the matcher could not resolve (%d): %s",
                        len(stuck), stuck[:40])
        else:
            log.info("kalshi: %d board games had no listed market (likely FCS or not yet posted)",
                     len(out) - hits)
    return out


def _multiplier(ask, prob):
    """What the contract pays per unit risked, and what it *should* pay.

    Buying YES costs ask + fee and returns 1.00, so the payout multiple is 1/(ask+fee). The
    fair multiple implied by our probability is 1/prob. Paying more than fair is the edge, and
    the ratio between them is exactly the ROI - it's the same number as EV/cost, just in a
    form that reads like odds instead of cents.
    """
    if ask is None or ask != ask or ask <= 0:
        return np.nan, np.nan, np.nan
    cost = ask + kalshi.fee(ask)
    pays = 1.0 / cost if cost > 0 else np.nan
    fair = 1.0 / prob if prob and prob == prob and prob > 0 else np.nan
    edge = (pays / fair - 1.0) * 100 if pays == pays and fair == fair else np.nan
    return round(pays, 3), round(fair, 3), round(edge, 1)


def _price_ladders(out: pd.DataFrame, games: pd.DataFrame,
                   sp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Price every rung of Kalshi's spread and total ladders against the model distribution.

    Both series expose ``floor_strike`` with ``strike_type: "greater"``, so each contract pays
    when the quantity exceeds the strike:

        total  "Over X points"            -> P(total  > X)  = sf(X, total_pred,  total_sigma)
        spread "TEAM wins by over X"      -> P(margin > X)  if TEAM is home
                                             P(margin < -X) if TEAM is away

    Because it's a ladder, we score every rung and keep the best tradeable positive-EV one
    rather than assuming a single line.
    """
    text_cols = ["kt_pick", "kt_ticker", "ks_pick", "ks_ticker"]
    num_cols = ["kt_strike", "kt_ask", "kt_prob", "kt_ev", "kt_book_gap", "kt_rungs",
                "ks_strike", "ks_ask", "ks_prob", "ks_ev", "ks_book_gap", "ks_rungs",
                # reference quote: the rung nearest the book number, recorded for EVERY game
                # whether or not it is playable, so the exchange price is always visible
                "kt_ref_strike", "kt_ref_ask", "kt_ref_prob", "kt_ref_ev", "kt_ref_spread_c",
                "ks_ref_strike", "ks_ref_ask", "ks_ref_prob", "ks_ref_ev", "ks_ref_spread_c"]
    text_cols = text_cols + ["ks_ref_side"]
    for c in text_cols:
        out[c] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["kt_ref_tradeable"] = False
    out["ks_ref_tradeable"] = False
    for c in num_cols:
        out[c] = np.nan
    out["kalshi_incoherent"] = False

    try:
        matcher = odds.build_matcher(sorted(board_teams(out, games, config.season_of(config.today_et()), sp)))
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
        # Never publish an exchange pick on a game the model itself flags as thin. In week 1 no
        # team has played a snap; the model correctly says "pass", and the Kalshi path must not
        # contradict it.
        thin_row = bool(g.get("thin_data", False))
        book_total = g.get("total_line", np.nan)

        # ---- totals ladder -------------------------------------------------------
        if len(tot) and g.get("total_pred") == g.get("total_pred"):
            rungs = tot[(tot["date"] == key[0]) & (tot["home_team"] == key[1])
                        & (tot["away_team"] == key[2])]
            # Reference quote: the rung closest to the sportsbook total. Always recorded, even
            # when nothing is playable, so the page can show what Kalshi is actually charging.
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
                out.at[i, "kt_book_gap"] = round(r.strike - g["total_line"], 1) \
                    if g.get("total_line") == g.get("total_line") else np.nan
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
                    p_ref = float(norm.sf(ref["strike"], loc=g["margin_pred"], scale=g["margin_sigma"]))
                else:
                    p_ref = float(norm.cdf(-ref["strike"], loc=g["margin_pred"], scale=g["margin_sigma"]))
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
                # how far Kalshi's strike sits from the sportsbook number - a stale exchange
                # strike is a likelier source of edge than the model outsmarting the book
                if g.get("spread_home") == g.get("spread_home"):
                    book_fav_margin = -g["spread_home"] if r.team == g["home_team"] else g["spread_home"]
                    out.at[i, "ks_book_gap"] = round(r.strike - book_fav_margin, 1)
                if r.event_ticker in breaks:
                    out.at[i, "kalshi_incoherent"] = True

    for pre in ("kt", "ks", "ml"):
        ask_col = f"{pre}_ref_ask"
        prob_col = f"{pre}_ref_prob"
        if ask_col not in out or prob_col not in out:
            continue
        trio = [_multiplier(a, p) for a, p in zip(out[ask_col], out[prob_col])]
        out[f"{pre}_pays"] = [t[0] for t in trio]
        out[f"{pre}_fair"] = [t[1] for t in trio]
        out[f"{pre}_edge_pct"] = [t[2] for t in trio]

    log.info("kalshi quotes recorded: %d totals, %d spreads (reference rung nearest the book "
             "number, shown regardless of playability)",
             int(out["kt_ref_ask"].notna().sum()), int(out["ks_ref_ask"].notna().sum()))
    log.info("kalshi ladders: %d total picks, %d spread picks "
             "(from %d/%d eligible rungs after guards; min EV %.0fc, prob band %.2f-%.2f)",
             int(out["kt_pick"].notna().sum()), int(out["ks_pick"].notna().sum()),
             int(out["kt_rungs"].fillna(0).sum() + out["ks_rungs"].fillna(0).sum()),
             len(tot) + len(spr), config.KALSHI_MIN_EV * 100,
             config.KALSHI_PROB_MIN, config.KALSHI_PROB_MAX)
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
