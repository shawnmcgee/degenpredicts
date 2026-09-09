"""Generate the week's Premier League picks.

    python -m epl.predict              # normal run
    python -m epl.predict --dry-run    # print, write nothing

The board is every fixture kicking off in the next BOARD_DAYS days. Numbers come from
football-data.co.uk; if ODDS_API_KEY is set we also pull live prices from books you can
actually bet, which is what makes EV and Kelly meaningful.

**Every market is priced off one scoreline distribution.** The two models predict supremacy and
total goals, those two numbers become a Dixon-Coles grid, and the 1X2 price, the Asian handicap
and the over/under are all read off that same grid. This is the structural payoff of modelling
goals rather than a margin: the three markets cannot contradict each other, so when our
handicap number and our draw price disagree with the book in the same direction, that is one
piece of evidence rather than two.

Three deliberate differences from the other two pipelines:

* **Odds are decimal**, so the profit multiple is ``price - 1`` and there is no American
  conversion anywhere.
* **There is a third outcome.** The 1X2 market gets its own pick line, priced against a
  three-way de-vig that corrects for favourite-longshot bias. It is the market where a public
  model has the best structural chance, because the draw is the leg nobody enjoys pricing.
* **Stakes are eighth-Kelly** and the published number sits close to the market, because the
  fitted shrink against a closing Asian handicap is small by construction.
"""
from __future__ import annotations

import argparse
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config
from .features import BASE_FEATURES, MARKET_FEATURES, build, epl_clubs
from .odds_math import devig_three, devig_two, payout
from .poisson import (asian_handicap, both_teams_score, fair_handicap, grid, match_odds,
                      most_likely_score, over_under)
from .sources import active, odds
from .train import load_models


def source():
    """The configured history/prices backend, resolved at call time.

    Never bound at import: the tests and the docs both switch DEGEN_EPL_SOURCE, and a module
    captured at import would ignore them - the same trap the config paths already avoid.
    """
    return active()


log = logging.getLogger("epl.predict")


def kelly(p_win, mult) -> float:
    """Fractional Kelly on a decimal price. `mult` is the profit multiple (price - 1)."""
    if p_win is None or p_win != p_win or mult is None or mult != mult or mult <= 0:
        return 0.0
    f = (p_win * (mult + 1) - 1) / mult
    return max(0.0, f) * config.KELLY_FRACTION


def _strength(edge, minimum, thin) -> str:
    e = abs(edge) if edge == edge else 0.0
    if e < minimum:
        return "pass"
    if thin:
        return "thin"
    return "bold" if e >= minimum * config.BOLD_MULT else "play"


def _ev(p_win, p_push, mult) -> float:
    """Expected value per unit staked, with the push leg returned rather than lost.

    The push term is not a rounding detail here. A whole-number Asian handicap or a
    whole-number goal line pushes on an exact hit, and on a level handicap that is the ~24% of
    matches that end drawn. Folding those into the loss column would understate EV by more than
    any edge in this file.
    """
    if any(x != x for x in (p_win, p_push, mult)):
        return np.nan
    p_loss = max(0.0, 1.0 - p_win - p_push)
    return round(p_win * mult - p_loss, 4)


def build_board(fixtures: pd.DataFrame, days: int | None = None) -> pd.DataFrame:
    today = config.today_uk()
    horizon = today + timedelta(days=days if days is not None else config.BOARD_DAYS)
    if fixtures.empty:
        return fixtures
    f = fixtures.copy()
    f["date"] = pd.to_datetime(f["date"]).dt.date
    return f[(f["date"] >= today) & (f["date"] <= horizon)].copy()


def run(dry_run: bool = False, days: int | None = None) -> pd.DataFrame:
    today = config.today_uk()
    models, meta = load_models()
    if not models:
        raise SystemExit("no trained models - run python -m epl.train")

    games = source().update_games()
    lines = source().load_lines()
    # Strength is READ here, never rebuilt. It is a season-static quantity - prior-season
    # ratings cannot change mid-season - so refreshing it daily bought nothing and re-fetched
    # the whole division below every morning. The Tuesday retrain owns building it.
    strength = source().load_strength()
    if strength.empty:
        log.warning("no strength table - preseason features will be empty. "
                    "Run `python -m epl.train` to build it.")

    # The source supplies fixtures if it can; the matchdata archive holds played matches only
    # and returns empty, which is its way of saying "ask the live feed". That feed knows about
    # unplayed matches and carries their prices, so it fills both roles at once.
    matcher = odds.build_matcher(sorted(epl_clubs(games, config.season_of(today))))
    fixtures = source().fetch_fixtures()
    if fixtures.empty:
        fixtures = odds.board(matcher)
    board = build_board(fixtures, days)
    if board.empty:
        log.info("no upcoming fixtures on the board")
        return board

    # Only re-pull when the board came from the source rather than from the feed itself;
    # otherwise this would spend a second API call to fetch numbers already in hand.
    from_feed = str(board.get("odds_source", pd.Series(dtype=str)).iloc[0]
                    if len(board) else "").startswith("odds-api")
    live = pd.DataFrame() if from_feed else odds.snapshot(matcher)
    if not dry_run and len(live):
        odds.append_snapshot(live)
    if len(live):
        board = board.merge(
            live[["date", "home_team", "away_team", "live_price_home", "live_price_draw",
                  "live_price_away", "live_total_line", "live_price_over", "live_price_under",
                  "book_1x2", "book_ou"]],
            on=["date", "home_team", "away_team"], how="left")
        # Prefer the live number where we have one - it is fresher than the fixtures file.
        for src, dst in (("live_price_home", "price_home"), ("live_price_draw", "price_draw"),
                         ("live_price_away", "price_away"), ("live_price_over", "price_over"),
                         ("live_price_under", "price_under"),
                         ("live_total_line", "total_line")):
            if src in board:
                board[dst] = board[src].combine_first(board[dst]) if dst in board else board[src]
        # Re-derive the market view from whatever prices actually won, so mkt_* always
        # describes the number the rest of this function is comparing against.
        board = _refresh_market(board)
    for c in ("book_1x2", "book_ou"):
        if c not in board:
            board[c] = np.nan

    _, up, _ = build(games, board, lines=lines, strength=strength)

    keep = ["game_id", "season", "matchweek", "date", "kickoff", "kickoff_uk", "league",
            "home_team", "away_team", "is_derby", "travel_km", "no_crowd",
            "h_promoted", "a_promoted", "h_in_europe", "a_in_europe",
            "h_rest", "a_rest", "rest_diff", "h_games", "a_games",
            "exp_sup", "exp_total", "ah_home", "total_line", "mkt_sup", "mkt_total",
            "mkt_p_home", "mkt_p_draw", "mkt_p_away", "odds_source", "is_closing",
            "price_home", "price_draw", "price_away", "price_over", "price_under",
            "price_ah_home", "price_ah_away", "book_1x2", "book_ou"]
    out = up[[c for c in keep if c in up.columns]].copy()

    # ---- the two point predictions -------------------------------------------------
    for kind, mkt_col in (("total", "mkt_total"), ("sup", "mkt_sup")):
        market_name, base_name = f"{kind}_market", f"{kind}_nomarket"
        has_line = up[mkt_col].notna() if mkt_col in up else pd.Series(False, index=up.index)
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

        line = up[mkt_col] if mkt_col in up else pd.Series(np.nan, index=up.index)
        shrink = meta["shrink"].get(name, config.DEFAULT_SHRINK)
        blended = np.where(line.notna(), line.fillna(0) + shrink * (raw - line.fillna(0)), raw)

        out[f"{kind}_pred"] = np.round(blended, 3)
        out[f"{kind}_raw"] = np.round(raw, 3)
        out[f"{kind}_edge"] = np.round(blended - line, 3)
        # Side is decided from the UNROUNDED difference. With a fitted shrink near 0.2 a raw
        # disagreement of a tenth of a goal shrinks below 0.005 and rounds to 0.000, and
        # `0.0 > 0` is False - which would silently take Under (or the away side) on every
        # near-tie, a real directional bias on exactly the matches that are closest.
        out[f"{kind}_side_val"] = blended - line
        # Selection uses the RAW disagreement, which is what cover_by_disagreement in
        # models/meta.json is bucketed on, so thresholds set from that table apply to the same
        # quantity. The shrunk edge is the honest expected difference and is tiny by design.
        out[f"{kind}_disagree"] = np.round(raw - line, 3)
        out[f"{kind}_model"] = name

    if "sup_pred" not in out or "total_pred" not in out:
        log.warning("need both a supremacy and a total model to price anything")
        return out

    # ---- one scoreline grid per match, and every market read off it -----------------
    out = _price_from_grid(out)

    thin = (out["h_games"] < config.MIN_GAMES) | (out["a_games"] < config.MIN_GAMES)
    out["thin_data"] = thin
    out["ah_strength"] = [_strength(d, config.SUP_EDGE_MIN, t)
                          for d, t in zip(out["sup_disagree"], thin)]
    out["total_strength"] = [_strength(d, config.GOALS_EDGE_MIN, t)
                             for d, t in zip(out["total_disagree"], thin)]
    # Kept under the other pipelines' column names so core.landing, which reads every sport's
    # picks.csv with the csv module and no knowledge of any of them, counts EPL plays too.
    out["spread_strength"] = out["ah_strength"]
    out["prediction_date"] = str(today)
    out = out.sort_values(["date", "kickoff"]).reset_index(drop=True)

    log.info("%d fixtures | handicap %s | totals %s | 1X2 value %d",
             len(out), out.ah_strength.value_counts().to_dict(),
             out.total_strength.value_counts().to_dict(),
             int(out["x2_pick"].notna().sum()))

    if dry_run:
        cols = ["kickoff_uk", "home_team", "away_team", "mkt_sup", "sup_raw", "sup_disagree",
                "ah_pick", "ah_strength", "mkt_total", "total_raw", "total_disagree",
                "total_pick", "total_strength", "p_home", "p_draw", "p_away", "score",
                "x2_pick", "x2_ev"]
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


def _refresh_market(board: pd.DataFrame) -> pd.DataFrame:
    """Recompute the de-vigged market view after live prices have overwritten the cached ones."""
    from .odds_math import supremacy_from_prices, total_from_prices
    for i, r in board.iterrows():
        ph, pdw, pa = r.get("price_home"), r.get("price_draw"), r.get("price_away")
        if ph == ph and pdw == pdw and pa == pa:
            mh, md, ma = devig_three(ph, pdw, pa)
            board.at[i, "mkt_p_home"] = round(mh, 4) if mh == mh else np.nan
            board.at[i, "mkt_p_draw"] = round(md, 4) if md == md else np.nan
            board.at[i, "mkt_p_away"] = round(ma, 4) if ma == ma else np.nan
            ah = r.get("ah_home")
            if not (ah == ah):
                s = supremacy_from_prices(ph, pdw, pa)
                if s == s:
                    board.at[i, "mkt_sup"] = s
        po, pu, tl = r.get("price_over"), r.get("price_under"), r.get("total_line")
        if po == po and pu == pu and tl == tl:
            t = total_from_prices(po, pu, tl)
            if t == t:
                board.at[i, "mkt_total"] = t
    return board


def _price_from_grid(out: pd.DataFrame) -> pd.DataFrame:
    """Turn each match's (supremacy, total) prediction into every market we quote.

    One grid, read four ways. The columns fall into three groups: what we think will happen
    (probabilities and the likeliest scoreline), what the market is offering, and the EV of
    taking it. A pick only surfaces where a real price exists - an edge against a price nobody
    is quoting is not an edge.
    """
    n = len(out)
    text = ["ah_pick", "total_pick", "x2_pick", "x2_side", "score"]
    num = ["p_home", "p_draw", "p_away", "p_btts", "score_prob", "fair_ah",
           "ah_p_win", "ah_p_push", "ah_price", "ah_ev", "ah_stake",
           "total_p_win", "total_p_push", "total_price", "total_ev", "total_stake",
           "x2_p_win", "x2_price", "x2_ev", "x2_stake", "x2_mkt_p", "x2_edge_pct"]
    for c in text:
        out[c] = pd.Series([None] * n, index=out.index, dtype="object")
    for c in num:
        out[c] = np.nan

    for i, g in out.iterrows():
        sup, tot = g.get("sup_pred"), g.get("total_pred")
        if sup != sup or tot != tot:
            continue
        m = grid(float(sup), float(tot))
        ph, pdw, pa = match_odds(m)
        hi, hj, hp = most_likely_score(m)
        out.at[i, "p_home"] = round(ph, 4)
        out.at[i, "p_draw"] = round(pdw, 4)
        out.at[i, "p_away"] = round(pa, 4)
        out.at[i, "p_btts"] = round(both_teams_score(m), 4)
        out.at[i, "score"] = f"{hi}-{hj}"
        out.at[i, "score_prob"] = round(hp, 4)
        out.at[i, "fair_ah"] = fair_handicap(m)

        # ---- Asian handicap --------------------------------------------------------
        ah = g.get("ah_home")
        if ah == ah:
            side_home = bool(g.get("sup_side_val", 0) > 0)
            # Backing the away side means taking the opposite handicap, which is the mirror of
            # the home one: home -1.0 is away +1.0. Reading the away side's probability off
            # the home leg's loss column would be wrong on a quarter line, where part of the
            # stake pushes on both sides.
            hc = float(ah) if side_home else -float(ah)
            w, pu_, _l = asian_handicap(m, hc) if side_home else asian_handicap(_mirror(m), hc)
            price = g.get("price_ah_home") if side_home else g.get("price_ah_away")
            team = g["home_team"] if side_home else g["away_team"]
            out.at[i, "ah_pick"] = f"{team} {hc:+.2f}"
            out.at[i, "ah_p_win"] = round(w, 4)
            out.at[i, "ah_p_push"] = round(pu_, 4)
            if price == price:
                mult = payout(price)
                out.at[i, "ah_price"] = price
                out.at[i, "ah_ev"] = _ev(w, pu_, mult)
                out.at[i, "ah_stake"] = round(
                    kelly(w / max(1 - pu_, 1e-9), mult) * config.BANKROLL_UNITS, 2)

        # ---- over / under ----------------------------------------------------------
        tl = g.get("total_line")
        if tl == tl:
            o, pp, u = over_under(m, float(tl))
            took_over = bool(g.get("total_side_val", 0) > 0)
            w = o if took_over else u
            price = g.get("price_over") if took_over else g.get("price_under")
            out.at[i, "total_pick"] = f"{'Over' if took_over else 'Under'} {float(tl):g}"
            out.at[i, "total_p_win"] = round(w, 4)
            out.at[i, "total_p_push"] = round(pp, 4)
            if price == price:
                mult = payout(price)
                out.at[i, "total_price"] = price
                out.at[i, "total_ev"] = _ev(w, pp, mult)
                out.at[i, "total_stake"] = round(
                    kelly(w / max(1 - pp, 1e-9), mult) * config.BANKROLL_UNITS, 2)

        # ---- 1X2: the market with a third outcome ----------------------------------
        prices = (g.get("price_home"), g.get("price_draw"), g.get("price_away"))
        if all(p == p for p in prices):
            ours = (ph, pdw, pa)
            mkt = devig_three(*prices)
            best = None
            for k, (label, team) in enumerate((("home", g["home_team"]), ("draw", "Draw"),
                                               ("away", g["away_team"]))):
                mult = payout(prices[k])
                if mult != mult:
                    continue
                ev = round(ours[k] * mult - (1 - ours[k]), 4)
                if best is None or ev > best[0]:
                    best = (ev, k, label, team, mult)
            if best:
                ev, k, label, team, mult = best
                out.at[i, "x2_side"] = label
                out.at[i, "x2_pick"] = (team if label != "draw" else "Draw")
                out.at[i, "x2_p_win"] = round(ours[k], 4)
                out.at[i, "x2_mkt_p"] = round(mkt[k], 4) if mkt[k] == mkt[k] else np.nan
                out.at[i, "x2_price"] = prices[k]
                out.at[i, "x2_ev"] = ev
                out.at[i, "x2_stake"] = round(kelly(ours[k], mult) * config.BANKROLL_UNITS, 2)
                if mkt[k] == mkt[k] and mkt[k] > 0:
                    out.at[i, "x2_edge_pct"] = round(100 * (ours[k] / mkt[k] - 1), 1)
    return out


def _mirror(m):
    """Transpose the grid, so the away side reads as if it were the home side.

    ``asian_handicap`` is written from the home team's point of view. Rather than duplicate its
    quarter-line logic for the away side - which is exactly the sort of near-copy that drifts
    out of sync - the grid is transposed and the same function is reused.
    """
    return m.T


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--days", type=int, default=None)
    a = ap.parse_args(argv)
    run(dry_run=a.dry_run, days=a.days)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
