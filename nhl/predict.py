"""Generate the NHL board: today's and tomorrow's games, a puck line and a total for each.

    python -m nhl.predict              # normal run (morning and evening)
    python -m nhl.predict --dry-run    # print, write nothing
    python -m nhl.predict --days 3     # post further ahead

For every game:

1. **The market's view** - the live moneyline and total, de-vigged per book and averaged, are
   inverted through the scoreline model into the goals each side is expected to score. That is
   the baseline, in the model's own units.
2. **The model's view** - the market-aware Poisson model moves each side's goals by what the
   ratings, the likely goalies and the schedule say the market has missed. Without a price,
   the no-market model predicts from ratings alone.
3. **The published view** - the market moved part of the way to the model, separately for the
   split and the total, by the shrinks the backtest fitted.
4. **One grid, every market** - the scoreline model turns the published goals into a final-score
   distribution, and the puck line, the total and the win probability are all read off it.

A pick is the side with the higher expected value at the posted price. It is staked only when
that EV clears the threshold and both teams have played enough games for their ratings to mean
something; everything else is still published, graded and CLV-tracked, so the evidence builds
whether or not money is at risk.

The job runs twice a day. The morning run records the number we FIRST published; the evening
run, a couple of hours before puck drop, refreshes prices that by then know the starting
goalies - and its snapshot is the closing line CLV is measured against.
"""
from __future__ import annotations

import argparse
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config
from .features import build, sides, unstack
from .odds_math import decimal_to_american, ev, kelly
from .scoreline import Table, cover, grids, moneyline, over_under, regulation_split, expected
from .sources import nhle, odds
from .train import blend, load_models

log = logging.getLogger("nhl.predict")
PLAYABLE = ("play", "bold")
MARKET_COLS = ["commence_time", "home_dec", "away_dec", "p_home", "n_ml", "total_line",
               "over_dec", "under_dec", "p_over", "n_total", "pl_home", "pl_home_dec",
               "pl_away_dec", "p_pl_home", "n_pl"]


def _strength(ev_val, minimum, thin, priced) -> str:
    if not priced or ev_val != ev_val or ev_val < minimum:
        return "pass"
    if thin:
        return "thin"
    return "bold" if ev_val >= minimum * config.BOLD_MULT else "play"


def _american(dec) -> str:
    a = decimal_to_american(dec)
    if a != a:
        return ""
    return f"{a:+.0f}"


def puck_drop(row) -> tuple[str, str]:
    """(display label, sortable ISO) from the feed's UTC start, else the NHL's Eastern time."""
    ts = pd.to_datetime(row.get("commence_time"), utc=True, errors="coerce")
    if ts is not pd.NaT and ts == ts:
        et = ts.tz_convert(config.ET)
        return et.strftime("%a %b %-d, %-I:%M %p"), ts.isoformat()
    s = str(row.get("start_et") or "")
    t = pd.to_datetime(s, errors="coerce")
    if t is not pd.NaT and t == t and len(s) > 10:
        loc = t.tz_localize(config.ET) if t.tzinfo is None else t
        return loc.strftime("%a %b %-d, %-I:%M %p"), loc.tz_convert("UTC").isoformat()
    return "", ""


def started(board: pd.DataFrame, now) -> pd.Series:
    """Games whose puck drop has passed - by the feed's start time, else the NHL's."""
    now = pd.Timestamp(now)
    return pd.Series([bool(iso) and pd.Timestamp(iso) <= now
                      for iso in (puck_drop(r)[1] for r in board.to_dict("records"))],
                     index=board.index, dtype=bool)


def build_board(games: pd.DataFrame, days: int | None = None) -> pd.DataFrame:
    today = config.today_et()
    horizon = today + timedelta(days=config.BOARD_DAYS if days is None else days)
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    return g[(~g["completed"].astype(bool)) & (g["date"] >= today) & (g["date"] <= horizon)].copy()


def price_game(F, R, g: dict) -> dict:
    """Every number the card shows, read off one final-score grid."""
    out = {}
    ph = moneyline(F)
    rh, ot, ra = regulation_split(R)
    e_m, e_t = expected(F)
    out.update(p_home_win=round(ph, 4), p_away_win=round(1 - ph, 4), p_reg_home=round(rh, 4),
               p_ot=round(ot, 4), p_reg_away=round(ra, 4), exp_margin=round(e_m, 3),
               exp_total=round(e_t, 3))
    thin = bool(g.get("thin_data"))

    # ---- puck line ----------------------------------------------------------------
    hl = g.get("pl_home")
    priced_pl = hl == hl and hl is not None and g.get("pl_home_dec") == g.get("pl_home_dec") \
        and g.get("pl_away_dec") == g.get("pl_away_dec")
    if not (hl == hl and hl is not None):
        hl = -1.5 if ph >= 0.5 else 1.5          # display only: the conventional line
    p_hc, p_push, p_ac = cover(F, float(hl))
    mkt_hc = g.get("p_pl_home")
    if priced_pl:
        ev_h = ev(p_hc, p_push, g["pl_home_dec"])
        ev_a = ev(p_ac, p_push, g["pl_away_dec"])
        home_side = ev_h >= ev_a
    else:
        ev_h = ev_a = np.nan
        ref = mkt_hc if mkt_hc == mkt_hc and mkt_hc is not None else 0.5
        home_side = p_hc >= ref
    side_line = float(hl) if home_side else -float(hl)
    p_side = p_hc if home_side else p_ac
    dec = g.get("pl_home_dec") if home_side else g.get("pl_away_dec")
    evv = ev_h if home_side else ev_a
    mkt_p = (mkt_hc if home_side else 1 - mkt_hc) if mkt_hc == mkt_hc and mkt_hc is not None else np.nan
    team = g["home_team"] if home_side else g["away_team"]
    st = _strength(evv, config.SPREAD_EV_MIN, thin, priced_pl)
    out.update(spread_side="home" if home_side else "away", spread_team=team,
               spread_line=side_line, spread_pick=f"{team} {side_line:+.1f}",
               spread_p_win=round(p_side, 4), spread_p_push=round(p_push, 4),
               spread_mkt_p=round(mkt_p, 4) if mkt_p == mkt_p else np.nan,
               spread_edge_pp=round(100 * (p_side - mkt_p), 2) if mkt_p == mkt_p else np.nan,
               spread_dec=dec if priced_pl else np.nan,
               spread_price=_american(dec) if priced_pl else "",
               spread_ev=round(evv, 4) if evv == evv else np.nan, spread_strength=st,
               spread_stake=round(kelly(p_side, p_push, dec, config.KELLY_FRACTION)
                                  * config.BANKROLL_UNITS, 2) if st in PLAYABLE else 0.0,
               p_home_cover=round(p_hc, 4), pl_home_used=float(hl))

    # ---- total ------------------------------------------------------------------------
    tl = g.get("total_line")
    priced_t = tl == tl and tl is not None and g.get("over_dec") == g.get("over_dec") \
        and g.get("under_dec") == g.get("under_dec")
    line = float(tl) if tl == tl and tl is not None else float(np.floor(e_t * 2) / 2 + 0.0)
    if line == np.floor(line) and not (tl == tl and tl is not None):
        line += 0.5                               # an unpriced display line never pushes
    po, pp, pu = over_under(F, line)
    mkt_o = g.get("p_over")
    if priced_t:
        ev_o, ev_u = ev(po, pp, g["over_dec"]), ev(pu, pp, g["under_dec"])
        take_over = ev_o >= ev_u
    else:
        ev_o = ev_u = np.nan
        live = max(po + pu, 1e-9)
        ref = mkt_o if mkt_o == mkt_o and mkt_o is not None else 0.5
        take_over = po / live >= ref
    p_side = po if take_over else pu
    dec = g.get("over_dec") if take_over else g.get("under_dec")
    evv = ev_o if take_over else ev_u
    live = max(1 - pp, 1e-9)
    mkt_p = (mkt_o if take_over else 1 - mkt_o) if mkt_o == mkt_o and mkt_o is not None else np.nan
    st = _strength(evv, config.TOTAL_EV_MIN, thin, priced_t)
    out.update(total_side="over" if take_over else "under", total_line_used=line,
               total_pick=f"{'Over' if take_over else 'Under'} {line:g}",
               total_p_win=round(p_side, 4), total_p_push=round(pp, 4),
               total_mkt_p=round(mkt_p, 4) if mkt_p == mkt_p else np.nan,
               total_edge_pp=round(100 * (p_side / live - mkt_p), 2) if mkt_p == mkt_p else np.nan,
               total_dec=dec if priced_t else np.nan,
               total_price=_american(dec) if priced_t else "",
               total_ev=round(evv, 4) if evv == evv else np.nan, total_strength=st,
               total_stake=round(kelly(p_side, pp, dec, config.KELLY_FRACTION)
                                 * config.BANKROLL_UNITS, 2) if st in PLAYABLE else 0.0)
    return out


def run(dry_run: bool = False, days: int | None = None) -> pd.DataFrame:
    today = config.today_et()
    models, meta = load_models()
    games = nhle.update_games()
    board = build_board(games, days)
    if board.empty:
        log.info("no NHL games on the board for %s", today)
        return board

    snap = odds.match_games(odds.snapshot(), games)
    if not dry_run:
        odds.append_snapshot(snap)
    if len(snap):
        s = snap.dropna(subset=["game_id"]).drop_duplicates("game_id", keep="last")
        board = board.merge(s[["game_id"] + MARKET_COLS], on="game_id", how="left")
    for c in MARKET_COLS:
        if c not in board:
            board[c] = np.nan
    # A game already under way keeps the row published before puck drop. Weekend matinees are
    # mid-game when the evening run fires, and re-pricing one then would replace the pick with
    # a pre-game model read against no price at all.
    live = started(board, config.now_et())
    if live.any():
        log.info("%d games already under way keep the rows published before puck drop",
                 int(live.sum()))
        board = board[~live]
        if board.empty:
            return board

    theta = meta.get("theta") or {}
    table = Table(theta)
    _, up, _ = build(games, upcoming=board, table=table)
    S = sides(up)
    gid = up["game_id"].values
    lh, la = unstack(S, models["nomarket"].predict(S), gid)
    model_name = np.full(len(up), "nomarket", dtype=object)
    if "market" in models:
        mk = S["log_mkt"].notna().values
        if mk.any():
            pred = np.full(len(S), np.nan)
            pred[mk] = models["market"].predict(S[mk])
            mh, ma = unstack(S, pred, gid)
            use = np.isfinite(mh) & np.isfinite(ma)
            lh, la = np.where(use, mh, lh), np.where(use, ma, la)
            model_name[use] = "market"
    raw_h, raw_a = lh.copy(), la.copy()
    sh = meta.get("shrink") or {}
    lh, la = blend(up["m_lh"], up["m_la"], lh, la,
                   sh.get("split", config.DEFAULT_SPLIT_SHRINK),
                   sh.get("total", config.DEFAULT_TOTAL_SHRINK))
    F, R = grids(lh, la, theta)

    up = up.copy()
    up["thin_data"] = (up["h_games"] < config.MIN_GAMES) | (up["a_games"] < config.MIN_GAMES)
    rows = []
    for i, g in enumerate(up.to_dict("records")):
        r = price_game(F[i], R[i], g)
        label, iso = puck_drop(g)
        r.update(puck_drop=label, puck_drop_utc=iso, lam_home=round(float(lh[i]), 3),
                 lam_away=round(float(la[i]), 3), raw_lam_home=round(float(raw_h[i]), 3),
                 raw_lam_away=round(float(raw_a[i]), 3), model=model_name[i])
        rows.append(r)
    priced = pd.DataFrame(rows, index=up.index)
    keep = ["game_id", "season", "game_type", "date", "home_team", "away_team", "h_games",
            "a_games", "h_rest", "a_rest", "h_b2b", "a_b2b", "h_3in4", "a_3in4", "h_km", "a_km",
            "h_tz", "a_tz", "g_h_top", "g_a_top", "g_h_conf", "g_a_conf", "thin_data",
            "m_lh", "m_la"] + MARKET_COLS
    out = pd.concat([up[[c for c in keep if c in up.columns]], priced], axis=1)
    out = out.rename(columns={"p_home": "mkt_p_home", "p_over": "mkt_p_over",
                              "p_pl_home": "mkt_p_pl_home"})
    out["prediction_date"] = str(today)
    out["model_trained_at"] = meta.get("trained_at", "")
    out = out.sort_values(["date", "puck_drop_utc", "game_id"]).reset_index(drop=True)
    log.info("%d games | puck line %s | totals %s", len(out),
             out["spread_strength"].value_counts().to_dict(),
             out["total_strength"].value_counts().to_dict())

    if dry_run:
        cols = ["puck_drop", "away_team", "home_team", "p_home_win", "spread_pick", "spread_price",
                "spread_p_win", "spread_mkt_p", "spread_ev", "spread_strength", "total_pick",
                "total_price", "total_p_win", "total_mkt_p", "total_ev", "total_strength"]
        print(out[[c for c in cols if c in out]].to_string(index=False))
        return out

    config.ensure_dirs()
    out["first_seen_at"] = str(today)
    for c in ("spread_line", "spread_dec", "spread_mkt_p", "total_line_used", "total_dec",
              "total_mkt_p", "m_lh", "m_la"):
        out[f"first_{c}"] = out[c]
    if config.PICKS.exists():
        prev = pd.read_csv(config.PICKS, dtype={"game_id": str}, low_memory=False)
        out = _carry_first_seen(prev, out)
        prev = prev[~prev["game_id"].astype(str).isin(out["game_id"].astype(str))]
        pd.concat([prev, out], ignore_index=True).to_csv(config.PICKS, index=False)
    else:
        out.to_csv(config.PICKS, index=False)
    log.info("wrote %d picks to %s", len(out), config.PICKS)
    return out


def _carry_first_seen(prev: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Keep the number we FIRST published for a game across every later rewrite of its row.

    The evening run rewrites each game's row with fresher prices. Without this the only surviving
    record would be the evening one, and closing-line value measured against it would compare
    the close with a number taken an hour before it rather than with what we put our name to.
    """
    firsts = [c for c in prev.columns if c.startswith("first_")]
    if not len(prev) or "first_seen_at" not in prev.columns:
        return out
    seen = prev.dropna(subset=["first_seen_at"]).drop_duplicates("game_id", keep="first")
    seen = seen.set_index(seen["game_id"].astype(str))
    ids = out["game_id"].astype(str)
    for c in firsts:
        if c in out.columns:
            earlier = ids.map(seen[c]) if c in seen else pd.Series(np.nan, index=out.index)
            out[c] = earlier.where(earlier.notna(), out[c]).values
    return out


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
