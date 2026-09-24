"""Generate the NBA board: today's and tomorrow's games - a spread, a total and a moneyline each.

    python -m nba.predict              # normal run (morning and evening)
    python -m nba.predict --dry-run    # print, write nothing
    python -m nba.predict --days 3     # post further ahead

For every game:

1. **Who is playing** - the day's injury report and the current rosters set each rotation
   player's chance of sitting (:mod:`nba.players`): an Out player is out, a questionable one
   half out, a traded one gone. That becomes each side's availability delta.
2. **The model's view** - the no-market model predicts the margin and total from ratings,
   availability and schedule alone; the market-aware models then say how far the posted line
   should move for what it has missed.
3. **The published view** - the line moved part of the way to the market-aware model, by the
   shrink the backtest fitted. Expect that to be close to the line itself.
4. **Prices** - cover, over/under and win probabilities off the published numbers, and the EV
   of each side at the posted price, pushes included.

A pick is the market-aware model's side of the line (the moneyline: the side with the larger
EV). It is staked only when its disagreement clears the threshold, both teams have played enough
games to rate, and no unresolved injury could still move the number by more than
``DEGEN_NBA_WAIT_POINTS`` - until then it is marked as waiting on injury news. Everything is
still published, graded and CLV-tracked, so the evidence builds whether or not money is at risk.

The job runs twice a day. The morning run records the number we FIRST published; the evening
run, an hour or two before tip, refreshes prices and the injury report - and its snapshot is the
closing line CLV is measured against.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config
from .features import build
from .odds_math import cover, decimal_to_american, ev, kelly, over_under, win_prob
from .sources import espn, history, hoopr, odds
from .train import load_models

log = logging.getLogger("nba.predict")
PLAYABLE = ("play", "bold")
MARKET_COLS = ["commence_time", "provider", "source", "spread_home", "spread_home_dec",
               "spread_away_dec", "p_spread_home", "n_spread", "total_line", "over_dec",
               "under_dec", "p_over", "n_total", "home_dec", "away_dec", "p_home", "n_ml"]
DEFAULT_DEC = 1 + 100 / 110          # -110, assumed only when a line has no posted price


def _strength(size, minimum, thin, wait, priced=True) -> str:
    if not priced or size != size or size < minimum:
        return "pass"
    if thin:
        return "thin"
    if wait:
        return "wait"               # clears the bar, but not while injury news is unresolved
    return "bold" if size >= minimum * config.BOLD_MULT else "play"


def _american(dec) -> str:
    a = decimal_to_american(dec)
    return "" if a != a else f"{a:+.0f}"


def _f(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f


def tip(row) -> tuple[str, str]:
    """(display label, sortable ISO) from the price feed's start, else the schedule's."""
    for key in ("commence_time", "tip_utc"):
        ts = pd.to_datetime(row.get(key), utc=True, errors="coerce")
        if ts is not pd.NaT and ts == ts:
            et = ts.tz_convert(config.ET)
            return et.strftime("%a %b %-d, %-I:%M %p"), ts.isoformat()
    return "", ""


def started(board: pd.DataFrame, now) -> pd.Series:
    """Games whose tip has passed - they keep the row published before it."""
    now = pd.Timestamp(now)
    return pd.Series([bool(iso) and pd.Timestamp(iso) <= now
                      for iso in (tip(r)[1] for r in board.to_dict("records"))],
                     index=board.index, dtype=bool)


def build_board(games: pd.DataFrame, days: int | None = None) -> pd.DataFrame:
    today = config.today_et()
    horizon = today + timedelta(days=config.BOARD_DAYS if days is None else days)
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    return g[(~g["completed"].astype(bool)) & (g["date"] >= today) & (g["date"] <= horizon)].copy()


def board_news(board: pd.DataFrame, report: pd.DataFrame, rosters: dict) -> dict:
    """{game_id: {"h": (report, roster), "a": (report, roster)}} for features.build."""
    by_team: dict[str, dict] = {}
    for r in report.itertuples(index=False):
        by_team.setdefault(r.team, {})[int(r.athlete_id)] = r.status
    return {str(g.game_id): {"h": (by_team.get(g.home_team, {}), rosters.get(g.home_team)),
                             "a": (by_team.get(g.away_team, {}), rosters.get(g.away_team))}
            for g in board.itertuples(index=False)}


def price_notes(notes: list[dict], model, side: str) -> list[dict]:
    """What the fitted no-market model charges for each absence or question mark, in points of
    margin to the player's own team.

    The ratings' own scale (``kappa``) says what an absence does to a rating update; the model
    on top of it decides what it does to a prediction, and on the 2019-26 data it charges the
    league's best players 4-8 points - the range the market prices them at. That is the number
    the card shows and the one the wait rule is measured in.
    """
    s = "h" if side == "h" else "a"
    c_net, c_off, c_top = (model.raw(f"{s}_d_net"), model.raw(f"{s}_d_off"),
                           model.raw(f"{s}_miss_top"))
    sign = 1.0 if s == "h" else -1.0
    out = []
    for n in notes:
        if "d_net" in n:
            n = {**n, "impact": round(sign * (c_net * n["d_net"] + c_off * n["d_off"]
                                              - c_top * n["d_net"]), 1)}
        out.append(n)
    return sorted(out, key=lambda n: -n["impact"] * n["p_out"])


def pending_points(notes: list[dict]) -> float:
    """How many points still hang on unresolved news: the largest impact among players who are
    neither confirmed in nor confirmed out."""
    return max([n["impact"] for n in notes if 0.0 < n["p_out"] < 1.0], default=0.0)


def price_game(g: dict, sig_win: float, sig_total: float) -> dict:
    """Every number the card shows, off the published margin and total."""
    out = {}
    thin, wait = bool(g.get("thin_data")), bool(g.get("injury_pending"))
    pm, pt = _f(g.get("pub_margin")), _f(g.get("pub_total"))
    ph = float(win_prob(pm, sig_win)[0])
    out.update(p_home_win=round(ph, 4), p_away_win=round(1 - ph, 4))

    # ---- spread -----------------------------------------------------------------------
    line = _f(g.get("spread_home"))
    if line == line:
        dis = _f(g.get("raw_margin")) - _f(g.get("mkt_margin"))
        lean = dis if dis == dis and dis != 0 else _f(g.get("nm_margin")) - _f(g.get("mkt_margin"))
        home = not (lean == lean and lean < 0)
        hc, push, ac = (float(x[0]) for x in cover(pm, sig_win, line))
        p_side = hc if home else ac
        price = _f(g.get("spread_home_dec") if home else g.get("spread_away_dec"))
        priced = price == price
        dec = price if priced else DEFAULT_DEC
        evv = ev(p_side, push, dec)
        mkt = _f(g.get("p_spread_home"))
        mkt_side = (mkt if home else 1 - mkt) if mkt == mkt else np.nan
        team = g["home_team"] if home else g["away_team"]
        side_line = line if home else -line
        st = _strength(abs(dis) if dis == dis else np.nan, config.SPREAD_EDGE_MIN, thin, wait)
        out.update(spread_side="home" if home else "away", spread_team=team,
                   spread_line=side_line, spread_pick=f"{team} {side_line:+g}" if side_line else f"{team} PK",
                   spread_disagree=round(dis, 2) if dis == dis else np.nan,
                   spread_p_win=round(p_side, 4), spread_p_push=round(push, 4),
                   spread_mkt_p=round(mkt_side, 4) if mkt_side == mkt_side else np.nan,
                   spread_dec=dec, spread_price=_american(dec) if priced else "",
                   spread_ev=round(float(evv), 4), spread_strength=st,
                   spread_stake=round(kelly(p_side, push, dec, config.KELLY_FRACTION)
                                      * config.BANKROLL_UNITS, 2) if st in PLAYABLE else 0.0)
    else:
        out.update(spread_side="", spread_team="", spread_line=np.nan, spread_pick="",
                   spread_disagree=np.nan, spread_p_win=np.nan, spread_p_push=np.nan,
                   spread_mkt_p=np.nan, spread_dec=np.nan, spread_price="", spread_ev=np.nan,
                   spread_strength="pass", spread_stake=0.0)

    # ---- total ------------------------------------------------------------------------
    tl = _f(g.get("total_line"))
    if tl == tl:
        dis = _f(g.get("raw_total")) - tl
        lean = dis if dis == dis and dis != 0 else _f(g.get("nm_total")) - tl
        over = not (lean == lean and lean < 0)
        po, pp, pu = (float(x[0]) for x in over_under(pt, sig_total, tl))
        p_side = po if over else pu
        price = _f(g.get("over_dec") if over else g.get("under_dec"))
        priced = price == price
        dec = price if priced else DEFAULT_DEC
        evv = ev(p_side, pp, dec)
        mkt = _f(g.get("p_over"))
        mkt_side = (mkt if over else 1 - mkt) if mkt == mkt else np.nan
        st = _strength(abs(dis) if dis == dis else np.nan, config.TOTAL_EDGE_MIN, thin, wait)
        out.update(total_side="over" if over else "under", total_line_used=tl,
                   total_pick=f"{'Over' if over else 'Under'} {tl:g}",
                   total_disagree=round(dis, 2) if dis == dis else np.nan,
                   total_p_win=round(p_side, 4), total_p_push=round(pp, 4),
                   total_mkt_p=round(mkt_side, 4) if mkt_side == mkt_side else np.nan,
                   total_dec=dec, total_price=_american(dec) if priced else "",
                   total_ev=round(float(evv), 4), total_strength=st,
                   total_stake=round(kelly(p_side, pp, dec, config.KELLY_FRACTION)
                                     * config.BANKROLL_UNITS, 2) if st in PLAYABLE else 0.0)
    else:
        out.update(total_side="", total_line_used=np.nan, total_pick="", total_disagree=np.nan,
                   total_p_win=np.nan, total_p_push=np.nan, total_mkt_p=np.nan, total_dec=np.nan,
                   total_price="", total_ev=np.nan, total_strength="pass", total_stake=0.0)

    # ---- moneyline --------------------------------------------------------------------
    hd, ad = _f(g.get("home_dec")), _f(g.get("away_dec"))
    if hd == hd and ad == ad:
        ev_h, ev_a = ev(ph, 0.0, hd), ev(1 - ph, 0.0, ad)
        home = ev_h >= ev_a
        p_side, dec, evv = (ph, hd, ev_h) if home else (1 - ph, ad, ev_a)
        team = g["home_team"] if home else g["away_team"]
        mkt = _f(g.get("p_home"))
        mkt_side = (mkt if home else 1 - mkt) if mkt == mkt else np.nan
        st = _strength(float(evv), config.ML_EV_MIN, thin, wait)
        out.update(ml_side="home" if home else "away", ml_team=team, ml_pick=f"{team} to win",
                   ml_p_win=round(p_side, 4),
                   ml_mkt_p=round(mkt_side, 4) if mkt_side == mkt_side else np.nan,
                   ml_dec=dec, ml_price=_american(dec), ml_ev=round(float(evv), 4),
                   ml_strength=st,
                   ml_stake=round(kelly(p_side, 0.0, dec, config.KELLY_FRACTION)
                                  * config.BANKROLL_UNITS, 2) if st in PLAYABLE else 0.0)
    else:
        out.update(ml_side="", ml_team="", ml_pick="", ml_p_win=np.nan, ml_mkt_p=np.nan,
                   ml_dec=np.nan, ml_price="", ml_ev=np.nan, ml_strength="pass", ml_stake=0.0)
    return out


def run(dry_run: bool = False, days: int | None = None) -> pd.DataFrame:
    today = config.today_et()
    models, meta = load_models()
    games = hoopr.update()
    board = build_board(games, days)
    if board.empty:
        log.info("no NBA games on the board for %s", today)
        return board

    snap = odds.match_games(odds.snapshot(), games)
    if not dry_run:
        odds.append_snapshot(snap)
    if len(snap):
        s = snap.dropna(subset=["game_id"]).drop_duplicates("game_id", keep="last")
        s = s.assign(game_id=s["game_id"].astype(str))
        board = board.merge(s[["game_id"] + MARKET_COLS], on="game_id", how="left")
    for c in MARKET_COLS:
        if c not in board:
            board[c] = np.nan
    # A game already under way keeps the row published before tip. Weekend matinees are
    # mid-game when the evening run fires, and re-pricing one then would replace the pick with
    # a pre-game model read against no price at all.
    live = started(board, config.now_et())
    if live.any():
        log.info("%d games already under way keep the rows published before tip", int(live.sum()))
        board = board[~live]
        if board.empty:
            return board

    # Who is playing: the day's injury report where it could be read, the last one logged today
    # where it could not, and the current rosters where hoopR has them.
    fresh = espn.injuries(write=not dry_run)
    report = espn.latest_injuries(fresh)
    report_ok = fresh is not None or len(report) > 0
    rosters = hoopr.rosters() if config.INJURIES_ON else {}
    news = board_news(board, report, rosters)

    # the report names a player the box scores may not have yet - a debut, a two-way call-up
    names = {**hoopr.load_names(),
             **{int(i): str(n) for i, n in zip(report.get("athlete_id", []), report.get("name", []))}}
    _, up, _, _ = build(games, hoopr.load_players(), history.load_lines(), upcoming=board,
                        board_news=news, names=names)
    up["mkt_margin"] = -pd.to_numeric(up["spread_home"], errors="coerce")
    up["mkt_total"] = pd.to_numeric(up["total_line"], errors="coerce")
    for t in ("margin", "total"):
        up[f"nm_{t}"] = models[f"{t}_nomarket"].predict(up)
        up[f"dev_{t}"] = up[f"nm_{t}"] - up[f"mkt_{t}"]
        has = up[f"mkt_{t}"].notna()
        raw = up[f"nm_{t}"].copy()
        if f"{t}_market" in models and has.any():
            raw[has] = up.loc[has, f"mkt_{t}"] + models[f"{t}_market"].predict(up[has])
        w = (meta.get("shrink") or {}).get(t, config.DEFAULT_SHRINK)
        up[f"raw_{t}"] = raw
        up[f"pub_{t}"] = np.where(has, up[f"mkt_{t}"] + w * (raw - up[f"mkt_{t}"]), raw)

    sig = meta.get("sigma") or {}
    sig_win, sig_total = float(sig.get("win", 12.0)), float(sig.get("total", 18.0))
    up = up.copy()
    m = models["margin_nomarket"]
    up["h_notes"] = [price_notes(n or [], m, "h") for n in up["h_notes"]]
    up["a_notes"] = [price_notes(n or [], m, "a") for n in up["a_notes"]]
    up["h_pending"] = [pending_points(n) for n in up["h_notes"]]
    up["a_pending"] = [pending_points(n) for n in up["a_notes"]]
    up["thin_data"] = (up["h_games"] < config.MIN_GAMES) | (up["a_games"] < config.MIN_GAMES)
    pend = np.maximum(up["h_pending"].astype(float), up["a_pending"].astype(float))
    # No report at all is not "nobody is hurt". The NHL board stakes on its own guess when the
    # goalie page is down, because the books price starters and knowing one barely moves the
    # market-aware model; here who plays is the largest thing the model knows, so no news means
    # nothing is staked - every pick still publishes, marked as waiting.
    if not report_ok and config.REQUIRE_NEWS:
        log.warning("no injury report today - every pick waits until one can be read")
    up["injury_pending"] = config.REQUIRE_NEWS & ((pend >= config.WAIT_POINTS) | (not report_ok))
    rows = []
    for g in up.to_dict("records"):
        r = price_game(g, sig_win, sig_total)
        label, iso = tip(g)
        r.update(tip_label=label, tip_sort=iso,
                 exp_home_pts=round((g["pub_total"] + g["pub_margin"]) / 2, 1),
                 exp_away_pts=round((g["pub_total"] - g["pub_margin"]) / 2, 1),
                 h_notes=json.dumps([{k: v for k, v in n.items() if k not in ("d_net", "d_off")}
                                     for n in g.get("h_notes") or []]),
                 a_notes=json.dumps([{k: v for k, v in n.items() if k not in ("d_net", "d_off")}
                                     for n in g.get("a_notes") or []]))
        rows.append(r)
    priced = pd.DataFrame(rows, index=up.index)
    keep = ["game_id", "season", "game_type", "date", "home_team", "away_team", "neutral",
            "venue_city", "h_games", "a_games", "h_rest", "a_rest", "h_b2b", "a_b2b", "h_3in4",
            "a_3in4", "h_km", "a_km", "a_tz", "altitude", "h_d_net", "a_d_net", "h_pending",
            "a_pending", "injury_pending", "thin_data", "nm_margin", "nm_total", "raw_margin",
            "raw_total", "pub_margin", "pub_total", "mkt_margin", "mkt_total"] + MARKET_COLS
    out = pd.concat([up[[c for c in keep if c in up.columns]], priced], axis=1)
    out = out.rename(columns={"p_home": "mkt_p_home", "p_over": "mkt_p_over",
                              "p_spread_home": "mkt_p_spread_home"})
    for c in ("nm_margin", "nm_total", "raw_margin", "raw_total", "pub_margin", "pub_total"):
        out[c] = out[c].astype(float).round(2)
    out["prediction_date"] = str(today)
    out["model_trained_at"] = meta.get("trained_at", "")
    out = out.sort_values(["date", "tip_sort", "game_id"]).reset_index(drop=True)
    log.info("%d games | spreads %s | totals %s | moneylines %s | %d waiting on injury news",
             len(out), out["spread_strength"].value_counts().to_dict(),
             out["total_strength"].value_counts().to_dict(),
             out["ml_strength"].value_counts().to_dict(), int(out["injury_pending"].sum()))

    if dry_run:
        cols = ["tip_label", "away_team", "home_team", "spread_home", "nm_margin", "pub_margin",
                "spread_pick", "spread_strength", "total_line", "nm_total", "pub_total",
                "total_pick", "total_strength", "p_home_win", "ml_pick", "ml_ev", "ml_strength"]
        print(out[[c for c in cols if c in out]].to_string(index=False))
        return out

    config.ensure_dirs()
    out["first_seen_at"] = str(today)
    for c in ("spread_home", "spread_line", "spread_dec", "total_line", "total_dec", "ml_dec",
              "mkt_p_home", "mkt_margin", "mkt_total"):
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
