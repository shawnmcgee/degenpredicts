"""Price today's (and tomorrow's) college basketball slate.

    python -m ncaab.predict              # normal run: update finals, price the slate, write picks
    python -m ncaab.predict --dry-run    # print, write nothing
    python -m ncaab.predict --days 0     # today only

Every game with a line gets the side of the spread and of the total the model thinks is likelier,
the score it projects, and how far it sits from the line - the same card the NFL board shows.

The models are the ones the bake-off chose (``data/ncaab/models/meta.json``), refitted here on
every Division I game through last night: nothing fitted is committed, so the models are always
current on the season in progress. With no ``meta.json`` yet - before the first training run -
the board still works off the rating engine alone, and says so in ``*_model``.

**A game that has tipped is never re-priced.** The slate drops anything under way, and a game's
row in ``picks.csv`` is only replaced while it is still to tip - so the evening run cannot
overwrite a noon game's pre-tip number with an in-play one.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config, models
from .features import TARGETS, build, d1_rows, market_line
from .sources import espn
from .train import features_for, job_name, load_meta

log = logging.getLogger("ncaab.predict")

DEFAULT_SIGMA = {"margin": 10.5, "total": 12.5}
FIRST_SEEN = ("first_seen_spread", "first_seen_total", "first_seen_at")


def _spec(meta: dict, kind: str, market: bool) -> dict | None:
    job = (meta.get("jobs") or {}).get(job_name(kind, market)) or {}
    if not job or job.get("skipped") or not job.get("chosen"):
        return None
    return job


def _fit_chosen(spec: dict | None, D: pd.DataFrame, kind: str, market: bool):
    """Refit the chosen model on every usable game. Falls back down the leaderboard if the
    chosen model cannot be built here (xgboost missing), and to the ratings if nothing can."""
    exp_col = f"exp_{kind}"
    if spec is None:
        return (None if market else models.Fitted("ratings", False, exp_col)), "ratings"
    order = [spec["chosen"]] + [r["model"] for r in spec.get("leaderboard", [])
                                if not r.get("reference")]
    pool = D[market_line(D, kind).notna()] if market else D
    if len(pool) < 200:
        log.warning("%s: only %d training rows - using the ratings", job_name(kind, market), len(pool))
        return (None if market else models.Fitted("ratings", False, exp_col)), "ratings"
    y = pool[TARGETS[kind]].to_numpy(float)
    if market:
        y = y - market_line(pool, kind).to_numpy(float)
    for name in order:
        if name in models.BASES and not models.available(name):
            continue
        m = models.Fitted(name, market, exp_col, spec.get("stack_weights"))
        if name == "stack" and not m.members():
            continue
        if name != spec["chosen"]:
            log.warning("%s: %s unavailable here - using %s, next on the leaderboard",
                        job_name(kind, market), spec["chosen"], name)
        return m.fit(pool[features_for(market)], y), name
    return (None if market else models.Fitted("ratings", False, exp_col)), "ratings"


def price(up: pd.DataFrame, D: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Add the model's numbers and the picks to the slate's feature rows. Pure given its inputs."""
    out = up.copy()
    for kind in ("margin", "total"):
        line = market_line(out, kind).to_numpy(float)
        nm, nm_name = _fit_chosen(_spec(meta, kind, False), D, kind, False)
        out[f"nm_{kind}"] = np.round(nm.predict(out[features_for(False)]), 2)
        mk_spec = _spec(meta, kind, True)
        mk, mk_name = _fit_chosen(mk_spec, D, kind, True)
        raw = out[f"nm_{kind}"].to_numpy(float).copy()
        used = np.full(len(out), f"{nm_name}/nomarket", dtype=object)
        has = np.isfinite(line)
        if mk is not None and has.any():
            X = out.loc[has, features_for(True)]
            raw[has] = line[has] + mk.predict(X)
            used[has] = f"{mk_name}/market"
        shrink = float((mk_spec or {}).get("shrink", config.DEFAULT_SHRINK))
        sigma = float((mk_spec or _spec(meta, kind, False) or {}).get("sigma",
                                                                      DEFAULT_SIGMA[kind]))
        pub = np.where(has, line + shrink * (raw - line), raw)
        out[f"{kind}_raw"] = np.round(raw, 1)
        out[f"{kind}_pred"] = np.round(pub, 1)
        out[f"{kind}_disagree"] = np.round(raw - line, 1)     # NaN where there is no line
        out[f"{kind}_sigma"] = sigma
        out[f"{kind}_model"] = used
        with np.errstate(invalid="ignore"):
            out[f"{kind}_p_over"] = np.round(norm.cdf((pub - line) / sigma), 4)

    # ---- the picks: the side of each line the model's raw number falls on -------------
    tl = out["total_line"]
    over = out["total_disagree"] > 0
    out["total_pick"] = np.where(tl.notna(), np.where(over, "Over", "Under"), None)
    out["total_p_win"] = np.where(over, out["total_p_over"], 1 - out["total_p_over"]).round(4)
    sp = out["spread_home"]
    took_home = out["margin_disagree"] > 0
    out["spread_side"] = np.where(sp.notna(), np.where(took_home, out["home_team"],
                                                       out["away_team"]), None)
    out["spread_number"] = np.where(took_home, sp, -sp)
    out["spread_pick"] = [_spread_label(s, n) for s, n in zip(out["spread_side"], out["spread_number"])]
    out["spread_p_win"] = np.where(took_home, out["margin_p_over"],
                                   1 - out["margin_p_over"]).round(4)
    out["thin_data"] = (out["h_games"] < config.MIN_GAMES) | (out["a_games"] < config.MIN_GAMES)
    # Every game gets a pick; nothing is staked. Kept for the chooser, which counts "play"s.
    out["spread_strength"] = np.where(sp.notna(), "lean", "none")
    out["total_strength"] = np.where(tl.notna(), "lean", "none")
    return out


def _spread_label(side, number) -> str:
    try:
        n = float(number)
    except (TypeError, ValueError):
        return ""
    if not side or n != n:
        return ""
    return f"{side} PK" if n == 0 else f"{side} {n:+g}"


def _tip_label(kickoff_utc) -> str:
    ts = pd.to_datetime(kickoff_utc, utc=True, errors="coerce")
    if ts is pd.NaT or ts != ts:
        return ""
    return ts.tz_convert(config.ET).strftime("%a %b %d, %-I:%M %p")


PICK_COLS = [
    "game_id", "season", "date", "tip_et", "kickoff_utc", "home_team", "away_team",
    "home_id", "away_id", "home_abbr", "away_abbr", "home_conf", "away_conf",
    "home_rank", "away_rank", "neutral_site", "conf_game", "postseason", "note",
    "h_games", "a_games", "h_rest", "a_rest", "h_new", "a_new",
    "provider", "spread_home", "spread_home_price", "spread_away_price", "open_spread_home",
    "total_line", "over_price", "under_price", "open_total",
    "exp_margin", "exp_total", "nm_margin", "nm_total",
    "margin_raw", "margin_pred", "margin_disagree", "margin_sigma", "margin_model", "margin_p_over",
    "total_raw", "total_pred", "total_disagree", "total_sigma", "total_model", "total_p_over",
    "spread_side", "spread_number", "spread_pick", "spread_p_win", "spread_strength",
    "total_pick", "total_p_win", "total_strength", "thin_data",
    "prediction_date", "predicted_at", "model_trained_at", *FIRST_SEEN,
]


def _carry_first_seen(prev: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Keep the number we FIRST published for a game across every later rewrite, so closing-line
    value is measured from it rather than from a number taken an hour before tip."""
    if not len(prev) or "first_seen_at" not in prev.columns:
        return out
    seen = (prev.dropna(subset=["first_seen_at"]).drop_duplicates("game_id", keep="first")
                .set_index("game_id"))
    for col in FIRST_SEEN:
        if col in seen.columns:
            earlier = out["game_id"].map(seen[col])
            out[col] = earlier.where(earlier.notna(), out[col])
    return out


def write_picks(out: pd.DataFrame) -> pd.DataFrame:
    config.ensure_dirs()
    out = out.reindex(columns=PICK_COLS)
    if config.PICKS.exists():
        prev = pd.read_csv(config.PICKS, dtype={"game_id": str, "home_id": str, "away_id": str})
        out = _carry_first_seen(prev, out)
        # only games still to tip are in `out`, so a started game's row is never touched
        prev = prev[~prev["game_id"].isin(out["game_id"])]
        allp = pd.concat([prev, out], ignore_index=True) if len(prev) else out
    else:
        allp = out
    allp.to_csv(config.PICKS, index=False)
    return out


def run(dry_run: bool = False, days: int | None = None, fetch=espn.fetch_day, now=None,
        update_games: bool = True) -> pd.DataFrame:
    today = config.today_et()
    meta = load_meta()
    if not meta:
        log.warning("no data/ncaab/models/meta.json yet - pricing off the ratings alone. "
                    "Run `python -m ncaab.train` (the ncaab-train workflow) to pick the models.")
    games = espn.update(recheck_days=3, fetch=fetch) if update_games else espn.load()
    slate = espn.slate(days, fetch=fetch, now=now)
    if slate.empty:
        log.info("no games to come on the board")
        return slate
    slate = slate[slate["spread_home"].notna() | slate["total_line"].notna()]
    if slate.empty:
        log.info("games to come, but none has a line yet")
        return slate

    train_rows, up, _ = build(games, slate)
    D = d1_rows(train_rows.dropna(subset=list(TARGETS.values()))) if len(train_rows) else train_rows
    out = price(up, D, meta)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["tip_et"] = out["kickoff_utc"].map(_tip_label)
    out["prediction_date"] = str(today)
    out["predicted_at"] = stamp
    out["model_trained_at"] = meta.get("trained_at", "")
    out["first_seen_spread"] = out["spread_home"]
    out["first_seen_total"] = out["total_line"]
    out["first_seen_at"] = stamp
    out = out.sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)

    log.info("%d games priced: %d spreads, %d totals; models %s / %s", len(out),
             int(out["spread_home"].notna().sum()), int(out["total_line"].notna().sum()),
             out["margin_model"].value_counts().to_dict(), out["total_model"].value_counts().to_dict())
    if dry_run:
        cols = ["tip_et", "away_team", "home_team", "spread_home", "margin_raw", "margin_disagree",
                "spread_pick", "total_line", "total_raw", "total_disagree", "total_pick"]
        print(out[cols].to_string(index=False))
        return out
    out = write_picks(out)
    log.info("wrote %d picks to %s", len(out), config.PICKS)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args(argv)
    run(dry_run=args.dry_run, days=args.days)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
