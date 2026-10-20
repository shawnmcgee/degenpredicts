"""Grade logged NBA prop rungs against what the players did.

    python -m nba.props.grade

For every rung in ``data/nba/props/ladders/``, the last quote pulled before tip is the one
graded - the price a reader could have taken. Outcomes come from the same player files the
models train on, so a rung grades as soon as hoopR publishes the night's box scores.

Three questions, in the order they matter (the NFL props grade's, unchanged):

1. **Calibration against Kalshi** - Brier score of the model's own probability, of the published
   one and of Kalshi's mid, on the same rungs. If the market's is lowest, the market knows more.
   ``shrink`` reports how far toward the model the graded rungs say to move - the number
   ``DEGEN_NBA_PROPS_SHRINK`` should be set from once there are enough of them.
2. **Flagged rungs** - how the side the board called cheap did, at the price it was cheap at.
   Nothing was staked; ``units`` is what one contract on each flag would have made.
3. **Line movement** - whether Kalshi's mid moved toward the model between the first pull and
   the last one before tip: prop CLV, the number that settles things fastest.

A player who did not play grades as zero, because Kalshi settles "records N+": he recorded
nothing.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from .. import config
from ..sources import hoopr, kalshi
from .features import TARGETS

log = logging.getLogger("nba.props.grade")
MIN_SHRINK_RUNGS = 2000


def _ladders() -> pd.DataFrame:
    frames = []
    for p in sorted(config.PROPS_LADDERS.glob("*.csv.gz")):
        try:
            frames.append(pd.read_csv(p, dtype={"game_id": str}))
        except Exception as e:
            log.warning("could not read %s: %s", p.name, e)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def outcomes(players: pd.DataFrame) -> dict:
    """{(game_id, athlete_id): {market: stat}} for every player who logged minutes."""
    p = players.copy()
    p["reb"] = pd.to_numeric(p["oreb"], errors="coerce") + pd.to_numeric(p["dreb"], errors="coerce")
    out = {}
    for r in p.itertuples(index=False):
        out[(str(r.game_id), int(r.athlete_id))] = {m: float(getattr(r, c)) for m, c in TARGETS.items()}
    return out


def grade() -> pd.DataFrame:
    lad = _ladders()
    if lad.empty:
        log.info("no prop rungs logged yet")
        return lad
    lad["pulled_at"] = pd.to_datetime(lad["pulled_at"], utc=True, errors="coerce")
    lad["tip"] = pd.to_datetime(lad["tip_utc"], utc=True, errors="coerce")
    lad = lad[lad["tip"].isna() | (lad["pulled_at"] < lad["tip"])].sort_values("pulled_at")
    first, last = lad.groupby("ticker").first(), lad.groupby("ticker").last()
    g = last.copy()
    g["first_mid"] = (first["yes_bid"] + first["yes_ask"]) / 2
    g["last_mid"] = (g["yes_bid"] + g["yes_ask"]) / 2
    g = g.reset_index()

    games = hoopr.load_games()
    done_ids = set(games.loc[games["completed"].astype(bool), "game_id"].astype(str))
    season = config.season_of(config.today_et())
    players = hoopr.load_players(seasons=[season - 1, season])
    have_box = set(players["game_id"].astype(str))
    done = g[g["game_id"].astype(str).isin(done_ids & have_box)].copy()
    if done.empty:
        log.info("%d rungs logged, none from a game with a box score yet", len(g))
        return done
    stats = outcomes(players)
    done["actual"] = [stats.get((str(r.game_id), int(r.athlete_id)), {}).get(r.market, 0.0)
                      for r in done.itertuples()]
    done["hit"] = (done["actual"] >= done["k"]).astype(int)
    done["mid"] = done["last_mid"]
    cost_yes = done["yes_ask"].map(kalshi.cost)
    cost_no = (1 - done["yes_bid"]).map(kalshi.cost)
    done["units"] = np.select(
        [done["flag"] == "yes", done["flag"] == "no"],
        [np.where(done["hit"] == 1, 1 - cost_yes, -cost_yes),
         np.where(done["hit"] == 0, 1 - cost_no, -cost_no)], default=0.0)
    toward = np.sign(done["p_model"] - done["first_mid"])
    done["clv_cents"] = np.round(100 * toward * (done["last_mid"] - done["first_mid"]), 2)
    config.PROPS.mkdir(parents=True, exist_ok=True)
    done.to_csv(config.PROPS_RESULTS, index=False)
    metrics(done)
    return done


def fit_shrink(ok: pd.DataFrame) -> dict:
    """How far toward the model the logged rungs say Kalshi's price should move. Reported,
    never applied automatically: set DEGEN_NBA_PROPS_SHRINK from it once `ready`."""
    from .board import publish
    y = ok["hit"].to_numpy(float)
    best, best_ll = 0.0, np.inf
    for w in np.round(np.arange(0, 1.0001, 0.05), 2):
        p = np.clip(publish(ok["p_model"], ok["mid"], w), 1e-4, 1 - 1e-4)
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if ll < best_ll:
            best, best_ll = float(w), ll
    return {"best": best, "n": int(len(ok)), "ready": bool(len(ok) >= MIN_SHRINK_RUNGS),
            "in_use": config.PROPS_SHRINK}


def metrics(done: pd.DataFrame) -> dict:
    out = {"updated": str(config.today_et()), "n_rungs": int(len(done))}
    ok = done.dropna(subset=["p_model", "mid"])
    ok = ok[(ok["mid"] > 0) & (ok["mid"] < 1)]
    if len(ok):
        b = lambda p: round(float(np.mean((p - ok["hit"]) ** 2)), 4)
        out.update(brier_model=b(ok["p_model"]), brier_published=b(ok["p_yes"]),
                   brier_kalshi=b(ok["mid"]), clv_cents=round(float(ok["clv_cents"].mean()), 2),
                   shrink=fit_shrink(ok))
    flagged = done[done["flag"].isin(["yes", "no"])]
    if len(flagged):
        won = np.where(flagged["flag"] == "yes", flagged["hit"] == 1, flagged["hit"] == 0)
        out["flagged"] = {"n": int(len(flagged)), "wins": int(won.sum()),
                          "win_pct": round(100 * float(won.mean()), 1),
                          "units": round(float(flagged["units"].sum()), 2),
                          "clv_cents": round(float(flagged["clv_cents"].mean()), 2)}
    out["by_market"] = {m: {"n": int(len(x)),
                            "brier_model": round(float(np.mean((x["p_model"] - x["hit"]) ** 2)), 4),
                            "brier_kalshi": round(float(np.mean((x["mid"] - x["hit"]) ** 2)), 4)}
                        for m, x in ok.groupby("market")}
    (config.PROPS / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    log.info("props graded: %s", {k: v for k, v in out.items() if k != "by_market"})
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    grade()
