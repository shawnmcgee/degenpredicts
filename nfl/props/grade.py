"""Grade logged prop rungs against what the players did.

    python -m nfl.props.grade

For every rung in ``data/nfl/props/ladders/``, the last quote pulled before kickoff is the
one graded - that is the price a reader could have taken. Outcomes come from the same
player-week files the models train on, so a rung grades as soon as nflverse publishes the
game's stats (usually the next morning).

Three questions, in the order they matter:

1. **Calibration against Kalshi** - Brier score of the model's probability against the Kalshi
   mid's on the same rungs. Lower is better; if the market's is lower, the market knows more.
2. **Flagged rungs** - how the side the board called cheap did, at the price it was cheap at.
   Nothing was staked; ``units`` is what one contract a flag would have made.
3. **Line movement** - whether the Kalshi mid moved toward the model's side between the first
   pull and the last one before kickoff: prop CLV, the number that settles things fastest.

A player who did not play grades as zero, because Kalshi's rules read "records N+": he
recorded nothing.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from .. import config
from ..sources import kalshi
from . import data

log = logging.getLogger("nfl.props.grade")

STAT = {"pass_yds": "passing_yards", "rush_yds": "rushing_yards", "rec_yds": "receiving_yards",
        "receptions": "receptions", "anytime_td": "any_td"}


def _ladders() -> pd.DataFrame:
    frames = []
    for p in sorted(config.PROPS_LADDERS.glob("*.csv.gz")):
        try:
            frames.append(pd.read_csv(p, dtype={"game_id": str, "player_id": str}))
        except Exception as e:
            log.warning("could not read %s: %s", p.name, e)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def grade() -> pd.DataFrame:
    lad = _ladders()
    if lad.empty:
        log.info("no prop rungs logged yet")
        return lad
    lad["pulled_at"] = pd.to_datetime(lad["pulled_at"], utc=True, errors="coerce")
    lad["kickoff"] = pd.to_datetime(lad["kickoff_utc"], utc=True, errors="coerce")
    lad = lad[lad["kickoff"].isna() | (lad["pulled_at"] < lad["kickoff"])]
    lad = lad.sort_values("pulled_at")
    first = lad.groupby("ticker").first()
    last = lad.groupby("ticker").last()
    g = last.copy()
    g["first_mid"] = (first["yes_bid"] + first["yes_ask"]) / 2
    g["last_mid"] = (g["yes_bid"] + g["yes_ask"]) / 2
    g = g.reset_index()

    players = data.load(first=config.season_of(config.today_et()) - 1)
    if players.empty:
        return pd.DataFrame()
    players["any_td"] = ((players["rushing_tds"] + players["receiving_tds"]) > 0).astype(int)
    played_games = set(players["game_id"].astype(str))
    stats = players.set_index(["game_id", "player_id"])
    done = g[g["game_id"].astype(str).isin(played_games)].copy()
    if done.empty:
        log.info("%d rungs logged, none from a game with stats yet", len(g))
        return done
    vals = []
    for r in done.itertuples():
        key = (str(r.game_id), str(r.player_id))
        vals.append(float(stats.loc[key, STAT[r.market]]) if key in stats.index else 0.0)
    done["actual"] = vals
    done["hit"] = (done["actual"] >= done["k"]).astype(int)
    done["mid"] = done["last_mid"]
    side_yes = done["flag"] == "yes"
    side_no = done["flag"] == "no"
    cost_yes = done["yes_ask"] + done["yes_ask"].map(lambda p: kalshi.fee(p) if p == p else np.nan)
    no_px = 1 - done["yes_bid"]
    cost_no = no_px + no_px.map(lambda p: kalshi.fee(p) if p == p else np.nan)
    done["units"] = np.select(
        [side_yes, side_no],
        [np.where(done["hit"] == 1, 1 - cost_yes, -cost_yes),
         np.where(done["hit"] == 0, 1 - cost_no, -cost_no)], default=0.0)
    # prop CLV in cents: the mid moving toward the model's side after the first pull
    toward = np.sign(done["p_yes"] - done["first_mid"])
    done["clv_cents"] = np.round(100 * toward * (done["last_mid"] - done["first_mid"]), 2)
    config.PROPS.mkdir(parents=True, exist_ok=True)
    done.to_csv(config.PROPS_RESULTS, index=False)
    metrics(done)
    return done


def metrics(done: pd.DataFrame) -> dict:
    out = {"updated": str(config.today_et()), "n_rungs": int(len(done))}
    ok = done.dropna(subset=["p_yes", "mid"])
    if len(ok):
        out["brier_model"] = round(float(np.mean((ok["p_yes"] - ok["hit"]) ** 2)), 4)
        out["brier_kalshi"] = round(float(np.mean((ok["mid"] - ok["hit"]) ** 2)), 4)
        out["clv_cents"] = round(float(ok["clv_cents"].mean()), 2)
    flagged = done[done["flag"].isin(["yes", "no"])]
    if len(flagged):
        won = np.where(flagged["flag"] == "yes", flagged["hit"] == 1, flagged["hit"] == 0)
        out["flagged"] = {"n": int(len(flagged)), "wins": int(won.sum()),
                          "win_pct": round(100 * float(won.mean()), 1),
                          "units": round(float(flagged["units"].sum()), 2),
                          "clv_cents": round(float(flagged["clv_cents"].mean()), 2)}
    out["by_market"] = {}
    for m, g in ok.groupby("market"):
        out["by_market"][m] = {"n": int(len(g)),
                               "brier_model": round(float(np.mean((g["p_yes"] - g["hit"]) ** 2)), 4),
                               "brier_kalshi": round(float(np.mean((g["mid"] - g["hit"]) ** 2)), 4)}
    (config.PROPS / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    log.info("props graded: %s", {k: v for k, v in out.items() if k != "by_market"})
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    grade()
