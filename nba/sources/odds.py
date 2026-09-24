"""Live prices, from ESPN's book by default or The Odds API's consensus on request.

One snapshot row per game, whichever source produced it:

    spread_home / spread_home_dec / spread_away_dec   the home line (book convention: -6.5
                                                       means home gives 6.5) and both prices
    total_line / over_dec / under_dec                  the total and both prices
    home_dec / away_dec                                the moneyline
    p_*                                                the same markets de-vigged

``DEGEN_NBA_ODDS`` picks the source: ``espn`` (the default - free, one book, see
:mod:`nba.sources.espn`), ``oddsapi`` (``basketball_nba``, 3 credits a run from the key the
other boards share) or ``none``. The Odds API is opt-in because the other four boards already
use most of the free tier in the months they overlap with basketball, and a board that quietly
exhausts the shared quota takes their prices down with it.

From The Odds API the consensus is formed the way the NHL board forms it:

* **probabilities are de-vigged per book, then averaged**, so a book with a fat margin does not
  drag the consensus toward itself;
* **the line is the modal line** across books, and only books at that line contribute to its
  price and probability - a -6.5 at three books and a -7 at one is a -6.5 game;
* **prices are aggregated in decimal**, as medians. Never American: see :mod:`nba.odds_math`.

Every pull is appended to snapshots.csv. The morning run records the number we first published;
the evening run, an hour or two before tip, is the closest thing to a closing line this free
setup sees, and it becomes that game's row in lines.csv once the game is final - which is how
the training set keeps growing after the historical import runs out.
"""
from __future__ import annotations

import logging
import statistics
from collections import Counter
from datetime import timezone

import numpy as np
import pandas as pd

from .. import config
from ..http import get
from ..odds_math import devig_two
from ..teams import UnknownTeam, from_name

log = logging.getLogger(__name__)
URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

SNAP_COLS = ["pulled_at", "source", "game_id", "event_id", "commence_time", "date", "home_team",
             "away_team", "provider", "spread_home", "spread_home_dec", "spread_away_dec",
             "p_spread_home", "n_spread", "total_line", "over_dec", "under_dec", "p_over",
             "n_total", "home_dec", "away_dec", "p_home", "n_ml", "open_spread_home",
             "open_total"]


def _market(bm: dict, key: str) -> list[dict]:
    for m in bm.get("markets", []):
        if m.get("key") == key:
            return m.get("outcomes", [])
    return []


def parse_event(ev: dict) -> dict | None:
    """One Odds API event -> one consensus row. Pure; None for an unmappable team."""
    try:
        home, away = from_name(ev["home_team"]), from_name(ev["away_team"])
    except (UnknownTeam, KeyError) as e:
        log.warning("odds: %s", e)
        return None
    h_name, a_name = ev["home_team"], ev["away_team"]
    ml, tot, spr = [], [], []
    for bm in ev.get("bookmakers", []):
        o = {x.get("name"): x for x in _market(bm, "h2h")}
        if h_name in o and a_name in o:
            ml.append((float(o[h_name]["price"]), float(o[a_name]["price"])))
        o = {x.get("name"): x for x in _market(bm, "totals")}
        if "Over" in o and "Under" in o and o["Over"].get("point") is not None:
            tot.append((float(o["Over"]["point"]), float(o["Over"]["price"]),
                        float(o["Under"]["price"])))
        o = {x.get("name"): x for x in _market(bm, "spreads")}
        if h_name in o and a_name in o and o[h_name].get("point") is not None:
            spr.append((float(o[h_name]["point"]), float(o[h_name]["price"]),
                        float(o[a_name]["price"])))

    def consensus(rows, lined: bool):
        if not rows:
            return (np.nan,) * 4 + (0,)
        if lined:
            line = Counter(r[0] for r in rows).most_common(1)[0][0]
            rows = [r for r in rows if r[0] == line]
            a, b = [r[1] for r in rows], [r[2] for r in rows]
        else:
            line = np.nan
            a, b = [r[0] for r in rows], [r[1] for r in rows]
        p = [devig_two(x, y) for x, y in zip(a, b)]
        p = [v for v in p if v == v]
        return (line, statistics.median(a), statistics.median(b),
                float(np.mean(p)) if p else np.nan, len(rows))

    _, hd, ad, ph, n_ml = consensus(ml, lined=False)
    tl, od, ud, po, n_t = consensus(tot, lined=True)
    sl, shd, sad, ps, n_s = consensus(spr, lined=True)
    ts = pd.Timestamp(ev["commence_time"])
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    return {"event_id": ev.get("id"), "commence_time": ts.isoformat(),
            "date": ts.tz_convert(config.ET).date(), "home_team": home, "away_team": away,
            "provider": "consensus", "spread_home": sl, "spread_home_dec": shd,
            "spread_away_dec": sad, "p_spread_home": ps, "n_spread": n_s, "total_line": tl,
            "over_dec": od, "under_dec": ud, "p_over": po, "n_total": n_t, "home_dec": hd,
            "away_dec": ad, "p_home": ph, "n_ml": n_ml}


def oddsapi_rows() -> pd.DataFrame:
    if not config.ODDS_API_KEY:
        log.info("DEGEN_NBA_ODDS=oddsapi but ODDS_API_KEY is unset - no prices")
        return pd.DataFrame(columns=SNAP_COLS)
    r = get(URL, params={"apiKey": config.ODDS_API_KEY, "regions": "us",
                         "markets": "h2h,spreads,totals", "oddsFormat": "decimal",
                         "dateFormat": "iso"})
    if r is None or r.status_code != 200:
        log.warning("Odds API unavailable: %s", getattr(r, "status_code", "no response"))
        return pd.DataFrame(columns=SNAP_COLS)
    log.info("Odds API quota used=%s remaining=%s",
             r.headers.get("x-requests-used"), r.headers.get("x-requests-remaining"))
    now = config.now_et()
    rows = [x for x in (parse_event(ev) for ev in r.json()) if x]
    # The feed also lists games already under way, at in-play prices. Those are not pre-game
    # numbers: on a weekend the evening run would otherwise price a matinee in its third quarter.
    live = [x for x in rows if pd.Timestamp(x["commence_time"]) <= now]
    if live:
        log.info("odds: skipped %d games already under way (in-play prices)", len(live))
    rows = [x for x in rows if pd.Timestamp(x["commence_time"]) > now]
    df = pd.DataFrame(rows)
    df["pulled_at"] = now.astimezone(timezone.utc).isoformat(timespec="seconds")
    df["source"] = "oddsapi"
    return df.reindex(columns=SNAP_COLS)


def snapshot() -> pd.DataFrame:
    """Every posted game that has not tipped, from the configured source. Empty on failure."""
    src = config.ODDS_SOURCE
    if src == "none":
        return pd.DataFrame(columns=SNAP_COLS)
    if src == "oddsapi":
        return oddsapi_rows()
    if src != "espn":
        log.warning("unknown DEGEN_NBA_ODDS=%r - using espn", src)
    from .espn import odds_rows
    try:
        return odds_rows()
    except Exception as e:                      # a third-party layout change must not kill the run
        log.warning("ESPN prices unavailable (%s) - no prices this run", e)
        return pd.DataFrame(columns=SNAP_COLS)


def match_games(snap: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach game ids: ESPN rows carry them already; Odds API rows join on (date, home, away)."""
    if snap.empty:
        return snap.reindex(columns=SNAP_COLS)
    s = snap.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.date
    g = games[["game_id", "date", "home_team", "away_team"]].copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    g["game_id"] = g["game_id"].astype(str)
    have = s["game_id"].notna() & (s["game_id"].astype(str) != "") & \
        s["game_id"].astype(str).isin(set(g["game_id"]))
    joined = s[~have].drop(columns=["game_id"]).merge(g, on=["date", "home_team", "away_team"],
                                                      how="left")
    out = pd.concat([s[have], joined], ignore_index=True)
    miss = out["game_id"].isna()
    if miss.any():
        log.info("odds: %d posted games not in the schedule cache: %s", int(miss.sum()),
                 [f"{a}@{h} {d}" for d, h, a in out.loc[miss, ["date", "home_team", "away_team"]]
                  .itertuples(index=False)][:8])
    out["game_id"] = out["game_id"].astype("string")
    return out.reindex(columns=SNAP_COLS)


def append_snapshot(df: pd.DataFrame) -> None:
    if df.empty:
        return
    config.ensure_dirs()
    df.reindex(columns=SNAP_COLS).to_csv(config.SNAPSHOTS, mode="a",
                                         header=not config.SNAPSHOTS.exists(), index=False)


def load_snapshots() -> pd.DataFrame:
    if not config.SNAPSHOTS.exists():
        return pd.DataFrame(columns=SNAP_COLS)
    df = pd.read_csv(config.SNAPSHOTS, dtype={"game_id": str, "event_id": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def last_before_start(snaps: pd.DataFrame) -> pd.DataFrame:
    """Per game, the latest snapshot taken before tip - our closing line."""
    if snaps.empty:
        return snaps
    s = snaps.dropna(subset=["game_id"]).copy()
    s["_pulled"] = pd.to_datetime(s["pulled_at"], utc=True, errors="coerce")
    s["_start"] = pd.to_datetime(s["commence_time"], utc=True, errors="coerce")
    s = s[s["_pulled"] < s["_start"]]
    s = s.sort_values("_pulled").drop_duplicates("game_id", keep="last")
    s["minutes_before"] = (s["_start"] - s["_pulled"]).dt.total_seconds() / 60
    return s.drop(columns=["_pulled", "_start"])


def consolidate(games: pd.DataFrame) -> int:
    """Promote each completed game's last pre-tip snapshot into lines.csv.

    Only completed games: a line for a game still to be played would be a snapshot, not a
    close, and training on it would put a morning number where the history holds closing ones.
    A game the historical archives already close is left alone.
    """
    from .history import load_lines, upsert_lines
    snaps = last_before_start(load_snapshots())
    if snaps.empty:
        return 0
    done = set(games.loc[games["completed"].astype(bool), "game_id"].astype(str))
    s = snaps[snaps["game_id"].astype(str).isin(done)].copy()
    old = load_lines()
    archived = set(old.loc[old["source"].astype(str).str.contains("close"), "game_id"].astype(str))
    s = s[~s["game_id"].astype(str).isin(archived)]
    if s.empty:
        return 0
    meta = games.set_index(games["game_id"].astype(str))
    s["season"] = s["game_id"].astype(str).map(meta["season"])
    s["source"] = s["source"].astype(str) + "_pregame"
    s["is_closing"] = s["minutes_before"] <= 120
    s["n_books"] = s["n_spread"]
    s["ml_source"] = s["source"]
    upsert_lines(s)
    log.info("lines: promoted %d pre-tip snapshots (%d rows before)", len(s), len(old))
    return len(s)
