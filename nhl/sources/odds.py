"""Live prices from The Odds API (sport key ``icehockey_nhl``): moneyline, puck line, total.

One call a run returns every NHL game the books have posted, with three markets from each US
book. What comes back is turned into ONE market view per game:

* **probabilities are de-vigged per book, then averaged**, so a book with a fat margin does not
  drag the consensus toward itself;
* **the line is the modal line** across books (5.5 at three books and 6 at one is a 5.5 game),
  and only books at that line contribute to its price and probability;
* **prices are aggregated in decimal**, as medians. Never American: see :mod:`nhl.odds_math`.

The median price is what EV and stakes are computed against, not the best price on the board.
The best price is what you would actually bet, but it is also partly just the noisiest book,
and measuring an edge against it manufactures edge out of variance.

Every pull is appended to snapshots.csv. The morning run records the number we first published
(for closing-line value); the evening run, a couple of hours before puck drop, is the closest
thing to a closing line this free setup can see, and it becomes that game's row in lines.csv
once the game is final - which is how the training set keeps growing after the historical
import runs out.
"""
from __future__ import annotations

import logging
import statistics
from collections import Counter
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

from .. import config
from ..http import get
from ..odds_math import devig_two
from ..teams import UnknownTeam, canon, from_name
from . import espn_odds

log = logging.getLogger(__name__)
URL = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
SPORTS_URL = "https://api.the-odds-api.com/v4/sports"      # free: carries the quota headers
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
# A puck line is 1.5 (2.5 now and then); ESPN's flat `details` is sometimes the moneyline
# ("TOR -150"), and anything past this is that, not a handicap.
MAX_SPREAD = 3.0

SNAP_COLS = ["pulled_at", "event_id", "commence_time", "date", "home_team", "away_team",
             "home_dec", "away_dec", "p_home", "n_ml", "total_line", "over_dec", "under_dec",
             "p_over", "n_total", "pl_home", "pl_home_dec", "pl_away_dec", "p_pl_home", "n_pl"]


def _market(bm: dict, key: str) -> list[dict]:
    for m in bm.get("markets", []):
        if m.get("key") == key:
            return m.get("outcomes", [])
    return []


def parse_event(ev: dict) -> dict | None:
    """One Odds API event -> one consensus row. Pure; returns None for an unmappable team."""
    try:
        home, away = from_name(ev["home_team"]), from_name(ev["away_team"])
    except (UnknownTeam, KeyError) as e:
        log.warning("odds: %s", e)
        return None
    h_name, a_name = ev["home_team"], ev["away_team"]
    ml, tot, pl = [], [], []
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
            pl.append((float(o[h_name]["point"]), float(o[h_name]["price"]),
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
    pl_line, phd, pad, ppl, n_pl = consensus(pl, lined=True)
    ts = pd.Timestamp(ev["commence_time"])
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    et = ts.tz_convert(config.ET)
    return {"event_id": ev.get("id"), "commence_time": ts.isoformat(), "date": et.date(),
            "home_team": home, "away_team": away,
            "home_dec": hd, "away_dec": ad, "p_home": ph, "n_ml": n_ml,
            "total_line": tl, "over_dec": od, "under_dec": ud, "p_over": po, "n_total": n_t,
            "pl_home": pl_line, "pl_home_dec": phd, "pl_away_dec": pad, "p_pl_home": ppl,
            "n_pl": n_pl}


def credits_left() -> int | None:
    """Credits left on the shared key, from a call The Odds API does not charge for. None when
    the balance could not be read - which is not the same as zero."""
    r = get(SPORTS_URL, params={"apiKey": config.ODDS_API_KEY})
    try:
        return int(float(r.headers.get("x-requests-remaining")))
    except (AttributeError, TypeError, ValueError):
        return None


def use_odds_api() -> bool:
    """Spend credits this run? Not without a key, and not once fewer than ODDS_RESERVE are
    left: the other boards share the key, and running it dry takes their prices down too."""
    if not config.ODDS_API_KEY:
        return False
    left = credits_left()
    if left is not None and left < config.ODDS_RESERVE:
        log.warning("Odds API: %d credits left, under the %d-credit reserve - using ESPN's free "
                    "prices until they reset on the 1st", left, config.ODDS_RESERVE)
        return False
    return True


def snapshot() -> pd.DataFrame:
    """Every posted NHL game with its prices: The Odds API's consensus while the key has
    credits to spare, ESPN's free scoreboard (its partner book) once it is under the reserve or
    the call fails. Empty without a key."""
    if not config.ODDS_API_KEY:
        log.info("ODDS_API_KEY unset - no live prices, so nothing on the board can be staked")
        return pd.DataFrame(columns=SNAP_COLS)
    if use_odds_api():
        r = get(URL, params={"apiKey": config.ODDS_API_KEY, "regions": "us",
                             "markets": "h2h,spreads,totals", "oddsFormat": "decimal",
                             "dateFormat": "iso"})
        if r is not None and r.status_code == 200:
            return _oddsapi_frame(r)
        log.warning("Odds API unavailable (%s) - using ESPN's free prices this run",
                    getattr(r, "status_code", "no response"))
    if config.ODDS_FALLBACK != "espn":
        return pd.DataFrame(columns=SNAP_COLS)
    return espn_snapshot()


def _oddsapi_frame(r) -> pd.DataFrame:
    log.info("Odds API quota used=%s remaining=%s",
             r.headers.get("x-requests-used"), r.headers.get("x-requests-remaining"))
    now = config.now_et()
    pulled = now.astimezone(timezone.utc).isoformat(timespec="seconds")
    rows = [x for x in (parse_event(ev) for ev in r.json()) if x]
    # The feed also lists games already under way, at in-play prices. Those are not pre-game
    # lines: on a matinee day the evening run would otherwise price a game in its second period.
    live = [x for x in rows if pd.Timestamp(x["commence_time"]) <= now]
    if live:
        log.info("odds: skipped %d games already under way (in-play prices)", len(live))
    rows = [x for x in rows if pd.Timestamp(x["commence_time"]) > now]
    for x in rows:
        x["pulled_at"] = pulled
    return pd.DataFrame(rows, columns=SNAP_COLS)


def _club(team: dict) -> str | None:
    for name in (team["displayName"], team["abbreviation"]):
        try:
            return from_name(name) if name == team["displayName"] else canon(name)
        except UnknownTeam:
            continue
    log.warning("ESPN: unmapped NHL club %r (%s)", team["displayName"], team["abbreviation"])
    return None


def _espn_day(day) -> list[dict]:
    r = get(ESPN_URL, params={"dates": day.strftime("%Y%m%d")})
    if r is None or r.status_code != 200:
        return []
    try:
        return espn_odds.parse_scoreboard(r.json(), MAX_SPREAD)
    except ValueError:
        return []


def espn_snapshot() -> pd.DataFrame:
    """The same consensus rows from ESPN's scoreboard - one book, so every n_* is 1 - with
    prices turned to decimal and de-vigged the way the Odds API rows are."""
    today = config.today_et()
    events = {}
    for i in range(config.BOARD_DAYS + 1):
        for e in _espn_day(today + timedelta(days=i)):
            events[e["event_id"]] = e
    now = config.now_et()
    pulled = now.astimezone(timezone.utc).isoformat(timespec="seconds")
    rows, in_play = [], 0
    d = espn_odds.to_decimal
    for e in events.values():
        if not any(espn_odds.finite(e[k]) for k in ("home_ml", "total_line", "spread_home")):
            continue
        if e["state"] not in ("pre", "") or e["start"] <= now:
            in_play += 1
            continue
        home, away = _club(e["home"]), _club(e["away"])
        if home is None or away is None:
            continue
        hd, ad = d(e["home_ml"]), d(e["away_ml"])
        od, ud = d(e["over_price"]), d(e["under_price"])
        phd, pad = d(e["spread_home_price"]), d(e["spread_away_price"])
        tl, pl = e["total_line"], e["spread_home"]
        n = lambda *v: int(all(espn_odds.finite(x) for x in v))
        rows.append({"pulled_at": pulled, "event_id": f"espn:{e['event_id']}",
                     "commence_time": e["start"].isoformat(),
                     "date": e["start"].tz_convert(config.ET).date(),
                     "home_team": home, "away_team": away,
                     "home_dec": hd, "away_dec": ad, "p_home": devig_two(hd, ad), "n_ml": n(hd, ad),
                     "total_line": tl, "over_dec": od, "under_dec": ud,
                     "p_over": devig_two(od, ud), "n_total": n(tl, od, ud),
                     "pl_home": pl, "pl_home_dec": phd, "pl_away_dec": pad,
                     "p_pl_home": devig_two(phd, pad), "n_pl": n(pl, phd, pad)})
    if in_play:
        log.info("ESPN: skipped %d games already under way", in_play)
    log.info("ESPN scoreboard: %d NHL games priced, %d with a moneyline, %d with a puck line",
             len(rows), sum(r["n_ml"] for r in rows), sum(r["n_pl"] for r in rows))
    return pd.DataFrame(rows, columns=SNAP_COLS)


def match_games(snap: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach NHL game ids to snapshot rows by (date, home, away)."""
    if snap.empty:
        return snap.assign(game_id=pd.Series(dtype=str))
    g = games[["game_id", "date", "home_team", "away_team"]].copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    s = snap.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.date
    out = s.merge(g, on=["date", "home_team", "away_team"], how="left")
    miss = out["game_id"].isna()
    if miss.any():
        log.info("odds: %d posted games not in the NHL schedule cache: %s", int(miss.sum()),
                 [f"{a}@{h} {d}" for d, h, a in out.loc[miss, ["date", "home_team", "away_team"]]
                  .itertuples(index=False)][:8])
    return out


def append_snapshot(df: pd.DataFrame) -> None:
    if df.empty:
        return
    config.ensure_dirs()
    cols = ["game_id"] + SNAP_COLS
    df.reindex(columns=cols).to_csv(config.SNAPSHOTS, mode="a",
                                    header=not config.SNAPSHOTS.exists(), index=False)


def load_snapshots() -> pd.DataFrame:
    if not config.SNAPSHOTS.exists():
        return pd.DataFrame(columns=["game_id"] + SNAP_COLS)
    df = pd.read_csv(config.SNAPSHOTS, dtype={"game_id": str}, low_memory=False)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def last_before_start(snaps: pd.DataFrame) -> pd.DataFrame:
    """Per game, the latest snapshot taken before puck drop - our closing line."""
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
    """Promote each completed game's last pre-game snapshot into lines.csv.

    Only completed games: a line for a game still to be played would be a snapshot, not a
    close, and training on it would put a morning number where the history holds closing ones.
    """
    from .history import load_lines, upsert_lines
    snaps = last_before_start(load_snapshots())
    if snaps.empty:
        return 0
    done = set(games.loc[games["completed"].astype(bool), "game_id"].astype(str))
    s = snaps[snaps["game_id"].astype(str).isin(done)].copy()
    if s.empty:
        return 0
    meta = games.set_index(games["game_id"].astype(str))
    s["season"] = s["game_id"].map(meta["season"])
    s["source"] = "oddsapi_pregame"
    s["is_closing"] = s["minutes_before"] <= 120
    before = len(load_lines())
    upsert_lines(s)
    log.info("lines: promoted %d pre-game snapshots (%d rows before)", len(s), before)
    return len(s)
