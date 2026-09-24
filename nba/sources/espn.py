"""Today's slate, the ESPN book's prices, and the league injury report. No key, no quota.

    python -m nba.sources.espn               # print today's parsed scoreboard and report

Two endpoints of ESPN's public JSON API:

| Endpoint | What it gives |
|---|---|
| ``scoreboard?dates=YYYYMMDD`` | every game that day: ESPN's game id (hoopR's too), tip time, state (pre / in / post), and the prices of ESPN's partner book - spread, total and moneyline, each with an opening and a current number |
| ``injuries`` | every club's injury list: player id, status (Out, Doubtful, Questionable, Day-To-Day, Probable), and what is wrong |

**Why ESPN's prices by default.** One book is not a consensus, but it is a real price you could
bet, it costs no quota, and the four other boards already use most of the free Odds API tier.
``DEGEN_NBA_ODDS=oddsapi`` switches the board to The Odds API's consensus instead (see
:mod:`nba.sources.odds`); both produce the same snapshot rows.

**Written without reaching ESPN from the machine it was built on.** The field names follow the
two shapes ESPN's scoreboard is known to use for odds - a flat one (``spread``, ``overUnder``,
``homeTeamOdds.moneyLine``) and a nested one (``pointSpread.home.close.line``,
``total.over.close.line``, ``moneyline.home.close.odds``) - and both are parsed, preferring the
nested current price. ``tests/test_nba.py`` parses payloads of both shapes, but the first
Actions run is the real test: look for ``ESPN scoreboard: N games`` and ``injury report: N
players`` in the log. Every failure degrades to "no prices" or "no news", never to a crash.

Player and game ids are ESPN's, the same ids hoopR publishes, so the injury report joins the
player book on ids with no name matching at all.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

from .. import config
from ..http import get_json
from ..odds_math import american_to_decimal, devig_two
from ..teams import UnknownTeam, canon, from_name

log = logging.getLogger(__name__)

INJURY_COLS = ["pulled_at", "team", "athlete_id", "name", "status", "p_out", "detail"]
STATUS = {"out": "out", "o": "out", "suspension": "suspension", "suspended": "suspension",
          "doubtful": "doubtful", "d": "doubtful", "questionable": "questionable",
          "q": "questionable", "game time decision": "questionable", "gtd": "questionable",
          "day-to-day": "day-to-day", "day to day": "day-to-day", "dtd": "day-to-day",
          "probable": "probable", "p": "probable", "available": "available",
          "active": "available"}


def _num(x):
    """'+7.5', 'o226.5', '-110', 'EVEN', 7.5 -> float; anything else -> nan."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x) if x == x else np.nan
    s = str(x).strip().lower()
    if s in ("even", "ev", "pk", "pick", "pick'em"):
        return 0.0 if s in ("pk", "pick", "pick'em") else 100.0
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


def _pick(d: dict, *keys):
    """The first of `keys` present on `d` (current before close before open)."""
    for k in keys:
        if isinstance(d, dict) and isinstance(d.get(k), dict):
            return d[k]
    return {}


def _team(c: dict) -> str | None:
    t = c.get("team") or {}
    for key in ("abbreviation", "displayName", "name"):
        v = t.get(key)
        if not v:
            continue
        try:
            return canon(v) if key == "abbreviation" else from_name(v)
        except UnknownTeam:
            continue
    return None


def parse_odds(odds: list, home_abbr: str) -> dict:
    """The first priced entry of an event's ``odds`` list -> one row of prices. Pure."""
    out = {"provider": "", "spread_home": np.nan, "spread_home_dec": np.nan,
           "spread_away_dec": np.nan, "total_line": np.nan, "over_dec": np.nan,
           "under_dec": np.nan, "home_dec": np.nan, "away_dec": np.nan,
           "open_spread_home": np.nan, "open_total": np.nan}
    for o in odds or []:
        if not isinstance(o, dict):
            continue
        out["provider"] = str((o.get("provider") or {}).get("name") or "")
        # ---- nested shape: pointSpread / total / moneyline, each with open and close -------
        ps, tot, ml = o.get("pointSpread") or {}, o.get("total") or {}, o.get("moneyline") or {}
        hc = _pick(ps.get("home") or {}, "current", "close", "open")
        ac = _pick(ps.get("away") or {}, "current", "close", "open")
        out["spread_home"] = _num(hc.get("line"))
        out["spread_home_dec"] = american_to_decimal(_num(hc.get("odds")))
        out["spread_away_dec"] = american_to_decimal(_num(ac.get("odds")))
        out["open_spread_home"] = _num(_pick(ps.get("home") or {}, "open").get("line"))
        oc = _pick(tot.get("over") or {}, "current", "close", "open")
        uc = _pick(tot.get("under") or {}, "current", "close", "open")
        out["total_line"] = _num(oc.get("line"))
        out["over_dec"] = american_to_decimal(_num(oc.get("odds")))
        out["under_dec"] = american_to_decimal(_num(uc.get("odds")))
        out["open_total"] = _num(_pick(tot.get("over") or {}, "open").get("line"))
        out["home_dec"] = american_to_decimal(_num(_pick(ml.get("home") or {}, "current", "close",
                                                         "open").get("odds")))
        out["away_dec"] = american_to_decimal(_num(_pick(ml.get("away") or {}, "current", "close",
                                                         "open").get("odds")))
        # ---- flat shape: spread / overUnder / details / team odds ---------------------------
        if out["total_line"] != out["total_line"]:
            out["total_line"] = _num(o.get("overUnder"))
        if out["spread_home"] != out["spread_home"]:
            out["spread_home"] = _flat_spread(o, home_abbr)
        hto, ato = o.get("homeTeamOdds") or {}, o.get("awayTeamOdds") or {}
        if out["home_dec"] != out["home_dec"]:
            out["home_dec"] = american_to_decimal(_num(hto.get("moneyLine")))
            out["away_dec"] = american_to_decimal(_num(ato.get("moneyLine")))
        if out["spread_home_dec"] != out["spread_home_dec"]:
            out["spread_home_dec"] = american_to_decimal(_num(hto.get("spreadOdds")))
            out["spread_away_dec"] = american_to_decimal(_num(ato.get("spreadOdds")))
        if out["spread_home"] == out["spread_home"] or out["total_line"] == out["total_line"]:
            break
    return out


def _flat_spread(o: dict, home_abbr: str) -> float:
    """The home line from ``details`` ("BOS -7.5" names the favourite), cross-checked with the
    favourite flags; ``spread`` alone is not trusted for its sign."""
    det = str(o.get("details") or "").strip()
    m = re.match(r"^([A-Za-z]{2,4})\s+([-+]?\d+(\.\d+)?)$", det)
    if m:
        try:
            fav = canon(m.group(1))
        except UnknownTeam:
            fav = None
        mag = abs(float(m.group(2)))
        if fav is not None:
            return -mag if fav == home_abbr else mag
    if det.upper() in ("EVEN", "PK", "PICK"):
        return 0.0
    s = _num(o.get("spread"))
    if s != s:
        return np.nan
    home_fav = bool((o.get("homeTeamOdds") or {}).get("favorite"))
    away_fav = bool((o.get("awayTeamOdds") or {}).get("favorite"))
    if home_fav != away_fav:
        return -abs(s) if home_fav else abs(s)
    return np.nan


def parse_scoreboard(payload: dict) -> pd.DataFrame:
    """Scoreboard JSON -> one row per game with state and prices. Pure."""
    rows = []
    for ev in (payload or {}).get("events", []) or []:
        comp = (ev.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors", []) or []}
        if "home" not in sides or "away" not in sides:
            continue
        home, away = _team(sides["home"]), _team(sides["away"])
        if home is None or away is None:
            log.info("ESPN: skipped %s - a side maps to no NBA club", ev.get("shortName"))
            continue
        st = ((comp.get("status") or ev.get("status") or {}).get("type") or {})
        ts = pd.to_datetime(comp.get("date") or ev.get("date"), utc=True, errors="coerce")
        row = {"game_id": str(ev.get("id") or comp.get("id")), "home_team": home, "away_team": away,
               "commence_time": ts.isoformat() if ts is not pd.NaT else "",
               "date": ts.tz_convert(config.ET).date() if ts is not pd.NaT else None,
               "state": str(st.get("state") or ""), "completed": bool(st.get("completed")),
               **parse_odds(comp.get("odds") or ev.get("odds") or [], home)}
        rows.append(row)
    return pd.DataFrame(rows)


def scoreboard(day) -> pd.DataFrame:
    j = get_json(config.ESPN_SCOREBOARD, params={"dates": day.strftime("%Y%m%d")})
    if j is None:
        log.warning("ESPN scoreboard for %s unavailable", day)
        return pd.DataFrame()
    return parse_scoreboard(j)


def odds_rows(days: int | None = None) -> pd.DataFrame:
    """Snapshot rows for every game from today through `days` ahead that has not tipped."""
    from .odds import SNAP_COLS
    today = config.today_et()
    horizon = config.BOARD_DAYS if days is None else days
    frames = [scoreboard(today + timedelta(days=i)) for i in range(horizon + 1)]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=SNAP_COLS)
    s = pd.concat(frames, ignore_index=True)
    now = config.now_et()
    live = (s["state"] != "pre") | (pd.to_datetime(s["commence_time"], utc=True) <= now)
    if live.any():
        log.info("ESPN: skipped %d games already under way (in-play prices)", int(live.sum()))
    s = s[~live].copy()
    s["pulled_at"] = now.astimezone(timezone.utc).isoformat(timespec="seconds")
    s["source"] = "espn"
    s["event_id"] = s["game_id"]
    s["p_home"] = devig_two(s["home_dec"].values, s["away_dec"].values)
    s["p_over"] = devig_two(s["over_dec"].values, s["under_dec"].values)
    s["p_spread_home"] = devig_two(s["spread_home_dec"].values, s["spread_away_dec"].values)
    for c in ("n_ml", "n_total", "n_spread"):
        s[c] = 1
    log.info("ESPN scoreboard: %d games, %d with a spread, %d with a total, %d with a moneyline "
             "(%s)", len(s), int(s["spread_home"].notna().sum()), int(s["total_line"].notna().sum()),
             int(s["home_dec"].notna().sum()), ", ".join(sorted(set(s["provider"]) - {""})) or "-")
    return s.reindex(columns=SNAP_COLS)


# ---------------------------------------------------------------------------------
# the injury report
# ---------------------------------------------------------------------------------
def status(raw) -> str | None:
    """ESPN's status text -> one of config.MISS_PROB's keys, or None if unrecognised."""
    s = str(raw or "").strip().lower().replace("_", " ")
    s = re.sub(r"^injury status\s+", "", s)
    return STATUS.get(s) or STATUS.get(s.replace(" ", "-"))


def parse_injuries(payload: dict) -> pd.DataFrame:
    """Injuries JSON -> one row per listed player. Pure."""
    rows = []
    for team_block in (payload or {}).get("injuries", []) or []:
        team = None
        for key in ("displayName", "name", "abbreviation"):
            v = team_block.get(key) or (team_block.get("team") or {}).get(key)
            if v:
                try:
                    team = from_name(v)
                    break
                except UnknownTeam:
                    continue
        for inj in team_block.get("injuries", []) or []:
            ath = inj.get("athlete") or {}
            pid = ath.get("id") or inj.get("athleteId")
            t = team
            if t is None:
                try:
                    t = from_name((ath.get("team") or {}).get("displayName"))
                except UnknownTeam:
                    t = None
            st = status(inj.get("status")) or status((inj.get("type") or {}).get("description")) \
                or status((inj.get("type") or {}).get("name"))
            if t is None or pid is None or st is None:
                continue
            det = inj.get("details") or {}
            rows.append({"team": t, "athlete_id": int(pid),
                         "name": str(ath.get("displayName") or ath.get("fullName") or pid),
                         "status": st, "p_out": config.MISS_PROB.get(st, 0.5),
                         "detail": " ".join(str(det.get(k) or "") for k in ("side", "type")).strip()
                         or str(inj.get("shortComment") or "")[:80]})
    return pd.DataFrame(rows, columns=INJURY_COLS[1:])


def injuries(write: bool = True) -> pd.DataFrame | None:
    """Today's report, logged to injuries.csv. None when the report could not be read at all -
    which is different from an empty report, and is treated differently downstream."""
    if not config.INJURIES_ON:
        return None
    j = get_json(config.ESPN_INJURIES)
    if j is None:
        log.warning("injury report unavailable - falling back to the last one logged")
        return None
    df = parse_injuries(j)
    df.insert(0, "pulled_at", config.now_et().astimezone(timezone.utc).isoformat(timespec="seconds"))
    log.info("injury report: %d players on %d clubs (%s)", len(df), df["team"].nunique(),
             df["status"].value_counts().to_dict())
    if write and len(df):
        config.ensure_dirs()
        df.to_csv(config.INJURIES, mode="a", header=not config.INJURIES.exists(), index=False)
    return df


def latest_injuries(fresh: pd.DataFrame | None) -> pd.DataFrame:
    """The fresh report, or the last one logged today when the fresh pull failed - a failed
    evening read falls back to the morning's news rather than to nothing."""
    if fresh is not None:
        return fresh
    if not config.INJURIES.exists():
        return pd.DataFrame(columns=INJURY_COLS)
    log_ = pd.read_csv(config.INJURIES)
    if log_.empty:
        return log_
    ts = pd.to_datetime(log_["pulled_at"], utc=True, errors="coerce")
    day = ts.dt.tz_convert(config.ET).dt.date
    today = log_[day == config.today_et()]
    if today.empty:
        return pd.DataFrame(columns=INJURY_COLS)
    return today[today["pulled_at"] == today["pulled_at"].max()]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(odds_rows().to_string())
    inj = injuries(write=False)
    print(json.dumps({"injured": 0 if inj is None else len(inj)}, indent=2))
