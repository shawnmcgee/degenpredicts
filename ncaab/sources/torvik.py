"""Bart Torvik adjusted tempo / efficiency, pulled as *daily snapshots*.

Why snapshots: a season-final rating applied to a November game leaks the future - that was
the core flaw of the old pipeline. Torvik publishes a dated file per day:

    https://barttorvik.com/timemachine/team_results/YYYYMMDD_team_results.json.gz

so the rating we attach to a game is the rating that existed the morning of that game.

The main site sits behind browser verification, so treat this as best-effort: if a fetch
fails, the columns come back NaN, the model still trains (it has its own Elo/pace ratings),
and the failure is logged rather than crashing the run.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import time
from datetime import date, timedelta

import pandas as pd

from .. import config
from ..http import get
from ..team_names import normalize_team_name

log = logging.getLogger(__name__)

COLUMNS = ["snapshot_date", "team", "adj_o", "adj_d", "adj_t", "barthag", "wins", "games"]

# Torvik's team_results rows are positional arrays; these are the fields we use. Index layout
# has been stable for years but VERIFY once against a live file before the season.
IDX = {"rank": 0, "team": 1, "conf": 2, "record": 3, "adj_o": 4, "adj_d": 6, "barthag": 8, "adj_t": 26}


def fetch_snapshot(day: date) -> pd.DataFrame:
    url = f"{config.TORVIK_BASE}/timemachine/team_results/{day:%Y%m%d}_team_results.json.gz"
    r = get(url)
    if r is None or r.status_code != 200:
        log.debug("torvik snapshot %s unavailable (HTTP %s)", day, getattr(r, "status_code", "-"))
        return pd.DataFrame(columns=COLUMNS)
    try:
        raw = gzip.decompress(r.content) if r.content[:2] == b"\x1f\x8b" else r.content
        data = json.loads(raw)
    except Exception as e:
        log.warning("torvik snapshot %s unreadable: %s", day, e)
        return pd.DataFrame(columns=COLUMNS)

    rows = []
    for rec in data:
        try:
            if isinstance(rec, dict):
                team = rec.get("team") or rec.get("Team")
                adj_o, adj_d, adj_t = rec.get("adj_o"), rec.get("adj_d"), rec.get("adj_t")
                barthag = rec.get("barthag")
            else:
                team = rec[IDX["team"]]
                adj_o, adj_d = rec[IDX["adj_o"]], rec[IDX["adj_d"]]
                adj_t = rec[IDX["adj_t"]] if len(rec) > IDX["adj_t"] else None
                barthag = rec[IDX["barthag"]]
            if not team:
                continue
            rows.append({"snapshot_date": day, "team": normalize_team_name(str(team)),
                         "adj_o": _f(adj_o), "adj_d": _f(adj_d), "adj_t": _f(adj_t),
                         "barthag": _f(barthag), "wins": None, "games": None})
        except (IndexError, TypeError, KeyError):
            continue
    if rows:
        log.debug("torvik %s: %d teams", day, len(rows))
    return pd.DataFrame(rows, columns=COLUMNS)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load() -> pd.DataFrame:
    if config.TORVIK.exists():
        df = pd.read_csv(config.TORVIK, parse_dates=["snapshot_date"])
        df["snapshot_date"] = df["snapshot_date"].dt.date
        return df
    return pd.DataFrame(columns=COLUMNS)


def update(start: date | None = None, end: date | None = None, every: int = 3) -> pd.DataFrame:
    """Fetch snapshots every `every` days (ratings barely move day to day; this keeps the
    file small and polite). Missing days are forward-filled at join time."""
    cache = load()
    have = set(cache["snapshot_date"]) if len(cache) else set()
    end = end or config.today_et()
    start = start or config.season_start(config.FIRST_SEASON)

    days, d = [], start
    while d <= end:
        s = config.season_of(d)
        in_season = config.season_start(s) + timedelta(days=14) <= d <= config.season_end(s)
        if in_season and d not in have and ((d - start).days % every == 0 or (end - d).days <= 2):
            days.append(d)
        d += timedelta(days=1)

    if not days:
        return cache
    log.info("fetching %d torvik snapshots", len(days))
    frames = []
    for i, day in enumerate(days, 1):
        f = fetch_snapshot(day)
        if len(f):
            frames.append(f)
        if i % 25 == 0:
            log.info("  torvik %d/%d", i, len(days))
        time.sleep(0.4)
    if frames:
        cache = pd.concat([cache, *frames], ignore_index=True)
        config.ensure_dirs()
        cache.drop_duplicates(["snapshot_date", "team"], keep="last").to_csv(config.TORVIK, index=False)
    else:
        log.warning("no torvik snapshots retrieved - model will run without efficiency features")
    return load()


def as_of(cache: pd.DataFrame, day: date) -> dict[str, dict]:
    """Most recent snapshot strictly on or before `day`, as {team: {adj_o, adj_d, adj_t}}."""
    if cache.empty:
        return {}
    sub = cache[cache["snapshot_date"] <= day]
    if sub.empty:
        return {}
    latest = sub["snapshot_date"].max()
    rows = sub[sub["snapshot_date"] == latest]
    return {r.team: {"adj_o": r.adj_o, "adj_d": r.adj_d, "adj_t": r.adj_t} for r in rows.itertuples()}
