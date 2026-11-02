"""ESPN's line on every past game, one game at a time - the history the market models train on.

    python -m ncaab.sources.lines                  # back-fill, newest games first, for up to
                                                   # DEGEN_NCAAB_LINES_BUDGET minutes
    python -m ncaab.sources.lines --check 401638580   # print one game's parsed line, write nothing

**Why this exists.** The scoreboard carries a game's odds only until it finishes: the first
training run read 47,142 finals and found a line on none of them, so the four market-aware
models had nothing to train on. ESPN keeps past odds on two per-game endpoints:

| Endpoint | |
|---|---|
| core API ``.../events/{id}/competitions/{id}/odds`` | a few KB: every provider's line on the game. Tried first. |
| summary ``.../summary?event={id}`` | the whole game page, whose ``pickcenter`` holds the same lines. Used when the core API cannot answer for a game (an error, or a listing that holds only links). |

Both are read by :mod:`ncaab.sources.espn_odds`, the parser every board shares, which takes the
favourite from the ``details`` string ("DUKE -7.5") and never trusts a bare ``spread`` for its
sign. Live in-game providers ("... Live Odds") are skipped: their number is whatever it was at
the final buzzer.

**Resumable.** ``data/ncaab/lines.csv`` holds one row per game asked about: ``ok`` with the line,
or ``none`` when ESPN answered and had no line (most low-major games), so it is not asked again.
A game whose endpoints both failed is not recorded and is retried next run. Progress is saved
every few thousand games, so a run that hits its time budget or is cancelled keeps what it got.
Games go newest first, Division I against Division I only, and never the ratings' warm-up season.

**Written without reaching ESPN from the machine it was built on.** The first run's log line
``lines: N asked, X ok (Y%), Z none, W failed`` is the real test. If a whole season comes back
``none``, set ``DEGEN_NCAAB_LINES_SOURCE=summary`` and run the training job again.
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config
from ..http import get_json
from . import espn, espn_odds

log = logging.getLogger(__name__)

LINE_COLS = ["game_id", "season", "status", "source", "provider", "spread_home", "total_line",
             "spread_home_price", "spread_away_price", "over_price", "under_price",
             "open_spread_home", "open_total", "fetched_at"]
PRICED = ("details", "spread", "overUnder", "pointSpread", "total", "homeTeamOdds", "awayTeamOdds")
TOTAL_RANGE = (90.0, 230.0)          # a college total outside this is a parsing error
SAVE_EVERY = 2000
CHUNK = 300


# ---------------------------------------------------------------------------------
# parsing - pure
# ---------------------------------------------------------------------------------
def _usable(items) -> list[dict]:
    """Entries that carry a price, live in-game providers dropped."""
    out = []
    for o in items or []:
        if not isinstance(o, dict) or not any(k in o for k in PRICED):
            continue
        prov = o.get("provider") if isinstance(o.get("provider"), dict) else {}
        if "live" in str(prov.get("name") or "").lower():
            continue
        out.append(o)
    return out


def _row(items: list[dict], home_abbr, away_abbr) -> dict:
    r = espn_odds.parse_odds(items, str(home_abbr or ""), str(away_abbr or ""), config.MAX_SPREAD)
    tl = r.get("total_line")
    if tl == tl and not (TOTAL_RANGE[0] <= tl <= TOTAL_RANGE[1]):
        r["total_line"] = r["over_price"] = r["under_price"] = np.nan
    ot = r.get("open_total")
    if ot == ot and not (TOTAL_RANGE[0] <= ot <= TOTAL_RANGE[1]):
        r["open_total"] = np.nan
    has = r["spread_home"] == r["spread_home"] or r["total_line"] == r["total_line"]
    return {"status": "ok" if has else "none", **{k: r.get(k, np.nan) for k in LINE_COLS if k in r}}


def parse_core(payload, home_abbr, away_abbr) -> dict | None:
    """The core API's odds listing -> a line row, or None when it cannot answer for the game:
    not a listing at all, or a listing of links with no prices in it."""
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        return None
    items = payload["items"]
    usable = _usable(items)
    if items and not usable and not any(
            isinstance(o, dict) and any(k in o for k in PRICED) for o in items):
        return None                     # only $ref links: ask the summary instead
    return {"source": "core", **_row(usable, home_abbr, away_abbr)}


def parse_summary(payload, home_abbr, away_abbr) -> dict | None:
    """The summary page's ``pickcenter`` -> a line row, or None if it is not a game page."""
    if not isinstance(payload, dict) or not any(k in payload for k in ("header", "pickcenter",
                                                                         "boxscore")):
        return None
    items = payload.get("pickcenter") or payload.get("odds") or []
    return {"source": "summary", **_row(_usable(items), home_abbr, away_abbr)}


def fetch_one(game: dict, get=get_json, source: str | None = None) -> dict | None:
    """One game's line row, or None if nothing answered (so it is retried on a later run)."""
    source = (source or config.LINES_SOURCE).lower()
    gid, h, a = str(game["game_id"]), game.get("home_abbr"), game.get("away_abbr")
    row = None
    if source in ("auto", "core"):
        row = parse_core(get(config.ESPN_CORE_ODDS.format(id=gid)), h, a)
    if row is None and source in ("auto", "summary"):
        row = parse_summary(get(config.ESPN_SUMMARY, params={"event": gid}), h, a)
    if row is None:
        return None
    row.update({"game_id": gid, "season": game.get("season"),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return row


# ---------------------------------------------------------------------------------
# the lines file
# ---------------------------------------------------------------------------------
def load() -> pd.DataFrame:
    if not config.LINES.exists():
        return pd.DataFrame(columns=LINE_COLS)
    return pd.read_csv(config.LINES, dtype={"game_id": str, "status": str, "source": str,
                                            "provider": str})


def save(rows: pd.DataFrame) -> None:
    config.ensure_dirs()
    (rows.reindex(columns=LINE_COLS).drop_duplicates("game_id", keep="last")
         .sort_values(["season", "game_id"]).to_csv(config.LINES, index=False))


def _d1(games: pd.DataFrame) -> pd.Series:
    """Both sides Division I: at least D1_MIN_GAMES that season or the one before. The previous
    season covers November, when nobody has played ten games yet."""
    n = pd.concat([games[["season", "home_id"]].rename(columns={"home_id": "t"}),
                   games[["season", "away_id"]].rename(columns={"away_id": "t"})])
    counts = n.groupby(["season", "t"]).size()
    def ok(season, team):
        k = config.D1_MIN_GAMES
        return counts.get((season, team), 0) >= k or counts.get((season - 1, team), 0) >= k
    return pd.Series([ok(s, h) and ok(s, a) for s, h, a in
                      zip(games["season"], games["home_id"], games["away_id"])], index=games.index)


def todo(games: pd.DataFrame, have: pd.DataFrame) -> pd.DataFrame:
    """Finals with no line row yet: D1 against D1, past the warm-up season, newest first."""
    if games.empty:
        return games
    first = int(games["season"].min()) + config.WARMUP_SEASONS
    g = games[(games["season"] >= first)]
    g = g[_d1(g)]
    done = set(have["game_id"]) if len(have) else set()
    g = g[~g["game_id"].astype(str).isin(done)]
    sort = ["date", "kickoff_utc"] if "kickoff_utc" in g else ["date"]
    return g.sort_values(sort, ascending=False)


def backfill(games: pd.DataFrame | None = None, budget_min: float | None = None,
             workers: int | None = None, get=get_json, only: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ask ESPN for every missing line until done or the budget runs out. ``only`` skips the
    selection and asks for exactly those games (grading uses it for last night's)."""
    have = load()
    if only is None:
        games = espn.load() if games is None else games
        want = todo(games, have)
    else:
        done = set(have["game_id"]) if len(have) else set()
        want = only[~only["game_id"].astype(str).isin(done)]
    budget = config.LINES_BUDGET if budget_min is None else budget_min
    workers = max(1, config.LINES_WORKERS if workers is None else workers)
    if want.empty:
        log.info("lines: nothing to ask for (%d games on file)", len(have))
        return have

    log.info("lines: %d games to ask ESPN for, %d workers, %.0f-minute budget", len(want),
             workers, budget)
    deadline = time.monotonic() + budget * 60
    recs = want.to_dict("records")
    new, asked, failed, saved_at = [], 0, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(recs), CHUNK):
            if time.monotonic() >= deadline:
                log.info("lines: budget used up after %d games - the next run carries on", asked)
                break
            batch = recs[i:i + CHUNK]
            for r in ex.map(lambda rec: fetch_one(rec, get), batch):
                if r is None:
                    failed += 1
                else:
                    new.append(r)
            asked += len(batch)
            if len(new) - saved_at >= SAVE_EVERY:
                have = _merge(have, new[saved_at:])
                saved_at = len(new)
                save(have)
                log.info("lines: %d/%d asked, %d on file", asked, len(recs), len(have))
    have = _merge(have, new[saved_at:])
    save(have)
    ok = sum(1 for r in new if r["status"] == "ok")
    log.info("lines: %d asked, %d ok (%.0f%%), %d none, %d failed - %d games on file", asked, ok,
             100 * ok / max(asked, 1), len(new) - ok, failed, len(have))
    if len(have):
        by = have.assign(ok=have["status"] == "ok").groupby("season")["ok"].agg(["sum", "size"])
        log.info("lines by season (ok / asked): %s", ", ".join(
            f"{int(s)}: {int(r['sum'])}/{int(r['size'])}" for s, r in by.iterrows()))
    return have


def _merge(have: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return have
    add = pd.DataFrame(rows).reindex(columns=LINE_COLS)
    return pd.concat([have, add], ignore_index=True) if len(have) else add


# ---------------------------------------------------------------------------------
# attaching lines to games
# ---------------------------------------------------------------------------------
def attach(games: pd.DataFrame, lines: pd.DataFrame | None = None,
           picks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fill each final's line: what the scoreboard had, else ESPN's line from lines.csv, else
    the last pre-tip line this board published for it. Never writes games.csv."""
    g = games.copy()
    if g.empty:
        return g
    lines = load() if lines is None else lines
    if picks is None and config.PICKS.exists():
        picks = pd.read_csv(config.PICKS, dtype={"game_id": str})
    sources = []
    if len(lines):
        ok = lines[lines["status"] == "ok"].drop_duplicates("game_id", keep="last")
        sources.append(ok.set_index(ok["game_id"].astype(str)))
    if picks is not None and len(picks):
        p = picks.drop_duplicates("game_id", keep="last")
        sources.append(p.set_index(p["game_id"].astype(str)))
    ids = g["game_id"].astype(str)
    for src in sources:
        for col in ("spread_home", "total_line"):
            if col in src:
                g[col] = g[col].where(g[col].notna(), ids.map(src[col]))
    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None, help="minutes (default: the knob)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--check", help="one ESPN event id: print its parsed line, write nothing")
    args = ap.parse_args(argv)
    if args.check:
        g = espn.load()
        rec = g[g["game_id"] == args.check]
        game = rec.iloc[0].to_dict() if len(rec) else {"game_id": args.check}
        print(fetch_one(game))
        return
    games = espn.update()
    backfill(games, budget_min=args.budget, workers=args.workers)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
