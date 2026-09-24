"""Historical closing lines, imported once and committed to data/nba/lines.csv.

    python -m nba.sources.history            # rebuild lines.csv from all three archives

No free source publishes NBA closing lines going back years, and the Odds API's own history is a
paid endpoint. Three public archives on GitHub cover it between them, and each was checked
against the others before any of it was trusted:

| Seasons | Source | What it carries |
|---|---|---|
| 2019-20 (from the bubble) to 2025-26 | The Odds API's historical endpoint, a consensus of 10-21 books snapped **5-15 minutes before tip**, published by hoopR (`closing_lines_odds_api.parquet`) | closing spread and total, keyed by ESPN's game id; no prices |
| 2017-18 to 2022-23 | a scraped archive also published by hoopR (`games-archive.json`) | closing spread and total |
| 2007-08 to 2025-26 | SportsbookReviewsOnline's archive via `sbrscrape`, compiled into `kyleskom/NBA-Machine-Learning-Sports-Betting` (`OddsData.sqlite`) | the closing **moneyline**, both sides |

**How they were checked.** Where the first two overlap (2019-20 to 2022-23) they agree to a
median of 0.0 points on both spread and total. The third was the interesting one:

* its spreads before 2022-23 are **unsigned magnitudes** - every favourite and every underdog is
  printed as a positive number - and even with the sign restored from the moneyline they sit a
  median point from the close, and their totals 2-4 points from it. They are opening numbers,
  not closing ones, and in 2019-20 many are simply wrong. **They are not imported.**
* its moneylines are the close: across 2017-18 to 2021-22 they agree with the closing spread
  to 1.7 points of win probability, against 3.5 for the opener. So moneylines come from here.

The result is closing spreads and totals for **2017-18 to 2025-26** (about 11,500 games, nine
seasons) and closing moneylines from **2007-08**. Each row records where each half came from.

From the 2026-27 season on, the pipeline's own evening snapshots are promoted into this file as
games finish (see :func:`nba.sources.odds.consolidate`), so it keeps growing on its own.

The Odds API file is parquet, so this one-time import needs ``pyarrow`` (``pip install pyarrow``);
nothing the daily jobs run does. Credit to all three archives' authors; if any disappears, the
committed lines.csv is unaffected - this only runs when asked.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sqlite3
import tempfile

import numpy as np
import pandas as pd

from .. import config
from ..http import get
from ..odds_math import american_to_decimal, devig_two
from ..teams import UnknownTeam, canon, from_name
from .hoopr import load_games

log = logging.getLogger(__name__)

LINE_COLS = ["game_id", "season", "date", "home_team", "away_team", "source", "is_closing",
             "spread_home", "total_line", "n_books", "spread_home_dec", "spread_away_dec",
             "over_dec", "under_dec", "ml_source", "home_dec", "away_dec", "p_home"]


def load_lines() -> pd.DataFrame:
    if not config.LINES.exists():
        return pd.DataFrame(columns=LINE_COLS)
    return pd.read_csv(config.LINES, dtype={"game_id": str}, low_memory=False)


def upsert_lines(rows: pd.DataFrame) -> pd.DataFrame:
    """Add or replace rows by game_id. A later source for the same game wins."""
    config.ensure_dirs()
    new = rows.reindex(columns=LINE_COLS).copy()
    new["game_id"] = new["game_id"].astype(str)
    old = load_lines()
    if len(old):
        old = old[~old["game_id"].astype(str).isin(set(new["game_id"]))]
        new = pd.concat([old.reindex(columns=LINE_COLS), new], ignore_index=True)
    new["date"] = pd.to_datetime(new["date"]).dt.date.astype(str)
    new = new.sort_values(["date", "game_id"]).reset_index(drop=True)
    new.to_csv(config.LINES, index=False)
    return new


def _fetch(url_or_path: str) -> bytes:
    if url_or_path.startswith("http"):
        r = get(url_or_path)
        if r is None or r.status_code != 200:
            raise SystemExit(f"could not fetch {url_or_path}: {getattr(r, 'status_code', 'no response')}")
        return r.content
    with open(url_or_path, "rb") as fh:
        return fh.read()


def _code(fn, x):
    try:
        return fn(x)
    except UnknownTeam:
        return None


def oddsapi_close(raw: bytes, games: pd.DataFrame) -> pd.DataFrame:
    """The consensus close, keyed by ESPN game id. ``home_point`` is the home side's line in
    book convention (-2.5 = home gives 2.5) - verified against the favourite flag, not assumed."""
    try:
        d = pd.read_parquet(io.BytesIO(raw))
    except ImportError as e:
        raise SystemExit("the one-time line import reads parquet: pip install pyarrow") from e
    d["game_id"] = d["game_id"].astype("int64").astype(str)
    # One game in the file was matched with its sides flipped. It is one game; dropping it
    # beats trusting a sign someone else had to repair.
    d = d[d["match_method"].astype(str) == "exact"]
    fav = d["home_favorite"].astype(bool)
    bad = (fav & (d["home_point"] > 0)) | (~fav & (d["home_point"] < 0))
    if bad.any():
        log.warning("history: %d closing rows whose sign disagrees with the favourite - dropped",
                    int(bad.sum()))
    d = d[~bad]
    out = pd.DataFrame({"game_id": d["game_id"], "source": "oddsapi_close", "is_closing": True,
                        "spread_home": d["home_point"].astype(float),
                        "total_line": pd.to_numeric(d["over_under"], errors="coerce"),
                        "n_books": d["n_books"].astype(int)})
    return out[out["game_id"].isin(set(games["game_id"].astype(str)))]


def archive_close(raw: bytes, games: pd.DataFrame) -> pd.DataFrame:
    """The scraped closing archive, joined to games on (date, home, away)."""
    d = pd.DataFrame(json.loads(raw))
    d["home_team"] = d["home_team_abbrev"].map(lambda x: _code(canon, x))
    d["away_team"] = d["visit_team_abbrev"].map(lambda x: _code(canon, x))
    d["date"] = pd.to_datetime(d["game_date"]).dt.date.astype(str)
    d = d.dropna(subset=["home_team", "away_team"])
    g = games[["game_id", "date", "home_team", "away_team"]].copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date.astype(str)
    m = d.merge(g, on=["date", "home_team", "away_team"], how="inner")
    return pd.DataFrame({"game_id": m["game_id"].astype(str), "source": "archive_close",
                         "is_closing": True, "spread_home": pd.to_numeric(m["line"], errors="coerce"),
                         "total_line": pd.to_numeric(m["game_over_under"], errors="coerce"),
                         "n_books": np.nan})


def sbr_moneylines(raw: bytes, games: pd.DataFrame) -> pd.DataFrame:
    """Closing moneylines from the SBR-derived sqlite, joined on (date, home, away).

    Only the tables with ISO dates are read; the legacy ones write dates as "2007-08-1030".
    Spreads and totals in this file are deliberately ignored - see the module docstring.
    """
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as fh:
        fh.write(raw)
        fh.flush()
        con = sqlite3.connect(fh.name)
        tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
        use = [t for t in tables if t.endswith("_new") or t in ("2023-24", "2024-25", "odds_2025-26")]
        frames = [pd.read_sql(f'select * from "{t}"', con).assign(_t=t) for t in use]
        con.close()
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["Date"].astype(str).str.slice(0, 10), format="%Y-%m-%d",
                               errors="coerce").dt.date.astype(str)
    d["home_team"] = d["Home"].map(lambda x: _code(from_name, x))
    d["away_team"] = d["Away"].map(lambda x: _code(from_name, x))
    d = d.dropna(subset=["home_team", "away_team"])
    d = d[d["date"] != "NaT"]
    # a "_new" table outranks the older copy of the same season
    d["_rank"] = d["_t"].str.endswith("_new").astype(int)
    d = d.sort_values("_rank").drop_duplicates(["date", "home_team", "away_team"], keep="last")
    d["home_dec"] = american_to_decimal(pd.to_numeric(d["ML_Home"], errors="coerce"))
    d["away_dec"] = american_to_decimal(pd.to_numeric(d["ML_Away"], errors="coerce"))
    d["p_home"] = devig_two(d["home_dec"].values, d["away_dec"].values)
    d = d[np.isfinite(d["p_home"])]
    g = games[["game_id", "date", "home_team", "away_team"]].copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date.astype(str)
    m = d.merge(g, on=["date", "home_team", "away_team"], how="inner")
    return pd.DataFrame({"game_id": m["game_id"].astype(str), "ml_source": "sbr_close",
                         "home_dec": m["home_dec"], "away_dec": m["away_dec"], "p_home": m["p_home"]})


def import_lines(close: str | None = None, archive: str | None = None,
                 ml: str | None = None) -> pd.DataFrame:
    games = load_games()
    if games.empty:
        raise SystemExit("games.csv is empty - load games first (python -m nba.sources.hoopr)")
    games["game_id"] = games["game_id"].astype(str)
    a = oddsapi_close(_fetch(close or config.HISTORY_CLOSE_URL), games)
    b = archive_close(_fetch(archive or config.HISTORY_ARCHIVE_URL), games)
    c = sbr_moneylines(_fetch(ml or config.HISTORY_ML_URL), games)
    b = b[~b["game_id"].isin(set(a["game_id"]))]
    spreads = pd.concat([a, b], ignore_index=True)
    out = spreads.merge(c, on="game_id", how="outer")
    meta = games.set_index("game_id")[["season", "date", "home_team", "away_team"]]
    out = out.join(meta, on="game_id")
    out["is_closing"] = out["is_closing"].fillna(False).astype(bool) | out["ml_source"].notna()
    out["source"] = out["source"].fillna("")
    lines = upsert_lines(out)
    log.info("lines: %d consensus closes, %d archive closes, %d closing moneylines -> %d rows",
             len(a), len(b), len(c), len(lines))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--close", help="local copy of closing_lines_odds_api.parquet")
    ap.add_argument("--archive", help="local copy of games-archive.json")
    ap.add_argument("--ml", help="local copy of OddsData.sqlite")
    a = ap.parse_args(argv)
    import_lines(a.close, a.archive, a.ml)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
