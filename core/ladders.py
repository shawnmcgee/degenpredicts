"""Every rung of every Kalshi spread and total ladder, kept.

The board records two rungs per ladder: the reference rung nearest the book number and the
best playable one. That prices a pick, but it cannot answer the question the Big Game Card
raises: are the rungs far from the main line ("Alabama by 14+", "Over 62.5") priced worse, or
better, than history says they should be? Answering that needs the whole ladder at the moment
it was quoted, next to the sportsbook's number at the same moment, so it can be graded once
the game is final.

Storage is one gzip CSV per pull day under ``data/<sport>/ladders/``. A college Saturday is a
couple of thousand rungs, which compresses to tens of kilobytes; one plain CSV for the season
would grow by megabytes a week in a repo that is also the database. The gzip header carries no
timestamp, so rewriting a day file with unchanged rows leaves git with nothing to commit.

Sport-neutral on purpose: each sport's predict step hands over the ladder frames its own
``kalshi.ladder_board`` produced plus its board, and this module only joins and writes.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

KEY = ["date", "home_team", "away_team"]

# Rung fields as parsed by <sport>.sources.kalshi.parse_ladder, then the board's number at the
# same moment, so a rung can be compared with the sportsbook line it was quoted against.
RUNG_COLS = ["kind", "team", "strike", "yes_bid", "yes_ask", "ask_size", "volume",
             "open_interest", "tradeable", "ticker", "event_ticker"]
BOOK_COLS = ["game_id", "spread_home", "total_line", "spread_home_price", "spread_away_price",
             "over_price", "under_price"]
COLUMNS = ["pulled_at", *KEY, *RUNG_COLS, *BOOK_COLS]

GZIP = {"method": "gzip", "mtime": 0}


def _day_key(v) -> str:
    """Board and ladder dates arrive as date objects, Timestamps or ISO strings."""
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    return str(v)[:10]


def join_rungs(ladders: list[pd.DataFrame], board: pd.DataFrame,
               pulled_at: str | None = None) -> pd.DataFrame:
    """Rungs that belong to a game on the board, each carrying the book number beside it.

    Rungs for games the board does not carry (other days, FCS games, unmatched names) are
    dropped: without the book's number at the same moment they cannot be graded.
    """
    frames = [f for f in (ladders or []) if f is not None and len(f)]
    if not frames or board is None or board.empty:
        return pd.DataFrame(columns=COLUMNS)
    rungs = pd.concat(frames, ignore_index=True)
    if "kind" not in rungs:
        return pd.DataFrame(columns=COLUMNS)
    # The board is the authority for the game and its book number; a ladder frame that already
    # carries any of those columns would collide in the merge and lose them.
    rungs = rungs.drop(columns=[c for c in BOOK_COLS if c in rungs])
    for c in RUNG_COLS:
        if c not in rungs:
            rungs[c] = None
    rungs["_k"] = rungs["date"].map(_day_key)

    b = board.copy()
    for c in BOOK_COLS:
        if c not in b:
            b[c] = None
    b["_k"] = b["date"].map(_day_key)
    b = b[["_k", "home_team", "away_team", *BOOK_COLS]].drop_duplicates(
        subset=["_k", "home_team", "away_team"], keep="last")

    out = rungs.merge(b, on=["_k", "home_team", "away_team"], how="inner")
    out["date"] = out["_k"]
    out["pulled_at"] = pulled_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out[COLUMNS].reset_index(drop=True)


def log_rungs(ladders: list[pd.DataFrame], board: pd.DataFrame, directory: Path,
              day: date | str, pulled_at: str | None = None) -> int:
    """Append this pull's rungs to ``directory/<day>.csv.gz``. Returns rows written.

    Never raises: losing a day of ladder history must not cost the board its picks.
    """
    try:
        rows = join_rungs(ladders, board, pulled_at)
        if rows.empty:
            log.info("ladders: no rungs matched the board - nothing logged")
            return 0
        n_new, n_games = len(rows), rows["game_id"].nunique()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_day_key(day)}.csv.gz"
        rows["game_id"] = rows["game_id"].astype(str)
        if path.exists():
            old = pd.read_csv(path, dtype={"game_id": str})
            rows = pd.concat([old, rows], ignore_index=True)
        rows = rows.drop_duplicates(subset=["pulled_at", "ticker"], keep="last")
        rows.to_csv(path, index=False, compression=GZIP)
        log.info("ladders: %d rungs for %d games -> %s", n_new, n_games, path.name)
        return n_new
    except Exception as e:                       # pragma: no cover - logged, never fatal
        log.warning("ladders: could not log rungs (%s)", e)
        return 0


def load(directory: Path, days: int | None = None) -> pd.DataFrame:
    """Every logged rung, newest day files last. ``days`` limits it to the most recent N."""
    directory = Path(directory)
    if not directory.exists():
        return pd.DataFrame(columns=COLUMNS)
    files = sorted(directory.glob("*.csv.gz"))
    if days:
        files = files[-days:]
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_csv(f, dtype={"game_id": str}) for f in files], ignore_index=True)


def latest(directory: Path, days: int = 3) -> pd.DataFrame:
    """The most recent pull of each game's ladders - what a page should show right now."""
    df = load(directory, days=days)
    if df.empty:
        return df
    last = df.groupby("game_id")["pulled_at"].transform("max")
    return df[df["pulled_at"] == last].reset_index(drop=True)
