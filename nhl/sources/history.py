"""Historical closing lines, imported once and committed to data/nhl/lines.csv.

    python -m nhl.sources.history --lines            # rebuild lines.csv from both archives
    python -m nhl.sources.history --games            # bootstrap games.csv without the NHL API

No free source publishes NHL closing lines going back years, and the Odds API's own history is a
paid endpoint. Two public archives on GitHub cover most of it between them:

| Seasons | Source | What it carries |
|---|---|---|
| 2007-08 to 2022-23 | SportsbookReviewsOnline's closing archive, as compiled into `ethanbell528-cmd/fda-project-1` (the SBR files are only reachable through the Internet Archive now) | closing moneyline and total; the ±1.5 puck line from 2014-15, **without its price**; no over/under price |
| 2023-24 to Jan 2026 | The Odds API's historical endpoint, pulled at noon Eastern each day by `nielsenz/odds-api-current-save` | moneyline, puck line and total **with prices**, BetMGM and Caesars |

The difference matters and is kept visible rather than smoothed over. The older rows are
genuine closing numbers but carry no over price, so their market total is read at the 2023-26
average over price and flagged ``p_over_assumed``; they also carry no puck-line price, so they
can score the model's puck-line probabilities but not its puck-line ROI. The newer rows are
fully priced but are NOON prices, softer than a close - an edge measured against them is an edge
against the morning market, which is also exactly what the morning board bets into.

From the 2026-27 season on, the pipeline's own evening snapshots are promoted into this file as
games finish (see :func:`nhl.sources.odds.consolidate`), so it keeps growing on its own.

Both archives are fetched from raw.githubusercontent.com. Credit to their authors; if either
disappears, the committed lines.csv is unaffected - this only runs when asked.
"""
from __future__ import annotations

import argparse
import io
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .. import config
from ..http import get
from ..odds_math import american_to_decimal, devig_two
from ..teams import UnknownTeam, canon, from_name
from .nhle import GAME_COLS, _merge, load_games

log = logging.getLogger(__name__)

LINE_COLS = ["game_id", "season", "date", "home_team", "away_team", "source", "is_closing",
             "home_dec", "away_dec", "p_home", "total_line", "over_dec", "under_dec", "p_over",
             "pl_home", "pl_home_dec", "pl_away_dec", "p_pl_home", "n_books"]


def load_lines() -> pd.DataFrame:
    if not config.LINES.exists():
        return pd.DataFrame(columns=LINE_COLS)
    df = pd.read_csv(config.LINES, dtype={"game_id": str}, low_memory=False)
    return df


def upsert_lines(rows: pd.DataFrame) -> pd.DataFrame:
    """Add or replace rows by game_id. A later source for the same game wins."""
    config.ensure_dirs()
    new = rows.copy()
    if "n_books" not in new and "n_ml" in new:
        new["n_books"] = new["n_ml"]
    new = new.reindex(columns=LINE_COLS)
    new["game_id"] = new["game_id"].astype(str)
    old = load_lines()
    if len(old):
        old = old[~old["game_id"].astype(str).isin(set(new["game_id"]))]
        new = pd.concat([old.reindex(columns=LINE_COLS), new], ignore_index=True)
    new = new.sort_values(["date", "game_id"]).reset_index(drop=True)
    new.to_csv(config.LINES, index=False)
    return new


def _read_csv(url_or_path: str) -> pd.DataFrame:
    if url_or_path.startswith("http"):
        r = get(url_or_path)
        if r is None or r.status_code != 200:
            raise SystemExit(f"could not fetch {url_or_path}: {getattr(r, 'status_code', 'no response')}")
        return pd.read_csv(io.StringIO(r.text), low_memory=False)
    return pd.read_csv(url_or_path, low_memory=False)


def _code(x):
    try:
        return canon(x)
    except UnknownTeam:
        return None


def sbr_lines(panel: pd.DataFrame) -> pd.DataFrame:
    """Closing moneyline, total and puck line per game from the two-rows-per-game panel."""
    p = panel[panel["season"] >= 2005]
    h = p[p["home_away"] == "H"]
    a = p[p["home_away"] == "A"][["game_id", "moneyline"]].rename(columns={"moneyline": "away_ml"})
    g = h.merge(a, on="game_id", how="inner")
    g = g[g["moneyline"].notna() | g["total_line"].notna()].copy()
    g["home_dec"] = american_to_decimal(g["moneyline"])
    g["away_dec"] = american_to_decimal(g["away_ml"])
    out = pd.DataFrame({
        "game_id": g["game_id"].astype(str), "season": g["season"].astype(int),
        "date": g["date"], "home_team": g["team"].map(_code), "away_team": g["opponent"].map(_code),
        "source": "sbr_close", "is_closing": True,
        "home_dec": g["home_dec"], "away_dec": g["away_dec"],
        "p_home": devig_two(g["home_dec"].values, g["away_dec"].values),
        "total_line": g["total_line"], "over_dec": np.nan, "under_dec": np.nan, "p_over": np.nan,
        "pl_home": g["line"].where(g["line"].abs() == 1.5), "pl_home_dec": np.nan,
        "pl_away_dec": np.nan, "p_pl_home": np.nan, "n_books": 1})
    return out.dropna(subset=["home_team", "away_team"])


def oddsapi_lines(frames: list[pd.DataFrame], games: pd.DataFrame) -> pd.DataFrame:
    """Consensus noon prices per game from the daily Odds API history files."""
    if not frames:
        return pd.DataFrame(columns=LINE_COLS)
    o = pd.concat(frames, ignore_index=True)
    names = {}
    for n in set(o["home_team"]) | set(o["away_team"]):
        try:
            names[n] = from_name(n)
        except UnknownTeam as e:
            log.error("history: %s", e)
    o["home"] = o["home_team"].map(names)
    o["away"] = o["away_team"].map(names)
    o = o.dropna(subset=["home", "away"])
    ct = pd.to_datetime(o["commence_time"], utc=True).dt.tz_convert(config.ET)
    o["gdate"] = ct.dt.date.astype(str)
    o = o[o["file_date"] <= o["gdate"]]
    o = o.sort_values("file_date").drop_duplicates(["gdate", "home", "away", "bookmaker"], keep="last")
    for c in ("ml_home", "ml_away", "spread_home_odds", "spread_away_odds", "total_over_odds",
              "total_under_odds"):
        o[c + "_d"] = american_to_decimal(pd.to_numeric(o[c], errors="coerce"))
    rows = []
    for (d, hm, aw), grp in o.groupby(["gdate", "home", "away"]):
        r = {"date": d, "home_team": hm, "away_team": aw, "n_books": len(grp)}
        p = devig_two(grp["ml_home_d"].values, grp["ml_away_d"].values)
        r.update(home_dec=float(grp["ml_home_d"].median()), away_dec=float(grp["ml_away_d"].median()),
                 p_home=float(np.nanmean(p)) if np.isfinite(p).any() else np.nan)
        tl = grp["total_line"].mode()
        if len(tl):
            at = grp[grp["total_line"] == tl.iloc[0]]
            po = devig_two(at["total_over_odds_d"].values, at["total_under_odds_d"].values)
            r.update(total_line=float(tl.iloc[0]), over_dec=float(at["total_over_odds_d"].median()),
                     under_dec=float(at["total_under_odds_d"].median()),
                     p_over=float(np.nanmean(po)) if np.isfinite(po).any() else np.nan)
        pl = grp[grp["spread_home"].abs() == 1.5]
        plm = pl["spread_home"].mode()
        if len(plm):
            pl = pl[pl["spread_home"] == plm.iloc[0]]
            pp = devig_two(pl["spread_home_odds_d"].values, pl["spread_away_odds_d"].values)
            r.update(pl_home=float(plm.iloc[0]), pl_home_dec=float(pl["spread_home_odds_d"].median()),
                     pl_away_dec=float(pl["spread_away_odds_d"].median()),
                     p_pl_home=float(np.nanmean(pp)) if np.isfinite(pp).any() else np.nan)
        rows.append(r)
    c = pd.DataFrame(rows)
    g = games[["game_id", "season", "date", "home_team", "away_team"]].copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date.astype(str)
    c = c.merge(g, on=["date", "home_team", "away_team"], how="inner")
    c["source"] = "oddsapi_noon"
    c["is_closing"] = False
    return c.reindex(columns=LINE_COLS)


def fetch_oddsapi_history(first: date = date(2023, 10, 1), last: date | None = None,
                          local_dir: str | None = None) -> list[pd.DataFrame]:
    """One CSV per day. ~600 exist across three seasons; missing days are simply skipped."""
    import concurrent.futures as cf
    from pathlib import Path
    last = last or config.today_et()
    days = [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]

    def one(d):
        if local_dir:
            p = Path(local_dir) / f"odds_{d}.csv"
            return pd.read_csv(p).assign(file_date=d) if p.exists() else None
        r = get(config.HISTORY_ODDSAPI_URL.format(date=d))
        if r is None or r.status_code != 200 or not r.text.strip():
            return None
        return pd.read_csv(io.StringIO(r.text)).assign(file_date=d)

    with cf.ThreadPoolExecutor(6) as ex:
        frames = [f for f in ex.map(one, days) if f is not None and len(f)]
    log.info("history: %d daily Odds API files", len(frames))
    return frames


def import_lines(sbr: str | None = None, oddsapi_dir: str | None = None) -> pd.DataFrame:
    games = load_games()
    if games.empty:
        raise SystemExit("games.csv is empty - load games first (nhl.sources.nhle or --games)")
    panel = _read_csv(sbr or config.HISTORY_SBR_URL)
    a = sbr_lines(panel)
    a = a[a["game_id"].isin(set(games["game_id"].astype(str)))]
    b = oddsapi_lines(fetch_oddsapi_history(local_dir=oddsapi_dir), games)
    out = upsert_lines(pd.concat([a, b], ignore_index=True))
    log.info("lines: %d SBR closing rows, %d Odds API noon rows, %d total", len(a), len(b), len(out))
    return out


def bootstrap_games(sbr: str | None = None, goalies: str | None = None) -> pd.DataFrame:
    """games.csv from the same archive's NHL-API-derived results and goalie logs.

    For when the NHL API itself cannot be reached. The game ids are the NHL's own, so a later
    refresh from the API lines up row for row and simply replaces these.
    """
    panel = _read_csv(sbr or config.HISTORY_SBR_URL)
    p = panel[(panel["season"] >= config.LOAD_FROM_SEASON) & (panel["home_away"] == "H")]
    gl = _read_csv(goalies or config.HISTORY_GOALIES_URL)
    gl = gl[gl["season"] >= config.LOAD_FROM_SEASON].copy()
    gl["team"] = gl["team"].map(_code)
    gl["game_id"] = gl["game_id"].astype(str)
    agg = gl.groupby(["game_id", "team"]).agg(sa=("shots_against", "sum"),
                                                ga=("goals_against", "sum")).reset_index()
    st = (gl[gl["started"] == 1].sort_values("toi", ascending=False)
            .drop_duplicates(["game_id", "team"])[["game_id", "team", "goalie_id", "goalie"]])
    gt = agg.merge(st, on=["game_id", "team"], how="left")
    games = pd.DataFrame({
        "game_id": p["game_id"].astype(str), "season": p["season"].astype(int),
        "game_type": np.where(p["game_type"] == "playoff", "P", "R"), "date": p["date"],
        "start_et": "", "home_team": p["team"].map(_code), "away_team": p["opponent"].map(_code),
        "home_goals": p["score_for"].astype(float), "away_goals": p["score_against"].astype(float),
        "decided_in": p["decided_in"], "completed": True, "source": "archive"})
    games = games.dropna(subset=["home_team", "away_team"])
    from .nhle import attach_goalies
    games = attach_goalies(games, gt).reindex(columns=GAME_COLS)
    merged = _merge(load_games(), games)
    config.ensure_dirs()
    merged.to_csv(config.GAMES, index=False)
    log.info("games: bootstrapped %d games from the archive (%d with a starter)", len(games),
             int(games["home_goalie_id"].notna().sum()))
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", action="store_true")
    ap.add_argument("--games", action="store_true")
    ap.add_argument("--sbr", help="local copy of the SBR/NHL panel csv")
    ap.add_argument("--goalies", help="local copy of the goalie starts csv")
    ap.add_argument("--oddsapi-dir", help="local folder of odds_YYYY-MM-DD.csv files")
    a = ap.parse_args(argv)
    if a.games:
        bootstrap_games(a.sbr, a.goalies)
    if a.lines:
        import_lines(a.sbr, a.oddsapi_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
