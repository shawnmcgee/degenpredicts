"""Match history with prices, from a GitHub-hosted aggregate of the football-data archives.

This is the pipeline's primary source, and it exists because the original one could not be
reached. ``football-data.co.uk`` is the canonical archive for this sport, but from a GitHub
Actions runner it accepts the TCP connection and then stalls - three separate runs returned zero
bytes - so nothing built on it can train. See ``sources/footballdata.py``, which is kept working
and can be re-selected with ``DEGEN_EPL_SOURCE=footballdata`` if the host ever comes back.

The replacement is one CSV on ``raw.githubusercontent.com`` - the same host nflverse is served
from, which the NFL pipeline has been using reliably for months. It carries every division this
project cares about, every season since 2000, and crucially the three price columns the model
needs:

| Column                              | What it gives                         |
|-------------------------------------|---------------------------------------|
| ``OddHome/OddDraw/OddAway``         | 1X2, plus ``Max*`` across the panel   |
| ``Over25/Under25``                  | over/under 2.5 goals                  |
| ``HandiSize/HandiHome/HandiAway``   | Asian handicap line and both prices   |

Three properties make this a straight swap rather than a rewrite:

* **The club names are football-data's own spelling**, so ``epl.teams.canon`` resolves every
  name in E0 and E1 with no new aliases at all.
* **The handicap sign convention is the same** - negative means the home side gives goals -
  which is verified on the data itself rather than assumed, in
  ``tests/test_epl.py::test_matchdata_handicap_sign_matches_the_repo_convention``.
* **It is one request instead of fifty.** The old source needed a file per league per season;
  this is a single 45 MB fetch covering all of them, which is why the whole class of
  "unreachable host multiplied by fifty files" problem disappears rather than being tuned.

**The one real downgrade, stated plainly.** football-data.co.uk publishes explicit *closing*
columns (``PSCH``, ``AHCh``, ``PC>2.5``) and this aggregate does not distinguish opening from
closing. So ``mae_market_baseline`` here is measured against a number that may be softer than
the true close, and a model that appears to beat it may only be beating an opening price. Every
row is therefore marked ``is_closing = False``, the training metadata records it, and any edge
this source appears to show deserves more suspicion than the same edge measured against a close.
"""
from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd

from .. import config
from ..config import ensure_dirs
from ..teams import UnknownClub, canon
from ..http import get
from .footballdata import (GAME_COLS, LINE_COLS, _load, _market_view, _merge, _num, _text,
                           make_game_id)

log = logging.getLogger(__name__)

# Their column -> ours. Kept as data rather than inline so the mapping is auditable in one
# place, which is the lesson the era-change in the other source taught.
PRICE_1X2 = [("OddHome", "OddDraw", "OddAway"), ("MaxHome", "MaxDraw", "MaxAway")]
PRICE_OU = [("Over25", "Under25"), ("MaxOver25", "MaxUnder25")]
PRICE_AH = [("HandiSize", "HandiHome", "HandiAway")]

_cache: pd.DataFrame | None = None


def fetch_all(force: bool = False) -> pd.DataFrame:
    """The whole aggregate: every division, every season, one request.

    Held in memory for the life of the process because a retrain reads it once for the target
    division and again for the division below, and re-downloading 45 MB to do that would be
    the same mistake the old source made in a different costume.
    """
    global _cache
    if _cache is not None and not force:
        return _cache
    r = get(config.MATCHDATA_URL)
    if r is None or r.status_code != 200:
        log.error("match archive unavailable (%s) at %s",
                  getattr(r, "status_code", "no response"), config.MATCHDATA_URL)
        return pd.DataFrame()
    try:
        _cache = pd.read_csv(io.StringIO(r.text), low_memory=False)
    except (ValueError, pd.errors.ParserError) as e:
        log.error("could not parse the match archive (%s)", e)
        return pd.DataFrame()
    log.info("match archive: %d rows, %d divisions", len(_cache),
             _cache["Division"].nunique() if "Division" in _cache else 0)
    return _cache


def _pick(row, candidates, n):
    for cand in candidates:
        vals = [_num(row.get(c)) for c in cand[:n]]
        if all(v == v for v in vals):
            return "/".join(cand[:n]), vals
    return "", [np.nan] * n


def parse(raw: pd.DataFrame, league: str | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One division out of the aggregate, in this pipeline's schema. Pure: no network, no disk."""
    league = league or config.LEAGUE
    if raw.empty or "Division" not in raw.columns:
        return pd.DataFrame(columns=GAME_COLS), pd.DataFrame(columns=LINE_COLS)
    d = raw[raw["Division"].astype(str).str.strip() == league].copy()
    if d.empty:
        return pd.DataFrame(columns=GAME_COLS), pd.DataFrame(columns=LINE_COLS)
    # Dates here are ISO, unlike football-data's day-first British ones. Forced rather than
    # inferred for the same reason: a silently reordered season breaks the leak-free replay.
    d["_date"] = pd.to_datetime(d["MatchDate"], format="ISO8601", errors="coerce")
    d = d[d["_date"].notna()]

    games, lines = [], []
    for r in d.to_dict("records"):
        try:
            home, away = canon(r["HomeTeam"]), canon(r["AwayTeam"])
        except (UnknownClub, KeyError, TypeError) as e:
            log.error("skipping %s v %s: %s", r.get("HomeTeam"), r.get("AwayTeam"), e)
            continue
        gdate = r["_date"].date()
        season = config.season_of(gdate)
        hg, ag = _num(r.get("FTHome")), _num(r.get("FTAway"))
        done = hg == hg and ag == ag
        gid = make_game_id(season, home, away)
        kt = _text(r.get("MatchTime"))
        games.append({
            "game_id": gid, "season": season, "league": league, "date": gdate,
            "kickoff": kt,
            "kickoff_uk": f"{gdate:%a %d %b}" + (f", {kt}" if kt else ""),
            "matchweek": config.matchweek(gdate, season),
            "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag,
            "supremacy": hg - ag if done else np.nan,
            "total_goals": hg + ag if done else np.nan,
            "result": _text(r.get("FTResult")),
            "ht_home_goals": _num(r.get("HTHome")), "ht_away_goals": _num(r.get("HTAway")),
            "home_shots": _num(r.get("HomeShots")), "away_shots": _num(r.get("AwayShots")),
            "home_sot": _num(r.get("HomeTarget")), "away_sot": _num(r.get("AwayTarget")),
            "home_corners": _num(r.get("HomeCorners")),
            "away_corners": _num(r.get("AwayCorners")),
            "home_cards": _num(r.get("HomeYellow")) + 2 * _num(r.get("HomeRed")),
            "away_cards": _num(r.get("AwayYellow")) + 2 * _num(r.get("AwayRed")),
            "referee": "",          # not carried by this aggregate
            "no_crowd": int(config.no_crowd(gdate)),
            "completed": done,
        })

        src = []
        lbl, (ph, pdw, pa) = _pick(r, PRICE_1X2, 3)
        if lbl:
            src.append(lbl)
        lbl_ou, (po, pu) = _pick(r, PRICE_OU, 2)
        if lbl_ou:
            src.append(lbl_ou)
        lbl_ah, (ah, pah_h, pah_a) = _pick(r, PRICE_AH, 3)
        if lbl_ah:
            src.append(lbl_ah)
        if not src:
            continue
        o = {"odds_source": ",".join(src),
             # This aggregate does not distinguish opening from closing prices. Claiming
             # otherwise would overstate how sharp the baseline the model is measured against
             # actually is, which is the one number in this project that must not be flattered.
             "is_closing": False,
             "price_home": ph, "price_draw": pdw, "price_away": pa,
             "ah_home": ah, "price_ah_home": pah_h, "price_ah_away": pah_a,
             "total_line": 2.5 if (po == po and pu == pu) else np.nan,
             "price_over": po, "price_under": pu}
        lines.append({"game_id": gid, "season": season, "date": gdate,
                      "home_team": home, "away_team": away, **o, **_market_view(o)})

    g = pd.DataFrame(games)
    ln = pd.DataFrame(lines)
    return (g.reindex(columns=GAME_COLS) if len(g) else pd.DataFrame(columns=GAME_COLS),
            ln.reindex(columns=LINE_COLS) if len(ln) else pd.DataFrame(columns=LINE_COLS))


def update_games(seasons: list[int] | None = None, league: str | None = None) -> pd.DataFrame:
    """Refresh the cache from the archive and return the games table."""
    ensure_dirs()
    league = league or config.LEAGUE
    raw = fetch_all()
    if raw.empty:
        log.warning("archive returned nothing - keeping the existing cache")
        return load_games()
    g, ln = parse(raw, league)
    if g.empty:
        log.warning("archive holds no %s matches", league)
        return load_games()
    if seasons is not None:
        g = g[g["season"].isin(seasons)]
        ln = ln[ln["season"].isin(seasons)]
    g = g[g["season"] >= config.LOAD_FROM_SEASON]
    ln = ln[ln["season"] >= config.LOAD_FROM_SEASON]
    games = _merge(load_games(), g)
    lines = _merge(load_lines(), ln)
    games.to_csv(config.GAMES, index=False)
    lines.to_csv(config.LINES, index=False)
    log.info("%s: %d matches, %d priced, seasons %s", league, len(games), len(lines),
             f"{int(games.season.min())}-{int(games.season.max())}" if len(games) else "none")
    return games


def lower_games(seasons: list[int]) -> pd.DataFrame:
    """The division below, for the promoted-club prior. Free: the archive is already in memory."""
    if not config.LEAGUE_BELOW:
        return pd.DataFrame(columns=GAME_COLS)
    g, _ln = parse(fetch_all(), config.LEAGUE_BELOW)
    return g[g["season"].isin(seasons)] if len(g) else g


def load_games() -> pd.DataFrame:
    return _load(config.GAMES)


def load_lines() -> pd.DataFrame:
    return _load(config.LINES)


def load_strength() -> pd.DataFrame:
    from .footballdata import load_strength as _s
    return _s()


def build_strength(games: pd.DataFrame | None = None, fetch: bool = True) -> pd.DataFrame:
    from .footballdata import build_strength as _b
    games = load_games() if games is None else games
    lower = lower_games(sorted(int(s) for s in games["season"].unique())) if fetch \
        else pd.DataFrame()
    return _b(games, fetch=False, lower=lower)


def coverage_report(games: pd.DataFrame, lines: pd.DataFrame) -> list[dict]:
    from .footballdata import coverage_report as _c
    return _c(games, lines)


def fetch_fixtures(league: str | None = None) -> pd.DataFrame:
    """The archive holds played matches only, so the board comes from the live price feed.

    Returning empty here is not a failure - it is this source telling ``predict`` to fall back
    to :func:`epl.sources.odds.board`, which is the only feed in the project that knows about
    matches that have not been played yet.
    """
    return pd.DataFrame()
