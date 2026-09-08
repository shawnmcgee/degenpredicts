"""nflverse client - the spine of the NFL pipeline.

There is no CFBD equivalent for the NFL, and nflverse is better than one for this design's
purposes: it is a set of static CSVs served from GitHub, free, keyless and unmetered. That
means no quota to blow, no secret to rotate, and a backfill that is bounded by bandwidth
rather than by an API plan.

Three feeds, each answering one of the questions the college pipeline answered from CFBD:

``games.csv`` (nflverse/nfldata)
    Schedule and results back to 1999, and - the part that matters - the **closing spread,
    total and moneyline** for essentially every game. That is what lets the market-aware models
    train from day one instead of after a season of self-logging. It also carries rest days,
    divisional flag, roof, surface, temperature, wind, stadium and the starting quarterbacks,
    including for games that have not been played yet.

``stats_team_week_<season>.csv`` (nflverse-data)
    Team-week offensive EPA and play counts. Aggregated and opponent-adjusted here into a
    season-level offence/defence EPA-per-play rating - this pipeline's SP+ analogue. Joined
    only from season-1, so it cannot leak.

``snap_counts`` + ``rosters`` + ``players`` (nflverse-data)
    Snap-weighted roster continuity - the returning-production analogue. Snap counts key on
    Pro-Football-Reference ids and rosters key on GSIS ids, so the two are joined through the
    ``players`` crosswalk. Doing it naively on ``pfr_id`` looks like it works and silently
    drops almost every offensive lineman, which makes offensive continuity read ~0.39 instead
    of ~0.74. That is exactly the kind of quiet wrongness this module exists to prevent.

**Sign convention.** nflverse quotes ``spread_line`` as the home team's expected margin:
+3 means the home team is favoured by 3. CFBD quotes the opposite: -3 means home favoured by 3.
Everything downstream of this module - features, predict, grade, the site - was written against
the CFBD convention, so the sign is flipped exactly once, here, in :func:`_to_spread_home`, and
:func:`tests/test_nfl.py::test_spread_sign_convention` pins it.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# Everything below reads paths and settings as ``config.NAME`` at call time rather than
# importing the values. `from ..config import config.GAMES` would freeze the path at import, so a
# later `DEGEN_DATA` override - which the tests and the docs both rely on - would silently be
# ignored by whichever module happened to be imported first, and the pipeline would read and
# write the real repo data while believing it was sandboxed.
from .. import config
from ..config import ensure_dirs, season_of, today_et
from ..teams import DEFAULT_HOME_STADIUM, UnknownTeam, canon
from core.http import get

log = logging.getLogger(__name__)

GAME_COLS = ["game_id", "season", "week", "season_type", "playoff_round", "date", "tip_et",
             "kickoff_utc", "start_time_tbd", "weekday", "home_team", "away_team",
             "home_points", "away_points", "total_points", "home_margin", "neutral_site",
             "div_game", "home_div", "away_div", "home_rest", "away_rest", "roof", "surface",
             "temp", "wind", "stadium_id", "home_qb", "away_qb", "completed"]

LINE_COLS = ["game_id", "season", "week", "date", "home_team", "away_team", "provider",
             "spread_home", "spread_open", "total_line", "total_open", "home_ml", "away_ml",
             "n_providers"]


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------
def _csv(url: str, **kw) -> pd.DataFrame:
    """Fetch a CSV into a DataFrame, or an empty frame if the URL is unavailable.

    Missing seasons are normal, not exceptional: ``stats_team_week`` for the current season
    does not exist until the first Sunday, and ``snap_counts`` does not exist before 2013.
    """
    r = get(url)
    if r is None or r.status_code != 200:
        log.info("nflverse: %s unavailable (%s)", url.rsplit("/", 1)[-1],
                 getattr(r, "status_code", "no response"))
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(r.text), low_memory=False, **kw)
    except (ValueError, pd.errors.ParserError) as e:
        log.warning("nflverse: could not parse %s (%s)", url, e)
        return pd.DataFrame()


def _release(asset: str, name: str) -> pd.DataFrame:
    return _csv(f"{config.NFLVERSE_RELEASES}/{asset}/{name}")


def _str(v) -> str:
    """Text field, with pandas' float NaN rendered as empty rather than the string "nan".

    ``str(float("nan"))`` is ``"nan"``, which is truthy, renders on the page and compares
    equal to itself - so a missing quarterback would silently look like a real one named nan.
    """
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    t = str(v).strip()
    return "" if t.lower() in ("nan", "none", "<na>") else t


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _to_spread_home(spread_line) -> float:
    """nflverse ``spread_line`` (+3 = home favoured by 3) -> CFBD ``spread_home`` (-3 = same).

    The single point in this pipeline where the sign is flipped. Pinned by a test.
    """
    v = _num(spread_line)
    return -v if v == v else float("nan")


def _kickoff(gameday, gametime) -> tuple[object, str, str, bool]:
    """(date, display label, ISO timestamp, time-is-TBD). All times are Eastern.

    nflverse leaves ``gametime`` blank for games whose slot has not been announced - most of
    the late-season flexed Sunday slate, weeks ahead. Rendering a placeholder kickoff is worse
    than saying nothing, so those carry no time and the site shows "Time TBD".
    """
    try:
        d = pd.Timestamp(gameday).date()
    except (ValueError, TypeError):
        return None, "", "", True
    t = str(gametime or "").strip()
    if not t or t.lower() in ("nan", "none"):
        return d, "", "", True
    try:
        ts = pd.Timestamp(f"{d} {t}").tz_localize(config.ET)
    except (ValueError, TypeError):
        return d, "", "", True
    return d, ts.strftime("%a %b %d, %-I:%M %p"), ts.isoformat(), False


# ---------------------------------------------------------------------------------
# Games and lines
# ---------------------------------------------------------------------------------
def fetch_schedule() -> pd.DataFrame:
    """The whole nflverse schedule file: every season, one HTTP call, ~2 MB."""
    raw = _csv(config.NFLVERSE_SCHEDULE)
    if raw.empty:
        log.warning("nflverse schedule fetch returned nothing")
    return raw


def _explode(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the schedule file into our games and lines tables."""
    from ..teams import DIVISIONS

    games, lines = [], []
    for g in raw.itertuples(index=False):
        gdate, label, iso, tbd = _kickoff(getattr(g, "gameday", None),
                                          getattr(g, "gametime", None))
        if gdate is None:
            continue
        try:
            home, away = canon(g.home_team), canon(g.away_team)
        except UnknownTeam:
            # Loud, and skip the row rather than inventing a franchise. CI asserts this
            # never fires on the committed data.
            log.error("skipping %s: unmapped team code (%s / %s)",
                      getattr(g, "game_id", "?"), g.home_team, g.away_team)
            continue
        season = int(g.season)
        hp, ap = _num(getattr(g, "home_score", None)), _num(getattr(g, "away_score", None))
        done = hp == hp and ap == ap
        gtype = _str(getattr(g, "game_type", None)) or "REG"
        surface = _str(getattr(g, "surface", None)).lower()
        games.append({
            "game_id": str(g.game_id), "season": season, "week": int(g.week),
            "season_type": gtype, "playoff_round": config.playoff_round(gtype),
            "date": gdate, "tip_et": label, "kickoff_utc": iso, "start_time_tbd": tbd,
            "weekday": _str(getattr(g, "weekday", None)),
            "home_team": home, "away_team": away,
            "home_points": hp, "away_points": ap,
            "total_points": hp + ap if done else float("nan"),
            "home_margin": hp - ap if done else float("nan"),
            "neutral_site": _str(getattr(g, "location", None)).lower() == "neutral",
            "div_game": bool(_num(getattr(g, "div_game", 0)) == 1),
            "home_div": DIVISIONS.get(home, ""), "away_div": DIVISIONS.get(away, ""),
            "home_rest": _num(getattr(g, "home_rest", None)),
            "away_rest": _num(getattr(g, "away_rest", None)),
            "roof": _str(getattr(g, "roof", None)).lower(),
            "surface": surface,
            "temp": _num(getattr(g, "temp", None)), "wind": _num(getattr(g, "wind", None)),
            "stadium_id": _str(getattr(g, "stadium_id", None)).upper(),
            "home_qb": _str(getattr(g, "home_qb_name", None)),
            "away_qb": _str(getattr(g, "away_qb_name", None)),
            "completed": done,
        })
        lines.append({
            "game_id": str(g.game_id), "season": season, "week": int(g.week), "date": gdate,
            "home_team": home, "away_team": away, "provider": "nflverse_close",
            "spread_home": _to_spread_home(getattr(g, "spread_line", None)),
            # nflverse publishes the closing number only. There is no opening line in this
            # feed, so the line-movement features the CFB model uses have no historical
            # counterpart and are deliberately absent rather than filled with zeros.
            "spread_open": float("nan"), "total_open": float("nan"),
            "total_line": _num(getattr(g, "total_line", None)),
            "home_ml": _num(getattr(g, "home_moneyline", None)),
            "away_ml": _num(getattr(g, "away_moneyline", None)),
            "n_providers": 1,
        })
    return pd.DataFrame(games, columns=GAME_COLS), pd.DataFrame(lines, columns=LINE_COLS)


def load_games() -> pd.DataFrame:
    if config.GAMES.exists():
        df = pd.read_csv(config.GAMES, dtype={"game_id": str, "stadium_id": str}, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        return df
    return pd.DataFrame(columns=GAME_COLS)


def load_lines() -> pd.DataFrame:
    if config.LINES.exists():
        df = pd.read_csv(config.LINES, dtype={"game_id": str}, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        return df
    return pd.DataFrame(columns=LINE_COLS)


def update_games(first_season: int | None = None) -> pd.DataFrame:
    """Refresh the whole schedule. One request; there is no incremental endpoint and no need
    for one - the file is a couple of megabytes and always current."""
    raw = fetch_schedule()
    if raw.empty:
        log.warning("keeping cached games: nflverse schedule unavailable")
        return load_games()
    first = config.FIRST_SEASON if first_season is None else first_season
    raw = raw[raw["season"] >= first]
    games, lines = _explode(raw)
    if games.empty:
        return load_games()
    ensure_dirs()
    games.sort_values(["date", "game_id"]).to_csv(config.GAMES, index=False)
    lines.sort_values(["date", "game_id"]).to_csv(config.LINES, index=False)
    log.info("games: %d rows, seasons %s-%s, %d final, %d with a closing line",
             len(games), games.season.min(), games.season.max(),
             int(games.completed.sum()), int(lines.spread_home.notna().sum()))
    return load_games()


def update_lines(*_a, **_kw) -> pd.DataFrame:
    """Lines ship inside the schedule file, so this is just a read.

    Kept as a separate entry point so the NFL pipeline's call sequence reads the same as the
    college one, where lines really are a second endpoint.
    """
    return load_lines()


# ---------------------------------------------------------------------------------
# Prior-season EPA: this pipeline's SP+
# ---------------------------------------------------------------------------------
EPA_COLS = ["season", "team", "off_epa_play", "def_epa_play", "off_epa_adj", "def_epa_adj",
            "net_epa_adj", "plays"]


def fetch_team_week(season: int) -> pd.DataFrame:
    return _release("stats_team", f"stats_team_week_{season}.csv")


def season_epa(season: int, weekly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Opponent-adjusted offence and defence EPA per play for one regular season.

    ``passing_epa`` and ``rushing_epa`` are EPA *totals*, so they are divided by plays -
    dropbacks (attempts + sacks taken) plus carries - to get a rate. Defence is the same
    quantity measured from the other side of the ledger: the EPA per play a team's opponents
    generated against it, so **lower is better** for ``def_epa_*``.

    The adjustment is the obvious iterative one: subtract the average quality of the units you
    faced, repeat until it stops moving. With 17 games and a partly divisional schedule this
    matters - an NFC South offence and an AFC North offence did not face the same defences.
    """
    d = weekly if weekly is not None else fetch_team_week(season)
    if d.empty or "opponent_team" not in d.columns:
        return pd.DataFrame(columns=EPA_COLS)
    d = d[d["season_type"].astype(str).str.upper() == "REG"].copy()
    if d.empty:
        return pd.DataFrame(columns=EPA_COLS)
    for c in ("attempts", "sacks_suffered", "carries", "passing_epa", "rushing_epa"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce").fillna(0.0)
    d["team"] = d["team"].map(canon)
    d["opponent_team"] = d["opponent_team"].map(canon)
    d["plays"] = d["attempts"] + d["sacks_suffered"] + d["carries"]
    d["epa"] = d["passing_epa"] + d["rushing_epa"]
    d = d[d["plays"] > 0]
    if d.empty:
        return pd.DataFrame(columns=EPA_COLS)

    off = d.groupby("team").agg(epa=("epa", "sum"), plays=("plays", "sum"))
    off["off_epa_play"] = off["epa"] / off["plays"]
    dfn = d.groupby("opponent_team").agg(epa=("epa", "sum"), plays=("plays", "sum"))
    dfn["def_epa_play"] = dfn["epa"] / dfn["plays"]
    tbl = off[["off_epa_play", "plays"]].join(dfn[["def_epa_play"]], how="inner")
    if tbl.empty:
        return pd.DataFrame(columns=EPA_COLS)

    faced = d.groupby("team")["opponent_team"].apply(list)
    off_adj, def_adj = tbl["off_epa_play"].copy(), tbl["def_epa_play"].copy()
    lg_off, lg_def = tbl["off_epa_play"].mean(), tbl["def_epa_play"].mean()
    for _ in range(25):
        opp_def = pd.Series({t: float(np.mean([def_adj.get(o, lg_def) - lg_def for o in ops]))
                             for t, ops in faced.items()})
        opp_off = pd.Series({t: float(np.mean([off_adj.get(o, lg_off) - lg_off for o in ops]))
                             for t, ops in faced.items()})
        new_off = tbl["off_epa_play"] - opp_def.reindex(tbl.index).fillna(0.0)
        new_def = tbl["def_epa_play"] - opp_off.reindex(tbl.index).fillna(0.0)
        if (new_off - off_adj).abs().max() < 1e-6 and (new_def - def_adj).abs().max() < 1e-6:
            off_adj, def_adj = new_off, new_def
            break
        off_adj, def_adj = new_off, new_def

    out = tbl.reset_index().rename(columns={"index": "team"})
    out["season"] = int(season)
    out["off_epa_adj"] = off_adj.values
    out["def_epa_adj"] = def_adj.values
    out["net_epa_adj"] = out["off_epa_adj"] - out["def_epa_adj"]
    return out[EPA_COLS]


def load_epa() -> pd.DataFrame:
    return pd.read_csv(config.EPA_PRIOR) if config.EPA_PRIOR.exists() else pd.DataFrame(columns=EPA_COLS)


def update_epa(seasons: list[int] | None = None) -> pd.DataFrame:
    """Backfill and refresh the EPA table.

    Like SP+ in the college pipeline this is an END-OF-SEASON summary and is only ever joined
    from season-1, never the current one, so it cannot leak. The current season is still
    refreshed each run so that next season's join is ready the moment the season ends.
    """
    this = season_of(today_et())
    cache = load_epa()
    have = set(cache["season"].astype(int)) if len(cache) else set()
    if seasons is None:
        # FIRST_SEASON-1 because season N's features need season N-1's ratings.
        seasons = [s for s in range(config.FIRST_SEASON - 1, this + 1)
                   if s not in have or s >= this - 1]
    frames = [cache[~cache["season"].isin(seasons)]] if len(cache) else []
    for s in seasons:
        f = season_epa(s)
        if len(f):
            frames.append(f)
            log.info("  EPA %s: %d teams", s, len(f))
    if not frames:
        log.warning("no team EPA retrieved")
        return cache
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["season", "team"], keep="last")
    ensure_dirs()
    out.sort_values(["season", "team"]).to_csv(config.EPA_PRIOR, index=False)
    log.info("team EPA: %d team-seasons", len(out))
    return load_epa()


# ---------------------------------------------------------------------------------
# Roster continuity: this pipeline's returning production
# ---------------------------------------------------------------------------------
CONT_COLS = ["season", "team", "cont_off", "cont_def", "snaps_off", "snaps_def"]
_PLAYER_XWALK: dict[str, str] | None = None


def player_crosswalk() -> dict[str, str]:
    """pfr_id -> gsis_id, from nflverse's master players table (~7 MB, fetched once).

    Snap counts identify players by Pro-Football-Reference id; rosters identify them by GSIS
    id. Rosters *also* carry a ``pfr_id`` column and joining on that is the obvious move -
    but it is ~30% null and the nulls are almost entirely offensive linemen, who are a huge
    share of offensive snaps. Joining that way reports offensive continuity around 0.39
    league-wide instead of the correct ~0.74, and nothing about it looks broken.
    """
    global _PLAYER_XWALK
    if _PLAYER_XWALK is None:
        p = _release("players", "players.csv")
        if p.empty or "pfr_id" not in p or "gsis_id" not in p:
            log.warning("players crosswalk unavailable - roster continuity will be empty")
            _PLAYER_XWALK = {}
        else:
            p = p[["pfr_id", "gsis_id"]].dropna()
            _PLAYER_XWALK = dict(zip(p["pfr_id"], p["gsis_id"]))
            log.info("players crosswalk: %d pfr->gsis ids", len(_PLAYER_XWALK))
    return _PLAYER_XWALK


def season_continuity(season: int, snaps: pd.DataFrame | None = None,
                      roster: pd.DataFrame | None = None,
                      xwalk: dict[str, str] | None = None) -> pd.DataFrame:
    """Share of season-1's snaps being played by players on the season-N roster.

    Reported separately for offence and defence because they churn at very different rates -
    in a typical offseason offensive continuity runs high-60s to mid-70s and defensive
    continuity lower. It is known before week 1, which is the whole point: it is what carries
    the model through the weeks when in-season data does not exist yet.
    """
    prev = season - 1
    s = snaps if snaps is not None else _release("snap_counts", f"snap_counts_{prev}.csv")
    r = roster if roster is not None else _release("rosters", f"roster_{season}.csv")
    if s.empty or r.empty or "pfr_player_id" not in s or "gsis_id" not in r:
        return pd.DataFrame(columns=CONT_COLS)
    xw = player_crosswalk() if xwalk is None else xwalk
    if not xw:
        return pd.DataFrame(columns=CONT_COLS)

    s = s[s["game_type"].astype(str).str.upper() == "REG"].copy()
    if s.empty:
        return pd.DataFrame(columns=CONT_COLS)
    s["gsis"] = s["pfr_player_id"].map(xw)
    s = s.dropna(subset=["gsis"])
    s["team"] = s["team"].map(canon)
    for c in ("offense_snaps", "defense_snaps"):
        s[c] = pd.to_numeric(s.get(c), errors="coerce").fillna(0.0)
    tot = s.groupby(["team", "gsis"])[["offense_snaps", "defense_snaps"]].sum().reset_index()

    r = r.dropna(subset=["gsis_id"]).copy()
    r["team"] = r["team"].map(canon)
    on_roster = set(zip(r["team"], r["gsis_id"]))
    tot["back"] = [(t, g) in on_roster for t, g in zip(tot["team"], tot["gsis"])]

    team_tot = tot.groupby("team")[["offense_snaps", "defense_snaps"]].sum()
    ret = tot[tot["back"]].groupby("team")[["offense_snaps", "defense_snaps"]].sum()
    ret = ret.reindex(team_tot.index).fillna(0.0)
    out = pd.DataFrame({
        "season": int(season), "team": team_tot.index,
        "cont_off": (ret["offense_snaps"] / team_tot["offense_snaps"].replace(0, np.nan)).values,
        "cont_def": (ret["defense_snaps"] / team_tot["defense_snaps"].replace(0, np.nan)).values,
        "snaps_off": team_tot["offense_snaps"].values,
        "snaps_def": team_tot["defense_snaps"].values,
    })
    return out[CONT_COLS]


def load_continuity() -> pd.DataFrame:
    return pd.read_csv(config.CONTINUITY) if config.CONTINUITY.exists() else pd.DataFrame(columns=CONT_COLS)


def update_continuity(seasons: list[int] | None = None) -> pd.DataFrame:
    this = season_of(today_et())
    cache = load_continuity()
    have = set(cache["season"].astype(int)) if len(cache) else set()
    if seasons is None:
        # Needs season-1 snap counts, and snap counts start in 2013.
        start = max(config.FIRST_SEASON, config.FIRST_SNAP_SEASON + 1)
        seasons = [s for s in range(start, this + 1) if s not in have or s >= this]
    frames = [cache[~cache["season"].isin(seasons)]] if len(cache) else []
    for s in seasons:
        f = season_continuity(s)
        if len(f):
            frames.append(f)
            log.info("  continuity %s: %d teams (off %.2f / def %.2f league mean)",
                     s, len(f), f["cont_off"].mean(), f["cont_def"].mean())
    if not frames:
        log.warning("no roster continuity retrieved")
        return cache
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["season", "team"], keep="last")
    ensure_dirs()
    out.sort_values(["season", "team"]).to_csv(config.CONTINUITY, index=False)
    log.info("roster continuity: %d team-seasons", len(out))
    return load_continuity()


# ---------------------------------------------------------------------------------
# Home venue by season - used for travel, and correct across relocations by construction
# ---------------------------------------------------------------------------------
def home_stadiums(games: pd.DataFrame) -> dict[tuple[int, str], str]:
    """(season, team) -> the stadium the team actually played its home games in.

    Derived from the schedule rather than hard-coded, which is what makes the Raiders' move to
    Las Vegas, the Rams' two years in the Coliseum and a team's designated "home" game in
    London all come out right with no special cases. Neutral-site games are excluded so a
    London game does not become a team's home venue.
    """
    if games.empty:
        return {}
    d = games[~games["neutral_site"].astype(bool)]
    d = d[d["stadium_id"].astype(str).str.len() > 0]
    if d.empty:
        return {}
    modal = (d.groupby(["season", "home_team"])["stadium_id"]
               .agg(lambda s: s.value_counts().idxmax()))
    return {(int(s), t): v for (s, t), v in modal.items()}


def origin_stadium(lookup: dict, season: int, team: str) -> str | None:
    """Where a team travels from. Falls back a season, then to the franchise default."""
    return (lookup.get((season, team))
            or lookup.get((season - 1, team))
            or DEFAULT_HOME_STADIUM.get(team))
