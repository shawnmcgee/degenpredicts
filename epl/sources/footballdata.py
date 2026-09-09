"""football-data.co.uk client - the spine of the EPL pipeline.

This is the sport's nflverse: one static CSV per league per season, no key, no quota, served
over plain HTTPS. And like nflverse it carries the part that actually matters, which is not the
results but the **bookmakers' prices**. Every season file holds the 1X2 prices from a panel of
books, the Asian handicap line and its two prices, and the over/under 2.5 goals prices. That is
what lets the market-aware models train from day one rather than after a season of self-logging.

Two feeds:

``mmz4281/<season>/<league>.csv``
    One season of one division. Results, shots, shots on target, corners, cards, referee, and
    the odds columns described below. ~30 KB.

``fixtures.csv``
    Every forthcoming match across all the divisions this site covers, with current prices.
    This is the board source, and it is the reason the pipeline needs no separate schedule feed.

**The odds columns changed shape in 2019-20 and that is the single biggest trap in this file.**
Before then the aggregate columns were Betbrain's (``BbAvH``, ``BbAHh``, ``BbAv>2.5``); from
2019-20 Betbrain was dropped and replaced with ``AvgH`` / ``AHh`` / ``Avg>2.5``, and a whole
parallel set of **closing** columns appeared (``PSCH``, ``AHCh``, ``PC>2.5``). Reading only one
era's spelling does not raise - it silently yields a NaN for every row in the other era, and the
market-aware models then train on whichever half of history happened to match, with no
indication anything is wrong. So every quantity is resolved through an ordered candidate list,
the winner is recorded per row in ``odds_source``, and :func:`coverage_report` exists so a
training run can print what it actually found.

**Two sign and format conventions, each fixed in exactly one place:**

* **Dates are day-first.** football-data writes ``01/02/2024`` for 1 February. Parsed
  month-first that is 2 January - which does not raise, it just reorders the season, and the
  whole leak-free chronological replay silently trains on the future.
* **The Asian handicap is quoted on the home team**: ``-1`` means the home side gives a goal.
  That already matches the ``spread_home`` convention the rest of this repo uses (negative =
  home favoured), so unlike the NFL feed there is no flip. :func:`tests.test_epl` pins it.
"""
from __future__ import annotations

import io
import logging
import os
import re

import numpy as np
import pandas as pd

# Settings are read as ``config.NAME`` at call time, never imported by value: binding
# `config.GAMES` at import would freeze the path, so a later DEGEN_DATA override - which the
# tests rely on - would be ignored by whichever module imported first, and the suite would
# train on real repo data while believing it was sandboxed.
from .. import config
from ..config import ensure_dirs
from ..teams import UnknownClub, canon
from ..http import get

log = logging.getLogger(__name__)

GAME_COLS = ["game_id", "season", "league", "date", "kickoff", "kickoff_uk", "matchweek",
             "home_team", "away_team", "home_goals", "away_goals", "supremacy", "total_goals",
             "result", "ht_home_goals", "ht_away_goals", "home_shots", "away_shots",
             "home_sot", "away_sot", "home_corners", "away_corners", "home_cards",
             "away_cards", "referee", "no_crowd", "completed"]

LINE_COLS = ["game_id", "season", "date", "home_team", "away_team", "odds_source",
             "is_closing", "price_home", "price_draw", "price_away", "overround_1x2",
             "ah_home", "price_ah_home", "price_ah_away",
             "total_line", "price_over", "price_under",
             "mkt_p_home", "mkt_p_draw", "mkt_p_away", "mkt_sup", "mkt_total"]

# ---------------------------------------------------------------------------------
# Odds column resolution, newest-and-sharpest first
# ---------------------------------------------------------------------------------
# Ordering rationale, applied to all three markets: an explicit CLOSING price beats an opening
# one, because closing is the number every honest evaluation in this repo measures against.
# Pinnacle beats the market average, because it is the sharpest book on the panel and takes the
# largest limits. The market average beats a single soft book. Bet365 is last because it is
# available in every era, which makes it the right backstop and the wrong first choice.
CLOSING_1X2 = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"),
               ("MaxCH", "MaxCD", "MaxCA"), ("B365CH", "B365CD", "B365CA"),
               ("WHCH", "WHCD", "WHCA"), ("VCCH", "VCCD", "VCCA")]
OPENING_1X2 = [("PSH", "PSD", "PSA"), ("PH", "PD", "PA"), ("AvgH", "AvgD", "AvgA"),
               ("BbAvH", "BbAvD", "BbAvA"), ("B365H", "B365D", "B365A"),
               ("WHH", "WHD", "WHA"), ("VCH", "VCD", "VCA"), ("BWH", "BWD", "BWA"),
               ("IWH", "IWD", "IWA"), ("LBH", "LBD", "LBA")]

CLOSING_OU = [("PC>2.5", "PC<2.5"), ("AvgC>2.5", "AvgC<2.5"), ("MaxC>2.5", "MaxC<2.5"),
              ("B365C>2.5", "B365C<2.5")]
OPENING_OU = [("P>2.5", "P<2.5"), ("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5"),
              ("B365>2.5", "B365<2.5"), ("BbMx>2.5", "BbMx<2.5"), ("Max>2.5", "Max<2.5")]

# (handicap column, home price, away price)
CLOSING_AH = [("AHCh", "PCAHH", "PCAHA"), ("AHCh", "AvgCAHH", "AvgCAHA"),
              ("AHCh", "MaxCAHH", "MaxCAHA"), ("AHCh", "B365CAHH", "B365CAHA")]
OPENING_AH = [("AHh", "PAHH", "PAHA"), ("AHh", "AvgAHH", "AvgAHA"),
              ("AHh", "MaxAHH", "MaxAHA"), ("AHh", "B365AHH", "B365AHA"),
              ("BbAHh", "BbAvAHH", "BbAvAHA"), ("BbAHh", "BbMxAHH", "BbMxAHA"),
              ("LBAH", "LBAHH", "LBAHA"), ("GBAH", "GBAHH", "GBAHA")]

# The over/under columns football-data publishes are all struck at 2.5 goals.
OU_LINE = 2.5


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------
# --- primary-host circuit breaker -------------------------------------------------
# A first backfill fetches ~50 files. If the primary host refuses - which is exactly what
# football-data.co.uk does from a cloud runner, answering HTTP 503 - then retrying it once per
# file spends the whole retry budget fifty times over. That is what
# turned a six-file schema check into a twenty-five minute job and would have made a first
# retrain a three-hour one.
#
# So the host gets a small number of chances to answer, and after that it is treated as down for
# the remainder of the process and every later fetch goes straight to the mirror. This is
# per-process state rather than persisted: a new run always re-tests the host, so an outage
# heals by itself on the next scheduled job.
_primary_failures = 0
_primary_wasted = 0.0
_primary_down = False


def reset_primary() -> None:
    """Forget that the primary host was down. Used by the tests, and between CLI invocations."""
    global _primary_failures, _primary_wasted, _primary_down
    _primary_failures, _primary_wasted, _primary_down = 0, 0.0, False


def primary_is_down() -> bool:
    return _primary_down


def primary_wasted() -> float:
    return round(_primary_wasted, 1)


def _note_primary(ok: bool, elapsed: float = 0.0) -> None:
    """Record the outcome of one primary-host fetch, and trip the breaker if either bound is hit.

    Time from SUCCESSFUL requests is deliberately not counted: a host that is merely slow but
    working is still the only source of odds columns, and abandoning it would silently downgrade
    every market-aware model to nothing.
    """
    global _primary_failures, _primary_wasted, _primary_down
    if ok:
        _primary_failures = 0
        return
    _primary_failures += 1
    _primary_wasted += max(elapsed, 0.0)
    if _primary_down:
        return
    over_budget = _primary_wasted >= config.PRIMARY_TIME_BUDGET
    if _primary_failures >= config.PRIMARY_FAILURE_LIMIT or over_budget:
        _primary_down = True
        log.error(
            "football-data.co.uk: %d consecutive failures, %.0fs wasted (%s) - treating it as "
            "DOWN for the rest of this run and serving everything from the results-only mirror. "
            "The mirror carries NO odds columns, so the market-aware models will have no rows. "
            "Re-run once the host is reachable.",
            _primary_failures, _primary_wasted,
            "time budget exceeded" if over_budget else "failure limit reached")


def _csv(url: str) -> pd.DataFrame:
    r = get(url)
    if r is None or r.status_code != 200:
        log.info("football-data: %s unavailable (%s)", url.rsplit("/", 2)[-1],
                 getattr(r, "status_code", "no response"))
        return pd.DataFrame()
    try:
        # These files carry trailing all-empty columns and the occasional stray row; the C
        # parser copes where the default one raises on a ragged line.
        return pd.read_csv(io.StringIO(r.text), encoding_errors="replace",
                           on_bad_lines="skip", low_memory=False)
    except (ValueError, pd.errors.ParserError) as e:
        log.warning("football-data: could not parse %s (%s)", url, e)
        return pd.DataFrame()


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if f == f else float("nan")


def _text(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    t = str(v).strip()
    return "" if t.lower() in ("nan", "none", "<na>") else t


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def make_game_id(season: int, home: str, away: str) -> str:
    """A stable id for a fixture: season plus the two clubs.

    Deliberately does NOT include the date. English football postpones and rearranges matches
    constantly - weather, cup runs, European fixtures, and in one recent season a national
    period of mourning. A date-based id would mint a *new* id when a postponed match is finally
    played, so the pick published for it would never grade and would sit in picks.csv forever.
    In a round robin each ordered pair meets exactly once per season, so this is unique.
    """
    return f"{season}_{_slug(home)}_{_slug(away)}"


def season_url(season: int, league: str | None = None) -> str:
    return f"{config.FOOTBALL_DATA}/{config.season_code(season)}/{league or config.LEAGUE}.csv"


def mirror_url(season: int, league: str | None = None) -> str:
    slug = config.MIRROR_SLUGS.get(league or config.LEAGUE, "")
    if not slug:
        return ""
    return f"{config.FOOTBALL_DATA_MIRROR}/{slug}/season-{config.season_code(season)}.csv"


def _pick(row, candidates, n: int):
    """First candidate tuple whose columns are all present and numeric on this row.

    Returns ``(label, values)`` so the caller can record WHICH source it used. That label is
    written to every line row and summarised by :func:`coverage_report`, because "the market
    columns are populated" and "the market columns are populated from the source I think" are
    different claims, and only the second one is worth training on.
    """
    for cand in candidates:
        vals = [_num(row.get(c)) for c in cand[:n]]
        if all(v == v for v in vals):
            return "/".join(cand[:n]), vals
    return "", [float("nan")] * n


def _resolve_odds(row) -> dict:
    """Everything price-shaped for one match, with its provenance."""
    src, used_closing = [], False

    lbl, (ph, pd_, pa) = _pick(row, CLOSING_1X2, 3)
    if lbl:
        used_closing = True
    else:
        lbl, (ph, pd_, pa) = _pick(row, OPENING_1X2, 3)
    if lbl:
        src.append(lbl)

    lbl_ou, (po, pu) = _pick(row, CLOSING_OU, 2)
    if lbl_ou:
        used_closing = True
    else:
        lbl_ou, (po, pu) = _pick(row, OPENING_OU, 2)
    if lbl_ou:
        src.append(lbl_ou)

    lbl_ah, (ah, pah_h, pah_a) = _pick(row, CLOSING_AH, 3)
    if lbl_ah:
        used_closing = True
    else:
        lbl_ah, (ah, pah_h, pah_a) = _pick(row, OPENING_AH, 3)
    if lbl_ah:
        src.append(lbl_ah)

    return {"odds_source": ",".join(src), "is_closing": bool(used_closing),
            "price_home": ph, "price_draw": pd_, "price_away": pa,
            "ah_home": ah, "price_ah_home": pah_h, "price_ah_away": pah_a,
            "total_line": OU_LINE if (po == po and pu == pu) else float("nan"),
            "price_over": po, "price_under": pu}


def _market_view(o: dict) -> dict:
    """Turn raw prices into vig-free probabilities and the market's implied goal numbers.

    The 1X2 de-vig is three-way and uses Shin's method by default; see :mod:`epl.odds_math` for
    why proportional de-vigging is the wrong default in a market with this much
    favourite-longshot bias.
    """
    from ..odds_math import devig_three, overround, supremacy_from_prices, total_from_prices

    ph, pdw, pa = o["price_home"], o["price_draw"], o["price_away"]
    mh, md, ma = devig_three(ph, pdw, pa)
    # The Asian handicap is the sharpest supremacy number when it exists, because that is the
    # market books actually manage risk on. Backing supremacy out of a 1X2 price is the
    # fallback for the older seasons, where handicaps are not published at all.
    ah = o["ah_home"]
    sup = -float(ah) if ah == ah else supremacy_from_prices(ph, pdw, pa)
    return {"overround_1x2": round(overround(ph, pdw, pa), 4) if ph == ph else float("nan"),
            "mkt_p_home": round(mh, 4) if mh == mh else float("nan"),
            "mkt_p_draw": round(md, 4) if md == md else float("nan"),
            "mkt_p_away": round(ma, 4) if ma == ma else float("nan"),
            "mkt_sup": round(sup, 3) if sup == sup else float("nan"),
            "mkt_total": total_from_prices(o["price_over"], o["price_under"], o["total_line"])}


ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _dates(raw: pd.DataFrame) -> pd.Series:
    """Parse the Date column, choosing the convention rather than letting pandas guess.

    football-data writes BRITISH dates - ``01/02/2024`` is 1 February. Parsed month-first that
    becomes 2 January, which raises nothing, silently reorders a third of every season, and
    breaks the one guarantee this whole project rests on: that a feature for match N only ever
    saw matches 1..N-1. So day-first is forced rather than inferred, because pandas will
    otherwise happily infer a different convention for different chunks of the same column.

    The GitHub mirror writes ISO dates instead, and ISO parsed day-first is both wrong for any
    day past the 12th and noisy about it. The two are unambiguous to tell apart, so the format
    is detected once per file and then forced.
    """
    col = raw["Date"].astype(str).str.strip()
    sample = col[col.ne("") & col.ne("nan")]
    if len(sample) and ISO_DATE.match(sample.iloc[0]):
        return pd.to_datetime(col, format="ISO8601", errors="coerce")
    return pd.to_datetime(col, dayfirst=True, errors="coerce")


def parse_season(raw: pd.DataFrame, season: int, league: str | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one season file into our games and lines tables. Pure: no network, no disk."""
    league = league or config.LEAGUE
    if raw.empty or "HomeTeam" not in raw.columns:
        return pd.DataFrame(columns=GAME_COLS), pd.DataFrame(columns=LINE_COLS)

    raw = raw[raw["HomeTeam"].notna()].copy()
    raw["_date"] = _dates(raw)
    raw = raw[raw["_date"].notna()]

    games, lines = [], []
    for r in raw.to_dict("records"):
        try:
            home, away = canon(r["HomeTeam"]), canon(r["AwayTeam"])
        except UnknownClub as e:
            # Loud, and skip rather than invent a club. CI asserts this never fires on the
            # committed data, so a newly promoted side is caught the week it appears.
            log.error("skipping %s v %s in %s: %s", r.get("HomeTeam"), r.get("AwayTeam"),
                      season, e)
            continue
        gdate = r["_date"].date()
        hg, ag = _num(r.get("FTHG")), _num(r.get("FTAG"))
        done = hg == hg and ag == ag
        gid = make_game_id(season, home, away)
        kt = _text(r.get("Time"))
        games.append({
            "game_id": gid, "season": season, "league": league, "date": gdate,
            "kickoff": kt,
            "kickoff_uk": f"{gdate:%a %d %b}" + (f", {kt}" if kt else ""),
            "matchweek": config.matchweek(gdate, season),
            "home_team": home, "away_team": away,
            "home_goals": hg, "away_goals": ag,
            "supremacy": hg - ag if done else float("nan"),
            "total_goals": hg + ag if done else float("nan"),
            "result": _text(r.get("FTR")),
            "ht_home_goals": _num(r.get("HTHG")), "ht_away_goals": _num(r.get("HTAG")),
            "home_shots": _num(r.get("HS")), "away_shots": _num(r.get("AS")),
            "home_sot": _num(r.get("HST")), "away_sot": _num(r.get("AST")),
            "home_corners": _num(r.get("HC")), "away_corners": _num(r.get("AC")),
            "home_cards": _num(r.get("HY")) + 2 * _num(r.get("HR")),
            "away_cards": _num(r.get("AY")) + 2 * _num(r.get("AR")),
            "referee": _text(r.get("Referee")),
            "no_crowd": int(config.no_crowd(gdate)),
            "completed": done,
        })
        o = _resolve_odds(r)
        if o["odds_source"]:
            lines.append({"game_id": gid, "season": season, "date": gdate,
                          "home_team": home, "away_team": away, **o, **_market_view(o)})

    g = pd.DataFrame(games)
    ln = pd.DataFrame(lines)
    return (g.reindex(columns=GAME_COLS) if len(g) else pd.DataFrame(columns=GAME_COLS),
            ln.reindex(columns=LINE_COLS) if len(ln) else pd.DataFrame(columns=LINE_COLS))


def fetch_season(season: int, league: str | None = None) -> pd.DataFrame:
    """One season file, preferring the primary host and falling back to the GitHub mirror.

    The mirror carries results but **no odds columns at all**, so a season served from it can
    rate teams and predict but cannot train or score anything market-aware. That is a big
    enough difference to be worth a warning every time it happens rather than a silent
    degradation, which is how it would otherwise present: as a season whose market models
    simply have no rows.
    """
    if not _primary_down:
        import time
        t0 = time.monotonic()
        raw = _csv(season_url(season, league))
        _note_primary(bool(len(raw)), time.monotonic() - t0)
        if len(raw):
            return raw
    url = mirror_url(season, league)
    if not url:
        return pd.DataFrame()
    raw = _csv(url)
    if len(raw):
        log.warning("season %s served from the results-only mirror - it carries NO odds "
                    "columns, so market-aware models will have no rows for it", season)
    return raw


def update_games(seasons: list[int] | None = None, league: str | None = None
                 ) -> pd.DataFrame:
    """Refresh the local cache and return the games table.

    Completed seasons never change, so only the current one is re-fetched on a normal run; a
    full backfill happens when the cache is empty or ``seasons`` is given explicitly.
    """
    ensure_dirs()
    league = league or config.LEAGUE
    today = config.today_uk()
    current = config.season_of(today)
    have_g = load_games()

    if seasons is None:
        if have_g.empty:
            seasons = list(range(config.LOAD_FROM_SEASON, current + 1))
            log.info("no cache - backfilling seasons %s-%s", seasons[0], seasons[-1])
        else:
            # The season just gone can still gain a rearranged fixture, so refresh two.
            seasons = [current - 1, current]

    fresh_g, fresh_l = [], []
    for s in seasons:
        g, ln = parse_season(fetch_season(s, league), s, league)
        if len(g):
            fresh_g.append(g)
            fresh_l.append(ln)
            log.info("season %s: %d matches, %d with prices (%s)", s, len(g), len(ln),
                     ln["odds_source"].iloc[0] if len(ln) else "none")
        else:
            log.warning("season %s: nothing returned", s)

    if fresh_g:
        new_g = pd.concat(fresh_g, ignore_index=True)
        new_l = pd.concat([f for f in fresh_l if len(f)], ignore_index=True) \
            if any(len(f) for f in fresh_l) else pd.DataFrame(columns=LINE_COLS)
        have_l = load_lines()
        games = _merge(have_g, new_g)
        lines = _merge(have_l, new_l)
        games.to_csv(config.GAMES, index=False)
        lines.to_csv(config.LINES, index=False)
        log.info("cache: %d matches, %d priced, seasons %s", len(games), len(lines),
                 sorted(games["season"].unique()) if len(games) else [])
        return games
    return have_g


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old.empty:
        return new
    if new.empty:
        return old
    keep = old[~old["game_id"].isin(set(new["game_id"]))]
    out = pd.concat([keep, new], ignore_index=True)
    return out.sort_values(["season", "date", "game_id"]).reset_index(drop=True)


def _load(path, dates=("date",)) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"game_id": str}, low_memory=False)
    for c in dates:
        if c in df:
            df[c] = pd.to_datetime(df[c]).dt.date
    return df


def load_games() -> pd.DataFrame:
    return _load(config.GAMES)


def load_lines() -> pd.DataFrame:
    return _load(config.LINES)


def coverage_report(games: pd.DataFrame, lines: pd.DataFrame) -> list[dict]:
    """Per season: how many matches, how many priced, and from WHICH columns.

    Printed by the training job. A season showing 380 matches and 0 priced, or a season that
    silently switched to a soft book's opening price, is the failure this whole module is built
    to make visible rather than quiet.
    """
    if games.empty:
        return []
    out = []
    by_line = lines.groupby("season") if len(lines) else None
    for s, grp in games.groupby("season"):
        row = {"season": int(s), "matches": int(len(grp)), "priced": 0,
               "closing": 0, "sources": [], "ah": 0, "ou": 0}
        if by_line is not None and s in by_line.groups:
            ln = by_line.get_group(s)
            row["priced"] = int(len(ln))
            row["closing"] = int(ln["is_closing"].fillna(False).astype(bool).sum())
            row["ah"] = int(ln["ah_home"].notna().sum())
            row["ou"] = int(ln["price_over"].notna().sum())
            row["sources"] = sorted({s for s in ln["odds_source"].dropna().unique()})[:3]
        out.append(row)
    return out


# ---------------------------------------------------------------------------------
# Fixtures (the board)
# ---------------------------------------------------------------------------------
def fetch_fixtures(league: str | None = None) -> pd.DataFrame:
    """Forthcoming matches with current prices.

    football-data publishes one fixtures file covering every division it carries, so this is
    filtered down to ours. It is the same column layout as a season file, which means the same
    resolver handles it - and that is deliberate, because a board built by a second, parallel
    parser is a board that can disagree with the training data about what a price means.
    """
    league = league or config.LEAGUE
    raw = _csv(f"{config.FOOTBALL_DATA.rsplit('/mmz4281', 1)[0]}/fixtures.csv")
    if raw.empty or "Div" not in raw.columns:
        log.info("no fixtures file available")
        return pd.DataFrame()
    raw = raw[raw["Div"].astype(str).str.strip() == league].copy()
    if raw.empty:
        log.info("fixtures file carries no %s matches", league)
        return pd.DataFrame()
    raw["_date"] = _dates(raw)
    raw = raw[raw["_date"].notna()]

    rows = []
    for r in raw.to_dict("records"):
        try:
            home, away = canon(r["HomeTeam"]), canon(r["AwayTeam"])
        except UnknownClub as e:
            log.error("fixture skipped: %s", e)
            continue
        gdate = r["_date"].date()
        season = config.season_of(gdate)
        o = _resolve_odds(r)
        kt = _text(r.get("Time"))
        rows.append({"game_id": make_game_id(season, home, away), "season": season,
                     "league": league, "date": gdate, "kickoff": kt,
                     "kickoff_uk": f"{gdate:%a %d %b}" + (f", {kt}" if kt else ""),
                     "matchweek": config.matchweek(gdate, season),
                     "home_team": home, "away_team": away, "completed": False,
                     "no_crowd": 0, **o, **_market_view(o)})
    df = pd.DataFrame(rows)
    log.info("fixtures: %d %s matches, %d priced", len(df), league,
             int(df["price_home"].notna().sum()) if len(df) else 0)
    return df


# ---------------------------------------------------------------------------------
# Prior-season strength: this pipeline's SP+ / prior-EPA analogue
# ---------------------------------------------------------------------------------
# Two ratings per club per season, both opponent-adjusted and both multiplicative:
#
#   att  - goals scored per match against an average defence
#   deff - goals conceded per match against an average attack
#
# and the same pair computed from SHOTS ON TARGET rather than goals. The shot version is the
# one that carries the preseason weight, for the reason every football analytics department
# rediscovered a decade ago: over 38 matches, shot volume is a substantially better predictor
# of next season's goals than this season's goals are. Goals are the outcome; shots are the
# process, and the process has a far better signal-to-noise ratio at this sample size.
#
# Joined only from season-1, so it cannot leak.
STRENGTH_COLS = ["season", "league", "team", "played", "att_goals", "def_goals",
                 "att_shots", "def_shots", "ppg", "gd_per_game", "position", "promoted"]

ADJUST_ITERS = 12
# How much worse a division is than the one above it, as a multiplier on scoring rate. Used to
# translate a promoted club's Championship season into a Premier League prior. 0.80 is close to
# what promoted clubs actually go on to do, and it is the difference between entering the
# league rated average and entering it rated roughly where the bookmakers have them.
DIVISION_GAP = 0.80


def _adjust(matches: list[tuple[str, str, float, float]]) -> dict[str, dict]:
    """Opponent-adjusted multiplicative attack and defence, by fixed-point iteration.

    Each club's attack is its scoring rate divided by the average defensive weakness of the
    clubs it happened to play, and vice versa; iterating to convergence is the standard
    multiplicative Poisson fit. It matters more here than in a balanced league would suggest,
    because the *unbalanced* part of a football season is the fixture ORDER, and this rating is
    read at the season boundary where a club's easy or hard run is fully baked in.
    """
    teams = sorted({t for m in matches for t in (m[0], m[1])})
    att = {t: 1.0 for t in teams}
    dfn = {t: 1.0 for t in teams}
    for _ in range(ADJUST_ITERS):
        sf = {t: 0.0 for t in teams}
        se = {t: 0.0 for t in teams}
        cf = {t: 0.0 for t in teams}
        ce = {t: 0.0 for t in teams}
        for h, a, hg, ag in matches:
            sf[h] += hg
            ce[h] += dfn[a]
            se[h] += ag
            cf[h] += att[a]
            sf[a] += ag
            ce[a] += dfn[h]
            se[a] += hg
            cf[a] += att[h]
        for t in teams:
            if ce[t] > 0:
                att[t] = sf[t] / ce[t]
            if cf[t] > 0:
                dfn[t] = se[t] / cf[t]
        m_a = np.mean([att[t] for t in teams]) or 1.0
        m_d = np.mean([dfn[t] for t in teams]) or 1.0
        att = {t: v / m_a for t, v in att.items()}
        dfn = {t: v / m_d for t, v in dfn.items()}
    return {t: {"att": att[t], "def": dfn[t]} for t in teams}


def _season_strength(g: pd.DataFrame) -> pd.DataFrame:
    """One season of one division -> a strength row per club."""
    done = g[g["completed"].astype(bool)]
    if done.empty:
        return pd.DataFrame(columns=STRENGTH_COLS)
    goals = [(r.home_team, r.away_team, float(r.home_goals), float(r.away_goals))
             for r in done.itertuples()]
    adj_g = _adjust(goals)
    shots = [(r.home_team, r.away_team, float(r.home_sot), float(r.away_sot))
             for r in done.itertuples()
             if r.home_sot == r.home_sot and r.away_sot == r.away_sot]
    adj_s = _adjust(shots) if len(shots) >= 20 else {}

    tally: dict[str, dict] = {}
    for r in done.itertuples():
        for team, gf, ga in ((r.home_team, r.home_goals, r.away_goals),
                             (r.away_team, r.away_goals, r.home_goals)):
            t = tally.setdefault(team, {"played": 0, "gf": 0.0, "ga": 0.0, "pts": 0})
            t["played"] += 1
            t["gf"] += gf
            t["ga"] += ga
            t["pts"] += 3 if gf > ga else (1 if gf == ga else 0)

    rows = []
    for team, t in tally.items():
        n = max(t["played"], 1)
        rows.append({
            "season": int(done["season"].iloc[0]), "league": done["league"].iloc[0],
            "team": team, "played": t["played"],
            "att_goals": round(adj_g.get(team, {}).get("att", np.nan), 4),
            "def_goals": round(adj_g.get(team, {}).get("def", np.nan), 4),
            "att_shots": round(adj_s.get(team, {}).get("att", np.nan), 4) if adj_s else np.nan,
            "def_shots": round(adj_s.get(team, {}).get("def", np.nan), 4) if adj_s else np.nan,
            "ppg": round(t["pts"] / n, 3),
            "gd_per_game": round((t["gf"] - t["ga"]) / n, 3),
            "promoted": 0,
        })
    df = pd.DataFrame(rows).sort_values("ppg", ascending=False).reset_index(drop=True)
    df["position"] = np.arange(1, len(df) + 1)
    return df


def build_strength(games: pd.DataFrame | None = None, fetch: bool = True,
                   lower: pd.DataFrame | None = None) -> pd.DataFrame:
    """Prior-season strength for every club, including clubs promoted from the division below.

    The promoted half is the piece with no NFL analogue. Three of twenty clubs each season have
    no top-flight row at all, and leaving them blank means the model's strongest preseason
    features are missing for exactly the clubs it knows least about. So the division below is
    fetched too and a promoted club's Championship season is carried up, scaled by
    ``DIVISION_GAP`` and flagged ``promoted`` so the model can learn how much to trust it
    rather than being told.
    """
    ensure_dirs()
    games = load_games() if games is None else games
    if games.empty:
        return pd.DataFrame(columns=STRENGTH_COLS)

    frames = [_season_strength(g) for _, g in games.groupby("season")]
    top = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames \
        else pd.DataFrame(columns=STRENGTH_COLS)

    below = pd.DataFrame(columns=STRENGTH_COLS)
    if config.LEAGUE_BELOW:
        # `lower` is injected by the matchdata backend, which already holds the whole archive in
        # memory and can slice the division below for free. Only the football-data backend has
        # to go and fetch it a season at a time.
        if lower is None:
            lower = _lower_division(sorted(int(s) for s in games["season"].unique()), fetch)
        b_frames = [_season_strength(g) for _, g in lower.groupby("season")] if len(lower) else []
        if b_frames:
            below = pd.concat([f for f in b_frames if len(f)], ignore_index=True)
            # Scale the lower division onto this one's terms. Attack is scaled down and defence
            # up, because a promoted side both scores less and concedes more than its
            # second-tier record suggests.
            for c in ("att_goals", "att_shots"):
                below[c] = below[c] * DIVISION_GAP
            for c in ("def_goals", "def_shots"):
                below[c] = below[c] / DIVISION_GAP
            below["promoted"] = 1
            log.info("carried %d club-seasons up from %s for promoted sides",
                     len(below), config.LEAGUE_BELOW)

    # A club present in the top division that season wins; the lower-division row is only ever
    # a fallback for a club with no top-flight history in that year.
    have = set(zip(top["season"], top["team"])) if len(top) else set()
    if len(below):
        below = below[[(s, t) not in have for s, t in zip(below["season"], below["team"])]]
    out = pd.concat([f for f in (top, below) if len(f)], ignore_index=True) \
        if len(top) or len(below) else pd.DataFrame(columns=STRENGTH_COLS)
    if len(out):
        out = out.reindex(columns=STRENGTH_COLS)
        out.to_csv(config.STRENGTH, index=False)
        log.info("strength: %d club-seasons (%d promoted-club rows)",
                 len(out), int(out["promoted"].sum()))
    return out


def _lower_division(seasons: list[int], fetch: bool = True) -> pd.DataFrame:
    """The division below, cached to disk so a retrain does not refetch a decade of it.

    Only the promoted clubs' single prior season is ever read out of this, but working out which
    clubs those are needs the whole division. Completed seasons never change, so the first run
    pays for the backfill and every later one refreshes the current season only - which on a
    weekly retrain is the difference between ~24 requests and one.
    """
    have = _load(config.LOWER_GAMES)
    cached = set(int(s) for s in have["season"].unique()) if len(have) else set()
    current = config.season_of(config.today_uk())
    # Re-fetch anything missing, plus the current season, which is still gaining results.
    wanted = [s for s in seasons if s not in cached or s >= current]
    if fetch and wanted:
        fresh = []
        for s in wanted:
            g, _ln = parse_season(fetch_season(s, config.LEAGUE_BELOW), s, config.LEAGUE_BELOW)
            if len(g):
                fresh.append(g)
        if fresh:
            have = _merge(have, pd.concat(fresh, ignore_index=True))
            ensure_dirs()
            have.to_csv(config.LOWER_GAMES, index=False)
        log.info("%s: fetched %d season(s), cache now holds %d matches",
                 config.LEAGUE_BELOW, len(fresh), len(have))
    elif len(have):
        log.info("%s: served %d matches from cache, no fetch needed",
                 config.LEAGUE_BELOW, len(have))
    return have[have["season"].isin(seasons)] if len(have) else have


def load_strength() -> pd.DataFrame:
    return _load(config.STRENGTH, dates=())


# ---------------------------------------------------------------------------------
# Schema check, runnable from a phone
# ---------------------------------------------------------------------------------
def _check(seasons: list[int], league: str) -> list[dict]:
    """Fetch a few seasons and report what the odds resolver actually found in each.

    This exists because the column layout of the source is the one thing in this pipeline that
    cannot be verified without reaching the live host, and it changed shape once already (in
    2019-20). A season that returns matches but no prices, or that silently falls back to a
    soft book's opening number, is a real failure that otherwise shows up only as market models
    with mysteriously few rows.
    """
    import time

    reset_primary()
    out = []
    for s in seasons:
        t0 = time.time()
        raw = fetch_season(s, league)
        g, ln = parse_season(raw, s, league)
        row = {"season": s, "code": config.season_code(s), "raw_rows": len(raw),
               "matches": len(g), "priced": len(ln), "closing": 0, "ah": 0, "ou": 0,
               "source": "", "columns": len(raw.columns) if len(raw) else 0,
               "secs": round(time.time() - t0, 1), "mirror": primary_is_down()}
        if len(ln):
            row["closing"] = int(ln["is_closing"].astype(bool).sum())
            row["ah"] = int(ln["ah_home"].notna().sum())
            row["ou"] = int(ln["price_over"].notna().sum())
            row["source"] = str(ln["odds_source"].iloc[0])
        out.append(row)
    return out


def _summary(rows: list[dict], league: str) -> str:
    reachable = any(not r.get("mirror") and r.get("priced") for r in rows)
    total = sum(r.get("secs", 0) for r in rows)
    lines = [f"# football-data.co.uk schema check — {league}", ""]
    if primary_is_down():
        lines += [
            "> **football-data.co.uk did not answer.** Everything below was served from the "
            "results-only GitHub mirror, which carries no odds columns at all. This is a "
            "reachability problem, not a layout problem — the primary host was tried "
            f"{config.PRIMARY_FAILURE_LIMIT} times and then skipped for the rest of the run so "
            "the job could finish rather than hang. Re-run when the host is up.", ""]
    elif reachable:
        lines += ["> football-data.co.uk answered and the odds columns resolved.", ""]
    lines += [f"Checked {len(rows)} season(s) in {total:.0f}s.", "",
              "| Season | Matches | Priced | Closing | AH | O/U | Secs | Columns resolved |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['season']}-{(r['season']+1) % 100:02d} | {r['matches']} | "
                     f"{r['priced']} | {r['closing']} | {r['ah']} | {r['ou']} | "
                     f"{r.get('secs', 0)} | `{r['source'] or '—'}` |")
    lines += ["", "## What this means", "",
              "| What you see | Meaning | What to do |", "|---|---|---|",
              "| Matches and priced both non-zero, `source` naming PSC*/AHCh/PC* | "
              "Everything resolved, closing Pinnacle prices | Nothing |",
              "| Priced non-zero but `source` names B365 or Bb* only | Older season, or the "
              "sharper columns are absent | Normal before 2019-20; investigate if recent |",
              "| Matches non-zero, priced 0 | The odds columns were not found at all | "
              "The layout changed — add the new spellings to CLOSING_*/OPENING_* in "
              "`epl/sources/footballdata.py` |",
              "| Matches 0 with columns 0 | Neither host answered | Re-run later |",
              "| The banner above says the host did not answer | Reachability, not layout | "
              "Re-run later; the pipeline still works, without market-aware models |", "",
              "The market-aware models train only on rows in the **Priced** column, so a "
              "season showing matches but no prices is contributing nothing to them."]
    return "\n".join(lines)


def main(argv=None):
    import argparse
    import logging

    ap = argparse.ArgumentParser(description="Check what the football-data source resolves.")
    ap.add_argument("--check", action="store_true", help="fetch a sample and report")
    ap.add_argument("--league", default=None)
    ap.add_argument("--seasons", default=None,
                    help="comma-separated start years, e.g. 2015,2019,2024")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    league = a.league or config.LEAGUE
    if a.seasons:
        seasons = [int(x) for x in a.seasons.split(",") if x.strip()]
    else:
        # One season either side of the 2019-20 layout change, plus the current one.
        cur = config.season_of(config.today_uk())
        seasons = sorted({2015, 2018, 2019, 2022, cur - 1, cur})
    rows = _check(seasons, league)
    text = _summary(rows, league)
    print(text)
    # GitHub renders this as a page in the mobile app, rather than as raw logs.
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
