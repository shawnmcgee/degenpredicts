"""Optional live-price source (The Odds API, sport key ``americanfootball_nfl``).

nflverse already gives us the closing number for history and a current number for the board.
What it does not give is the *price* on each side at a book you can actually bet, which is what
EV and Kelly need. Without ``ODDS_API_KEY`` the pipeline runs fine on nflverse alone and
assumes -110.

Name matching is a much smaller problem here than in college football: 32 teams, all
well-known, and the feed sends "Kansas City Chiefs" rather than an abbreviation. The matcher
maps display names onto our canonical codes and logs anything it cannot resolve - a miss here
costs one game's price, not a franchise's rating history, so unlike :func:`nfl.teams.canon`
this side is allowed to guess and report rather than raise.

Every pull is appended to snapshots.csv so closing-line value is measurable later.
"""
from __future__ import annotations

import difflib
import logging
import re
import statistics
import unicodedata
from datetime import datetime, timezone

import pandas as pd

# Read settings as ``config.NAME`` at call time, not by value at import: `config.ODDS_API_KEY` and
# `config.SNAPSHOTS` would otherwise freeze at whatever the environment was when this module first
# happened to be imported.
from .. import config
from ..config import ensure_dirs
from ..teams import TEAMS, canon, is_known
from ..http import get

log = logging.getLogger(__name__)
URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

# Full club names as every feed spells them, mapped to our canonical codes. Unlike the college
# matcher this list is short enough to be exhaustive, so most lookups never reach the fuzzy
# fallback. Nicknames and city-only forms are included because different feeds pick different
# halves of the name.
NAMES: dict[str, str] = {
    "arizona cardinals": "ARI", "atlanta falcons": "ATL", "baltimore ravens": "BAL",
    "buffalo bills": "BUF", "carolina panthers": "CAR", "chicago bears": "CHI",
    "cincinnati bengals": "CIN", "cleveland browns": "CLE", "dallas cowboys": "DAL",
    "denver broncos": "DEN", "detroit lions": "DET", "green bay packers": "GB",
    "houston texans": "HOU", "indianapolis colts": "IND", "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC", "las vegas raiders": "LV", "los angeles chargers": "LAC",
    "los angeles rams": "LA", "miami dolphins": "MIA", "minnesota vikings": "MIN",
    "new england patriots": "NE", "new orleans saints": "NO", "new york giants": "NYG",
    "new york jets": "NYJ", "philadelphia eagles": "PHI", "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF", "seattle seahawks": "SEA", "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN", "washington commanders": "WAS",
    # historical and alternate spellings feeds still emit
    "oakland raiders": "LV", "san diego chargers": "LAC", "st louis rams": "LA",
    "st. louis rams": "LA", "washington redskins": "WAS", "washington football team": "WAS",
    # bare nicknames
    "cardinals": "ARI", "falcons": "ATL", "ravens": "BAL", "bills": "BUF", "panthers": "CAR",
    "bears": "CHI", "bengals": "CIN", "browns": "CLE", "cowboys": "DAL", "broncos": "DEN",
    "lions": "DET", "packers": "GB", "texans": "HOU", "colts": "IND", "jaguars": "JAX",
    "chiefs": "KC", "raiders": "LV", "chargers": "LAC", "rams": "LA", "dolphins": "MIA",
    "vikings": "MIN", "patriots": "NE", "saints": "NO", "giants": "NYG", "jets": "NYJ",
    "eagles": "PHI", "steelers": "PIT", "49ers": "SF", "niners": "SF", "seahawks": "SEA",
    "buccaneers": "TB", "bucs": "TB", "titans": "TEN", "commanders": "WAS",

    # --- city only ------------------------------------------------------------------
    # Kalshi's spread ladder names the favourite by CITY: "If Kansas City wins by more than
    # 7.5 points...". Without these the entire ladder fails to match, which is how a live
    # discover run against the real API turned up "Denver" and "Kansas City" unmatched.
    # The fuzzy fallback cannot rescue them either - "kansas city" against "kansas city
    # chiefs" scores 0.76, below the 0.85 cutoff, and lowering that cutoff to catch it would
    # start matching genuinely different clubs.
    "arizona": "ARI", "atlanta": "ATL", "baltimore": "BAL", "buffalo": "BUF",
    "carolina": "CAR", "chicago": "CHI", "cincinnati": "CIN", "cleveland": "CLE",
    "dallas": "DAL", "denver": "DEN", "detroit": "DET", "green bay": "GB",
    "houston": "HOU", "indianapolis": "IND", "jacksonville": "JAX", "kansas city": "KC",
    "las vegas": "LV", "miami": "MIA", "minnesota": "MIN", "new england": "NE",
    "new orleans": "NO", "philadelphia": "PHI", "pittsburgh": "PIT", "seattle": "SEA",
    "san francisco": "SF", "tampa bay": "TB", "tennessee": "TEN", "washington": "WAS",
    "oakland": "LV", "san diego": "LAC", "st louis": "LA", "st. louis": "LA",

    # --- the four clubs that share a city ---------------------------------------------
    # Kalshi disambiguates with a single trailing letter ("New York G") in yes_sub_title and
    # with a short city form ("NY Giants") in the rules text. Both spellings appear in the
    # same payload for the same game, so both have to resolve.
    "new york g": "NYG", "new york j": "NYJ",
    "los angeles r": "LA", "los angeles c": "LAC",
    "ny giants": "NYG", "ny jets": "NYJ", "la rams": "LA", "la chargers": "LAC",
    "n y giants": "NYG", "n y jets": "NYJ",
}

# Bare "New York" and "Los Angeles" name two clubs each and must never be guessed. They are
# refused before the fuzzy fallback can pick one, because picking wrong here silently prices
# the wrong team's contract - a far worse outcome than reporting the name as unmatched.
AMBIGUOUS = {"new york", "los angeles", "la", "ny", "nyc"}


def _norm(name: str) -> str:
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    n = re.sub(r"[^a-z0-9 ]+", " ", n.lower())
    return re.sub(r"\s+", " ", n).strip()


def build_matcher(teams: list[str] | None = None):
    """Map a display name from any price feed onto our canonical team code.

    Records misses on ``matcher.unmatched`` instead of raising, and returns the input
    unchanged so the row simply fails to join rather than poisoning anything.
    """
    known = set(teams) if teams else set(TEAMS)
    unmatched: set[str] = set()

    def match(name: str) -> str:
        if not name:
            return name
        raw = str(name).strip()
        # An abbreviation in any spelling goes straight through the canonical map - that is
        # the one place codes are translated, and feeds do send bare codes (LAR, WSH, JAC).
        if is_known(raw):
            code = canon(raw)
            if not known or code in known:
                return code
        n = _norm(raw)
        if n in AMBIGUOUS:
            unmatched.add(raw)
            return raw
        if n in NAMES:
            return NAMES[n]
        # try dropping the city: "Kansas City Chiefs" -> "chiefs"
        words = n.split()
        for cut in range(1, len(words)):
            tail = " ".join(words[cut:])
            if tail in NAMES:
                return NAMES[tail]
        close = difflib.get_close_matches(n, list(NAMES), n=1, cutoff=0.85)
        if close:
            return NAMES[close[0]]
        unmatched.add(raw)
        return raw

    match.unmatched = unmatched  # type: ignore[attr-defined]
    return match


def american_to_prob(price):
    if price is None or price != price:
        return None
    p = float(price)
    return 100 / (p + 100) if p > 0 else -p / (-p + 100)


def devig(a, b):
    if not a or not b:
        return a, b
    s = a + b
    return a / s, b / s


def _market(bm, key, home):
    for m in bm.get("markets", []):
        if m.get("key") != key:
            continue
        if key == "totals":
            o = next((x for x in m["outcomes"] if x["name"] == "Over"), None)
            u = next((x for x in m["outcomes"] if x["name"] == "Under"), None)
            if o and o.get("point") is not None:
                return float(o["point"]), o.get("price"), (u or {}).get("price")
        else:
            h = next((x for x in m["outcomes"] if x["name"] == home), None)
            a = next((x for x in m["outcomes"] if x["name"] != home), None)
            if h and h.get("point") is not None:
                return float(h["point"]), h.get("price"), (a or {}).get("price")
    return None


def snapshot(matcher=None) -> pd.DataFrame:
    """Live spreads and totals.

    The Odds API quotes a spread from each side: a home outcome of -3.5 means the home team
    gives 3.5. That is already the CFBD/`spread_home` convention this pipeline uses everywhere
    downstream, so unlike the nflverse feed no sign flip is needed here.
    """
    if not config.ODDS_API_KEY:
        log.info("config.ODDS_API_KEY unset - skipping live prices, using nflverse numbers at -110")
        return pd.DataFrame()
    r = get(URL, params={"apiKey": config.ODDS_API_KEY, "regions": "us", "markets": "totals,spreads",
                         "oddsFormat": "american", "dateFormat": "iso"})
    if r is None or r.status_code != 200:
        log.warning("Odds API unavailable: %s", getattr(r, "status_code", "no response"))
        return pd.DataFrame()
    log.info("Odds API quota used=%s remaining=%s",
             r.headers.get("x-requests-used"), r.headers.get("x-requests-remaining"))
    pulled = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for ev in r.json():
        tot = {bm["title"]: _market(bm, "totals", ev["home_team"])
               for bm in ev.get("bookmakers", [])}
        spr = {bm["title"]: _market(bm, "spreads", ev["home_team"])
               for bm in ev.get("bookmakers", [])}
        tot = {k: v for k, v in tot.items() if v}
        spr = {k: v for k, v in spr.items() if v}
        if not tot and not spr:
            continue

        def choose(d):
            if not d:
                return (float("nan"), None, None, "none", 0)
            b = next((x for x in config.ODDS_BOOKS if x in d), None)
            return (*d[b], b, len(d)) if b else \
                   (statistics.median(v[0] for v in d.values()), None, None, "consensus", len(d))

        t_line, t_over, t_under, t_book, t_n = choose(tot)
        s_line, s_home, s_away, s_book, s_n = choose(spr)
        ts = pd.Timestamp(ev["commence_time"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        et = ts.tz_convert(config.ET)
        home = matcher(ev["home_team"]) if matcher else ev["home_team"]
        away = matcher(ev["away_team"]) if matcher else ev["away_team"]
        p_o, p_u = devig(american_to_prob(t_over), american_to_prob(t_under))
        p_h, p_a = devig(american_to_prob(s_home), american_to_prob(s_away))
        rows.append({"pulled_at": pulled, "event_id": ev["id"], "date": et.date(),
                     "tip_et": et.strftime("%a %b %d, %-I:%M %p"),
                     "home_team": home, "away_team": away,
                     "home_raw": ev["home_team"], "away_raw": ev["away_team"],
                     "live_total": t_line, "over_price": t_over, "under_price": t_under,
                     "total_book": t_book, "total_n_books": t_n,
                     "live_spread_home": s_line, "spread_home_price": s_home,
                     "spread_away_price": s_away, "spread_book": s_book, "spread_n_books": s_n,
                     "p_over_mkt": p_o, "p_under_mkt": p_u,
                     "p_home_mkt": p_h, "p_away_mkt": p_a})
    df = pd.DataFrame(rows)
    if matcher is not None and getattr(matcher, "unmatched", None):
        log.warning("%d Odds API team names unmatched: %s",
                    len(matcher.unmatched), sorted(matcher.unmatched))
    return df


def append_snapshot(df: pd.DataFrame) -> None:
    if df.empty:
        return
    ensure_dirs()
    df.to_csv(config.SNAPSHOTS, mode="a", header=not config.SNAPSHOTS.exists(), index=False)


def snapshot_history() -> pd.DataFrame:
    if not config.SNAPSHOTS.exists():
        return pd.DataFrame()
    df = pd.read_csv(config.SNAPSHOTS)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df
