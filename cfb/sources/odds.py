"""Optional live-price source (The Odds API, sport key ``americanfootball_ncaaf``).

CFBD already gives us the number. What it doesn't give is the *price* on each side from a book
you can actually bet, which is what EV and Kelly need. If ODDS_API_KEY is unset the pipeline
runs fine on CFBD alone and assumes -110.

Every pull is appended to snapshots.csv so closing-line value is measurable later.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
import unicodedata
import statistics
from datetime import datetime

import pandas as pd

from .. import config
from ..config import ET, ODDS_BOOKS, SNAPSHOTS, ODDS_API_KEY, ensure_dirs
from core.http import get

log = logging.getLogger(__name__)
URL = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"

# The Odds API sends "Ole Miss Rebels"; CFBD sends "Ole Miss". Strip the mascot by matching
# against the known CFBD team list, with a few names that don't reduce cleanly.
MANUAL = {
    "Ole Miss Rebels": "Ole Miss", "Miami Hurricanes": "Miami", "Miami (OH) RedHawks": "Miami (OH)",
    "Southern Miss Golden Eagles": "Southern Mississippi", "UMass Minutemen": "UMass",
    "Louisiana Ragin' Cajuns": "Louisiana", "UL Monroe Warhawks": "Louisiana Monroe",
    "Texas A&M Aggies": "Texas A&M", "Hawai'i Rainbow Warriors": "Hawai'i",
    "San José State Spartans": "San José State", "Sam Houston Bearkats": "Sam Houston",
    "Appalachian State Mountaineers": "Appalachian State", "Army Black Knights": "Army",
    "Connecticut Huskies": "UConn", "UConn Huskies": "UConn",
    "Middle Tennessee Blue Raiders": "Middle Tennessee", "UTSA Roadrunners": "UT San Antonio",
    "Florida International Golden Panthers": "Florida International",
    "Jacksonville State Gamecocks": "Jacksonville State", "Kennesaw State Owls": "Kennesaw State",
    # Kalshi says "UMass"; CFBD says "Massachusetts".
    "UMass": "Massachusetts", "UMass Minutemen": "Massachusetts",
}


def _norm(name: str) -> str:
    """Lower-case, strip accents and punctuation, collapse whitespace."""
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    n = n.replace("&", " and ")
    n = re.sub(r"[^a-z0-9 ]+", " ", n.lower())
    return re.sub(r"\s+", " ", n).strip()


def _variants(name: str) -> set[str]:
    """Every spelling of a school we might see.

    The live Kalshi board abbreviates "State" to "St" - all 46 unmatched names in the first
    run contained "State" - so each canonical name generates both forms, plus a few other
    abbreviations different feeds disagree on.
    """
    base = _norm(name)
    out = {base}
    subs = [
        (r"\bstate\b", "st"), (r"\bst\b", "state"),
        (r"\buniversity\b", ""), (r"\bsoutheastern\b", "se"), (r"\bse\b", "southeastern"),
        (r"\bnorthwestern\b", "nw"), (r"\bsouthern\b", "so"),
        (r"\bmiddle\b", "mid"), (r"\bflorida international\b", "fiu"),
        (r"\blong island university\b", "liu"), (r"\bappalachian\b", "app"),
        (r"\bapp\b", "appalachian"), (r"\bconnecticut\b", "uconn"),
        (r"\bmississippi\b", "ole miss"),
    ]
    for pat, rep in subs:
        for v in list(out):
            alt = re.sub(pat, rep, v)
            alt = re.sub(r"\s+", " ", alt).strip()
            if alt and alt != v:
                out.add(alt)
    # drop parenthetical qualifiers: "St. Thomas (MN)" -> "st thomas mn" -> "st thomas"
    out |= {re.sub(r"\b(mn|fl|oh|sc|nj|pa|ca|tx)\b", "", v).strip() for v in list(out)}
    return {v for v in out if v}


def build_matcher(cfbd_teams: list[str]):
    """Map a display name from Kalshi or The Odds API onto a CFBD team name.

    Handles three things the feeds disagree on: trailing mascots ("Ohio State Buckeyes"),
    "State" vs "St", and punctuation/accents ("San Jose St" vs "San José State").
    """
    alias: dict[str, str] = {}
    for team in cfbd_teams:
        for v in _variants(team):
            alias.setdefault(v, team)
    unmatched: set[str] = set()

    def match(name: str) -> str:
        if not name:
            return name
        if name in MANUAL:
            return MANUAL[name]
        n = _norm(name)
        if n in alias:
            return alias[n]
        # strip a trailing mascot: try progressively shorter prefixes of the words
        words = n.split()
        for cut in range(len(words) - 1, 0, -1):
            if " ".join(words[:cut]) in alias:
                return alias[" ".join(words[:cut])]
        close = difflib.get_close_matches(n, list(alias), n=1, cutoff=0.80)
        if close:
            return alias[close[0]]
        unmatched.add(name)
        return name

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


def quotes(ev: dict, books=None) -> list[dict]:
    """Every shop venue's number and price on each side of one event, for the Big Game Card.

    One entry per venue, market and side: ``{"book", "key", "m", "side", "point", "price"}``,
    where ``m`` is "spread" or "total", a spread side is "home" or "away" with its own point
    (an away +3.5 is ``point=3.5``) and a total side is "over" or "under". Same shape the NFL
    snapshot has carried since October, so one reader handles both.
    """
    books = config.SHOP_BOOKS if books is None else books
    out = []
    for bm in ev.get("bookmakers", []):
        key = str(bm.get("key", "")).lower()
        if key not in books:
            continue
        title = bm.get("title") or key
        for m in bm.get("markets", []):
            for o in m.get("outcomes", []):
                if o.get("point") is None or o.get("price") is None:
                    continue
                if m.get("key") == "spreads":
                    side = "home" if o.get("name") == ev["home_team"] else "away"
                    out.append({"book": title, "key": key, "m": "spread", "side": side,
                                "point": float(o["point"]), "price": float(o["price"])})
                elif m.get("key") == "totals" and o.get("name") in ("Over", "Under"):
                    out.append({"book": title, "key": key, "m": "total",
                                "side": o["name"].lower(), "point": float(o["point"]),
                                "price": float(o["price"])})
    return out


def _fetch():
    """The odds call. Books are asked for by name - the line books plus the exchanges, at the
    cost of one region - and a refusal falls back to the plain US region, so the board can
    never lose its prices to the book list."""
    base = {"apiKey": ODDS_API_KEY, "markets": "totals,spreads", "oddsFormat": "american",
            "dateFormat": "iso"}
    r = get(URL, params={**base, "bookmakers": ",".join(config.ODDS_BOOKMAKERS)})
    if r is not None and r.status_code == 200:
        return r
    log.warning("Odds API refused the book list (%s) - falling back to regions=us",
                getattr(r, "status_code", "no response"))
    return get(URL, params={**base, "regions": "us"})


def snapshot(matcher=None) -> pd.DataFrame:
    if not ODDS_API_KEY:
        log.info("ODDS_API_KEY unset - skipping live prices, using CFBD numbers at -110")
        return pd.DataFrame()
    r = _fetch()
    if r is None or r.status_code != 200:
        log.warning("Odds API unavailable: %s", getattr(r, "status_code", "no response"))
        return pd.DataFrame()
    log.info("Odds API quota used=%s remaining=%s",
             r.headers.get("x-requests-used"), r.headers.get("x-requests-remaining"))
    pulled = datetime.utcnow().isoformat(timespec="seconds")
    now = config.now_et()
    rows, in_play = [], 0
    for ev in r.json():
        tot = {bm["title"]: _market(bm, "totals", ev["home_team"]) for bm in ev.get("bookmakers", [])}
        spr = {bm["title"]: _market(bm, "spreads", ev["home_team"]) for bm in ev.get("bookmakers", [])}
        tot = {k: v for k, v in tot.items() if v}
        spr = {k: v for k, v in spr.items() if v}
        if not tot and not spr:
            continue

        def choose(d):
            if not d:
                return (float("nan"), None, None, "none", 0)
            b = next((x for x in ODDS_BOOKS if x in d), None)
            return (*d[b], b, len(d)) if b else \
                   (statistics.median(v[0] for v in d.values()), None, None, "consensus", len(d))

        t_line, t_over, t_under, t_book, t_n = choose(tot)
        s_line, s_home, s_away, s_book, s_n = choose(spr)
        ts = pd.Timestamp(ev["commence_time"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        # The feed also lists games already under way, at in-play prices. Those are not
        # pre-game lines: the daily run starts hours late, and on a Saturday it lands just
        # after the noon kickoffs.
        if ts <= now:
            in_play += 1
            continue
        et = ts.tz_convert(ET)
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
                     "p_home_mkt": p_h, "p_away_mkt": p_a,
                     "quotes": json.dumps(quotes(ev), separators=(",", ":"))})
    if in_play:
        log.info("odds: skipped %d games already under way (in-play prices)", in_play)
    df = pd.DataFrame(rows)
    if matcher is not None and getattr(matcher, "unmatched", None):
        log.warning("%d Odds API team names unmatched: %s",
                    len(matcher.unmatched), sorted(matcher.unmatched))
    return df


def append_snapshot(df: pd.DataFrame) -> None:
    """Append a pull to snapshots.csv. The per-venue `quotes` column is left out: at ~90 games a
    day it would grow this file fifteenfold in a season. It goes to append_quotes instead."""
    if df.empty:
        return
    ensure_dirs()
    df = df.drop(columns=["quotes"], errors="ignore")
    if SNAPSHOTS.exists():
        df = df.reindex(columns=pd.read_csv(SNAPSHOTS, nrows=0).columns.tolist())
    df.to_csv(SNAPSHOTS, mode="a", header=not SNAPSHOTS.exists(), index=False)


QUOTE_COLS = ["pulled_at", "date", "home_team", "away_team", "quotes"]


def append_quotes(df: pd.DataFrame) -> None:
    """Every venue's spread and total quotes from this pull, for the Big Game Card.

    One gzip CSV per pull day under data/cfb/quotes/, written without a gzip timestamp so an
    unchanged file is byte-identical and gives git nothing to commit. Never fatal.
    """
    if df is None or df.empty or "quotes" not in df:
        return
    try:
        rows = df[[c for c in QUOTE_COLS if c in df]].copy()
        rows["date"] = rows["date"].astype(str)
        config.QUOTES.mkdir(parents=True, exist_ok=True)
        path = config.QUOTES / f"{str(rows['pulled_at'].iloc[0])[:10]}.csv.gz"
        if path.exists():
            rows = pd.concat([pd.read_csv(path), rows], ignore_index=True)
        rows = rows.drop_duplicates(subset=["pulled_at", "home_team", "away_team"], keep="last")
        rows.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    except Exception as e:                       # pragma: no cover - logged, never fatal
        log.warning("odds: could not write quotes (%s)", e)


def quotes_history(days: int = 3) -> pd.DataFrame:
    if not config.QUOTES.exists():
        return pd.DataFrame(columns=QUOTE_COLS)
    files = sorted(config.QUOTES.glob("*.csv.gz"))[-days:]
    if not files:
        return pd.DataFrame(columns=QUOTE_COLS)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def snapshot_history() -> pd.DataFrame:
    if not SNAPSHOTS.exists():
        return pd.DataFrame()
    df = pd.read_csv(SNAPSHOTS)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df
