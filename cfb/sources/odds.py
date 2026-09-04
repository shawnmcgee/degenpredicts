"""Optional live-price source (The Odds API, sport key ``americanfootball_ncaaf``).

CFBD already gives us the number. What it doesn't give is the *price* on each side from a book
you can actually bet, which is what EV and Kelly need. If ODDS_API_KEY is unset the pipeline
runs fine on CFBD alone and assumes -110.

Every pull is appended to snapshots.csv so closing-line value is measurable later.
"""
from __future__ import annotations

import difflib
import logging
import statistics
from datetime import datetime

import pandas as pd

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
}


def build_matcher(cfbd_teams: list[str]):
    """Return a function mapping an Odds-API display name to a CFBD team name."""
    canon = {t.lower(): t for t in cfbd_teams}
    unmatched: set[str] = set()

    def match(name: str) -> str:
        if not name:
            return name
        if name in MANUAL:
            return MANUAL[name]
        low = name.lower()
        if low in canon:
            return canon[low]
        # longest CFBD name that prefixes the Odds name (mascot is the trailing words)
        best = max((t for t in cfbd_teams if low.startswith(t.lower() + " ")),
                   key=len, default=None)
        if best:
            return best
        close = difflib.get_close_matches(low, list(canon), n=1, cutoff=0.86)
        if close:
            return canon[close[0]]
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


def snapshot(matcher=None) -> pd.DataFrame:
    if not ODDS_API_KEY:
        log.info("ODDS_API_KEY unset - skipping live prices, using CFBD numbers at -110")
        return pd.DataFrame()
    r = get(URL, params={"apiKey": ODDS_API_KEY, "regions": "us", "markets": "totals,spreads",
                         "oddsFormat": "american", "dateFormat": "iso"})
    if r is None or r.status_code != 200:
        log.warning("Odds API unavailable: %s", getattr(r, "status_code", "no response"))
        return pd.DataFrame()
    log.info("Odds API quota used=%s remaining=%s",
             r.headers.get("x-requests-used"), r.headers.get("x-requests-remaining"))
    pulled = datetime.utcnow().isoformat(timespec="seconds")
    rows = []
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
    df.to_csv(SNAPSHOTS, mode="a", header=not SNAPSHOTS.exists(), index=False)


def snapshot_history() -> pd.DataFrame:
    if not SNAPSHOTS.exists():
        return pd.DataFrame()
    df = pd.read_csv(SNAPSHOTS)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df
