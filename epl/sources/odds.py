"""Optional live-price source (The Odds API, sport key ``soccer_epl``).

football-data.co.uk already gives us closing prices for history and current prices for the
board. What it does not give is a live number from a book you can bet right now, which is what
EV and Kelly need on the morning of a match. Without ``ODDS_API_KEY`` the pipeline runs fine on
football-data alone.

Two things differ from the NFL version of this module:

* **Decimal odds and a three-way h2h.** The ``h2h`` market for football returns three outcomes,
  not two, and the third one is spelled "Draw". Requesting american odds and assuming two
  outcomes - which is what the gridiron code does - silently drops the draw and leaves a
  two-way book that de-vigs to nonsense.
* **UK and EU regions.** The sharp prices for this league are at European books, not American
  ones. Pinnacle is the reference and is quoted first where available.

Name matching is lenient and logs misses, exactly as in the other pipelines: a miss here costs
one match's price, not a club's rating history, so unlike :func:`epl.teams.canon` this side is
allowed to guess and report rather than raise.
"""
from __future__ import annotations

import difflib
import logging
import statistics
from datetime import datetime

import pandas as pd

# Settings are read as ``config.NAME`` at call time, never bound at import - see the note in
# sources/footballdata.py for what freezing them silently breaks.
from .. import config
from ..config import ensure_dirs
from ..teams import CLUBS, canon, is_known
from ..odds_math import devig_three, devig_two
from ..http import get

log = logging.getLogger(__name__)
URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
DRAW = "Draw"


def build_matcher(clubs: list[str] | None = None):
    """Map a display name from any price feed onto our canonical club name.

    Records misses on ``matcher.unmatched`` instead of raising, and returns the input unchanged
    so the row simply fails to join rather than poisoning anything downstream.
    """
    known = set(clubs) if clubs else set(CLUBS)
    unmatched: set[str] = set()

    def match(name: str) -> str:
        if not name:
            return name
        raw = str(name).strip()
        if is_known(raw):
            code = canon(raw)
            if not known or code in known:
                return code
        # Fuzzy fallback, against the canonical spellings only. Cutoff is high on purpose:
        # "Sheffield United" and "Sheffield Weds" are 0.85 similar, and a wrong pick there
        # would price the wrong club.
        close = difflib.get_close_matches(raw, list(known), n=1, cutoff=0.9)
        if close:
            return close[0]
        unmatched.add(raw)
        return raw

    match.unmatched = unmatched  # type: ignore[attr-defined]
    return match


def _h2h(bm, home, away):
    """(home, draw, away) decimal prices from a football h2h market."""
    for m in bm.get("markets", []):
        if m.get("key") != "h2h":
            continue
        got = {o.get("name"): o.get("price") for o in m.get("outcomes", [])}
        h, d, a = got.get(home), got.get(DRAW), got.get(away)
        if h and d and a:
            return float(h), float(d), float(a)
    return None


def _totals(bm):
    """(line, over price, under price) from a totals market."""
    for m in bm.get("markets", []):
        if m.get("key") != "totals":
            continue
        o = next((x for x in m["outcomes"] if x.get("name") == "Over"), None)
        u = next((x for x in m["outcomes"] if x.get("name") == "Under"), None)
        if o and u and o.get("point") is not None:
            return float(o["point"]), float(o["price"]), float(u["price"])
    return None


def snapshot(matcher=None) -> pd.DataFrame:
    """Live 1X2 and over/under prices, in decimal."""
    if not config.ODDS_API_KEY:
        log.info("config.ODDS_API_KEY unset - skipping live prices, using football-data's")
        return pd.DataFrame()
    r = get(URL, params={"apiKey": config.ODDS_API_KEY, "regions": "uk,eu",
                         "markets": "h2h,totals", "oddsFormat": "decimal",
                         "dateFormat": "iso"})
    if r is None or r.status_code != 200:
        log.warning("Odds API unavailable: %s", getattr(r, "status_code", "no response"))
        return pd.DataFrame()
    log.info("Odds API quota used=%s remaining=%s",
             r.headers.get("x-requests-used"), r.headers.get("x-requests-remaining"))
    pulled = datetime.utcnow().isoformat(timespec="seconds")
    rows = []
    for ev in r.json():
        h2h = {bm["title"]: _h2h(bm, ev["home_team"], ev["away_team"])
               for bm in ev.get("bookmakers", [])}
        tot = {bm["title"]: _totals(bm) for bm in ev.get("bookmakers", [])}
        h2h = {k: v for k, v in h2h.items() if v}
        tot = {k: v for k, v in tot.items() if v}
        if not h2h and not tot:
            continue

        def choose(d, n):
            """Prefer a named sharp book; otherwise take the median across the panel.

            The median rather than the max: a maximum across books is the best available price,
            which is the right thing to BET but the wrong thing to measure an edge against -
            it is partly just the noisiest book on the panel, and treating it as the market's
            opinion manufactures edge out of variance.
            """
            if not d:
                return (*(float("nan"),) * n, "none", 0)
            b = next((x for x in config.ODDS_BOOKS if x in d), None)
            if b:
                return (*d[b], b, len(d))
            return (*(statistics.median(v[i] for v in d.values()) for i in range(n)),
                    "consensus", len(d))

        ph, pdw, pa, b1, n1 = choose(h2h, 3)
        tl, po, pu, b2, n2 = choose(tot, 3)
        ts = pd.Timestamp(ev["commence_time"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        uk = ts.tz_convert(config.UK)
        home = matcher(ev["home_team"]) if matcher else ev["home_team"]
        away = matcher(ev["away_team"]) if matcher else ev["away_team"]
        mh, md, ma = devig_three(ph, pdw, pa)
        p_o, p_u = devig_two(po, pu)
        rows.append({"pulled_at": pulled, "event_id": ev["id"], "date": uk.date(),
                     "kickoff": uk.strftime("%H:%M"),
                     "kickoff_uk": uk.strftime("%a %d %b, %H:%M"),
                     "home_team": home, "away_team": away,
                     "home_raw": ev["home_team"], "away_raw": ev["away_team"],
                     "live_price_home": ph, "live_price_draw": pdw, "live_price_away": pa,
                     "book_1x2": b1, "n_books_1x2": n1,
                     "live_total_line": tl, "live_price_over": po, "live_price_under": pu,
                     "book_ou": b2, "n_books_ou": n2,
                     "p_home_mkt": mh, "p_draw_mkt": md, "p_away_mkt": ma,
                     "p_over_mkt": p_o, "p_under_mkt": p_u})
    df = pd.DataFrame(rows)
    if matcher is not None and getattr(matcher, "unmatched", None):
        log.warning("%d Odds API club names unmatched: %s",
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
