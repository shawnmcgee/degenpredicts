"""Lines from The Odds API: totals *and* spreads.

Every snapshot is appended to data/lines.csv. After one season that file is your own
historical-lines database - which is what makes the residual model (and closing-line value)
possible without paying for historical odds.
"""
from __future__ import annotations

import logging
import statistics
from datetime import datetime

import pandas as pd

from .. import config
from ..http import get
from ..team_names import normalize_team_name

log = logging.getLogger(__name__)

URL = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"


def american_to_prob(price) -> float | None:
    if price is None or price != price:
        return None
    p = float(price)
    return 100 / (p + 100) if p > 0 else -p / (-p + 100)


def devig(p_a: float | None, p_b: float | None) -> tuple[float | None, float | None]:
    """Remove the book's hold so two sides sum to 1."""
    if not p_a or not p_b:
        return p_a, p_b
    s = p_a + p_b
    return p_a / s, p_b / s


def fetch(api_key: str | None = None) -> list[dict]:
    api_key = api_key or config.ODDS_API_KEY
    if not api_key:
        raise RuntimeError("ODDS_API_KEY is not set")
    r = get(URL, params={"apiKey": api_key, "regions": "us", "markets": config.MARKETS,
                         "oddsFormat": "american", "dateFormat": "iso"})
    if r is None or r.status_code != 200:
        log.error("Odds API failed: %s", getattr(r, "text", "no response")[:300])
        return []
    log.info("Odds API quota used=%s remaining=%s",
             r.headers.get("x-requests-used"), r.headers.get("x-requests-remaining"))
    return r.json()


def _totals(bm: dict):
    for m in bm.get("markets", []):
        if m.get("key") != "totals":
            continue
        o = next((x for x in m["outcomes"] if x["name"] == "Over"), None)
        u = next((x for x in m["outcomes"] if x["name"] == "Under"), None)
        if o and o.get("point") is not None:
            return float(o["point"]), o.get("price"), (u or {}).get("price")
    return None


def _spreads(bm: dict, home: str):
    for m in bm.get("markets", []):
        if m.get("key") != "spreads":
            continue
        h = next((x for x in m["outcomes"] if x["name"] == home), None)
        a = next((x for x in m["outcomes"] if x["name"] != home), None)
        if h and h.get("point") is not None:
            return float(h["point"]), h.get("price"), (a or {}).get("price")
    return None


def snapshot(api_key: str | None = None) -> pd.DataFrame:
    """One row per upcoming game with the chosen book's total and spread."""
    events = fetch(api_key)
    pulled = datetime.utcnow().isoformat(timespec="seconds")
    rows = []
    for ev in events:
        tot = {bm.get("title"): _totals(bm) for bm in ev.get("bookmakers", [])}
        spr = {bm.get("title"): _spreads(bm, ev["home_team"]) for bm in ev.get("bookmakers", [])}
        tot = {k: v for k, v in tot.items() if v}
        spr = {k: v for k, v in spr.items() if v}
        if not tot and not spr:
            continue

        def pick(d):
            if not d:
                return None, None, None, "none", 0
            b = next((x for x in config.PREFERRED_BOOKS if x in d), None)
            if b:
                return (*d[b], b, len(d))
            return (statistics.median(v[0] for v in d.values()), None, None, "consensus", len(d))

        t_line, t_over, t_under, t_book, t_n = pick(tot)
        s_line, s_home, s_away, s_book, s_n = pick(spr)

        commence = pd.Timestamp(ev["commence_time"])
        commence = commence.tz_localize("UTC") if commence.tzinfo is None else commence
        et = commence.tz_convert(config.ET)

        p_over, p_under = devig(american_to_prob(t_over), american_to_prob(t_under))
        p_home, p_away = devig(american_to_prob(s_home), american_to_prob(s_away))

        rows.append({
            "pulled_at": pulled, "event_id": ev["id"],
            "commence_utc": commence.isoformat(), "date": et.date(),
            "tip_et": et.strftime("%Y-%m-%d %H:%M"),
            "home_team_raw": ev["home_team"], "away_team_raw": ev["away_team"],
            "home_team": normalize_team_name(ev["home_team"]),
            "away_team": normalize_team_name(ev["away_team"]),
            "total_line": t_line, "over_price": t_over, "under_price": t_under,
            "total_book": t_book, "total_n_books": t_n,
            "spread_home": s_line, "spread_home_price": s_home, "spread_away_price": s_away,
            "spread_book": s_book, "spread_n_books": s_n,
            "p_over": p_over, "p_under": p_under, "p_home_cover": p_home, "p_away_cover": p_away,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = (df.sort_values("commence_utc")
            .drop_duplicates(["date", "home_team", "away_team"], keep="first")
            .reset_index(drop=True))
    log.info("%d games with lines (%d totals, %d spreads)",
             len(df), df["total_line"].notna().sum(), df["spread_home"].notna().sum())
    return df


def append_snapshot(df: pd.DataFrame) -> None:
    """Append to the permanent line history (this is how you build CLV data)."""
    if df.empty:
        return
    config.ensure_dirs()
    header = not config.LINES.exists()
    df.to_csv(config.LINES, mode="a", header=header, index=False)


def line_history() -> pd.DataFrame:
    if not config.LINES.exists():
        return pd.DataFrame()
    df = pd.read_csv(config.LINES)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def closing_lines() -> pd.DataFrame:
    """Last snapshot we took before each game - our proxy for the closing line."""
    h = line_history()
    if h.empty:
        return h
    return (h.sort_values("pulled_at")
             .groupby("event_id", as_index=False).last()
             .rename(columns={"total_line": "close_total", "spread_home": "close_spread_home"}))
