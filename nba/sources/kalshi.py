"""Kalshi's NBA player-prop ladders. Read-only market data, no key.

Four series, one ladder per player per game, each rung "records N+":

    KXNBAPTS   "Karl-Anthony Towns: 10+ points"
    KXNBAREB   "De'Aaron Fox: 8+ rebounds"
    KXNBAAST   "Jalen Brunson: 2+ assists"
    KXNBA3PT   "Shai Gilgeous-Alexander: 3+ threes"

The ticker carries the game and the player, the way the NFL's does::

    KXNBAPTS-26JUN13NYKSAS-NYKKTOWNS32-10
             └ date, away, home └ team, name, jersey └ threshold

and the rules name the matchup by city, away first, with Kalshi's own disambiguator for the two
Los Angeles clubs ("If Shai Gilgeous-Alexander records 3+ Three Pointers in the Oklahoma City at
Los Angeles L professional basketball game ... scheduled for May 9, 2026").

**What has and has not been checked.** The series names, ticker shape, titles and rules wording
above come from public market pages for the 2026 playoffs. This repo's own runner has not read a
live payload yet, so - like the NFL module before its first run - every parse is defensive: the
game is read from the event ticker first and the rules second, the threshold from
``floor_strike`` first and the title second, and a rung that cannot be read is dropped and
counted in the log rather than guessed. ``python -m nba.sources.kalshi`` prints what the live
API returns and how much of it parsed.

**Players are matched on team and jersey, then checked on surname** - the NFL board's rule, for
its reason: Kalshi writes "Karl-Anthony Towns", a box score may not, and two players on one club
can share a surname. A number that disagrees with the name falls back to a name match within the
two clubs playing, and is left unmatched rather than guessed if that is not unique.

Sport-local on purpose: :mod:`nba` imports no other sport, so this does not reuse the NFL's
client. It is the same eighty lines of paging and parsing, and a change to one cannot break the
other.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date, datetime, timezone

import pandas as pd

from .. import config
from ..http import get_json
from ..teams import TEAM_ALIASES, UnknownTeam, canon, from_name

log = logging.getLogger(__name__)

RULES_RE = re.compile(
    r"in the (?P<a>.+?) (?:at|vs\.?|@) (?P<b>.+?) (?:NBA |pro |professional )?basketball game"
    r".*?scheduled for (?P<date>[A-Za-z]{3,9}\.? \d{1,2}, \d{4})", re.I)
THRESH_RE = re.compile(r"(\d+)\s*\+")
PLAYER_SEG_RE = re.compile(r"^(?P<letters>[A-Z]+?)(?P<jersey>\d{1,2})$")
EVENT_RE = re.compile(r"-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<teams>[A-Z]{4,10})$")

# How Kalshi names clubs in its rules: by place, with a letter for the two that share one.
PLACES = {
    "atlanta": "ATL", "boston": "BOS", "brooklyn": "BKN", "charlotte": "CHA", "chicago": "CHI",
    "cleveland": "CLE", "dallas": "DAL", "denver": "DEN", "detroit": "DET",
    "golden state": "GSW", "houston": "HOU", "indiana": "IND", "los angeles c": "LAC",
    "la clippers": "LAC", "los angeles l": "LAL", "la lakers": "LAL", "memphis": "MEM",
    "miami": "MIA", "milwaukee": "MIL", "minnesota": "MIN", "new orleans": "NOP",
    "new york": "NYK", "oklahoma city": "OKC", "orlando": "ORL", "philadelphia": "PHI",
    "phoenix": "PHX", "portland": "POR", "sacramento": "SAC", "san antonio": "SAS",
    "toronto": "TOR", "utah": "UTA", "washington": "WAS",
}


def _f(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def norm_name(name) -> str:
    """Lower-case ASCII, no punctuation, no generational suffix: 'Jaren Jackson Jr.' and
    'Jaren Jackson' are one key, as are 'De'Aaron Fox' and 'DeAaron Fox'."""
    n = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z ]+", "", n.replace(".", "").replace("'", "").replace("-", " "))
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def surname(name) -> str:
    parts = norm_name(name).split()
    return parts[-1] if parts else ""


def place(text) -> str | None:
    """A club from Kalshi's place name ("Los Angeles L"), its full name, or a code."""
    key = re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", str(text or "").lower())).strip()
    if key in PLACES:
        return PLACES[key]
    for fn in (from_name, canon):
        try:
            return fn(text)
        except UnknownTeam:
            continue
    return None


def fee(price: float) -> float:
    """Kalshi's taker fee per contract: KALSHI_FEE x P x (1 - P)."""
    return config.KALSHI_FEE * price * (1 - price)


def cost(price: float) -> float:
    """All-in cost of one contract at `price`: the price plus the taker fee."""
    if price != price or not (0 < price < 1):
        return float("nan")
    return price + fee(price)


def fetch_markets(series_ticker: str, status: str = "open", max_pages: int = 40) -> list[dict]:
    """Page through /markets for a series. Kalshi paginates with an opaque `cursor`."""
    out, cursor, pages = [], None, 0
    while pages < max_pages:
        params = {"limit": 200, "status": status, "series_ticker": series_ticker}
        if cursor:
            params["cursor"] = cursor
        payload = get_json(f"{config.KALSHI_BASE}/markets", params=params)
        if not payload:
            break
        batch = payload.get("markets") or []
        out.extend(batch)
        cursor = payload.get("cursor")
        pages += 1
        if not cursor or not batch:
            break
    log.info("kalshi %s: %d markets over %d page(s)", series_ticker, len(out), pages)
    return out


def _event_game(event_ticker: str) -> tuple[date | None, str | None, str | None]:
    """(date, away, home) from an event ticker like KXNBAPTS-26JUN13NYKSAS. The team block is
    split at the first point where both halves are known clubs."""
    m = EVENT_RE.search(str(event_ticker or ""))
    if not m:
        return None, None, None
    try:
        d = datetime.strptime(f"{m['dd']} {m['mon']} 20{m['yy']}", "%d %b %Y").date()
    except ValueError:
        d = None
    block = m["teams"]
    # three-and-three first: "SACCHA" must not read as "SA" + "CCHA" before "SAC" + "CHA"
    for i in sorted(range(2, len(block) - 1), key=lambda i: (i != 3, i)):
        a, b = block[:i], block[i:]
        if a in TEAM_ALIASES and b in TEAM_ALIASES:
            return d, canon(a), canon(b)
    return d, None, None


def parse_rung(m: dict, market: str) -> dict | None:
    """One market -> one rung, or None if it is not a player prop this can read."""
    title = str(m.get("title") or m.get("yes_sub_title") or "")
    if ":" not in title:
        return None
    player = title.split(":", 1)[0].strip()
    strike = _f(m.get("floor_strike"))
    if strike == strike:
        k = strike + 0.5              # "10+" is floor_strike 9.5: YES when strictly above it
    else:
        hit = THRESH_RE.search(title)
        k = float(hit.group(1)) if hit else float("nan")
    if k != k:
        return None
    ticker = str(m.get("ticker") or "")
    event = str(m.get("event_ticker") or "-".join(ticker.split("-")[:2]))
    gdate, away, home = _event_game(event)
    rules = RULES_RE.search(str(m.get("rules_primary") or ""))
    if rules:
        if away is None or home is None:
            away, home = place(rules["a"]), place(rules["b"])
        if gdate is None:
            for fmt in ("%b %d, %Y", "%B %d, %Y"):
                try:
                    gdate = datetime.strptime(rules["date"].replace(".", ""), fmt).date()
                    break
                except ValueError:
                    continue
    parts = ticker.split("-")
    ps = PLAYER_SEG_RE.match(parts[2]) if len(parts) >= 4 else None
    yes_bid, yes_ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
    spread = yes_ask - yes_bid if yes_ask == yes_ask and yes_bid == yes_bid else float("nan")
    ask_size = _f(m.get("yes_ask_size_fp"), 0.0)
    return {
        "market": market, "ticker": ticker, "event_ticker": event, "player_raw": player, "k": k,
        "seg_letters": ps.group("letters") if ps else "",
        "jersey": int(ps.group("jersey")) if ps else None,
        "away_team": away, "home_team": home, "date": gdate,
        "yes_bid": yes_bid, "yes_ask": yes_ask, "quote_spread": spread, "ask_size": ask_size,
        "volume": _f(m.get("volume_fp"), 0.0), "open_interest": _f(m.get("open_interest_fp"), 0.0),
        "tradeable": bool(spread == spread and spread <= config.PROPS_MAX_SPREAD
                          and ask_size >= config.PROPS_MIN_SIZE),
        "close_time": m.get("close_time"),
        "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def fetch(series: dict[str, str] | None = None) -> pd.DataFrame:
    """Every open rung of every configured series. Never raises: no prices, no board."""
    series = series or config.KALSHI_PROP_SERIES
    rows = []
    for market, ticker in series.items():
        try:
            ms = fetch_markets(ticker)
        except Exception as e:                      # pragma: no cover - network
            log.warning("kalshi %s: %s", ticker, e)
            ms = []
        parsed = [parse_rung(m, market) for m in ms]
        ok = [r for r in parsed if r]
        if ms and len(ok) < len(ms):
            log.warning("kalshi %s: %d of %d markets parsed - read the dropped ones with "
                        "`python -m nba.sources.kalshi`", ticker, len(ok), len(ms))
        rows.extend(ok)
    return pd.DataFrame(rows)


def attach_games(rungs: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Each rung's game_id, from its date and the two clubs - either way round, so a listing
    that names the clubs in the other order still joins."""
    if rungs.empty:
        return rungs.assign(game_id=pd.Series(dtype=str))
    g = games[["game_id", "date", "home_team", "away_team"]].copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    g["game_id"] = g["game_id"].astype(str)
    key = {}
    for r in g.itertuples(index=False):
        key[(r.date, r.home_team, r.away_team)] = r.game_id
        key.setdefault((r.date, r.away_team, r.home_team), r.game_id)
    out = rungs.copy()
    out["game_id"] = [key.get((d, h, a)) for d, h, a in
                      zip(out["date"], out["home_team"], out["away_team"])]
    # the board's clubs are the schedule's, whichever way round Kalshi wrote them
    sched = g.set_index("game_id")
    hit = out["game_id"].notna()
    out.loc[hit, "home_team"] = out.loc[hit, "game_id"].map(sched["home_team"])
    out.loc[hit, "away_team"] = out.loc[hit, "game_id"].map(sched["away_team"])
    miss = int((~hit).sum())
    if miss:
        log.info("kalshi props: %d rungs are not on the board (already under way, or no scheduled "
                 "game matched)", miss)
    return out


def _aliases_for(team: str) -> list[str]:
    return sorted({a for a, c in TEAM_ALIASES.items() if c == team}, key=len, reverse=True)


def _team_from_segment(letters: str, home, away) -> str | None:
    best = None
    for team in (home, away):
        if not isinstance(team, str):
            continue
        for alias in _aliases_for(team):
            if letters.startswith(alias) and (best is None or len(alias) > best[1]):
                best = (team, len(alias))
    return best[0] if best else None


def match_players(rungs: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Attach ``athlete_id`` and ``team`` to every rung: jersey first, surname to confirm."""
    r = rungs.copy()
    if r.empty or roster is None or roster.empty:
        r["athlete_id"], r["team"], r["match"] = None, None, "no roster"
        return r
    ros = roster.dropna(subset=["athlete_id"]).copy()
    ros["jersey"] = pd.to_numeric(ros["jersey"], errors="coerce")
    ros["surname"] = ros["name"].map(surname)
    ros["key"] = ros["name"].map(norm_name)
    by_num = {(t, int(j)): g for (t, j), g in ros.dropna(subset=["jersey"]).groupby(["team", "jersey"])}
    by_team = {t: g for t, g in ros.groupby("team")}
    ids, teams, how, cache = [], [], [], {}
    for row in r.itertuples():
        key = (row.player_raw, row.seg_letters, row.jersey, row.home_team, row.away_team)
        if key not in cache:
            pid, team, h = None, None, "unmatched"
            want_sur, want_key = surname(row.player_raw), norm_name(row.player_raw)
            seg_team = _team_from_segment(row.seg_letters or "", row.home_team, row.away_team)
            if seg_team is not None and row.jersey is not None and row.jersey == row.jersey:
                cand = by_num.get((seg_team, int(row.jersey)))
                if cand is not None:
                    ok = cand[(cand["surname"] == want_sur) | (cand["key"] == want_key)]
                    if len(ok) == 1:
                        pid, team, h = int(ok["athlete_id"].iloc[0]), seg_team, "jersey"
            if pid is None:
                pool = [by_team[t] for t in (row.home_team, row.away_team)
                        if isinstance(t, str) and t in by_team]
                pool = pd.concat(pool) if pool else pd.DataFrame(columns=ros.columns)
                exact = pool[pool["key"] == want_key]
                if len(exact) == 1:
                    pid, team, h = int(exact["athlete_id"].iloc[0]), exact["team"].iloc[0], "name"
                else:
                    sur = pool[pool["surname"] == want_sur]
                    if seg_team is not None:
                        sur = sur[sur["team"] == seg_team]
                    if len(sur) == 1:
                        pid, team, h = int(sur["athlete_id"].iloc[0]), sur["team"].iloc[0], "surname"
                    elif len(sur) > 1:
                        h = "ambiguous"
            cache[key] = (pid, team, h)
        pid, team, h = cache[key]
        ids.append(pid)
        teams.append(team)
        how.append(h)
    r["athlete_id"], r["team"], r["match"] = ids, teams, how
    bad = r[r["athlete_id"].isna()]
    if len(bad):
        log.warning("kalshi props: %d rungs (%d players) unmatched: %s", len(bad),
                    bad["player_raw"].nunique(), sorted(bad["player_raw"].unique())[:15])
    return r


def _discover() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for market, ticker in config.KALSHI_PROP_SERIES.items():
        ms = fetch_markets(ticker)
        parsed = [parse_rung(m, market) for m in ms]
        print(f"\n{market} ({ticker}): {len(ms)} open markets, "
              f"{sum(1 for p in parsed if p)} parsed")
        for m, p in list(zip(ms, parsed))[:2]:
            print(f"  title: {m.get('title')}\n  rules: {str(m.get('rules_primary'))[:160]}")
            print(f"  parsed: {p}")


if __name__ == "__main__":
    _discover()
