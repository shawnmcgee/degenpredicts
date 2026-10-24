"""ESPN's scoreboard prices - the free fallback when The Odds API is low on credits.

Pure parsing, no network: the sport's ``sources/odds.py`` fetches the scoreboard and hands the
JSON here. Every board that prices off The Odds API shares one key and the free plan's 500
credits a month, so when fewer than ``DEGEN_ODDS_RESERVE`` (75) are left - or the Odds API call
fails outright - the board takes ESPN's partner book (DraftKings) instead until the credits
reset on the 1st. One book is not a consensus, but it is a real price you can bet, and it costs
nothing.

**One file, four identical copies** - ``cfb``, ``nfl``, ``nhl`` and ``epl`` each carry this
module, because a sport package never imports another (see "Isolation over shared code" in the
README). ``tests/test_odds_fallback.py`` asserts the copies are byte-for-byte the same, so a fix
to one is a fix to all.

ESPN publishes odds in two shapes and both are read, preferring the nested current price:

* nested - ``pointSpread.home.current.line`` / ``.odds``, ``total.over.current.line``,
  ``moneyline.home.current.odds`` (and ``.draw`` for football), each with open/close/current;
* flat - ``details`` ("KC -3.5"), ``spread``, ``overUnder``, ``overOdds`` / ``underOdds``,
  ``homeTeamOdds.moneyLine`` / ``.spreadOdds``, ``drawOdds.moneyLine``.

The flat ``details`` string names the favourite, but in hockey it is sometimes the MONEYLINE
("TOR -150"), so a flat spread larger than the sport's ``max_spread`` is refused rather than read
as a 150-point handicap.

Prices come back American, as floats. ``EVEN`` is +100; ``OFF`` and anything unreadable is NaN.
"""
from __future__ import annotations

import math
import re

import pandas as pd

NAN = float("nan")
PROVIDER = "DraftKings"          # ESPN's partner book; the provider name on the feed wins


def american(x) -> float:
    """'+150', '-110', 'EVEN', 150, 'o48.5' (no) -> float American odds; NaN otherwise."""
    if x is None:
        return NAN
    if isinstance(x, (int, float)):
        v = float(x)
    else:
        s = str(x).strip().lower()
        if s in ("even", "ev"):
            return 100.0
        m = re.fullmatch(r"[+-]?\d+(\.\d+)?", s)
        if not m:
            return NAN
        v = float(s)
    return v if v == v and abs(v) >= 100 else NAN


def line(x) -> float:
    """'-3.5', 'o48.5', 'u48.5', '+1.5', 'PK', 3.5 -> float; NaN otherwise."""
    if x is None:
        return NAN
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower()
    if s in ("pk", "pick", "pick'em", "even"):
        return 0.0
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else NAN


def to_decimal(a) -> float:
    a = american(a)
    if a != a:
        return NAN
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def _side(d: dict | None, *path) -> dict:
    """The first of current / close / open under ``d[path...]`` that is a dict."""
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    if not isinstance(d, dict):
        return {}
    for k in ("current", "close", "open"):
        if isinstance(d.get(k), dict):
            return d[k]
    return {}


def _open(d: dict | None, *path) -> dict:
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d.get("open") if isinstance(d, dict) and isinstance(d.get("open"), dict) else {}


def parse_odds(odds: list, home_abbr: str, away_abbr: str, max_spread: float = 40.0) -> dict:
    """The first priced entry of an event's ``odds`` list -> one row of American prices.

    ``spread_home`` is in book convention: -3.5 means the home side gives 3.5.
    """
    out = {"provider": "", "spread_home": NAN, "spread_home_price": NAN, "spread_away_price": NAN,
           "total_line": NAN, "over_price": NAN, "under_price": NAN, "home_ml": NAN,
           "away_ml": NAN, "draw_ml": NAN, "open_spread_home": NAN, "open_total": NAN}
    for o in odds or []:
        if not isinstance(o, dict):
            continue
        row = dict(out)
        row["provider"] = str((o.get("provider") or {}).get("name") or PROVIDER)
        ps, tot, ml = o.get("pointSpread"), o.get("total"), o.get("moneyline")
        # ---- nested ---------------------------------------------------------------------
        h = _side(ps, "home")
        row["spread_home"] = line(h.get("line"))
        row["spread_home_price"] = american(h.get("odds"))
        row["spread_away_price"] = american(_side(ps, "away").get("odds"))
        row["open_spread_home"] = line(_open(ps, "home").get("line"))
        ov = _side(tot, "over")
        row["total_line"] = abs(line(ov.get("line")))
        row["over_price"] = american(ov.get("odds"))
        row["under_price"] = american(_side(tot, "under").get("odds"))
        row["open_total"] = abs(line(_open(tot, "over").get("line")))
        row["home_ml"] = american(_side(ml, "home").get("odds"))
        row["away_ml"] = american(_side(ml, "away").get("odds"))
        row["draw_ml"] = american(_side(ml, "draw").get("odds"))
        # ---- flat -----------------------------------------------------------------------
        hto = o.get("homeTeamOdds") if isinstance(o.get("homeTeamOdds"), dict) else {}
        ato = o.get("awayTeamOdds") if isinstance(o.get("awayTeamOdds"), dict) else {}
        if row["total_line"] != row["total_line"]:
            row["total_line"] = line(o.get("overUnder"))
        if row["over_price"] != row["over_price"]:
            row["over_price"] = american(o.get("overOdds"))
            row["under_price"] = american(o.get("underOdds"))
        if row["spread_home"] != row["spread_home"]:
            row["spread_home"] = _flat_spread(o, hto, ato, home_abbr, away_abbr, max_spread)
        if row["spread_home_price"] != row["spread_home_price"]:
            row["spread_home_price"] = american(hto.get("spreadOdds"))
            row["spread_away_price"] = american(ato.get("spreadOdds"))
        if row["home_ml"] != row["home_ml"]:
            row["home_ml"] = american(hto.get("moneyLine"))
            row["away_ml"] = american(ato.get("moneyLine"))
        if row["draw_ml"] != row["draw_ml"]:
            row["draw_ml"] = american((o.get("drawOdds") or {}).get("moneyLine")
                                      if isinstance(o.get("drawOdds"), dict) else None)
        if abs(row["spread_home"]) > max_spread if row["spread_home"] == row["spread_home"] else False:
            row["spread_home"] = row["spread_home_price"] = row["spread_away_price"] = NAN
        if any(row[k] == row[k] for k in ("spread_home", "total_line", "home_ml")):
            return row
    return out


def _flat_spread(o, hto, ato, home_abbr, away_abbr, max_spread) -> float:
    """The home line from ``details`` ("KC -3.5" names the favourite), cross-checked with the
    favourite flags; ``spread`` alone is not trusted for its sign."""
    det = str(o.get("details") or "").strip()
    if det.upper() in ("EVEN", "PK", "PICK"):
        return 0.0
    m = re.match(r"^([A-Za-z.&']{2,5})\s+([-+]?\d+(\.\d+)?)$", det)
    if m:
        fav, mag = m.group(1).upper(), abs(float(m.group(2)))
        if mag > max_spread:
            return NAN                  # a moneyline in the details slot, not a handicap
        if fav == str(home_abbr or "").upper():
            return -mag
        if fav == str(away_abbr or "").upper():
            return mag
    s = line(o.get("spread"))
    if s != s or abs(s) > max_spread:
        return NAN
    home_fav, away_fav = bool(hto.get("favorite")), bool(ato.get("favorite"))
    if home_fav != away_fav:
        return -abs(s) if home_fav else abs(s)
    return NAN


def _team(c: dict) -> dict:
    t = c.get("team") if isinstance(c.get("team"), dict) else {}
    return {k: str(t.get(k) or "") for k in ("abbreviation", "displayName", "shortDisplayName",
                                             "location", "name")}


def parse_scoreboard(payload, max_spread: float = 40.0) -> list[dict]:
    """Scoreboard JSON -> one dict per event: id, start (UTC), state, both teams' names and the
    book's prices. Pure; anything that is not a scoreboard (an error page, a list) gives []."""
    if not isinstance(payload, dict):
        return []
    out = []
    for ev in payload.get("events") or []:
        if not isinstance(ev, dict):
            continue
        comp = (ev.get("competitions") or [{}])[0] or {}
        sides = {c.get("homeAway"): c for c in comp.get("competitors") or [] if isinstance(c, dict)}
        if "home" not in sides or "away" not in sides:
            continue
        home, away = _team(sides["home"]), _team(sides["away"])
        st = ((comp.get("status") or ev.get("status") or {}).get("type") or {})
        ts = pd.to_datetime(comp.get("date") or ev.get("date"), utc=True, errors="coerce")
        if ts is pd.NaT or ts != ts:
            continue
        odds = comp.get("odds") or ev.get("odds") or []
        out.append({"event_id": str(ev.get("id") or comp.get("id") or ""), "start": ts,
                    "state": str(st.get("state") or ""), "home": home, "away": away,
                    "neutral": bool(comp.get("neutralSite")),
                    **parse_odds(odds, home["abbreviation"], away["abbreviation"], max_spread)})
    return out


def devig_two(a, b) -> tuple[float, float]:
    """Vig-free probabilities of two American prices; (NaN, NaN) if either is missing."""
    pa, pb = _implied(a), _implied(b)
    if pa != pa or pb != pb or not (0.98 < pa + pb < 1.25):
        return NAN, NAN
    return pa / (pa + pb), pb / (pa + pb)


def devig_three(a, b, c) -> tuple[float, float, float]:
    ps = [_implied(x) for x in (a, b, c)]
    if any(p != p for p in ps) or not (0.98 < sum(ps) < 1.35):
        return NAN, NAN, NAN
    s = sum(ps)
    return tuple(p / s for p in ps)


def _implied(a) -> float:
    a = american(a)
    if a != a:
        return NAN
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
