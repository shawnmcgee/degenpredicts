"""Polymarket event-contract prices for the NFL.

Same job as :mod:`kalshi`, different exchange: read the ask on a binary contract that settles
at $1.00, so it can be compared directly against the model's probability. The fee schedule
happens to have the same shape (``coef * P * (1-P)``, sports coefficient 0.05 against Kalshi's
0.07, maker 0.00), so everything downstream of the ask is shared - see :mod:`venues`.

    >>> python -m nfl.sources.polymarket --discover

**Read that as a requirement, not a suggestion.** The Kalshi module was written against a
captured live response; this one was written against Polymarket's published API docs without
a live call, because the environment it was authored in could not reach their hosts. The
shapes below are therefore *expected*, not confirmed, and the module is built to be wrong
safely:

* every field is read through :func:`_get`, which accepts the camelCase and snake_case
  spellings the Gamma and CLOB APIs use in different places;
* nothing raises - a changed shape yields an empty frame, `venues` logs it, and the run
  continues on Kalshi alone;
* markets **fail closed**. No confirmed ask with real size behind it means ``tradeable`` is
  False, so an unverified shape costs visibility rather than producing a phantom edge.

Two APIs, because neither alone is enough:

``gamma-api.polymarket.com/markets``
    Discovery and metadata: the question text, the outcome names, the game start time, the
    handicap, and the sports market type. Its ``outcomes`` / ``outcomePrices`` /
    ``clobTokenIds`` are *stringified JSON inside JSON* and need a second parse.

``clob.polymarket.com``
    The order book. Gamma's ``outcomePrices`` is a last/mid-ish number, and this pipeline's
    whole discipline is that EV is measured against the **ask** you can actually pay, never
    the mid. Where Gamma carries ``bestAsk``/``bestBid`` those are used; otherwise the CLOB
    ``/books`` batch endpoint is asked for real depth. If neither answers, the market is not
    tradeable.

Team names arrive in prose (``question``, ``groupItemTitle``) exactly as they do from Kalshi,
so the same ``sources/odds.py`` matcher reduces them to nflverse abbreviations.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone

import pandas as pd

from .. import config
from ..http import get_json, session

log = logging.getLogger(__name__)

GAMMA = os.environ.get("DEGEN_POLYMARKET_GAMMA", "https://gamma-api.polymarket.com")
CLOB = os.environ.get("DEGEN_POLYMARKET_CLOB", "https://clob.polymarket.com")

# Gamma tags/slugs for the sport. Confirm with --discover; Polymarket has renamed these.
TAG = os.environ.get("DEGEN_POLYMARKET_NFL_TAG", "nfl")

# `sportsMarketType` values, mapped to the kinds this pipeline prices. Polymarket exposes the
# market shape as a field rather than in prose, which is why this module needs no equivalent of
# Kalshi's three rules regexes for the *shape* - only for the team names and the strike.
MARKET_TYPES = {
    "moneyline": ("moneyline", "winner", "money_line"),
    "spread": ("spread", "handicap", "spreads"),
    "total": ("total", "totals", "over_under", "overunder"),
}

# Liquidity gates. Polymarket sizes are USDC of depth rather than contract counts, so these are
# deliberately separate knobs from the Kalshi ones rather than shared numbers that happen to
# look alike.
_BOOK_BATCH = int(os.environ.get("DEGEN_PM_BOOK_BATCH", "100"))

MAX_SPREAD = float(os.environ.get("DEGEN_PM_MAX_SPREAD", "0.06"))
MIN_ASK_SIZE = float(os.environ.get("DEGEN_PM_MIN_SIZE", "100"))     # USDC at the best ask
MIN_VOLUME = float(os.environ.get("DEGEN_PM_MIN_VOLUME", "1000"))    # USDC traded

# "Will Syracuse beat Pittsburgh?" / "Syracuse vs. Pittsburgh"
VS_RE = re.compile(r"(?P<a>.+?)\s+(?:vs\.?|@|at)\s+(?P<b>.+?)\s*$", re.I)
BEAT_RE = re.compile(r"Will\s+(?P<winner>.+?)\s+beat\s+(?P<loser>.+?)\?", re.I)
# "Syracuse -6.5" / "Over 55.5" - the handicap when it is only in the title
NUM_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)")


def _get(d: dict, *names, default=None):
    """Read the first present key out of several spellings (camelCase / snake_case)."""
    for n in names:
        if isinstance(d, dict) and n in d and d[n] is not None:
            return d[n]
    return default


def _f(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _jlist(v) -> list:
    """Gamma returns `outcomes`, `outcomePrices` and `clobTokenIds` as JSON *strings*."""
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip():
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else [parsed]
        except ValueError:
            return []
    return []


def _date_from(v):
    """Gamma timestamps are ISO-8601 UTC. Games are keyed on the ET calendar date, matching
    the rest of the pipeline - a 8pm ET Saturday kickoff is 00:00 UTC Sunday, and keying it on
    the UTC date would put it on the wrong day of the board."""
    if not v:
        return None
    try:
        ts = pd.Timestamp(v)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        return ts.tz_convert(config.ET).date()
    except Exception:
        return None


def _kind_of(m: dict) -> str | None:
    raw = str(_get(m, "sportsMarketType", "sports_market_type", "marketType", default="")).lower()
    for kind, aliases in MARKET_TYPES.items():
        if raw in aliases:
            return kind
    return None


# ---------------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------------
def fetch_markets(tag: str | None = None, closed: bool = False,
                  max_pages: int = 20, limit: int = 500) -> list[dict]:
    """Page through Gamma's /markets for the sport's tag. Gamma pages by offset."""
    tag = tag or TAG
    out, offset, pages = [], 0, 0
    while pages < max_pages:
        params = {"limit": limit, "offset": offset,
                  "closed": str(bool(closed)).lower(), "tag_slug": tag}
        payload = get_json(f"{GAMMA}/markets", params=params)
        # Gamma returns a bare list; some deployments wrap it in {"data": [...]}
        batch = payload if isinstance(payload, list) else (payload or {}).get("data") or []
        if not batch:
            break
        out.extend(batch)
        offset += len(batch)
        pages += 1
        if len(batch) < limit:
            break
    if not out:
        log.warning("polymarket: no markets for tag %r. Confirm it with "
                    "`python -m nfl.sources.polymarket --discover`.", tag)
    else:
        log.info("polymarket %s: %d markets over %d page(s)", tag, len(out), pages)
    return out


def fetch_books(token_ids: list[str]) -> dict[str, dict]:
    """Best bid/ask and depth per token, from the CLOB order book.

    Batched: one POST for up to `_BOOK_BATCH` tokens rather than a call per market, because a
    college Saturday is ~50 games and a spread ladder multiplies that. Returns {} on any
    failure, which leaves every market untradeable rather than guessing at a price.
    """
    ids = [t for t in dict.fromkeys(token_ids) if t]
    if not ids:
        return {}
    out: dict[str, dict] = {}
    for i in range(0, len(ids), _BOOK_BATCH):
        chunk = ids[i:i + _BOOK_BATCH]
        try:
            r = session().post(f"{CLOB}/books", json=[{"token_id": t} for t in chunk],
                               timeout=config.HTTP_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            log.warning("polymarket CLOB /books failed (%s) - those markets stay untradeable", e)
            continue
        for book in (payload if isinstance(payload, list) else [payload]):
            tid = str(_get(book, "asset_id", "token_id", "assetId", default=""))
            if tid:
                out[tid] = book
    log.info("polymarket: order books for %d/%d tokens", len(out), len(ids))
    return out


def _side_top(book: dict, side: str) -> tuple[float, float]:
    """(price, size) at the top of one side of a CLOB book.

    CLOB returns `bids` ascending and `asks` descending by price, but that ordering is not
    worth trusting for something this consequential: the best ask is the *lowest* price
    offered and the best bid the highest, so both are computed rather than indexed.
    """
    levels = _get(book, side, default=[]) or []
    priced = [(_f(_get(lv, "price")), _f(_get(lv, "size"), 0.0)) for lv in levels]
    priced = [(p, s) for p, s in priced if p == p and 0 < p < 1]
    if not priced:
        return float("nan"), 0.0
    return min(priced) if side == "asks" else max(priced)


def _liquidity(m: dict, book: dict | None) -> dict:
    """Quote and liquidity for one outcome token.

    Prefers Gamma's `bestAsk`/`bestBid` when present, else the CLOB book. Volume comes from
    Gamma either way. Everything fails closed: an ask we could not confirm is not tradeable,
    however attractive the mid looked, because this whole module exists to price against a
    number someone is actually offering.
    """
    yes_bid = _f(_get(m, "bestBid", "best_bid"))
    yes_ask = _f(_get(m, "bestAsk", "best_ask"))
    ask_size = float("nan")
    if book:
        b_price, b_size = _side_top(book, "bids")
        a_price, a_size = _side_top(book, "asks")
        # the book is the authority when it answered; Gamma's copy can lag
        yes_bid = b_price if b_price == b_price else yes_bid
        yes_ask = a_price if a_price == a_price else yes_ask
        ask_size = a_size
    if ask_size != ask_size:
        # No book. `liquidity` is total USDC in the market, not size at the touch, so it is a
        # weak proxy - accepted only because the spread gate below still has to pass.
        ask_size = _f(_get(m, "liquidityNum", "liquidity"), 0.0)
    spread = yes_ask - yes_bid if yes_ask == yes_ask and yes_bid == yes_bid else float("nan")
    volume = _f(_get(m, "volumeNum", "volume"), 0.0)
    return {
        "yes_bid": yes_bid, "yes_ask": yes_ask, "quote_spread": spread,
        "ask_size": ask_size, "bid_size": float("nan"), "volume": volume,
        "open_interest": _f(_get(m, "openInterest", "open_interest"), 0.0),
        "tradeable": bool(yes_ask == yes_ask and spread == spread and spread <= MAX_SPREAD
                          and ask_size >= MIN_ASK_SIZE and volume >= MIN_VOLUME),
    }


# ---------------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------------
def _matchup(m: dict) -> tuple[str | None, str | None]:
    """(away_raw, home_raw). Polymarket phrases fixtures away-first, as Kalshi does."""
    for text in (_get(m, "gameTitle", "game_title"), _get(m, "title"),
                 _get(m, "question"), _get(m, "slug")):
        if not text:
            continue
        t = str(text).replace("-", " ") if text is _get(m, "slug") else str(text)
        hit = VS_RE.search(t.strip().rstrip("?"))
        if hit:
            return hit.group("a").strip(), hit.group("b").strip()
    hit = BEAT_RE.search(str(_get(m, "question", default="")))
    if hit:
        # "Will X beat Y?" names the sides but not which is home; the odds matcher resolves
        # names, and predict.py joins on the unordered pair, so both orders are tried there.
        return hit.group("winner").strip(), hit.group("loser").strip()
    return None, None


def _strike(m: dict, kind: str) -> float:
    """The handicap. Polymarket carries it as a field, which is why there is no ladder regex.

    Kalshi's ``strike_type: "greater"`` means "resolves YES above the strike", and this
    pipeline is built entirely around that convention, so a Polymarket line is normalised to
    the same meaning: the absolute handicap, with the side carried separately.
    """
    for key in ("line", "spread", "strike", "floorStrike", "floor_strike"):
        v = _f(_get(m, key))
        if v == v:
            return abs(v) if kind == "spread" else v
    hit = NUM_RE.search(str(_get(m, "groupItemTitle", "question", default="")))
    if hit:
        v = _f(hit.group(1))
        return abs(v) if kind == "spread" and v == v else v
    return float("nan")


def _outcome_tokens(m: dict) -> list[tuple[str, str]]:
    """[(outcome_name, clob_token_id)] for one market, from the stringified arrays."""
    names = [str(x) for x in _jlist(_get(m, "outcomes"))]
    tokens = [str(x) for x in _jlist(_get(m, "clobTokenIds", "clob_token_ids"))]
    if not names:
        return []
    return list(zip(names, tokens + [""] * (len(names) - len(tokens))))


def _base_row(m: dict) -> dict:
    away, home = _matchup(m)
    return {
        "ticker": str(_get(m, "conditionId", "condition_id", "id", default="")),
        "event_ticker": str(_get(m, "eventSlug", "event_slug", "gameId", "game_id",
                                 "slug", default="")),
        "away_raw": away, "home_raw": home,
        "date": _date_from(_get(m, "gameStartTime", "game_start_time",
                                "startDate", "start_date", "endDate")),
        "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def parse_market(m: dict, book_of=None) -> list[dict]:
    """One moneyline market -> one row per side.

    Polymarket lists a two-outcome market per game rather than Kalshi's one market per side,
    so this returns a list. Each outcome is a separate CLOB token with its own book, and the
    YES price on the "Team A" token *is* P(Team A wins) - the same reading Kalshi allows.
    """
    if _kind_of(m) not in (None, "moneyline"):
        return []
    base = _base_row(m)
    if not base["date"]:
        return []
    rows = []
    for name, token in _outcome_tokens(m):
        if name.strip().lower() in {"yes", "no"}:
            # A "Will X beat Y?" market prices one side only, so YES is the named team and NO
            # is its complement - which we skip, because P(NO) is 1 - P(YES) and pricing both
            # would double-count the same game.
            hit = BEAT_RE.search(str(_get(m, "question", default="")))
            if not hit or name.strip().lower() == "no":
                continue
            name = hit.group("winner").strip()
        book = book_of(token) if book_of else None
        rows.append({**base, "pm_team": name.strip(), "token_id": token,
                     **_liquidity(m, book)})
    return rows


def parse_ladder(m: dict, kind: str, book_of=None) -> list[dict]:
    """One spread/total market -> one row per rung side, normalised to "resolves above".

    The YES token of a "Over 55.5" market is P(total > 55.5); the YES token of a
    "Syracuse -6.5" market is P(Syracuse wins by more than 6.5). Both match Kalshi's
    ``strike_type: "greater"``, which is what lets `predict._price_ladders` treat the two
    venues' rungs identically.
    """
    if _kind_of(m) != kind:
        return []
    base = _base_row(m)
    strike = _strike(m, kind)
    if not base["date"] or strike != strike:
        return []
    team = None
    if kind == "spread":
        # the favoured side, named in the group title ("Syracuse -6.5")
        title = str(_get(m, "groupItemTitle", "question", default=""))
        team = NUM_RE.split(title)[0].strip(" -+") or None
    rows = []
    for name, token in _outcome_tokens(m):
        low = name.strip().lower()
        if low in {"no", "under"} or (kind == "total" and low.startswith("under")):
            continue                       # we price the "above" side, as Kalshi does
        book = book_of(token) if book_of else None
        rows.append({**base, "kind": kind, "strike": strike, "token_id": token,
                     "pm_team": team, **_liquidity(m, book)})
    return rows


# ---------------------------------------------------------------------------------
# Boards - the interface `venues.py` fans out to
# ---------------------------------------------------------------------------------
def _resolve(df: pd.DataFrame, matcher, team_col: str | None) -> pd.DataFrame:
    if matcher is not None:
        df["home_team"] = df["home_raw"].map(lambda x: matcher(x) if x else None)
        df["away_team"] = df["away_raw"].map(lambda x: matcher(x) if x else None)
        df["team"] = (df[team_col].map(lambda x: matcher(x) if x else None)
                      if team_col else None)
    else:
        df["home_team"], df["away_team"] = df["home_raw"], df["away_raw"]
        df["team"] = df[team_col] if team_col else None
    return df


def _board(kind: str | None, matcher=None) -> pd.DataFrame:
    markets = fetch_markets()
    if not markets:
        return pd.DataFrame()
    want = [m for m in markets
            if _kind_of(m) == kind or (kind is None and _kind_of(m) in (None, "moneyline"))]
    tokens = [t for m in want for _, t in _outcome_tokens(m)]
    books = fetch_books(tokens)
    of = books.get
    rows = []
    for m in want:
        rows.extend(parse_market(m, of) if kind is None else parse_ladder(m, kind, of))
    rows = [r for r in rows if r]
    if not rows:
        log.warning("polymarket: %d %s markets returned but none parsed - the shapes have "
                    "changed; run `python -m nfl.sources.polymarket --discover`.",
                    len(want), kind or "moneyline")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = _resolve(df, matcher, "pm_team")
    log.info("polymarket %s: %d rows, %d tradeable", kind or "moneyline", len(df),
             int(df["tradeable"].sum()))
    return df


def moneyline_board(matcher=None) -> pd.DataFrame:
    return _board(None, matcher)


def ladder_board(kind: str, matcher=None) -> pd.DataFrame:
    return _board(kind, matcher)


def fee_rate() -> float:
    """Polymarket's sports taker coefficient.

    Their documentation and the CLOB `/fee-rate` endpoint have disagreed per sport
    (Polymarket/py-clob-client#326), so this is configurable and worth confirming against the
    live endpoint before sizing anything. `venues.fee` reads the value from
    `config.FEE_COEF["polymarket_taker"]`; this helper exists so `--discover` can print what
    the API currently claims next to what we are charging.
    """
    return float(config.FEE_COEF.get("polymarket_taker", 0.05) or 0.0)


def _discover() -> None:
    """Print enough of the live shape to confirm or correct every assumption above."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(f"Gamma: {GAMMA}\nCLOB:  {CLOB}\ntag:   {TAG}\n")
    markets = fetch_markets()
    print(f"{len(markets)} open markets for tag {TAG!r}")
    if not markets:
        print("\n!! Nothing returned. Try another tag, e.g.:\n"
              "     DEGEN_POLYMARKET_NFL_TAG=nfl-football python -m nfl.sources.polymarket --discover\n"
              f"   or list what Gamma has: curl '{GAMMA}/tags?limit=500'")
        return
    kinds: dict[str, int] = {}
    for m in markets:
        kinds[str(_kind_of(m))] = kinds.get(str(_kind_of(m)), 0) + 1
    print(f"sportsMarketType breakdown: {kinds}")
    print(f"\nfirst market's keys:\n  {sorted(markets[0].keys())}\n")
    for kind in (None, "spread", "total"):
        want = [m for m in markets if _kind_of(m) == kind
                or (kind is None and _kind_of(m) in (None, "moneyline"))]
        print(f"--- {kind or 'moneyline'}: {len(want)} markets")
        for m in want[:2]:
            print(f"  question: {str(_get(m, 'question'))[:110]}")
            print(f"  groupItemTitle={_get(m, 'groupItemTitle')!r} "
                  f"line={_get(m, 'line')!r} spread={_get(m, 'spread')!r} "
                  f"gameStartTime={_get(m, 'gameStartTime')!r}")
            print(f"  outcomes={_jlist(_get(m, 'outcomes'))} "
                  f"bestBid={_get(m, 'bestBid')!r} bestAsk={_get(m, 'bestAsk')!r}")
            parsed = parse_market(m) if kind is None else parse_ladder(m, kind)
            print(f"  parsed: {parsed}")
        if want and not any(parse_market(m) if kind is None else parse_ladder(m, kind)
                            for m in want[:5]):
            print("  !! markets returned but none parsed - fix the readers in this module.")
    print(f"\nfee coefficient in use: {fee_rate()} (fee = coef * P * (1-P))")
    print(f"   confirm against: curl '{CLOB}/fee-rate-bps' and docs.polymarket.com/trading/fees")


if __name__ == "__main__":
    import sys
    if "--discover" in sys.argv:
        _discover()
    else:
        print(__doc__)
