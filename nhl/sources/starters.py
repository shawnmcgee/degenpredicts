"""Tonight's starting goalies - confirmed, likely or projected - from Daily Faceoff.

    python -m nhl.sources.starters          # print today's and tomorrow's starters

The NHL marks who started only once the puck drops, so the board needs the day's news. Daily
Faceoff tracks it game by game and labels each starter the way the industry does:
**Confirmed** (the team or a beat reporter has said so, usually after the morning skate),
**Likely** (strong signals - who left the ice first, who started last night) or
**Unconfirmed**, which is their projection. The label decides how far the board moves off the
model's own guess from recent starts (``config.STARTER_WEIGHT``):

| label | weight on the named goalie | the rest |
|---|---:|---|
| Confirmed | 100% | - |
| Likely | 85% | the model's own starter shares |
| Unconfirmed | 50% | the model's own starter shares |

The page is a Next.js app whose data sits in a ``__NEXT_DATA__`` JSON blob. Its field names are
matched by pattern rather than by exact name, so a renamed key costs starters instead of
inventing them, and every failure - a refused request, a changed page, a team that maps to
nothing - leaves the board on the model's own guess and says so in the log. A goalie is matched
to the NHL's player id by name, first within his team's recent goalies, then league-wide
(trades), then by surname; one nobody can match (a debut) is rated as a new goalie.

Every pull is appended to ``starters.csv``. That is what lets a failed evening pull fall back to
the morning's news, and it is the record that will show how often each label turned out right.
"""
from __future__ import annotations

import json
import logging
import re
import zlib
from datetime import date, timezone

import numpy as np
import pandas as pd

from .. import config
from ..http import get
from ..teams import NAMES, UnknownTeam, _norm, from_name

log = logging.getLogger(__name__)
COLS = ["pulled_at", "date", "game_id", "side", "team", "goalie", "goalie_id", "status",
        "matched_by"]
OK = ("confirmed", "likely")
UA = "Mozilla/5.0 (X11; Linux x86_64) degenpredicts/1.0 (+https://github.com/shawnmcgee/degenpredicts)"
_NEXT = re.compile(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


# ---------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------
def _nicknames() -> dict[str, str]:
    """'bruins', 'maple leafs', 'golden knights' -> code, wherever the nickname is unambiguous."""
    seen: dict[str, set] = {}
    for full, code in NAMES.items():
        words = full.split()
        for k in (1, 2):
            if len(words) > k:
                seen.setdefault(" ".join(words[-k:]), set()).add(code)
    return {nick: codes.pop() for nick, codes in seen.items() if len(codes) == 1}


NICKNAMES = _nicknames()


def team_code(value) -> str | None:
    """A code from any spelling a page might use: full name, slug, code or bare nickname."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return from_name(value)
    except UnknownTeam:
        return NICKNAMES.get(_norm(value))


def status(text) -> str:
    t = str(text or "").lower()
    if "unconfirm" in t or "not confirm" in t:
        return "projected"
    if "confirm" in t:
        return "confirmed"
    if any(w in t for w in ("likely", "expected", "probable")):
        return "likely"
    return "projected"


def _name(v) -> str | None:
    if isinstance(v, str):
        return v.strip() or None
    if isinstance(v, dict):
        for k in ("name", "fullName", "full_name", "displayName"):
            if isinstance(v.get(k), str) and v[k].strip():
                return v[k].strip()
        first, last = v.get("firstName") or v.get("first_name"), v.get("lastName") or v.get("last_name")
        if isinstance(first, str) and isinstance(last, str) and last.strip():
            return f"{first.strip()} {last.strip()}".strip()
    return None


def _goalie(d: dict, side: str) -> str | None:
    keys = [(k.lower(), v) for k, v in d.items()]
    mine = [(k, v) for k, v in keys if k.startswith(side) and "goalie" in k and "team" not in k]
    for suffix in ("goaliename", "goaliefullname", "goalie_name", "goalie_full_name"):
        for k, v in mine:
            if k.endswith(suffix) and isinstance(v, str) and v.strip():
                return v.strip()
    first = next((v for k, v in mine if "first" in k and isinstance(v, str)), None)
    last = next((v for k, v in mine if "last" in k and isinstance(v, str)), None)
    if first and last:
        return f"{first.strip()} {last.strip()}"
    for k, v in mine:
        if isinstance(v, dict) or (k in (f"{side}goalie", f"{side}_goalie") and isinstance(v, str)):
            n = _name(v)
            if n:
                return n
    return None


def _team(d: dict, side: str) -> str | None:
    cands = [(k.lower(), v) for k, v in d.items()
             if k.lower().startswith(side) and "team" in k.lower() and "goalie" not in k.lower()]
    for pref in ("name", "abbrev", "code", "slug", ""):
        for k, v in cands:
            if pref in k:
                code = team_code(v) if isinstance(v, str) else None
                if code is None and isinstance(v, dict):
                    code = next((c for c in (team_code(v.get(x)) for x in
                                             ("name", "fullName", "abbreviation", "abbrev",
                                              "triCode", "slug", "shortName")) if c), None)
                if code:
                    return code
    return None


def _label(d: dict, side: str) -> str:
    # the news-strength field first: a generic "status" (a player's roster status, say) must not
    # be read as the confirmation
    for word in ("strength", "confirm", "status"):
        for k, v in d.items():
            kl = k.lower()
            if kl.startswith(side) and word in kl and isinstance(v, str) and v.strip():
                return status(v)
    return "projected"          # no label at all is treated as a projection, never a confirmation


def _walk(obj):
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            yield x
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(reversed(x))


def parse(payload) -> list[dict]:
    """Every game-like object in a page's data: both teams and both goalies, with their labels.
    Pure, and indifferent to where in the payload the games sit."""
    out, seen = [], set()
    for d in _walk(payload):
        h_team, a_team = _team(d, "home"), _team(d, "away")
        h_g, a_g = _goalie(d, "home"), _goalie(d, "away")
        if not (h_team and a_team and h_g and a_g) or h_team == a_team:
            continue
        if (h_team, a_team) in seen:
            continue
        seen.add((h_team, a_team))
        out.append({"home_team": h_team, "away_team": a_team,
                    "h_goalie": h_g, "h_status": _label(d, "home"),
                    "a_goalie": a_g, "a_status": _label(d, "away")})
    return out


def next_data(html: str):
    m = _NEXT.search(html or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


def fetch(day: date) -> list[dict] | None:
    """One day's starters, or None when the page could not be had or read at all."""
    url = config.STARTERS_URL.format(date=day.isoformat())
    r = get(url, headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"})
    if r is None or r.status_code != 200:
        log.warning("starting goalies unavailable for %s: HTTP %s", day,
                    getattr(r, "status_code", "no response"))
        return None
    data = next_data(r.text)
    if data is None:
        log.warning("starting goalies: no __NEXT_DATA__ on %s - has the page changed?", url)
        return None
    games = parse(data)
    if not games:
        log.warning("starting goalies: %s read, but no games found in it", url)
    return games


# ---------------------------------------------------------------------------------
# Matching names to NHL player ids
# ---------------------------------------------------------------------------------
def norm_name(name) -> str:
    return _norm(name or "")


def goalie_index(games: pd.DataFrame, seasons_back: int = 2) -> pd.DataFrame:
    """Every goalie who has started recently, by team: id, name, and when he last did."""
    cols = ["goalie_id", "goalie", "team", "date", "norm", "last"]
    if games.empty or "home_goalie_id" not in games:
        return pd.DataFrame(columns=cols)
    done = games[games["completed"].astype(bool)]
    if len(done):
        done = done[done["season"].astype(int) >= int(done["season"].max()) - seasons_back]
    parts = []
    for side in ("home", "away"):
        sub = done[[f"{side}_goalie_id", f"{side}_goalie", f"{side}_team", "date"]]
        sub.columns = ["goalie_id", "goalie", "team", "date"]
        parts.append(sub.dropna(subset=["goalie_id", "goalie"]))
    idx = pd.concat(parts, ignore_index=True)
    if idx.empty:
        return pd.DataFrame(columns=cols)
    idx["date"] = idx["date"].astype(str)
    idx = idx.sort_values("date").drop_duplicates(["goalie_id", "team"], keep="last")
    idx["goalie_id"] = idx["goalie_id"].astype(float).astype(int)
    idx["norm"] = idx["goalie"].map(norm_name)
    idx["last"] = idx["norm"].str.split().str[-1]
    return idx[cols].reset_index(drop=True)


def new_goalie_id(name) -> int:
    """A stable stand-in id for a goalie the NHL data has never seen start - negative, so it can
    never collide with a real one. The ratings treat him as a new goalie."""
    return -1 - (zlib.crc32(norm_name(name).encode()) & 0x7FFFFFFF)


def resolve(name: str, team: str, idx: pd.DataFrame) -> tuple[int, str]:
    """(NHL player id, how it was matched) for a goalie named on a page."""
    n = norm_name(name)
    if n and len(idx):
        mine = idx[idx["team"] == team]
        for pool, how in ((mine, "name"), (idx, "name, other team")):
            hit = pool[pool["norm"] == n]
            if len(hit):
                return int(hit.sort_values("date")["goalie_id"].iloc[-1]), how
        last = n.split()[-1]
        for pool, how in ((mine, "surname"), (idx, "surname, other team")):
            hit = pool[pool["last"] == last].drop_duplicates("goalie_id")
            if len(hit) == 1:
                return int(hit["goalie_id"].iloc[0]), how
    return new_goalie_id(name), "new"


# ---------------------------------------------------------------------------------
# The board's view
# ---------------------------------------------------------------------------------
def pull(board: pd.DataFrame, games: pd.DataFrame, write: bool = True) -> tuple[pd.DataFrame, set]:
    """Fetch the starters for every date on the board and match them to its games.

    Returns this pull's rows (appended to starters.csv unless ``write`` is off) and the set of
    dates the feed answered for - a date it answered for is one where a game it did not list is
    still waiting on news, not one where the feed is down.
    """
    empty = pd.DataFrame(columns=COLS)
    if not config.STARTERS_ON or board.empty:
        return empty, set()
    b = board[["game_id", "date", "home_team", "away_team"]].copy()
    b["date"] = pd.to_datetime(b["date"]).dt.date
    b["game_id"] = b["game_id"].astype(str)
    idx = goalie_index(games)
    pulled = config.now_et().astimezone(timezone.utc).isoformat(timespec="seconds")
    rows, covered, unmatched = [], set(), []
    for day in sorted(set(b["date"])):
        found = fetch(day)
        if found is None:
            continue
        if found:
            covered.add(day)
        for g in found:
            m = b[(b["date"] == day) & (b["home_team"] == g["home_team"])
                  & (b["away_team"] == g["away_team"])]
            if m.empty:
                unmatched.append(f"{g['away_team']}@{g['home_team']} {day}")
                continue
            for side, team in (("h", g["home_team"]), ("a", g["away_team"])):
                gid, how = resolve(g[f"{side}_goalie"], team, idx)
                rows.append({"pulled_at": pulled, "date": day, "game_id": m["game_id"].iloc[0],
                             "side": side, "team": team, "goalie": g[f"{side}_goalie"],
                             "goalie_id": gid, "status": g[f"{side}_status"], "matched_by": how})
    df = pd.DataFrame(rows, columns=COLS)
    if unmatched:
        log.info("starting goalies: %d listed games not on the board: %s", len(unmatched),
                 unmatched[:6])
    if len(df) and write:
        config.ensure_dirs()
        df.to_csv(config.STARTERS, mode="a", header=not config.STARTERS.exists(), index=False)
    if len(df):
        new = df[df["matched_by"] == "new"]
        if len(new):
            log.warning("starting goalies: no NHL id for %s - rated as new goalies",
                        sorted(set(new["goalie"])))
    counts = df["status"].value_counts().to_dict() if len(df) else {}
    log.info("starting goalies: %d of %d board games (%s)", df["game_id"].nunique() if len(df) else 0,
             len(b), ", ".join(f"{counts.get(k, 0)} {k}" for k in ("confirmed", "likely", "projected")))
    return df, covered


def load() -> pd.DataFrame:
    if not config.STARTERS.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(config.STARTERS, dtype={"game_id": str}, low_memory=False)


def latest(game_ids, fresh: pd.DataFrame | None = None) -> pd.DataFrame:
    """The freshest label per game and side, from every pull so far plus ``fresh`` - so an
    evening pull that fails still leaves the board with the morning's news."""
    df = load()
    if fresh is not None and len(fresh):
        df = pd.concat([df, fresh], ignore_index=True) if len(df) else fresh.copy()
    if df.empty:
        return df
    df["game_id"] = df["game_id"].astype(str)
    df = df[df["game_id"].isin(set(map(str, game_ids)))]
    return df.sort_values("pulled_at").drop_duplicates(["game_id", "side"], keep="last")


def known(rows: pd.DataFrame) -> dict:
    """{game_id: {"h": {id, weight, status, name}, "a": {...}}} for features.build."""
    out: dict = {}
    for r in rows.itertuples(index=False):
        if r.goalie_id != r.goalie_id:
            continue
        out.setdefault(str(r.game_id), {})[r.side] = {
            "id": int(r.goalie_id), "status": r.status, "name": r.goalie,
            "weight": config.STARTER_WEIGHT.get(r.status, config.STARTER_WEIGHT["projected"])}
    return out


def pending(board: pd.DataFrame, rows: pd.DataFrame, covered: set) -> np.ndarray:
    """Games whose picks must wait for their goalies.

    A game waits when the feed knows it - or answered for its date without listing it - and a
    starter on either side is not yet confirmed or likely. A game the feed has never answered
    for is priced and staked on the model's own guess, as it would be with the feed switched off.
    """
    if not config.REQUIRE_STARTERS:
        return np.zeros(len(board), dtype=bool)
    lab = {(str(r.game_id), r.side): r.status for r in rows.itertuples(index=False)} \
        if len(rows) else {}
    out = []
    for r in board.itertuples(index=False):
        gid, day = str(r.game_id), pd.Timestamp(r.date).date()
        sts = [lab.get((gid, s)) for s in ("h", "a")]
        tracked = any(s is not None for s in sts) or day in covered
        out.append(bool(tracked and not all(s in OK for s in sts)))
    return np.asarray(out, dtype=bool)


if __name__ == "__main__":
    from datetime import timedelta

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    today = config.today_et()
    for d in (today, today + timedelta(days=1)):
        print(d, json.dumps(fetch(d), indent=2, default=str))
