"""The one place a team code is translated, plus arenas and time zones.

It *raises* on anything unmapped rather than passing it through, for the same reason the NFL
and Premier League maps do: a silently wrong code splits or merges a franchise's rating history
and nothing downstream can detect it.

Two relocations collapse onto the club playing today, so a rating follows the ROSTER:

* **Atlanta Thrashers -> Winnipeg Jets (2011).** The same franchise and players moved north.
* **Phoenix / Arizona Coyotes -> Utah (2024).** The NHL files Utah as a new franchise with no
  Coyotes history, which is right for the record books and wrong for a rating: the 2024-25
  Utah roster WAS the 2023-24 Coyotes roster, bought and moved over a summer. Starting Utah at
  league average would hand a 36-41 team a clean slate it had not earned.

Arenas are looked up by (team, season), so the Thrashers' games are measured from Atlanta and
the Coyotes' from Glendale and Tempe with no special cases downstream.
"""
from __future__ import annotations

import math
import re
import unicodedata

TEAMS = ["ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET", "EDM", "FLA",
         "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
         "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH"]

# Every spelling any feed sends, mapped to the code above. The NHL's own tri-codes come first;
# the rest are the short forms other feeds use (LA, NJ, SJ, TB are the common ones).
TEAM_ALIASES = {t: t for t in TEAMS}
TEAM_ALIASES.update({
    "ATL": "WPG", "PHX": "UTA", "ARI": "UTA", "UTAH": "UTA",
    "LA": "LAK", "L.A": "LAK", "NJ": "NJD", "N.J": "NJD", "SJ": "SJS", "S.J": "SJS",
    "TB": "TBL", "T.B": "TBL", "MON": "MTL", "WAS": "WSH", "CLB": "CBJ", "CLS": "CBJ",
    "NAS": "NSH", "CAL": "CGY", "VEG": "VGK", "LV": "VGK", "WIN": "WPG", "PHO": "UTA",
    "NYIS": "NYI", "NYRS": "NYR",
})
RELOCATIONS = {"ATL": "WPG", "PHX": "UTA", "ARI": "UTA"}

# Full names as price feeds spell them. Accents and punctuation are stripped before lookup, so
# "Montréal" and "St. Louis" need one entry each.
NAMES = {
    "anaheim ducks": "ANA", "boston bruins": "BOS", "buffalo sabres": "BUF",
    "carolina hurricanes": "CAR", "columbus blue jackets": "CBJ", "calgary flames": "CGY",
    "chicago blackhawks": "CHI", "colorado avalanche": "COL", "dallas stars": "DAL",
    "detroit red wings": "DET", "edmonton oilers": "EDM", "florida panthers": "FLA",
    "los angeles kings": "LAK", "minnesota wild": "MIN", "montreal canadiens": "MTL",
    "new jersey devils": "NJD", "nashville predators": "NSH", "new york islanders": "NYI",
    "new york rangers": "NYR", "ottawa senators": "OTT", "philadelphia flyers": "PHI",
    "pittsburgh penguins": "PIT", "seattle kraken": "SEA", "san jose sharks": "SJS",
    "st louis blues": "STL", "tampa bay lightning": "TBL", "toronto maple leafs": "TOR",
    "utah hockey club": "UTA", "utah mammoth": "UTA", "utah": "UTA",
    "vancouver canucks": "VAN", "vegas golden knights": "VGK", "winnipeg jets": "WPG",
    "washington capitals": "WSH",
    # historical names still in older feeds
    "arizona coyotes": "UTA", "phoenix coyotes": "UTA", "atlanta thrashers": "WPG",
}
# Bare city names that name two clubs. Refused rather than guessed: picking wrong would price
# the other team's game.
AMBIGUOUS = {"new york", "ny", "los angeles"}


class UnknownTeam(KeyError):
    pass


def canon(code) -> str:
    """Canonical team code for any NHL tri-code or common short form. Raises if unknown."""
    if code is None or (isinstance(code, float) and code != code):
        raise UnknownTeam(f"no team code: {code!r}")
    key = str(code).strip().upper()
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    raise UnknownTeam(f"unmapped NHL team code {code!r} - add it to nhl/teams.py TEAM_ALIASES")


def is_known(code) -> bool:
    try:
        canon(code)
        return True
    except UnknownTeam:
        return False


def _norm(name: str) -> str:
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    n = re.sub(r"[^a-z0-9 ]+", " ", n.lower())
    return re.sub(r"\s+", " ", n).strip()


def from_name(name) -> str:
    """Canonical code for a full club name as a price feed spells it. Raises if unknown."""
    n = _norm(name or "")
    if n in AMBIGUOUS:
        raise UnknownTeam(f"{name!r} names more than one club")
    if n in NAMES:
        return NAMES[n]
    if is_known(name):
        return canon(name)
    raise UnknownTeam(f"unmapped NHL team name {name!r} - add it to nhl/teams.py NAMES")


# --- arenas --------------------------------------------------------------------------
# team -> [(first season, lat, lon, UTC offset in standard time)]. Later rows win.
VENUES = {
    "ANA": [(1900, 33.8078, -117.8765, -8)],
    "BOS": [(1900, 42.3662, -71.0621, -5)],
    "BUF": [(1900, 42.8750, -78.8764, -5)],
    "CAR": [(1900, 35.8033, -78.7219, -5)],
    "CBJ": [(1900, 39.9693, -83.0061, -5)],
    "CGY": [(1900, 51.0374, -114.0519, -7)],
    "CHI": [(1900, 41.8807, -87.6742, -6)],
    "COL": [(1900, 39.7487, -105.0077, -7)],
    "DAL": [(1900, 32.7905, -96.8103, -6)],
    "DET": [(1900, 42.3253, -83.0514, -5), (2017, 42.3411, -83.0553, -5)],
    "EDM": [(1900, 53.5469, -113.4978, -7)],
    "FLA": [(1900, 26.1584, -80.3256, -5)],
    "LAK": [(1900, 34.0430, -118.2673, -8)],
    "MIN": [(1900, 44.9448, -93.1010, -6)],
    "MTL": [(1900, 45.4961, -73.5693, -5)],
    "NJD": [(1900, 40.7336, -74.1711, -5)],
    "NSH": [(1900, 36.1592, -86.7785, -6)],
    "NYI": [(1900, 40.7229, -73.5905, -5), (2015, 40.6826, -73.9754, -5),
            (2020, 40.7229, -73.5905, -5), (2021, 40.7117, -73.7256, -5)],
    "NYR": [(1900, 40.7505, -73.9934, -5)],
    "OTT": [(1900, 45.2969, -75.9272, -5)],
    "PHI": [(1900, 39.9012, -75.1720, -5)],
    "PIT": [(1900, 40.4394, -79.9892, -5)],
    "SEA": [(1900, 47.6221, -122.3540, -8)],
    "SJS": [(1900, 37.3327, -121.9012, -8)],
    "STL": [(1900, 38.6268, -90.2026, -6)],
    "TBL": [(1900, 27.9427, -82.4518, -5)],
    "TOR": [(1900, 43.6435, -79.3791, -5)],
    # Glendale, then Tempe, then Salt Lake City - one roster, three buildings
    "UTA": [(1900, 33.5319, -112.2611, -7), (2022, 33.4265, -111.9313, -7),
            (2024, 40.7683, -111.9011, -7)],
    "VAN": [(1900, 49.2778, -123.1089, -8)],
    "VGK": [(1900, 36.1029, -115.1784, -8)],
    # Atlanta until the move, Winnipeg after
    "WPG": [(1900, 33.7573, -84.3963, -5), (2011, 49.8928, -97.1436, -6)],
    "WSH": [(1900, 38.8981, -77.0209, -5)],
}


def venue(team: str, season: int):
    """(lat, lon, utc_offset) of the arena `team` played home games in during `season`."""
    row = None
    for r in VENUES.get(team, []):
        if season >= r[0]:
            row = r
    return None if row is None else (row[1], row[2], row[3])


def haversine_km(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))
