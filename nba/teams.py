"""The one place a team code is translated, plus where every game was played.

It *raises* on anything unmapped rather than passing it through, for the same reason the NFL,
NHL and Premier League maps do: a silently wrong code splits or merges a franchise's rating
history and nothing downstream can detect it. ESPN's schedule also carries the All-Star game
("EAST" v "WEST", "Team Chuck" v "Team Shaq") and unfilled knockout slots ("TBD"); refusing
unknown codes is what keeps those out of the ratings.

Codes are the league's own tri-codes. ESPN spells five of them differently (GS, NY, NO, SA,
UTAH, WSH) and those are aliases here. Two relocations collapse onto the club playing today, so
a rating follows the ROSTER:

* **New Jersey -> Brooklyn Nets (2012).** Same franchise, a river away.
* **Seattle SuperSonics -> Oklahoma City Thunder (2008).** The 2008-09 Thunder were the 2007-08
  Sonics, Kevin Durant included.

The Charlotte Hornets name belonged to the franchise now in New Orleans until 2002; the
current Charlotte club began as the Bobcats in 2004. Nothing loaded here is older than 2003, so
"CHA" is only ever the Bobcats/Hornets and "NOP" only ever New Orleans.

**Where a game was played** comes from the game's own venue city in the schedule, not from a
home-arena table - so the Hornets' two Katrina seasons in Oklahoma City, the Raptors' 2020-21
season in Tampa, the Orlando bubble and the London, Paris, Mexico City, Berlin, Manchester and
Las Vegas games all come out right with no special cases.
"""
from __future__ import annotations

import math
import re
import unicodedata

TEAMS = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND",
         "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
         "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]

TEAM_ALIASES = {t: t for t in TEAMS}
TEAM_ALIASES.update({
    # ESPN's spellings
    "GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS",
    # other feeds' short forms
    "BRK": "BKN", "BRO": "BKN", "CHO": "CHA", "CHH": "CHA", "GOS": "GSW", "GOL": "GSW",
    "PHO": "PHX", "NOR": "NOP", "NOH": "NOP", "NOK": "NOP", "SAN": "SAS", "UTH": "UTA",
    "WSN": "WAS", "NYC": "NYK",
    # relocations: the rating follows the roster
    "NJ": "BKN", "NJN": "BKN", "SEA": "OKC",
})
RELOCATIONS = {"NJ": "BKN", "NJN": "BKN", "SEA": "OKC"}

# Full names as price feeds and the injury report spell them. Accents and punctuation are
# stripped before lookup.
NAMES = {
    "atlanta hawks": "ATL", "boston celtics": "BOS", "brooklyn nets": "BKN",
    "charlotte hornets": "CHA", "charlotte bobcats": "CHA", "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE", "dallas mavericks": "DAL", "denver nuggets": "DEN",
    "detroit pistons": "DET", "golden state warriors": "GSW", "houston rockets": "HOU",
    "indiana pacers": "IND", "los angeles clippers": "LAC", "la clippers": "LAC",
    "los angeles lakers": "LAL", "la lakers": "LAL", "memphis grizzlies": "MEM",
    "miami heat": "MIA", "milwaukee bucks": "MIL", "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP", "new orleans hornets": "NOP", "new york knicks": "NYK",
    "oklahoma city thunder": "OKC", "orlando magic": "ORL", "philadelphia 76ers": "PHI",
    "phoenix suns": "PHX", "portland trail blazers": "POR", "sacramento kings": "SAC",
    "san antonio spurs": "SAS", "toronto raptors": "TOR", "utah jazz": "UTA",
    "washington wizards": "WAS",
    # historical names still in older feeds
    "new jersey nets": "BKN", "seattle supersonics": "OKC",
    "new orleans oklahoma city hornets": "NOP",
}
# Bare city names that name two clubs. Refused rather than guessed: picking wrong would price
# the other team's game.
AMBIGUOUS = {"los angeles", "la", "l a"}


class UnknownTeam(KeyError):
    pass


def canon(code) -> str:
    """Canonical code for any NBA tri-code or common short form. Raises if unknown."""
    if code is None or (isinstance(code, float) and code != code):
        raise UnknownTeam(f"no team code: {code!r}")
    key = str(code).strip().upper()
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    raise UnknownTeam(f"unmapped NBA team code {code!r} - add it to nba/teams.py TEAM_ALIASES")


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
    raise UnknownTeam(f"unmapped NBA team name {name!r} - add it to nba/teams.py NAMES")


# --- venues ----------------------------------------------------------------------------
# venue city -> (lat, lon, UTC offset in standard time, altitude in metres). Every city an NBA
# game has been played in since 2003, neutral sites included. Denver and Salt Lake City are the
# two that matter for altitude.
CITIES = {
    "Atlanta": (33.7573, -84.3963, -5, 320), "Austin": (30.2672, -97.7431, -6, 150),
    "Berlin": (52.5200, 13.4050, 1, 34), "Boston": (42.3662, -71.0621, -5, 5),
    "Brooklyn": (40.6826, -73.9754, -5, 10), "Charlotte": (35.2251, -80.8392, -5, 230),
    "Chicago": (41.8807, -87.6742, -6, 180), "Cleveland": (41.4965, -81.6882, -5, 200),
    "Dallas": (32.7905, -96.8103, -6, 130), "Denver": (39.7487, -105.0077, -7, 1609),
    "Detroit": (42.3411, -83.0553, -5, 190), "East Rutherford": (40.8128, -74.0742, -5, 5),
    "Houston": (29.7508, -95.3621, -6, 15), "Indianapolis": (39.7640, -86.1555, -5, 220),
    "Inglewood": (33.9450, -118.3430, -8, 30), "Lake Buena Vista": (28.3705, -81.5569, -5, 30),
    "Las Vegas": (36.1029, -115.1784, -8, 610), "London": (51.5072, -0.1276, 0, 20),
    "Los Angeles": (34.0430, -118.2673, -8, 90), "Manchester": (53.4808, -2.2426, 0, 40),
    "Memphis": (35.1382, -90.0506, -6, 80), "Mexico City": (19.4326, -99.1332, -6, 2240),
    "Miami": (25.7814, -80.1870, -5, 2), "Milwaukee": (43.0451, -87.9172, -6, 190),
    "Minneapolis": (44.9795, -93.2761, -6, 260), "New Orleans": (29.9490, -90.0821, -6, 2),
    "New York": (40.7505, -73.9934, -5, 10), "Newark": (40.7336, -74.1711, -5, 10),
    "Oakland": (37.7503, -122.2030, -8, 5), "Oklahoma City": (35.4634, -97.5151, -6, 370),
    "Orlando": (28.5392, -81.3839, -5, 30), "Paris": (48.8566, 2.3522, 1, 35),
    "Philadelphia": (39.9012, -75.1720, -5, 10), "Phoenix": (33.4457, -112.0712, -7, 330),
    "Portland": (45.5316, -122.6668, -8, 15), "Sacramento": (38.5802, -121.4997, -8, 10),
    "Salt Lake City": (40.7683, -111.9011, -7, 1288), "San Antonio": (29.4270, -98.4375, -6, 200),
    "San Francisco": (37.7680, -122.3877, -8, 5), "Seattle": (47.6221, -122.3540, -8, 50),
    "Tampa": (27.9427, -82.4518, -5, 5), "Toronto": (43.6435, -79.3791, -5, 80),
    "Washington": (38.8981, -77.0209, -5, 20),
}
# Where each club plays at home today - used only when a scheduled game carries no venue.
HOME_CITY = {
    "ATL": "Atlanta", "BOS": "Boston", "BKN": "Brooklyn", "CHA": "Charlotte", "CHI": "Chicago",
    "CLE": "Cleveland", "DAL": "Dallas", "DEN": "Denver", "DET": "Detroit",
    "GSW": "San Francisco", "HOU": "Houston", "IND": "Indianapolis", "LAC": "Inglewood",
    "LAL": "Los Angeles", "MEM": "Memphis", "MIA": "Miami", "MIL": "Milwaukee",
    "MIN": "Minneapolis", "NOP": "New Orleans", "NYK": "New York", "OKC": "Oklahoma City",
    "ORL": "Orlando", "PHI": "Philadelphia", "PHX": "Phoenix", "POR": "Portland",
    "SAC": "Sacramento", "SAS": "San Antonio", "TOR": "Toronto", "UTA": "Salt Lake City",
    "WAS": "Washington",
}


def city(name, home_team: str | None = None):
    """(lat, lon, utc_offset, altitude) of a venue city; the home club's city if none is given."""
    if isinstance(name, str) and name in CITIES:
        return CITIES[name]
    if home_team in HOME_CITY:
        return CITIES[HOME_CITY[home_team]]
    return None


def haversine_km(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))
