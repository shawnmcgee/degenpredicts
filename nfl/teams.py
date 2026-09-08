"""Single source of truth for NFL team codes and venues.

NFL abbreviations are a minefield and every feed disagrees. The same franchise appears as
OAK and LV, SD and LAC, STL and LA, WAS and WSH and WFT, JAC and JAX. Getting one of these
wrong does not raise - it silently splits one franchise's rating history in two, or merges two
teams, and the model quietly gets worse. So:

* :data:`TEAM_ALIASES` is the ONLY place a code is translated, and it is closed. Anything not
  in it raises :class:`UnknownTeam` rather than passing through.
* Relocations map to the franchise's CURRENT code (OAK -> LV). That is deliberate: it is the
  same franchise and its rating should carry across the move. Where the team physically played
  is a separate question, answered by the stadium of the game, not by the team code.
* :func:`test_teams.py` asserts every code in the committed schedule resolves, so a new or
  renamed franchise fails CI instead of corrupting a season of ratings.

External feeds (The Odds API, Kalshi) send full names, not codes. Those go through
:func:`build_matcher` in ``sources/odds.py``, which is deliberately lenient and logs misses -
a wrong guess there costs one game's price, not a franchise's history.
"""
from __future__ import annotations

import logging
import math

log = logging.getLogger(__name__)


class UnknownTeam(KeyError):
    """Raised for a team code that is not in the canonical map."""


# The 32 current franchises, in nflverse's spelling. This list IS the canon.
TEAMS: tuple[str, ...] = (
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
    "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
)

# Everything any feed has ever called a franchise -> its canonical code.
# Grouped by the reason the alias exists, because that is what makes them memorable.
TEAM_ALIASES: dict[str, str] = {
    # --- identity -----------------------------------------------------------------
    **{t: t for t in TEAMS},

    # --- relocations: the franchise keeps its rating across the move ---------------
    "OAK": "LV",     # Oakland Raiders -> Las Vegas, 2020
    "LVR": "LV",
    "RAI": "LV",
    "SD": "LAC",     # San Diego Chargers -> Los Angeles, 2017
    "SDG": "LAC",
    "STL": "LA",     # St. Louis Rams -> Los Angeles, 2016
    # nflverse spells St. Louis "STL" in the schedule and "SL" in the roster and snap-count
    # files. Both are the Rams. This is exactly the kind of split-brain spelling that makes
    # canon() raise instead of guess.
    "SL": "LA",
    "RAM": "LA",
    "LAR": "LA",     # most feeds outside nflverse spell the Rams LAR
    "PHO": "ARI",    # Phoenix Cardinals -> Arizona, 1994 (pre-window, kept for safety)
    "ARZ": "ARI",

    # --- renames ------------------------------------------------------------------
    "WSH": "WAS",    # ESPN
    "WFT": "WAS",    # Washington Football Team, 2020-21
    "WAS": "WAS",
    "JAC": "JAX",    # Jacksonville, pre-2013 nflverse and several feeds

    # --- pro-football-reference / SIS three-letter style --------------------------
    "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO", "SFO": "SF", "TAM": "TB",
    "NOS": "NO", "TBB": "TB", "GBP": "GB", "KCC": "KC",
    "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "ARI": "ARI",
    "CRD": "ARI", "RAV": "BAL", "OTI": "TEN", "HTX": "HOU",
}

# Franchises that changed code inside the data window, so a test can assert the
# relocation actually collapsed rather than merely being listed above.
RELOCATIONS: dict[str, str] = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def canon(code: str) -> str:
    """Map any spelling of a franchise onto its canonical code.

    Raises rather than guessing. A silently wrong team code is the single most damaging
    failure mode in this pipeline: it splits or merges rating histories, and nothing
    downstream can detect it.
    """
    if code is None:
        raise UnknownTeam("team code is None")
    key = str(code).strip().upper()
    try:
        return TEAM_ALIASES[key]
    except KeyError:
        raise UnknownTeam(
            f"unmapped NFL team code {code!r}. Add it to TEAM_ALIASES in nfl/teams.py - "
            "do not let it through, it will corrupt the ratings."
        ) from None


def is_known(code: str) -> bool:
    return str(code).strip().upper() in TEAM_ALIASES


DIVISIONS: dict[str, str] = {
    "BUF": "AFC East", "MIA": "AFC East", "NE": "AFC East", "NYJ": "AFC East",
    "BAL": "AFC North", "CIN": "AFC North", "CLE": "AFC North", "PIT": "AFC North",
    "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "TEN": "AFC South",
    "DEN": "AFC West", "KC": "AFC West", "LAC": "AFC West", "LV": "AFC West",
    "DAL": "NFC East", "NYG": "NFC East", "PHI": "NFC East", "WAS": "NFC East",
    "CHI": "NFC North", "DET": "NFC North", "GB": "NFC North", "MIN": "NFC North",
    "ATL": "NFC South", "CAR": "NFC South", "NO": "NFC South", "TB": "NFC South",
    "ARI": "NFC West", "LA": "NFC West", "SEA": "NFC West", "SF": "NFC West",
}


def conference(code: str) -> str:
    return DIVISIONS[canon(code)].split()[0]


# ---------------------------------------------------------------------------------
# Venues
# ---------------------------------------------------------------------------------
# Keyed on nflverse's ``stadium_id``, which is stable across a stadium's many sponsor
# renames (Heinz -> Acrisure is one id, not two). Coordinates are the playing surface;
# the timezone is what decides body-clock cost, which is the thing that actually moves
# a line for a west-to-east 1pm kickoff.
#
# An unmapped id is NOT fatal at runtime - it yields NaN travel, which the model handles.
# It IS fatal in CI, so a new stadium is noticed the week it appears rather than a season
# later. See tests/test_nfl.py::test_every_stadium_in_the_data_is_mapped.
STADIUMS: dict[str, tuple[float, float, str]] = {
    "ATL00": (33.7577, -84.4008, "America/New_York"),    # Georgia Dome
    "ATL97": (33.7554, -84.4009, "America/New_York"),    # Mercedes-Benz
    "BAL00": (39.2780, -76.6227, "America/New_York"),
    "BOS00": (42.0909, -71.2643, "America/New_York"),    # Gillette
    "BUF00": (42.7738, -78.7870, "America/New_York"),
    "BUF01": (43.6414, -79.3894, "America/Toronto"),     # Rogers Centre
    "CAR00": (35.2258, -80.8528, "America/New_York"),
    "CHI98": (41.8623, -87.6167, "America/Chicago"),
    "CIN00": (39.0955, -84.5161, "America/New_York"),
    "CLE00": (41.5061, -81.6995, "America/New_York"),
    "DAL00": (32.7473, -97.0945, "America/Chicago"),
    "DEN00": (39.7439, -105.0201, "America/Denver"),
    "DET00": (42.3400, -83.0456, "America/New_York"),
    "FRA00": (50.0685, 8.6455, "Europe/Berlin"),         # Deutsche Bank Park
    "GER00": (48.2188, 11.6247, "Europe/Berlin"),        # Allianz Arena
    "GNB00": (44.5013, -88.0622, "America/Chicago"),
    "HOU00": (29.6847, -95.4107, "America/Chicago"),
    "IND00": (39.7601, -86.1639, "America/Indiana/Indianapolis"),
    "JAX00": (30.3239, -81.6373, "America/New_York"),
    "KAN00": (39.0489, -94.4839, "America/Chicago"),
    "LAX01": (33.9535, -118.3392, "America/Los_Angeles"),   # SoFi
    "LAX97": (33.8644, -118.2611, "America/Los_Angeles"),   # Dignity Health
    "LAX99": (34.0141, -118.2879, "America/Los_Angeles"),   # LA Coliseum
    "LON00": (51.5560, -0.2795, "Europe/London"),           # Wembley
    "LON01": (51.4560, -0.3417, "Europe/London"),           # Twickenham
    "LON02": (51.6043, -0.0665, "Europe/London"),           # Tottenham
    "MAD01": (40.4530, -3.6883, "Europe/Madrid"),
    "MEL00": (-37.8200, 144.9834, "Australia/Melbourne"),
    "MEX00": (19.3029, -99.1505, "America/Mexico_City"),
    "MIA00": (25.9580, -80.2389, "America/New_York"),
    "MIN00": (44.9738, -93.2578, "America/Chicago"),        # Metrodome
    "MIN01": (44.9738, -93.2577, "America/Chicago"),        # U.S. Bank
    "MIN98": (44.9765, -93.2247, "America/Chicago"),        # TCF Bank
    "MUN01": (48.2188, 11.6247, "Europe/Berlin"),
    "NAS00": (36.1665, -86.7713, "America/Chicago"),
    "NOR00": (29.9511, -90.0812, "America/Chicago"),
    "NYC00": (40.8128, -74.0764, "America/New_York"),       # Giants Stadium, through 2009
    "NYC01": (40.8135, -74.0745, "America/New_York"),       # MetLife, 2010 on
    "OAK00": (37.7516, -122.2005, "America/Los_Angeles"),
    "PAR00": (48.9245, 2.3601, "Europe/Paris"),
    "PHI00": (39.9008, -75.1675, "America/New_York"),
    "PHO00": (33.5276, -112.2626, "America/Phoenix"),       # State Farm
    "PIT00": (40.4468, -80.0158, "America/New_York"),
    "RIO00": (-22.9121, -43.2302, "America/Sao_Paulo"),
    "SAO00": (-23.5453, -46.4742, "America/Sao_Paulo"),
    "SDG00": (32.7831, -117.1196, "America/Los_Angeles"),
    "SEA00": (47.5952, -122.3316, "America/Los_Angeles"),
    "SFO00": (37.7130, -122.3861, "America/Los_Angeles"),   # Candlestick
    "SFO01": (37.4030, -121.9698, "America/Los_Angeles"),   # Levi's
    "STL00": (38.6329, -90.1885, "America/Chicago"),
    "TAM00": (27.9759, -82.5033, "America/New_York"),
    "VEG00": (36.0909, -115.1833, "America/Los_Angeles"),
    "WAS00": (38.9076, -76.8645, "America/New_York"),
}

# Fallback origin for a franchise when the schedule has no home game to infer one from
# (week 1 of a season we hold no history for). Travel normally uses the stadium the team
# actually played its home games in that season, which is what makes relocations and
# London "home" games come out right without special-casing them.
DEFAULT_HOME_STADIUM: dict[str, str] = {
    "ARI": "PHO00", "ATL": "ATL97", "BAL": "BAL00", "BUF": "BUF00", "CAR": "CAR00",
    "CHI": "CHI98", "CIN": "CIN00", "CLE": "CLE00", "DAL": "DAL00", "DEN": "DEN00",
    "DET": "DET00", "GB": "GNB00", "HOU": "HOU00", "IND": "IND00", "JAX": "JAX00",
    "KC": "KAN00", "LA": "LAX01", "LAC": "LAX01", "LV": "VEG00", "MIA": "MIA00",
    "MIN": "MIN01", "NE": "BOS00", "NO": "NOR00", "NYG": "NYC01", "NYJ": "NYC01",
    "PHI": "PHI00", "PIT": "PIT00", "SEA": "SEA00", "SF": "SFO01", "TB": "TAM00",
    "TEN": "NAS00", "WAS": "WAS00",
}

EARTH_KM = 6371.0088


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km. Good enough - we need 'far' vs 'near', not navigation."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(h))


def venue(stadium_id: str | None) -> tuple[float, float, str] | None:
    if not stadium_id:
        return None
    return STADIUMS.get(str(stadium_id).strip().upper())


_TZ_OFFSET_HOURS: dict[str, float] = {
    # Standard-time UTC offsets. Using standard rather than local-on-the-day offsets keeps
    # the feature stable across the DST switch in November, which is a schedule artefact
    # rather than anything about the teams.
    "America/New_York": -5, "America/Chicago": -6, "America/Denver": -7,
    "America/Phoenix": -7, "America/Los_Angeles": -8,
    "America/Indiana/Indianapolis": -5, "America/Toronto": -5,
    "America/Mexico_City": -6, "America/Sao_Paulo": -3,
    "Europe/London": 0, "Europe/Berlin": 1, "Europe/Madrid": 1, "Europe/Paris": 1,
    "Australia/Melbourne": 10,
}


def tz_offset(tz: str | None) -> float:
    """Standard-time UTC offset in hours, or NaN for an unmapped zone."""
    if not tz:
        return float("nan")
    return _TZ_OFFSET_HOURS.get(tz, float("nan"))


def tz_shift(origin_tz: str | None, venue_tz: str | None) -> float:
    """Body-clock shift in hours, wrapped across the date line into [-12, +12].

    Plain subtraction is wrong for trans-Pacific trips: Los Angeles is UTC-8 and Melbourne is
    UTC+10, so a naive difference says a team crossed **18** time zones. Nobody's body clock
    moves 18 hours - it moves 6 the other way. Left unwrapped this produces a feature value no
    other game in the data comes close to, on exactly the handful of international games where
    the effect is supposed to matter most.
    """
    o, v = tz_offset(origin_tz), tz_offset(venue_tz)
    if o != o or v != v:
        return float("nan")
    return (v - o + 12) % 24 - 12
