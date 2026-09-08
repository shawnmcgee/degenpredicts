"""Single source of truth for club names.

Football club naming is worse than NFL abbreviations, not better. The same club is "Man
United", "Manchester United", "Man Utd" and "MUN" depending on the feed; "Nott'm Forest"
carries an apostrophe that half the sources drop; and - the one that actually bites - there
are *pairs of different clubs* whose short forms collide. "Sheffield" is two clubs. So does
"Manchester", "Bristol", "Nottingham" in the lower divisions. Guessing here does not raise; it
silently merges two clubs' rating histories, and nothing downstream can detect it.

So the rules match ``nfl/teams.py``:

* :data:`CLUB_ALIASES` is the ONLY place a name is translated, and it is closed. Anything not
  in it raises :class:`UnknownClub` rather than passing through.
* Canonical spelling is **football-data.co.uk's**, because that feed is the spine of this
  pipeline the way nflverse is the spine of the NFL one.
* A club keeps one identity across renames and relocations, so its rating carries.
* ``tests/test_epl.py`` asserts every name in the committed data resolves, so a newly promoted
  club fails CI instead of quietly entering the league as an unrated stranger.

External price feeds (The Odds API, Kalshi) send long-form names. Those go through
:func:`build_matcher` in ``sources/odds.py``, which is deliberately lenient and logs misses - a
wrong guess there costs one match's price, not a club's history.
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata

log = logging.getLogger(__name__)


class UnknownClub(KeyError):
    """Raised for a club name that is not in the canonical map."""


# Every club to have played in the Premier League, in football-data.co.uk's spelling.
# This tuple IS the canon for E0. Lower divisions add to CLUB_ALIASES below.
PREMIER_LEAGUE_CLUBS: tuple[str, ...] = (
    "Arsenal", "Aston Villa", "Barnsley", "Birmingham", "Blackburn", "Blackpool", "Bolton",
    "Bournemouth", "Bradford", "Brentford", "Brighton", "Burnley", "Cardiff", "Charlton",
    "Chelsea", "Coventry", "Crystal Palace", "Derby", "Everton", "Fulham", "Huddersfield",
    "Hull", "Ipswich", "Leeds", "Leicester", "Liverpool", "Luton", "Man City", "Man United",
    "Middlesbrough", "Newcastle", "Norwich", "Nott'm Forest", "Oldham", "Portsmouth", "QPR",
    "Reading", "Sheffield United", "Sheffield Weds", "Southampton", "Stoke", "Sunderland",
    "Swansea", "Swindon", "Tottenham", "Watford", "West Brom", "West Ham", "Wigan",
    "Wimbledon", "Wolves",
)

# Clubs that appear in the divisions below, so pointing DEGEN_EPL_LEAGUE at E1/E2 works
# without a second alias table. Not exhaustive for the whole pyramid - it does not need to be,
# because canon() raises and the test suite tells you exactly which name to add.
FOOTBALL_LEAGUE_CLUBS: tuple[str, ...] = (
    "Accrington", "Barrow", "Blackpool", "Bristol City", "Bristol Rvs", "Burton", "Bury",
    "Cambridge", "Carlisle", "Cheltenham", "Chesterfield", "Colchester", "Crewe", "Doncaster",
    "Exeter", "Fleetwood Town", "Forest Green", "Gillingham", "Grimsby", "Harrogate",
    "Hartlepool", "Leyton Orient", "Lincoln", "Mansfield", "Milton Keynes Dons", "Morecambe",
    "Newport County", "Northampton", "Notts County", "Oxford", "Peterboro", "Plymouth",
    "Port Vale", "Preston", "Rochdale", "Rotherham", "Salford", "Scunthorpe", "Shrewsbury",
    "Southend", "Stevenage", "Stockport", "Sutton", "Swindon", "Tranmere", "Walsall",
    "Wycombe", "Yeovil", "Millwall", "Blackburn", "Wrexham", "Bromley",
)

CLUBS: tuple[str, ...] = tuple(dict.fromkeys(PREMIER_LEAGUE_CLUBS + FOOTBALL_LEAGUE_CLUBS))


def _key(name: str) -> str:
    """Normalise for lookup: strip accents and punctuation, collapse whitespace, lowercase.

    This is what lets "Nott'm Forest", "Nottm Forest" and "nott m forest" all be the same key
    without three entries, while keeping the *canonical* spelling apostrophe and all.
    """
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    n = re.sub(r"[^a-z0-9 ]+", " ", n.lower())
    return re.sub(r"\s+", " ", n).strip()


# Everything any feed has ever called a club -> its canonical football-data spelling.
# Grouped by why the alias exists, because that is what makes them memorable.
_ALIAS_SOURCE: dict[str, str] = {
    # --- identity -----------------------------------------------------------------
    **{c: c for c in CLUBS},

    # --- long forms, as every price feed and the Odds API spell them ---------------
    "Manchester United": "Man United", "Manchester Utd": "Man United", "Man Utd": "Man United",
    "Manchester City": "Man City", "Man. City": "Man City",
    "Tottenham Hotspur": "Tottenham", "Spurs": "Tottenham",
    "Newcastle United": "Newcastle", "Newcastle Utd": "Newcastle",
    "Wolverhampton Wanderers": "Wolves", "Wolverhampton": "Wolves",
    "West Ham United": "West Ham", "West Ham Utd": "West Ham",
    "West Bromwich Albion": "West Brom", "West Bromwich": "West Brom", "WBA": "West Brom",
    "Brighton and Hove Albion": "Brighton", "Brighton & Hove Albion": "Brighton",
    "Brighton Hove Albion": "Brighton",
    "Leicester City": "Leicester", "Norwich City": "Norwich", "Swansea City": "Swansea",
    "Cardiff City": "Cardiff", "Stoke City": "Stoke", "Hull City": "Hull",
    "Birmingham City": "Birmingham", "Coventry City": "Coventry", "Bradford City": "Bradford",
    "Leeds United": "Leeds", "Leeds Utd": "Leeds",
    "Sheffield Utd": "Sheffield United", "Sheff United": "Sheffield United",
    "Sheff Utd": "Sheffield United",
    "Sheffield Wednesday": "Sheffield Weds", "Sheff Wed": "Sheffield Weds",
    "Sheff Weds": "Sheffield Weds", "Sheffield Wed": "Sheffield Weds",
    "Nottingham Forest": "Nott'm Forest", "Nottm Forest": "Nott'm Forest",
    "Notts Forest": "Nott'm Forest", "Forest": "Nott'm Forest",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Crystal Palace FC": "Crystal Palace", "Palace": "Crystal Palace",
    "Luton Town": "Luton", "Ipswich Town": "Ipswich", "Huddersfield Town": "Huddersfield",
    "Burnley FC": "Burnley", "Watford FC": "Watford", "Fulham FC": "Fulham",
    "Blackburn Rovers": "Blackburn", "Bolton Wanderers": "Bolton",
    "Blackpool FC": "Blackpool", "Charlton Athletic": "Charlton",
    "Wigan Athletic": "Wigan", "Oldham Athletic": "Oldham",
    "Queens Park Rangers": "QPR", "Queen's Park Rangers": "QPR",
    "Derby County": "Derby", "Reading FC": "Reading", "Barnsley FC": "Barnsley",
    "Portsmouth FC": "Portsmouth", "Southampton FC": "Southampton",
    "Sunderland AFC": "Sunderland", "Middlesbrough FC": "Middlesbrough",
    "Swindon Town": "Swindon", "Wimbledon FC": "Wimbledon", "AFC Wimbledon": "Wimbledon",
    "Aston Villa FC": "Aston Villa", "Villa": "Aston Villa",
    "Arsenal FC": "Arsenal", "Chelsea FC": "Chelsea", "Everton FC": "Everton",
    "Liverpool FC": "Liverpool",

    # --- lower-division long forms ------------------------------------------------
    "Bristol Rovers": "Bristol Rvs", "Bristol Rov": "Bristol Rvs",
    "Peterborough": "Peterboro", "Peterborough United": "Peterboro",
    "MK Dons": "Milton Keynes Dons", "Milton Keynes": "Milton Keynes Dons",
    "Newport": "Newport County", "Notts Co": "Notts County",
    "Fleetwood": "Fleetwood Town", "Oxford United": "Oxford", "Oxford Utd": "Oxford",
    "Plymouth Argyle": "Plymouth", "Preston North End": "Preston",
    "Rotherham United": "Rotherham", "Shrewsbury Town": "Shrewsbury",
    "Stockport County": "Stockport", "Wrexham AFC": "Wrexham",
    "Accrington Stanley": "Accrington", "Cambridge United": "Cambridge",
    "Carlisle United": "Carlisle", "Colchester United": "Colchester",
    "Crewe Alexandra": "Crewe", "Doncaster Rovers": "Doncaster", "Exeter City": "Exeter",
    "Grimsby Town": "Grimsby", "Lincoln City": "Lincoln", "Mansfield Town": "Mansfield",
    "Northampton Town": "Northampton", "Salford City": "Salford",
    "Scunthorpe United": "Scunthorpe", "Southend United": "Southend",
    "Walsall FC": "Walsall", "Wycombe Wanderers": "Wycombe", "Yeovil Town": "Yeovil",
    "Millwall FC": "Millwall", "Burton Albion": "Burton", "Cheltenham Town": "Cheltenham",
    "Harrogate Town": "Harrogate", "Hartlepool United": "Hartlepool",
    "Leyton Orient FC": "Leyton Orient", "Sutton United": "Sutton",
    "Tranmere Rovers": "Tranmere", "Bromley FC": "Bromley",

    # --- three-letter codes some feeds emit ---------------------------------------
    "ARS": "Arsenal", "AVL": "Aston Villa", "BOU": "Bournemouth", "BRE": "Brentford",
    "BHA": "Brighton", "BUR": "Burnley", "CHE": "Chelsea", "CRY": "Crystal Palace",
    "EVE": "Everton", "FUL": "Fulham", "IPS": "Ipswich", "LEE": "Leeds", "LEI": "Leicester",
    "LIV": "Liverpool", "LUT": "Luton", "MCI": "Man City", "MUN": "Man United",
    "NEW": "Newcastle", "NFO": "Nott'm Forest", "SHU": "Sheffield United",
    "SOU": "Southampton", "TOT": "Tottenham", "WHU": "West Ham", "WOL": "Wolves",
    "NOR": "Norwich", "WAT": "Watford", "SUN": "Sunderland",
}

# Built once, keyed on the normalised form, so lookups are punctuation- and case-insensitive
# while the values stay in football-data's exact spelling.
CLUB_ALIASES: dict[str, str] = {_key(k): v for k, v in _ALIAS_SOURCE.items()}

# Short forms that are genuinely ambiguous and must NOT be resolved. Each of these names more
# than one club in English football, so accepting it would merge two rating histories - the
# exact failure this module exists to prevent. They raise with a message naming the options.
AMBIGUOUS: dict[str, tuple[str, ...]] = {
    "sheffield": ("Sheffield United", "Sheffield Weds"),
    "sheff": ("Sheffield United", "Sheffield Weds"),
    "bristol": ("Bristol City", "Bristol Rvs"),
    "manchester": ("Man City", "Man United"),
    "united": ("Man United", "Sheffield United", "Leeds", "West Ham", "Newcastle"),
    "city": ("Man City", "Leicester", "Norwich", "Bristol City", "Hull"),
    "rovers": ("Blackburn", "Bristol Rvs", "Doncaster", "Tranmere"),
    "wanderers": ("Wolves", "Bolton", "Wycombe"),
    "town": ("Ipswich", "Luton", "Huddersfield", "Swindon", "Northampton"),
    "athletic": ("Charlton", "Wigan", "Oldham"),
    "county": ("Derby", "Notts County", "Newport County", "Stockport"),
}


def canon(name: str) -> str:
    """Map any spelling of a club onto its canonical football-data name.

    Raises rather than guessing. A silently wrong club name is the most damaging failure mode
    in this pipeline: it splits or merges rating histories and nothing downstream can detect it.
    """
    if name is None:
        raise UnknownClub("club name is None")
    k = _key(name)
    if not k:
        raise UnknownClub("club name is blank")
    if k in CLUB_ALIASES:
        return CLUB_ALIASES[k]
    if k in AMBIGUOUS:
        raise UnknownClub(
            f"ambiguous club name {name!r} - could be any of {', '.join(AMBIGUOUS[k])}. "
            "Resolve it upstream; do not let it through, it will merge rating histories."
        )
    raise UnknownClub(
        f"unmapped club name {name!r}. Add it to _ALIAS_SOURCE in epl/teams.py - "
        "do not let it through, it will corrupt the ratings."
    )


def is_known(name: str) -> bool:
    return _key(name) in CLUB_ALIASES


# ---------------------------------------------------------------------------------
# Grounds
# ---------------------------------------------------------------------------------
# Coordinates of each club's home ground, used for travel distance. English football has no
# time zones to cross, so unlike the NFL there is no body-clock term - what remains is the
# genuine north-south haul (Newcastle to Bournemouth is 500 km) and, more usefully, the
# derby/local-rivalry signal that falls out of a very short distance.
#
# An unmapped club is NOT fatal at runtime - it yields NaN travel, which the trees handle.
# It IS fatal in CI, so a promoted club is noticed the week it appears.
GROUNDS: dict[str, tuple[float, float]] = {
    "Arsenal": (51.5549, -0.1084), "Aston Villa": (52.5092, -1.8848),
    "Barnsley": (53.5522, -1.4668), "Birmingham": (52.4757, -1.8681),
    "Blackburn": (53.7286, -2.4892), "Blackpool": (53.8046, -3.0480),
    "Bolton": (53.5805, -2.5357), "Bournemouth": (50.7352, -1.8384),
    "Bradford": (53.8046, -1.7594), "Brentford": (51.4907, -0.2887),
    "Brighton": (50.8616, -0.0837), "Burnley": (53.7890, -2.2303),
    "Cardiff": (51.4728, -3.2031), "Charlton": (51.4865, 0.0364),
    "Chelsea": (51.4816, -0.1910), "Coventry": (52.4480, -1.4955),
    "Crystal Palace": (51.3983, -0.0855), "Derby": (52.9150, -1.4472),
    "Everton": (53.4388, -2.9664), "Fulham": (51.4749, -0.2217),
    "Huddersfield": (53.6540, -1.7683), "Hull": (53.7460, -0.3676),
    "Ipswich": (52.0550, 1.1449), "Leeds": (53.7778, -1.5722),
    "Leicester": (52.6203, -1.1422), "Liverpool": (53.4308, -2.9608),
    "Luton": (51.8842, -0.4317), "Man City": (53.4831, -2.2004),
    "Man United": (53.4631, -2.2913), "Middlesbrough": (54.5782, -1.2170),
    "Newcastle": (54.9756, -1.6216), "Norwich": (52.6220, 1.3092),
    "Nott'm Forest": (52.9400, -1.1328), "Oldham": (53.5551, -2.1284),
    "Portsmouth": (50.7963, -1.0637), "QPR": (51.5093, -0.2321),
    "Reading": (51.4223, -0.9827), "Sheffield United": (53.3703, -1.4709),
    "Sheffield Weds": (53.4114, -1.5006), "Southampton": (50.9058, -1.3911),
    "Stoke": (52.9884, -2.1757), "Sunderland": (54.9145, -1.3882),
    "Swansea": (51.6433, -3.9351), "Swindon": (51.5645, -1.7708),
    "Tottenham": (51.6043, -0.0665), "Watford": (51.6499, -0.4015),
    "West Brom": (52.5093, -1.9639), "West Ham": (51.5387, -0.0166),
    "Wigan": (53.5478, -2.6540), "Wimbledon": (51.4315, -0.1875),
    "Wolves": (52.5903, -2.1303),
    # lower divisions
    "Millwall": (51.4859, -0.0507), "Preston": (53.7728, -2.6880),
    "Bristol City": (51.4400, -2.6206), "Bristol Rvs": (51.4863, -2.5833),
    "Plymouth": (50.3881, -4.1508), "Rotherham": (53.4280, -1.3620),
    "Stockport": (53.4054, -2.1620), "Wrexham": (53.0511, -2.9950),
    "Oxford": (51.7161, -1.2080), "Peterboro": (52.5648, -0.2400),
}

EARTH_KM = 6371.0088


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km. Good enough - we need 'far' vs 'derby', not navigation."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(h))


def ground(club: str) -> tuple[float, float] | None:
    return GROUNDS.get(str(club).strip())


DERBY_KM = 20.0     # inside this, the two clubs share a city and the fixture is a derby


def travel_km(home: str, away: str) -> float:
    """How far the away side travelled. NaN if either ground is unmapped."""
    h, a = ground(home), ground(away)
    if h is None or a is None:
        return float("nan")
    return round(haversine_km(h, a), 1)


def is_derby(home: str, away: str) -> int:
    d = travel_km(home, away)
    return int(d == d and d <= DERBY_KM)
