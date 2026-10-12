"""Constants and small normalisers that define the query format.

Everything the prompt tells the model about units, spatial terms and operators
is generated from the tables in this file, so the prompt and the code that
checks the model's output can never drift apart.
"""

from __future__ import annotations

import os
import re
import unicodedata

# Query size SPOT is trained for (transcription guidelines): at most 3 entities
# and at most 3 properties per entity. Larger queries may work but are untested.
MAX_ENTITIES = int(os.getenv("SPOT_MAX_ENTITIES", "3"))
MAX_PROPERTIES = int(os.getenv("SPOT_MAX_PROPERTIES", "3"))

# The relative spatial term ("next to") is used internally to pick the distance from the table
# below. Only the distance matters downstream, so the term is left out of the YAML unless
# SPOT_INCLUDE_SPATIAL_TERM=1.
INCLUDE_SPATIAL_TERM = os.getenv("SPOT_INCLUDE_SPATIAL_TERM", "0").strip().lower() in ("1", "true", "yes")

# --------------------------------------------------------------------------- #
# Distances
# --------------------------------------------------------------------------- #

UNIT_ALIASES = {
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m", "mtr": "m",
    "km": "km", "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
    "mi": "mi", "mile": "mi", "miles": "mi",
    "yd": "yd", "yds": "yd", "yard": "yd", "yards": "yd",
    "ft": "ft", "foot": "ft", "feet": "ft",
}
UNITS = ("m", "km", "mi", "yd", "ft")

_DISTANCE_RE = re.compile(r"^\s*(\d[\d.,]*)\s*([^\W\d_]+)\.?\s*$", re.UNICODE)


def normalize_number(raw: str) -> str:
    """Turn '1,200' into '1200' and '2,4' into '2.4' (the gold-standard convention).

    A comma followed by exactly three digits is a thousands separator,
    any other comma is a decimal point. The original precision is kept
    ('50.50' stays '50.50').
    """
    num = raw.strip()
    if "," in num and "." in num:          # 1,200.5
        return num.replace(",", "")
    if "," in num:
        parts = num.split(",")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            return "".join(parts)           # 1,200 / 85,800 / 1,234,567
        if len(parts) == 2:
            return f"{parts[0]}.{parts[1]}"  # 2,4 / 1,01 / 125,20
        raise ValueError(f"cannot read the number '{raw}'")
    if num.count(".") > 1:
        raise ValueError(f"cannot read the number '{raw}'")
    return num


def normalize_distance(raw) -> str:
    """'400 meters' -> '400 m', '1,5km' -> '1.5 km'. Raises ValueError if unreadable."""
    if isinstance(raw, (int, float)):
        raise ValueError(f"distance {raw} has no unit; use e.g. '{raw} m'")
    match = _DISTANCE_RE.match(str(raw))
    if not match:
        raise ValueError(
            f"'{raw}' is not a distance; use digits and a unit ({', '.join(UNITS)}), e.g. '400 m'"
        )
    number, unit = match.groups()
    unit = UNIT_ALIASES.get(unit.lower())
    if unit is None:
        raise ValueError(f"unknown unit in '{raw}'; use one of {', '.join(UNITS)}")
    return f"{normalize_number(number)} {unit}"


METRES = {"m": 1.0, "km": 1000.0, "mi": 1609.344, "yd": 0.9144, "ft": 0.3048}
_DISTANCE_IN_TEXT_RE = re.compile(
    r"(?<![\w.,])(\d[\d.,]*?)[.,]?\s*(" + "|".join(sorted(UNIT_ALIASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE)


def in_metres(distance: str) -> float:
    number, unit = normalize_distance(distance).split()
    return float(number) * METRES[unit]


def distances_in(text: str) -> list[str]:
    """The distances written in a text, normalised ("0.5km" -> "0.5 km")."""
    found = []
    for number, unit in _DISTANCE_IN_TEXT_RE.findall(str(text)):
        try:
            found.append(normalize_distance(f"{number} {unit}"))
        except ValueError:
            pass
    return found


def as_written(distance: str, text: str) -> str:
    """The user's own form of a distance: "500 m" becomes "0.5 km" if that is what the text
    says and "500 m" is not in it. Unit and precision of the user are part of the query."""
    written = distances_in(text)
    if distance in written:
        return distance
    try:
        metres = in_metres(distance)
    except (ValueError, KeyError):
        return distance
    return next((d for d in written if abs(in_metres(d) - metres) < 1e-6 * max(metres, 1.0)), distance)


def looks_like_distance(raw) -> bool:
    try:
        normalize_distance(raw)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Relative spatial terms ("next to", "near", ...): SPOT's official lookup table.
# The term is stored in the YAML together with its distance ("up to").
# --------------------------------------------------------------------------- #

SPATIAL_TERM_TABLE: dict[str, list[str]] = {
    "50 m": ["among", "beside", "next to", "colocated", "very near", "along", "besides"],
    # "side by side" is listed under both 50 m and 100 m in the source table;
    # 100 m is used here because the gold standard annotates it that way.
    "100 m": ["at", "close to", "close", "in close distance to", "across from", "not far away",
              "enclosed by", "around it", "side by side", "in front of", "behind", "right of", "left of"],
    "250 m": ["adjacent to", "near", "nearby", "down the street", "a block away", "opposite to",
              "opposite from"],
    "500 m": ["around", "in vicinity of", "in surroundings"],
    "900 m": ["far", "distant", "distance"],
    "1500 m": ["east of", "west of", "north of", "south of", "north-east of", "north-west of",
               "south-east of", "south-west of"],
}
SPATIAL_TERMS: dict[str, str] = {
    term: distance for distance, terms in SPATIAL_TERM_TABLE.items() for term in terms
}

# Other wordings that mean one of the official terms above.
SPATIAL_TERM_ALIASES: dict[str, str] = {
    "in the vicinity of": "in vicinity of",
    "in the vicinity": "in vicinity of",
    "in the surroundings of": "in surroundings",
    "in the surroundings": "in surroundings",
    "opposite": "opposite from",
    "opposite of": "opposite from",
    "across the street from": "across from",
    "not far from": "not far away",
    "not far": "not far away",
    "adjacent": "adjacent to",
    "in close distance": "in close distance to",
    "right next to": "next to",
    "co-located": "colocated",
    "co located": "colocated",
    "left to": "left of",
    "right to": "right of",
    "a block from": "a block away",
    "one block away": "a block away",
    "far from": "far",
    "far away": "far",
    "far away from": "far",
    "northeast of": "north-east of",
    "northwest of": "north-west of",
    "southeast of": "south-east of",
    "southwest of": "south-west of",
}


def normalize_spatial_term(raw: str) -> tuple[str, str | None]:
    """Return (term, default_distance). default_distance is None for unknown terms."""
    term = " ".join(str(raw).lower().split())
    for suffix in (" each other", " one another"):      # "next to each other" -> "next to"
        if term.endswith(suffix) and term[: -len(suffix)] in SPATIAL_TERMS:
            term = term[: -len(suffix)]
    term = SPATIAL_TERM_ALIASES.get(term, term)
    return term, SPATIAL_TERMS.get(term)


# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #

OPERATORS = ("=", "~", "<", ">")
OPERATOR_ALIASES = {
    "==": "=", "is": "=", "equals": "=",
    ">=": ">", "=>": ">", "≥": ">", "more than": ">", "greater than": ">",
    "<=": "<", "=<": "<", "≤": "<", "less than": "<",
    "~=": "~", "=~": "~", "contains": "~", "like": "~",
}

# Free-text properties always use the '~' (partial match) operator.
TEXT_PROPERTIES = {
    "name", "brand", "brand name", "house number", "housenumber",
    "building number", "street", "street name",
}

# OSM keys and location words that must not appear as property names. The
# second pipeline step does the tagging; locations belong in the area.
FORBIDDEN_PROPERTY_NAMES = {
    "amenity", "shop", "leisure", "tourism", "building", "highway", "waterway",
    "natural", "landuse", "man_made", "railway", "historic", "office", "craft",
    "public_transport", "aeroway", "boundary", "place",
}
LOCATION_PROPERTY_NAMES = {
    "city", "town", "village", "country", "state", "region", "district",
    "location", "area", "place name", "municipality",
}

DEFAULT_CLUSTER_DISTANCE = "50 m"


def clean_text(raw) -> str:
    """Collapse whitespace and apply Unicode NFC so names compare reliably."""
    return " ".join(unicodedata.normalize("NFC", str(raw)).split())
