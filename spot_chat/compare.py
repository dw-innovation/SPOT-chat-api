"""Compare a predicted query with a gold-standard query.

Entity ids may differ between the two, so entities are first matched by
the best assignment, then relations are compared under that assignment.
Text is compared case-insensitively with light normalisation, and distances
numerically ("50.50 m" == "50.5 m"). Units are NOT converted.

With SPOT's descriptor tables loaded, the main score (`exact`) compares at
bundle level, i.e. whether both sides lead to the same OSM query: "pub" and
"bar" are the same bundle, so are "brand name" and "name", "floors" and
"levels", "pale blue" and "light blue". A cuisine given as a property is
folded into the name ("restaurant" + cuisine=italian == "italian restaurant").
`exact_literal` additionally requires the literal entity names to match.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from dataclasses import asdict, dataclass

from .descriptors import Descriptors
from .schema import normalize_distance, normalize_spatial_term

# Fallback when no descriptor table is loaded: wordings the gold standard uses
# interchangeably for the same property.
PROPERTY_SYNONYMS = {
    "brand name": "brand",
    "color": "colour",
    "roof color": "roof colour",
    "building color": "building colour",
    "outside seating area": "outdoor seating",
    "outside seating": "outdoor seating",
    "housenumber": "house number",
    "building levels": "floors",
    "levels": "floors",
    "wifi": "wi-fi",
}


def norm_text(value) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[\u2010-\u2015_\-]", " ", text)        # dashes / underscores -> space
    text = re.sub(r"[\"'\u2018\u2019\u201c\u201d`*]", "", text)  # quotes, apostrophes, wildcards
    return " ".join(text.split()).strip(" .,;")


def norm_distance(value):
    try:
        number, unit = normalize_distance(value).split()
        return float(number), unit
    except (ValueError, AttributeError):
        return norm_text(value)


def norm_prop_value(name: str, value):
    if value is None:
        return None
    distance = norm_distance(value)
    if isinstance(distance, tuple):
        return distance
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return norm_text(value)


def canon_property(prop: dict, descriptors: Descriptors | None = None) -> tuple:
    name = norm_text(prop.get("name", ""))
    spec = descriptors.property(name) if descriptors else None
    key = ("bundle", spec.bundle) if spec else PROPERTY_SYNONYMS.get(name, name)
    value = prop.get("value")
    if spec and spec.kind == "flag":
        return key, None, None
    operator = (spec.operator if spec and spec.kind == "text" else None) \
        or prop.get("operator") or ("=" if value is not None else None)
    if spec and spec.is_colour and value is not None and descriptors.colour_bundles(value):
        return key, operator, ("colour", descriptors.colour_bundles(value))
    return key, operator, norm_prop_value(name, value)


def _is_cuisine(prop: dict, descriptors: Descriptors | None) -> bool:
    spec = descriptors.property(prop.get("name", "")) if descriptors else None
    cuisine = descriptors.property("cuisine") if descriptors else None
    return (spec is not None and spec == cuisine) or norm_text(prop.get("name", "")) == "cuisine"


def canon_entity(entity: dict, descriptors: Descriptors | None = None) -> dict:
    is_cluster = entity.get("type") == "cluster"
    props = entity.get("properties") or []
    name = norm_text(entity.get("name", ""))
    cuisines = [p for p in props if _is_cuisine(p, descriptors) and p.get("value")]
    if len(cuisines) == 1:                  # "restaurant" + cuisine=italian -> "italian restaurant"
        name = f"{norm_text(cuisines[0]['value'])} {name}"
        props = [p for p in props if p is not cuisines[0]]
    bundles = descriptors.entity_bundles(name) if descriptors else frozenset({name})
    return {
        "name": name,
        "bundles": bundles,
        "type": "cluster" if is_cluster else "nwr",
        "minPoints": int(entity["minPoints"]) if is_cluster and entity.get("minPoints") else None,
        "maxDistance": norm_distance(entity["maxDistance"]) if is_cluster and entity.get("maxDistance") else None,
        "properties": frozenset(canon_property(p, descriptors) for p in props),
    }


def canon_relation(rel: dict, id_map: dict) -> tuple | None:
    source, target = id_map.get(rel.get("source")), id_map.get(rel.get("target"))
    if source is None or target is None:
        return None
    if rel.get("type") == "contains":
        return ("contains", source, target, None, None)
    pair = tuple(sorted((source, target)))                 # distance is symmetric
    # Only the distance counts; the spatial term's word is informational (see schema.py).
    return ("distance", *pair, norm_distance(rel.get("value", "")), None)


def canon_area(area: dict | None) -> tuple:
    area = area or {}
    if area.get("type") == "area" and area.get("value"):
        return "area", norm_text(area["value"])
    return ("bbox", None)


@dataclass
class Comparison:
    area: bool
    entity_count: bool
    entity_names: bool          # literal names (after normalisation and cuisine folding)
    entity_bundles: bool        # names lead to the same SPOT bundle
    entity_types: bool          # nwr / cluster incl. minPoints and maxDistance
    properties: bool
    relation_structure: bool    # which pairs are related, and how (ignores values)
    relations: bool             # full relation incl. value and spatial term
    exact: bool                 # same OSM query: bundles, types, properties, relations, area
    exact_literal: bool         # exact + literal entity names
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _entity_score(pred: dict, gold: dict) -> int:
    score = 0
    if pred["name"] == gold["name"]:
        score += 5
    elif pred["bundles"] & gold["bundles"]:
        score += 4
    elif pred["name"] in gold["name"] or gold["name"] in pred["name"]:
        score += 2
    score += pred["type"] == gold["type"]
    score += pred["properties"] == gold["properties"]
    return score


def _best_assignment(pred: list[dict], gold: list[dict]) -> dict[int, int]:
    """Map predicted positions to gold positions maximising the match score."""
    if not pred or not gold:
        return {}
    small, large, swapped = (pred, gold, False) if len(pred) <= len(gold) else (gold, pred, True)
    if len(large) > 7:                                     # greedy fallback for huge queries
        perms = [tuple(range(len(small)))]
    else:
        perms = itertools.permutations(range(len(large)), len(small))
    best, best_score = None, -1
    for perm in perms:
        score = sum(
            _entity_score(*( (large[j], small[i]) if swapped else (small[i], large[j]) ))
            for i, j in enumerate(perm)
        )
        if score > best_score:
            best, best_score = perm, score
    pairs = {i: j for i, j in enumerate(best)}
    return {j: i for i, j in pairs.items()} if swapped else pairs


def compare(pred_query: dict, gold_query: dict, descriptors: Descriptors | None = None) -> Comparison:
    pred_entities = pred_query.get("entities") or []
    gold_entities = gold_query.get("entities") or []
    pred = [canon_entity(e, descriptors) for e in pred_entities]
    gold = [canon_entity(e, descriptors) for e in gold_entities]
    assignment = _best_assignment(pred, gold)              # pred position -> gold position
    pairs = [(pred[p], gold[g]) for p, g in assignment.items()]
    same_count = len(pred) == len(gold)
    notes = []

    names = same_count and all(p["name"] == g["name"] for p, g in pairs)
    bundles = same_count and all(bool(p["bundles"] & g["bundles"]) for p, g in pairs)
    types = same_count and all(
        (p["type"], p["minPoints"], p["maxDistance"]) == (g["type"], g["minPoints"], g["maxDistance"])
        for p, g in pairs
    )
    props = same_count and all(p["properties"] == g["properties"] for p, g in pairs)
    for p, g in pairs:
        if p["name"] != g["name"]:
            same = " (same bundle)" if p["bundles"] & g["bundles"] else ""
            notes.append(f"name: '{p['name']}' vs gold '{g['name']}'{same}")
        if p["properties"] != g["properties"]:
            notes.append(f"props of '{g['name']}': {sorted(map(str, p['properties']))} vs gold {sorted(map(str, g['properties']))}")

    # Relations: express both sides in gold positions.
    pred_map = {e.get("id"): assignment.get(i) for i, e in enumerate(pred_entities)}
    gold_map = {e.get("id"): i for i, e in enumerate(gold_entities)}
    pred_rels = {canon_relation(r, pred_map) for r in pred_query.get("relations") or []}
    gold_rels = {canon_relation(r, gold_map) for r in gold_query.get("relations") or []}
    relations = same_count and pred_rels == gold_rels
    structure = same_count and {r[:3] if r else None for r in pred_rels} == {r[:3] if r else None for r in gold_rels}
    if not relations:
        notes.append(f"relations: {sorted(map(str, pred_rels))} vs gold {sorted(map(str, gold_rels))}")

    area = canon_area(pred_query.get("area")) == canon_area(gold_query.get("area"))
    if not area:
        notes.append(f"area: {canon_area(pred_query.get('area'))} vs gold {canon_area(gold_query.get('area'))}")
    if not same_count:
        notes.append(f"{len(pred)} entities vs gold {len(gold)}")

    exact = all((area, same_count, bundles, types, props, relations))
    return Comparison(
        area=area, entity_count=same_count, entity_names=names, entity_bundles=bundles,
        entity_types=types, properties=props, relation_structure=structure, relations=relations,
        exact=exact, exact_literal=exact and names, notes=" | ".join(notes),
    )
