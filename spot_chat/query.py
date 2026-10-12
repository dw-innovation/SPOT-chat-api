"""The search query: a plain dict with `area`, `entities` and `relations`.

All functions here are pure: they take plain Python data and either return
normalised data or raise `QueryError` with a message the LLM can act on.
"""

from __future__ import annotations

import copy

import yaml

from .descriptors import get_descriptors
from .schema import (
    DEFAULT_CLUSTER_DISTANCE,
    INCLUDE_SPATIAL_TERM,
    MAX_ENTITIES,
    MAX_PROPERTIES,
    FORBIDDEN_PROPERTY_NAMES,
    LOCATION_PROPERTY_NAMES,
    OPERATOR_ALIASES,
    OPERATORS,
    TEXT_PROPERTIES,
    clean_text,
    looks_like_distance,
    normalize_distance,
    normalize_spatial_term,
)


class QueryError(ValueError):
    """Invalid tool input. The message is sent back to the model verbatim."""


def _distance(raw, what: str) -> str:
    try:
        return normalize_distance(raw)
    except ValueError as exc:
        raise QueryError(f"{what}: {exc}") from None


def empty_query() -> dict:
    return {"area": {"type": "bbox"}, "entities": [], "relations": []}


def is_empty(query: dict) -> bool:
    return not query["entities"] and query["area"].get("type") == "bbox"


# --------------------------------------------------------------------------- #
# Area
# --------------------------------------------------------------------------- #

def normalize_area(raw) -> dict:
    if not isinstance(raw, dict):
        raise QueryError("area must be an object like {type: 'area', value: 'Cologne'}")
    area_type = str(raw.get("type") or "").strip().lower()
    value = clean_text(raw.get("value") or "")
    if area_type not in ("", "bbox", "area"):
        raise QueryError(f"unknown area type '{area_type}'; use 'bbox' or 'area'")
    if value:                                   # a place name always wins over 'bbox'
        return {"type": "area", "value": value}
    if area_type == "area":
        raise QueryError("area of type 'area' needs a value (the place name)")
    return {"type": "bbox"}


# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #

def normalize_property(raw) -> dict:
    if isinstance(raw, str):                      # tolerate "outdoor seating"
        raw = {"name": raw}
    if not isinstance(raw, dict) or not raw.get("name"):
        raise QueryError("every property needs a name, e.g. {name: 'outdoor seating'}")

    name = clean_text(raw["name"])
    key = name.lower()
    if "=" in name or ":" in name or key in FORBIDDEN_PROPERTY_NAMES:
        raise QueryError(
            f"property '{name}' looks like an OpenStreetMap tag. Do not tag: describe the "
            "object in the entity name (e.g. name 'bar') and use SPOT property names."
        )
    if key in LOCATION_PROPERTY_NAMES or key.startswith("addr"):
        raise QueryError(
            f"property '{name}' is a location. Put place names in the area "
            "(set_area / new_query area), not on entities."
        )

    value = raw.get("value")
    value = None if value is None or str(value).strip() == "" else clean_text(value)
    operator = raw.get("operator")
    operator = None if operator in (None, "") else str(operator).strip().lower()
    operator = OPERATOR_ALIASES.get(operator, operator)

    descriptors = get_descriptors()
    spec = descriptors.property(name) if descriptors else None
    if spec is not None:
        return _apply_property_spec(spec, name, operator, value)

    # Property not in SPOT's table (or no table loaded): generic rules.
    if value is None:
        if operator:
            raise QueryError(f"property '{name}' has operator '{operator}' but no value")
        return {"name": name}
    if key in TEXT_PROPERTIES:
        operator = "~"
        value = value.strip("*").strip()          # "starts with Zren*" -> "Zren"
    elif operator is None:
        operator = "="
    if operator not in OPERATORS:
        raise QueryError(f"operator '{operator}' is not allowed; use one of {', '.join(OPERATORS)}")
    if operator in ("<", ">") and not any(ch.isdigit() for ch in value):
        raise QueryError(f"operator '{operator}' needs a numeric value, got '{value}'")
    if key not in TEXT_PROPERTIES and looks_like_distance(value):
        value = normalize_distance(value)         # "20 meters" -> "20 m"
    return {"name": name, "operator": operator, "value": value}


def _apply_property_spec(spec, name: str, operator, value) -> dict:
    """Enforce the value type SPOT's tag table defines for this property."""
    if spec.kind == "flag":                       # fixed value, e.g. "outdoor seating"
        return {"name": name}
    if value is None:
        example = "'20 m' or '3'" if spec.kind == "numeric" else "'red'" if spec.is_colour else "'italian'"
        raise QueryError(f"property '{name}' needs a value, e.g. {example}")
    if spec.kind == "numeric":
        operator = "=" if operator in (None, "~") else operator
        if operator not in ("=", "<", ">"):
            raise QueryError(f"'{name}' takes the operators =, < or >, not '{operator}'")
        if not any(ch.isdigit() for ch in value):
            raise QueryError(f"'{name}' needs a number, got '{value}'")
        if looks_like_distance(value):
            value = normalize_distance(value)     # "20 meters" -> "20 m"
        return {"name": name, "operator": operator, "value": value}
    if spec.operator == "~":
        value = value.strip("*").strip()          # "starts with Zren*" -> "Zren"
    if spec.is_colour:
        value = value.lower()
    return {"name": name, "operator": spec.operator, "value": value}


def merge_properties(existing: list[dict], new: list[dict]) -> list[dict]:
    """Add properties, replacing any existing property with the same name."""
    merged = {p["name"].lower(): p for p in existing}
    for prop in new:
        merged[prop["name"].lower()] = prop
    return list(merged.values())


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #

def normalize_entity(raw, entity_id: int) -> dict:
    if not isinstance(raw, dict):
        raise QueryError("each entity must be an object with at least a name")
    name = clean_text(raw.get("name") or "")
    if not name:
        raise QueryError("every entity needs a name, e.g. 'restaurant'")
    if "=" in name or (":" in name and not name.lower().startswith("brand:")):
        raise QueryError(
            f"entity name '{name}' looks like an OpenStreetMap tag. Use the user's words "
            "in singular, e.g. 'bar' or 'train station' (or 'brand:<Brand>' for a bare brand)."
        )

    min_points = raw.get("minPoints")
    entity_type = str(raw.get("type") or "").strip().lower()
    if not entity_type:
        entity_type = "cluster" if min_points else "nwr"
    if entity_type not in ("nwr", "cluster"):
        raise QueryError(f"entity type '{entity_type}' is not allowed; use 'nwr' or 'cluster'")

    entity = {"id": entity_id, "name": name, "type": entity_type}
    if entity_type == "cluster":
        try:
            min_points = int(min_points)
        except (TypeError, ValueError):
            raise QueryError(f"cluster '{name}' needs minPoints (how many), e.g. 3") from None
        if min_points < 2:
            raise QueryError(f"cluster '{name}' needs minPoints >= 2; use type 'nwr' for one object")
        entity["minPoints"] = min_points
        if raw.get("maxDistance"):
            entity["maxDistance"] = _cluster_distance(raw["maxDistance"], name)

    props = raw.get("properties") or []
    if not isinstance(props, list):
        raise QueryError(f"properties of '{name}' must be a list")
    if props:
        entity["properties"] = merge_properties([], [normalize_property(p) for p in props])
    return entity


def _cluster_distance(raw, name: str) -> str:
    """A distance, or a relative spatial term ("three benches next to each other" -> 50 m)."""
    try:
        return normalize_distance(raw)
    except ValueError:
        term, default = normalize_spatial_term(raw)
        if default:
            return default
        raise QueryError(
            f"maxDistance of '{name}' must be a distance like '100 m' or a spatial term like "
            f"'next to', got '{raw}'"
        ) from None


def get_entity(query: dict, entity_id) -> dict:
    for entity in query["entities"]:
        if entity["id"] == entity_id:
            return entity
    ids = ", ".join(str(e["id"]) for e in query["entities"]) or "none"
    raise QueryError(f"there is no entity with id {entity_id} (existing ids: {ids})")


# --------------------------------------------------------------------------- #
# Relations
# --------------------------------------------------------------------------- #

def normalize_relation(raw, entity_ids: set[int]) -> dict:
    if not isinstance(raw, dict):
        raise QueryError("each relation must be an object with source, target and type")
    try:
        source, target = int(raw["source"]), int(raw["target"])
    except (KeyError, TypeError, ValueError):
        raise QueryError("a relation needs integer 'source' and 'target' entity ids") from None
    for eid in (source, target):
        if eid not in entity_ids:
            raise QueryError(f"relation refers to entity {eid}, which does not exist")
    if source == target:
        raise QueryError("a relation must connect two different entities")

    rel_type = str(raw.get("type") or "").strip().lower()
    if not rel_type:
        rel_type = "distance" if raw.get("value") or raw.get("spatial_term") else ""
    if rel_type == "contains":
        return {"source": source, "target": target, "type": "contains"}
    if rel_type != "distance":
        raise QueryError(f"relation type '{rel_type}' is not allowed; use 'distance' or 'contains'")

    relation = {"source": source, "target": target, "type": "distance"}
    term, default = (None, None)
    if raw.get("spatial_term"):
        term, default = normalize_spatial_term(raw["spatial_term"])
    if default:                                   # a table term always has its table distance
        relation["value"] = default
    elif raw.get("value"):
        relation["value"] = _distance(raw["value"], "relation value")
    elif term:
        raise QueryError(
            f"'{term}' is not a known spatial term, so give a value (e.g. '100 m') as well"
        )
    else:
        raise QueryError("a distance relation needs a value like '400 m' or a spatial_term like 'next to'")
    if term:
        relation["spatial_term"] = term
    return relation


def find_relation(query: dict, a: int, b: int) -> int:
    """Index of the relation between a and b (either direction)."""
    for i, rel in enumerate(query["relations"]):
        if {rel["source"], rel["target"]} == {a, b}:
            return i
    raise QueryError(f"there is no relation between entity {a} and entity {b}")


def check_new_relation(query: dict, relation: dict) -> None:
    pair = {relation["source"], relation["target"]}
    for rel in query["relations"]:
        if {rel["source"], rel["target"]} == pair:
            raise QueryError(
                f"entities {relation['source']} and {relation['target']} are already related; "
                "use update_relation to change that relation"
            )


# --------------------------------------------------------------------------- #
# Whole-query helpers
# --------------------------------------------------------------------------- #

def fill_cluster_distances(query: dict) -> list[str]:
    """Give clusters without maxDistance the distance of their relation, else 50 m."""
    notes = []
    for entity in query["entities"]:
        if entity["type"] != "cluster" or entity.get("maxDistance"):
            continue
        distance = next(
            (r["value"] for r in query["relations"]
             if r["type"] == "distance" and entity["id"] in (r["source"], r["target"])),
            DEFAULT_CLUSTER_DISTANCE,
        )
        entity["maxDistance"] = distance
        notes.append(f"cluster {entity['id']} ({entity['name']}) got maxDistance {distance}")
    return notes


def renumber(query: dict) -> dict[int, int]:
    """Make entity ids 0..n-1 again and update relations. Returns {old: new} for changed ids."""
    mapping = {e["id"]: i for i, e in enumerate(query["entities"])}
    for entity in query["entities"]:
        entity["id"] = mapping[entity["id"]]
    for rel in query["relations"]:
        rel["source"], rel["target"] = mapping[rel["source"]], mapping[rel["target"]]
    return {old: new for old, new in mapping.items() if old != new}


def warnings_for(query: dict) -> list[str]:
    """Problems that are allowed mid-edit but should be fixed before the turn ends."""
    warnings = []
    if len(query["entities"]) > MAX_ENTITIES:
        warnings.append(
            f"the query has {len(query['entities'])} entities, but SPOT supports at most "
            f"{MAX_ENTITIES}. Tell the user and ask which objects to keep."
        )
    for entity in query["entities"]:
        count = len(entity.get("properties", []))
        if count > MAX_PROPERTIES:
            warnings.append(
                f"entity {entity['id']} ({entity['name']}) has {count} properties, but SPOT supports "
                f"at most {MAX_PROPERTIES} per object. Tell the user and ask which to keep."
            )
    if query["relations"]:
        linked = {r["source"] for r in query["relations"]} | {r["target"] for r in query["relations"]}
        loose = [f"{e['id']} ({e['name']})" for e in query["entities"] if e["id"] not in linked]
        if loose:
            warnings.append(
                "entities " + ", ".join(loose) + " are not part of any relation. If the user "
                "related them to something, add that relation."
            )
    return warnings


def build_query(area, entities, relations) -> dict:
    """Validate a complete query given as plain data (used by new_query and manual edits).

    Entities may carry their own ids (any numbering); they are renumbered to
    0..n-1 in list order and relations are remapped accordingly.
    """
    if not isinstance(entities, list) or not entities:
        raise QueryError("a query needs at least one entity")
    relations = relations or []
    if not isinstance(relations, list):
        raise QueryError("relations must be a list")

    given_ids = [e.get("id") if isinstance(e, dict) else None for e in entities]
    use_given = all(isinstance(i, int) for i in given_ids) and len(set(given_ids)) == len(given_ids)
    id_map = {gid: pos for pos, gid in enumerate(given_ids)} if use_given else {}

    query = {
        "area": normalize_area(area if area is not None else {"type": "bbox"}),
        "entities": [normalize_entity(e, i) for i, e in enumerate(entities)],
        "relations": [],
    }
    ids = set(range(len(entities)))
    for raw in relations:
        if isinstance(raw, dict) and id_map:
            raw = dict(raw)
            for key in ("source", "target"):
                if raw.get(key) in id_map:
                    raw[key] = id_map[raw[key]]
                elif raw.get(key) is not None:
                    raise QueryError(f"relation {key} {raw[key]} does not match any entity id")
        relation = normalize_relation(raw, ids)
        check_new_relation(query, relation)
        query["relations"].append(relation)
    fill_cluster_distances(query)
    return query


# --------------------------------------------------------------------------- #
# YAML
# --------------------------------------------------------------------------- #

_ENTITY_KEY_ORDER = ("id", "name", "type", "minPoints", "maxDistance", "properties")
_RELATION_KEY_ORDER = ("source", "target", "type", "value") + (
    ("spatial_term",) if INCLUDE_SPATIAL_TERM else ())


def to_export(query: dict) -> dict:
    """Ordered copy in the gold-standard layout (no empty relations list). The spatial term
    stays internal unless SPOT_INCLUDE_SPATIAL_TERM is set; the distance is always exported."""
    out = {"area": dict(query["area"]), "entities": []}
    for entity in query["entities"]:
        out["entities"].append({k: copy.deepcopy(entity[k]) for k in _ENTITY_KEY_ORDER if k in entity})
    if query["relations"]:
        out["relations"] = [
            {k: rel[k] for k in _RELATION_KEY_ORDER if k in rel} for rel in query["relations"]
        ]
    return out


def to_yaml(query: dict) -> str:
    if not query["entities"] and query["area"]["type"] == "bbox":
        return "# No query yet. Describe what you are looking for in the chat.\n"
    return yaml.safe_dump(
        to_export(query), sort_keys=False, allow_unicode=True, default_flow_style=False, width=1000
    )


def from_yaml(text: str) -> dict:
    """Parse and validate YAML typed by the user (or taken from the gold standard)."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise QueryError(f"this is not valid YAML: {exc}") from None
    if data is None:
        return empty_query()
    if not isinstance(data, dict):
        raise QueryError("the YAML must be a mapping with area, entities and relations")
    if not data.get("entities"):
        if data.get("relations"):
            raise QueryError("relations need entities")
        return {"area": normalize_area(data.get("area") or {}), "entities": [], "relations": []}
    relations = [
        {k: v for k, v in r.items() if k != "id"} for r in (data.get("relations") or [])
        if isinstance(r, dict)
    ]
    return build_query(data.get("area"), data.get("entities") or [], relations)
