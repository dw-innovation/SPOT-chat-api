"""Tools the LLM can call to build and edit the query.

Each tool is a plain function `fn(query, **args) -> message` that edits a
*copy* of the query in place. `execute_tool` takes care of copying, argument
checking and error handling, so a failed call never leaves a half-edited
query behind.
"""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import dataclass, field
from typing import Callable

from .query import (
    QueryError,
    build_query,
    check_new_relation,
    fill_cluster_distances,
    find_relation,
    get_entity,
    merge_properties,
    normalize_area,
    normalize_entity,
    normalize_property,
    normalize_relation,
    renumber,
    to_yaml,
    warnings_for,
)
from .descriptors import get_descriptors, norm, notes_for
from .policy import LIMITS, REFUSAL_CATEGORIES, Refusal, code_limits, limitation, refusal_for
from .schema import OPERATORS, UNITS, as_written

# --------------------------------------------------------------------------- #
# Shared JSON schemas
# --------------------------------------------------------------------------- #

AREA_SCHEMA = {
    "type": "object",
    "description": "Where to search.",
    "properties": {
        "type": {
            "type": "string", "enum": ["bbox", "area"],
            "description": "'area' if the user names a place, otherwise 'bbox' (the visible map).",
        },
        "value": {
            "type": "string",
            "description": "Only for type 'area': the place name as the user wrote it, "
                           "e.g. 'Perugia, Italy'. Keep non-Latin scripts as they are.",
        },
    },
    "required": ["type"],
    "additionalProperties": False,
}

PROPERTY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Feature in the user's words, e.g. 'outdoor seating', 'name', 'brand', "
                           "'height', 'roof colour', 'cuisine'. Never an OpenStreetMap key.",
        },
        "operator": {
            "type": "string", "enum": list(OPERATORS),
            "description": "Only together with a value. '~' for name, brand, house number, street "
                           "name; '<' or '>' for less/more than; '=' for everything else.",
        },
        "value": {
            "type": "string",
            "description": "Only if the feature has a value, e.g. 'red', 'glass', '20 m', 'Esso'. "
                           "Leave out for plain features like 'outdoor seating'.",
        },
    },
    "required": ["name"],
    "additionalProperties": False,
}


def _entity_schema(with_id: bool) -> dict:
    props = {
        "name": {
            "type": "string",
            "description": "The object in singular, in the user's words (typos fixed), e.g. 'bar', "
                           "'train station'. For a bare brand without object type: "
                           "'brand:<Brand>', e.g. 'brand:McDonald's'.",
        },
        "type": {
            "type": "string", "enum": ["nwr", "cluster"],
            "description": "'nwr' for one object; 'cluster' when the user asks for a number "
                           "(2 or more) of this object close together.",
        },
        "minPoints": {"type": "integer", "description": "Cluster only: how many (>= 2)."},
        "maxDistance": {
            "type": "string",
            "description": f"Cluster only: max distance between them, number + unit "
                           f"({', '.join(UNITS)}), e.g. '300 m'.",
        },
        "properties": {"type": "array", "items": PROPERTY_SCHEMA},
    }
    required = ["name", "type"]
    if with_id:
        props = {"id": {"type": "integer", "description": "0, 1, 2, ... in list order."}, **props}
        required = ["id", *required]
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


RELATION_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "integer", "description": "Entity id. For 'contains': the container."},
        "target": {"type": "integer", "description": "Entity id. For 'contains': the contained object."},
        "type": {"type": "string", "enum": ["distance", "contains"]},
        "value": {
            "type": "string",
            "description": f"Distance only: number + unit ({', '.join(UNITS)}), e.g. '400 m'. "
                           "Omit for 'contains'.",
        },
        "spatial_term": {
            "type": "string",
            "description": "Distance only, and only if the user used a vague phrase instead of a "
                           "number: the phrase, e.g. 'next to', 'near', 'close to'.",
        },
    },
    "required": ["source", "target", "type"],
    "additionalProperties": False,
}

PAIR_SCHEMA = {
    "type": "object",
    "properties": {"source": {"type": "integer"}, "target": {"type": "integer"}},
    "required": ["source", "target"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable

    def spec(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters,
        }}


TOOLS: dict[str, Tool] = {}


def tool(description: str, properties: dict, required: list[str] | None = None):
    def register(fn):
        TOOLS[fn.__name__] = Tool(
            name=fn.__name__,
            description=description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required if required is not None else list(properties),
                "additionalProperties": False,
            },
            fn=fn,
        )
        return fn
    return register


def tool_specs() -> list[dict]:
    return [t.spec() for t in TOOLS.values()]


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

@tool(
    "Start a NEW search from one description. Replaces the whole current query. Pass the "
    "complete query at once: area, all entities, all relations.",
    {
        "area": AREA_SCHEMA,
        "entities": {"type": "array", "items": _entity_schema(with_id=True), "minItems": 1},
        "relations": {
            "type": "array", "items": RELATION_SCHEMA,
            "description": "Relations stated in the description. Empty list if none.",
        },
    },
    required=["area", "entities", "relations"],
)
def new_query(query, area, entities, relations=None):
    fresh = build_query(area, entities, relations)
    query.clear()
    query.update(fresh)
    # Say so when a cluster got its distance from the tool, or the model tries to "fix" it.
    defaulted = [f"cluster {e['id']} ({e['name']}) got maxDistance {e['maxDistance']}"
                 for raw, e in zip(entities, fresh["entities"])
                 if e["type"] == "cluster" and not (isinstance(raw, dict) and raw.get("maxDistance"))]
    note = (f" Note: {'; '.join(defaulted)}. That is the right value when the user gives none: "
            "leave it as it is." if defaulted else "")
    return (f"Started a new query with {len(fresh['entities'])} entities and "
            f"{len(fresh['relations'])} relations.{note}")


@tool(
    "Change where to search: a named place (type 'area' + value) or the visible map ('bbox').",
    {"type": AREA_SCHEMA["properties"]["type"], "value": AREA_SCHEMA["properties"]["value"]},
    required=["type"],
)
def set_area(query, type, value=None):
    query["area"] = normalize_area({"type": type, "value": value})
    area = query["area"]
    return f"Area set to {area.get('value', 'the visible map (bbox)')}."


@tool(
    "Add entities to the current query. They get the next free ids, in list order "
    "(if the query has entities 0 and 1, the first new one is 2).",
    {"entities": {"type": "array", "items": _entity_schema(with_id=False), "minItems": 1}},
)
def add_entities(query, entities):
    start = len(query["entities"])
    new = [normalize_entity(e, start + i) for i, e in enumerate(entities)]
    query["entities"].extend(new)
    notes = fill_cluster_distances(query)
    added = ", ".join(f"{e['id']} ({e['name']})" for e in new)
    return f"Added entities {added}." + (f" Note: {'; '.join(notes)}." if notes else "")


@tool(
    "Change the name or type of an entity, or turn it into a cluster / back into a single object.",
    {
        "entity_id": {"type": "integer"},
        "name": {"type": "string", "description": "New name, singular."},
        "type": {"type": "string", "enum": ["nwr", "cluster"]},
        "minPoints": {"type": "integer", "description": "Cluster only: how many (>= 2)."},
        "maxDistance": {"type": "string", "description": "Cluster only, e.g. '300 m'."},
    },
    required=["entity_id"],
)
def update_entity(query, entity_id, name=None, type=None, minPoints=None, maxDistance=None):
    entity = get_entity(query, entity_id)
    raw = {**entity}
    if name is not None:
        raw["name"] = name
    if type is not None:
        raw["type"] = type
    elif minPoints is not None:
        raw["type"] = "cluster"
    if minPoints is not None:
        raw["minPoints"] = minPoints
    if maxDistance is not None:
        raw["maxDistance"] = maxDistance
    updated = normalize_entity(raw, entity_id)
    entity.clear()
    entity.update(updated)
    fill_cluster_distances(query)
    return f"Updated entity {entity_id}."


@tool(
    "Remove entities and all their relations. Remaining entities are renumbered 0..n-1.",
    {"entity_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 1}},
)
def remove_entities(query, entity_ids):
    for eid in entity_ids:
        get_entity(query, eid)                      # raises if unknown
    doomed = set(entity_ids)
    query["entities"] = [e for e in query["entities"] if e["id"] not in doomed]
    query["relations"] = [
        r for r in query["relations"] if r["source"] not in doomed and r["target"] not in doomed
    ]
    mapping = renumber(query)
    msg = f"Removed entities {', '.join(map(str, sorted(doomed)))} and their relations."
    if mapping:
        msg += " Renumbered: " + ", ".join(f"{old} -> {new}" for old, new in mapping.items()) + "."
    return msg


@tool(
    "Add properties to an entity. A property with the same name is replaced.",
    {"entity_id": {"type": "integer"}, "properties": {"type": "array", "items": PROPERTY_SCHEMA, "minItems": 1}},
)
def add_properties(query, entity_id, properties):
    entity = get_entity(query, entity_id)
    new = [normalize_property(p) for p in properties]
    entity["properties"] = merge_properties(entity.get("properties", []), new)
    return f"Entity {entity_id} now has properties: {', '.join(p['name'] for p in entity['properties'])}."


@tool(
    "Change one property of an entity (its name, operator or value).",
    {
        "entity_id": {"type": "integer"},
        "property_name": {"type": "string", "description": "Current name of the property."},
        "new_name": {"type": "string"},
        "operator": PROPERTY_SCHEMA["properties"]["operator"],
        "value": {"type": "string", "description": "New value. Empty string removes the value."},
    },
    required=["entity_id", "property_name"],
)
def update_property(query, entity_id, property_name, new_name=None, operator=None, value=None):
    entity = get_entity(query, entity_id)
    props = entity.get("properties", [])
    match = next((p for p in props if p["name"].lower() == property_name.lower()), None)
    if match is None:
        names = ", ".join(p["name"] for p in props) or "none"
        raise QueryError(f"entity {entity_id} has no property '{property_name}' (it has: {names})")
    raw = {**match}
    if new_name is not None:
        raw["name"] = new_name
    if value is not None:
        raw["value"] = value
        if value == "":
            raw.pop("operator", None)
    if operator is not None:
        raw["operator"] = operator
    updated = normalize_property(raw)
    entity["properties"] = [updated if p is match else p for p in props]
    return f"Updated property '{property_name}' of entity {entity_id}."


@tool(
    "Remove properties from an entity by name.",
    {"entity_id": {"type": "integer"}, "property_names": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
)
def remove_properties(query, entity_id, property_names):
    entity = get_entity(query, entity_id)
    doomed = {n.lower() for n in property_names}
    before = entity.get("properties", [])
    missing = doomed - {p["name"].lower() for p in before}
    if missing:
        raise QueryError(f"entity {entity_id} has no property {', '.join(sorted(missing))}")
    kept = [p for p in before if p["name"].lower() not in doomed]
    if kept:
        entity["properties"] = kept
    else:
        entity.pop("properties", None)
    return f"Removed {', '.join(property_names)} from entity {entity_id}."


@tool(
    "Add relations between existing entities. Only one relation per pair of entities.",
    {"relations": {"type": "array", "items": RELATION_SCHEMA, "minItems": 1}},
)
def add_relations(query, relations):
    ids = {e["id"] for e in query["entities"]}
    for raw in relations:
        relation = normalize_relation(raw, ids)
        check_new_relation(query, relation)
        query["relations"].append(relation)
    fill_cluster_distances(query)
    return f"Added {len(relations)} relation(s)."


@tool(
    "Change the relation between two entities (found in either direction). Giving a value "
    "without spatial_term removes the old spatial term; giving a spatial_term without value "
    "uses its default distance.",
    {
        "source": {"type": "integer"},
        "target": {"type": "integer"},
        "type": RELATION_SCHEMA["properties"]["type"],
        "value": RELATION_SCHEMA["properties"]["value"],
        "spatial_term": RELATION_SCHEMA["properties"]["spatial_term"],
        "reverse": {"type": "boolean", "description": "Swap source and target, e.g. to fix who contains whom."},
    },
    required=["source", "target"],
)
def update_relation(query, source, target, type=None, value=None, spatial_term=None, reverse=False):
    index = find_relation(query, source, target)
    raw = dict(query["relations"][index])
    if type is not None:
        raw["type"] = type
    if value is not None or spatial_term is not None:
        raw.pop("value", None)
        raw.pop("spatial_term", None)
        if value:
            raw["value"] = value
        if spatial_term:
            raw["spatial_term"] = spatial_term
    if raw.get("type") == "distance" and not raw.get("value") and not raw.get("spatial_term"):
        raise QueryError("a distance relation needs a value or a spatial_term")
    if reverse:
        raw["source"], raw["target"] = raw["target"], raw["source"]
    query["relations"][index] = normalize_relation(raw, {e["id"] for e in query["entities"]})
    return f"Updated the relation between {source} and {target}."


@tool(
    "Remove relations, each given by the two entity ids it connects (either direction).",
    {"relations": {"type": "array", "items": PAIR_SCHEMA, "minItems": 1}},
)
def remove_relations(query, relations):
    indices = {find_relation(query, int(p["source"]), int(p["target"])) for p in relations}
    query["relations"] = [r for i, r in enumerate(query["relations"]) if i not in indices]
    return f"Removed {len(indices)} relation(s)."


# --------------------------------------------------------------------------- #
# Tools that do not edit the query: declining, and reporting what was left out
# --------------------------------------------------------------------------- #

@dataclass
class Outcome:
    """What a tool returns when it has more to say than a message."""
    message: str
    refusal: Refusal | None = None
    limitations: list[dict] = field(default_factory=list)


@tool(
    "Decline the user's message instead of building or editing the query. Call it alone, "
    "without any other tool. The chat ends the turn with a fixed reply.",
    {
        "category": {"type": "string", "enum": list(REFUSAL_CATEGORIES)},
        "reply": {
            "type": "string",
            "description": "Only for 'off_topic': one short sentence in the user's language saying "
                           "that you only build map searches. Ignored for all other categories.",
        },
    },
    required=["category"],
)
def decline(query, category, reply=None):
    try:
        refusal = refusal_for(category, reply)
    except ValueError as exc:
        raise QueryError(str(exc)) from None
    return Outcome(f"Declined ({refusal.category}).", refusal=refusal)


@tool(
    "Record the parts of the user's request that the query cannot express and that you left "
    "out or approximated. Call it in the same reply as the tools that build the rest.",
    {"items": {
        "type": "array", "minItems": 1,
        "items": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": list(LIMITS)},
                "detail": {
                    "type": "string",
                    "description": "The user's words for that part, e.g. 'at least 500 m from', "
                                   "'open after 10 pm', 'the nearest'.",
                },
            },
            "required": ["category", "detail"],
            "additionalProperties": False,
        },
    }},
)
def report_limitations(query, items):
    try:
        found = [limitation(item.get("category"), item.get("detail")) for item in items]
    except ValueError as exc:
        raise QueryError(str(exc)) from None
    told = "; ".join(f"\"{item['detail']}\": {item['message']}" for item in found)
    return Outcome(f"Recorded. Tell the user in your reply: {told}", limitations=found)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

@dataclass
class ToolResult:
    name: str
    arguments: dict
    ok: bool
    message: str
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)    # unknown descriptors, with suggestions
    refusal: Refusal | None = None                    # set by a successful decline
    limitations: list[dict] = field(default_factory=list)   # parts the query cannot express

    def for_model(self, query: dict) -> str:
        """Tool message sent back to the LLM: outcome, warnings, notes and the current query."""
        lines = [("OK: " if self.ok else "ERROR (nothing was changed): ") + self.message]
        lines += [f"Warning: {w}" for w in self.warnings]
        lines += [f"Note: {n}" for n in self.notes]
        if self.name != "report_limitations":           # its message already says this
            lines += [f"Limit: \"{item['detail'] or item['category']}\": {item['message']} "
                      "Tell the user." for item in self.limitations]
        lines += ["", "Current query:", to_yaml(query).rstrip()]
        return "\n".join(lines)


def parse_arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if raw is None or not str(raw).strip():
        return {}
    args = json.loads(raw)
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be a JSON object")
    # Some models double-encode nested lists/objects as strings.
    for key, value in args.items():
        if isinstance(value, str) and value.strip()[:1] in "[{":
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
    return args


def execute_tool(query: dict, name: str, raw_arguments, message: str | None = None) -> tuple[dict, ToolResult]:
    """Run one tool call. Returns the (possibly) new query and a result for the model/UI.
    `message` is the user's message of this turn, used to check names against it."""
    try:
        arguments = parse_arguments(raw_arguments)
    except (ValueError, json.JSONDecodeError) as exc:
        return query, ToolResult(name, {}, False, f"arguments are not valid JSON ({exc}). Try again.")

    selected = TOOLS.get(name)
    if selected is None:
        return query, ToolResult(name, arguments, False, f"unknown tool '{name}'. Available: {', '.join(TOOLS)}.")

    try:
        inspect.signature(selected.fn).bind(None, **arguments)
    except TypeError:
        expected = ", ".join(selected.parameters["properties"])
        return query, ToolResult(
            name, arguments, False,
            f"wrong arguments {sorted(arguments)}; {name} takes: {expected} "
            f"(required: {', '.join(selected.parameters['required'])}).",
        )

    if name == "report_limitations" and message:
        mistake = not_a_limit(arguments.get("items"), message)
        if mistake:
            return query, ToolResult(name, arguments, False, mistake)

    working = copy.deepcopy(query)
    try:
        outcome = selected.fn(working, **arguments)
    except QueryError as exc:
        return query, ToolResult(name, arguments, False, str(exc))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return query, ToolResult(name, arguments, False, f"could not apply {name}: {exc!r}")
    if not isinstance(outcome, Outcome):
        outcome = Outcome(outcome)
    if message:
        keep_user_units(working, message)
    return working, ToolResult(name, arguments, True, outcome.message, warnings_for(working),
                               new_notes(query, working, message), outcome.refusal,
                               outcome.limitations + code_limits(query, working))


def not_a_limit(items, message: str) -> str | None:
    """Why a report is wrong: the "unsupported detail" is an object or a property that SPOT
    knows and that the user wrote. Such a part belongs in the query, not in the limits."""
    descriptors = get_descriptors()
    if descriptors is None or not isinstance(items, list):
        return None
    found = descriptors.scan(message)
    objects = {norm(word): name for name, wrote in found.objects for word in (name, wrote)}
    properties = {norm(word): name for name, wrote in found.properties for word in (name, wrote)}
    for item in items:
        if not isinstance(item, dict) or item.get("category") != "unsupported_attribute":
            continue
        detail = norm(item.get("detail") or "")
        for article in ("a ", "an ", "the ", "with a ", "with an ", "with "):
            if detail.startswith(article):
                detail = detail[len(article):]
        if detail in objects:
            return (f"'{objects[detail]}' is an object SPOT can search for, not a limit. Add it as an "
                    "entity (add_entities) and connect it with a contains or distance relation "
                    "(add_relations). Report only the other items, if any.")
        if detail in properties:
            return (f"'{properties[detail]}' is a SPOT property, not a limit. Add it with "
                    "add_properties. Report only the other items, if any.")
    return None


def keep_user_units(query: dict, message: str) -> None:
    """Distances stay in the unit the user wrote ("0.5 km", not "500 m")."""
    for relation in query["relations"]:
        if relation.get("value"):
            relation["value"] = as_written(relation["value"], message)
    for entity in query["entities"]:
        if entity.get("maxDistance"):
            entity["maxDistance"] = as_written(entity["maxDistance"], message)


def new_notes(before: dict, after: dict, message: str | None = None) -> list[str]:
    """Descriptor notes caused by this call (not ones the model has already seen)."""
    seen = set(notes_for(before, message))
    return [note for note in notes_for(after, message) if note not in seen]
