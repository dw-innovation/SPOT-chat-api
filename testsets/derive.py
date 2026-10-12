"""Multi-turn test dialogues derived from the gold standard (roadmap section C).

Two kinds of dialogue come out of a gold sentence:

- An edit: the chat starts with the gold query already built and the user asks for one
  mechanical change (new distance, new area, drop an entity, ...). This isolates editing
  from the one-shot errors the benchmark already measures.
- A split build: the gold query is built over several messages, one object at a time.

The expected query is computed by the same tool functions the model calls, so it follows
their conventions (renumbering, cluster distances). Only the user's wording comes from the
templates below, and should be spot-checked: `python run_testsets.py --gold ... --list`.
"""

from __future__ import annotations

import random

from spot_chat.query import QueryError, build_query, from_yaml, to_export
from spot_chat.schema import MAX_ENTITIES, MAX_PROPERTIES
from spot_chat.tools import execute_tool

from .cases import Case, Expect, Turn

AREAS = ["Tartu, Estonia", "Bergamo", "Kraków", "Aarhus", "Valparaíso", "Graz"]
DISTANCES = ["80 m", "150 m", "350 m", "600 m", "1.2 km"]
OBJECTS = ["pharmacy", "bench", "bus stop", "fountain", "bakery", "post office"]
NAMES = ["Aurora", "Lindenhof", "Milano", "Nordstern"]
ASSISTANT_REPLY = "I set up that search."


def label(entity: dict) -> str:
    """How a user would refer to the entity."""
    name = entity["name"]
    return name.split(":", 1)[1] if name.lower().startswith("brand:") else name


def a(name: str) -> str:
    return f"{'an' if name[:1].lower() in 'aeiou' else 'a'} {name}"


def plural(name: str) -> str:
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    if name.endswith("y") and name[-2:-1] not in "aeiou":
        return name[:-1] + "ies"
    return name + "s"


def _plain(entity: dict) -> bool:
    return entity["type"] == "nwr" and not entity["name"].lower().startswith("brand:")


def _referable(query: dict) -> bool:
    """Every entity can be named without ambiguity."""
    names = [e["name"].lower() for e in query["entities"]]
    return len(set(names)) == len(names)


def apply(query: dict, *calls) -> dict:
    """The query after these tool calls, exactly as the tools would leave it."""
    for name, arguments in calls:
        query, result = execute_tool(query, name, arguments)
        if not result.ok:
            raise QueryError(f"{name}: {result.message}")
    return query


# --------------------------------------------------------------------------- #
# Edits: (query, rng) -> (message, tool calls), or None if the edit does not fit the query
# --------------------------------------------------------------------------- #

def change_distance(query, rng):
    relations, entities = query["relations"], query["entities"]
    if len(relations) != 1 or relations[0]["type"] != "distance" or not all(map(_plain, entities)):
        return None
    relation = relations[0]
    value = rng.choice([d for d in DISTANCES if d != relation["value"]])
    first, second = (label(entities[relation[k]]) for k in ("source", "target"))
    message = rng.choice([
        f"Make it {value} instead.",
        f"Change the distance between the {first} and the {second} to {value}.",
        f"The {first} should be within {value} of the {second}.",
    ])
    return message, [("update_relation", {"source": relation["source"], "target": relation["target"],
                                          "value": value})]


def swap_area(query, rng):
    area = rng.choice([p for p in AREAS if p != query["area"].get("value")])
    message = rng.choice([f"Search in {area} instead.", f"Same search, but in {area}.",
                          f"Move the search to {area}."])
    return message, [("set_area", {"type": "area", "value": area})]


def remove_entity(query, rng):
    candidates = [e for e in query["entities"] if e["type"] == "nwr"]
    if len(query["entities"]) < 2 or not _referable(query) or not candidates:
        return None
    entity = rng.choice(candidates)
    name = label(entity)
    message = rng.choice([f"Remove the {name}.", f"Forget about the {name}.",
                          f"I don't need the {name} anymore."])
    return message, [("remove_entities", {"entity_ids": [entity["id"]]})]


def add_entity(query, rng):
    entities = query["entities"]
    targets = [e for e in entities if e["type"] == "nwr"]
    if len(entities) >= MAX_ENTITIES or not _referable(query) or not targets:
        return None
    target = rng.choice(targets)
    new = rng.choice([o for o in OBJECTS if o not in {e["name"].lower() for e in entities}])
    value = rng.choice(DISTANCES)
    message = rng.choice([
        f"Also add {a(new)} within {value} of the {label(target)}.",
        f"There should also be {a(new)} no more than {value} from the {label(target)}.",
    ])
    return message, [
        ("add_entities", {"entities": [{"name": new, "type": "nwr"}]}),
        ("add_relations", {"relations": [{"source": len(entities), "target": target["id"],
                                          "type": "distance", "value": value}]}),
    ]


def add_property(query, rng):
    candidates = [
        e for e in query["entities"]
        if _plain(e) and len(e.get("properties", [])) < MAX_PROPERTIES
        and not any(p["name"].lower() in ("name", "brand", "brand name") for p in e.get("properties", []))
    ]
    if not candidates or not _referable(query):
        return None
    entity, value = rng.choice(candidates), rng.choice(NAMES)
    message = rng.choice([f"The {label(entity)} should have \"{value}\" in its name.",
                          f"The name of the {label(entity)} should contain \"{value}\"."])
    return message, [("add_properties", {"entity_id": entity["id"], "properties": [
        {"name": "name", "operator": "~", "value": value}]})]


def remove_property(query, rng):
    candidates = [e for e in query["entities"] if e.get("properties")]
    if not candidates or not _referable(query):
        return None
    entity = rng.choice(candidates)
    prop = rng.choice(entity["properties"])["name"]
    message = rng.choice([f"Remove the {prop} condition from the {label(entity)}.",
                          f"The {label(entity)} no longer needs the {prop} condition."])
    return message, [("remove_properties", {"entity_id": entity["id"], "property_names": [prop]})]


def to_cluster(query, rng):
    candidates = [e for e in query["entities"] if _plain(e)]
    if not candidates or not _referable(query):
        return None
    entity, count = rng.choice(candidates), rng.choice([2, 3, 4])
    message = rng.choice([f"I want {count} of those {plural(label(entity))}.",
                          f"Make it {count} {plural(label(entity))} instead of one."])
    return message, [("update_entity", {"entity_id": entity["id"], "type": "cluster", "minPoints": count})]


def reverse_contains(query, rng):
    relations = [r for r in query["relations"] if r["type"] == "contains"]
    if not relations or not _referable(query):
        return None
    relation = rng.choice(relations)
    outer, inner = (label(query["entities"][relation[k]]) for k in ("source", "target"))
    message = rng.choice([f"The other way round: the {inner} should contain the {outer}.",
                          f"Swap that: the {outer} is inside the {inner}."])
    return message, [("update_relation", {"source": relation["source"], "target": relation["target"],
                                          "reverse": True})]


EDITS = [change_distance, swap_area, remove_entity, add_entity, add_property, remove_property,
         to_cluster, reverse_contains]


# --------------------------------------------------------------------------- #
# Split build
# --------------------------------------------------------------------------- #

def split_build(query: dict) -> list[Turn] | None:
    """The gold query built one object per message, with the expected query after each."""
    entities = query["entities"]
    if (len(entities) < 2 or not _referable(query)
            or not all(_plain(e) and not e.get("properties") for e in entities)):
        return None
    place = f" in {query['area']['value']}" if query["area"]["type"] == "area" else ""
    messages = [f"Find {a(label(entities[0]))}{place}."]
    for new in range(1, len(entities)):
        name, parts = label(entities[new]), []
        for relation in query["relations"]:
            ends = (relation["source"], relation["target"])
            if new not in ends or max(ends) != new:
                continue
            other = label(entities[min(ends)])
            this = f"the {name}" if parts else a(name)          # introduced by the first part
            if relation["type"] == "distance":
                parts.append(f"The {name} should also be within {relation['value']} of the {other}."
                             if parts else f"Add {this} within {relation['value']} of the {other}.")
            elif relation["target"] == new:
                parts.append(f"There should be {this} inside the {other}.")
            else:
                parts.append(f"The {other} should be inside {this}.")
        messages.append(" ".join(parts) or f"Also look for {a(name)}.")

    turns = []
    for count, message in enumerate(messages, 1):
        part = build_query(query["area"], entities[:count],
                           [r for r in query["relations"] if max(r["source"], r["target"]) < count])
        turns.append(Turn(message, Expect(query=to_export(part))))
    return turns


# --------------------------------------------------------------------------- #
# Deriving a test set
# --------------------------------------------------------------------------- #

def derive(rows: list[tuple[str, str]], per_kind: int = 8, seed: int = 0) -> list[Case]:
    """Up to `per_kind` dialogues of every kind from (sentence, gold YAML) rows. The same rows
    and seed always give the same cases."""
    queries = []
    for number, (sentence, gold_yaml) in enumerate(rows):
        try:
            queries.append((number, str(sentence).strip(), from_yaml(gold_yaml)))
        except QueryError:
            continue                                  # a gold row the tools would not accept

    cases = []
    for kind in [*EDITS, split_build]:
        rng = random.Random(f"{seed}-{kind.__name__}")
        made = 0
        for number, sentence, query in rng.sample(queries, len(queries)):
            if made == per_kind:
                break
            case_id, group = f"{kind.__name__}-{number:03d}", kind.__name__
            if kind is split_build:
                turns = split_build(query)
                if turns is None:
                    continue
                cases.append(Case(case_id, "C", group, turns))
            else:
                edit = kind(query, rng)
                if edit is None:
                    continue
                message, calls = edit
                expected = to_export(apply(query, *calls))
                history = [{"role": "user", "content": sentence},
                           {"role": "assistant", "content": ASSISTANT_REPLY}]
                cases.append(Case(case_id, "C", group, [Turn(message, Expect(query=expected))],
                                  query=query, history=history))
            made += 1
    return cases
