"""SPOT's descriptor tables: object names, property names and colours.

Loaded from data/SPOT_OSM-tag-bundles.xlsx (or SPOT_BUNDLES_PATH) and
data/spot_colours.csv (or SPOT_COLOURS_PATH). They are used to
- list the known properties and their value types in the prompt,
- enforce property value types in the tools,
- tell the model when an entity name, property or colour is not in the tables,
  together with the closest known ones (the tables are too big for the prompt),
- compare predictions with the gold standard at bundle level.
Everything keeps working without the files, just without these checks.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
from pathlib import Path

from .schema import SPATIAL_TERM_ALIASES, SPATIAL_TERMS, normalize_spatial_term

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def norm(text) -> str:
    text = str(text).casefold().replace("-", " ").replace("_", " ")
    return " ".join(text.split())


@dataclass(frozen=True)
class PropertySpec:
    bundle: int
    kind: str                  # "flag" (no value), "numeric" or "text"
    operator: str | None       # "~" or "=" for text properties
    names: tuple[str, ...]     # all descriptors of the bundle; the first is the main name
    is_colour: bool


def _property_spec(bundle: int, names: list[str], tags) -> PropertySpec:
    tags = str(tags)
    if "~***" in tags:
        kind, operator = "text", "~"
    elif "***numeric***" in tags:
        kind, operator = "numeric", None
    elif "***example***" in tags:
        kind, operator = "text", "="
    else:
        kind, operator = "flag", None
    is_colour = any("colour" in n.lower() or "color" in n.lower() for n in names[:3])
    return PropertySpec(bundle, kind, operator, tuple(names), is_colour)


def _display(name: str) -> str:
    """Lowercase, but keep acronyms such as ATM or TV."""
    return " ".join(w if len(w) > 1 and w.isupper() else w.lower() for w in name.split())


def _token_index(keys) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for key in keys:
        for token in key.split():
            index.setdefault(token, set()).add(key)
    return index


def _suggest(query: str, keys: list[str], index: dict, display: dict, n: int) -> list[str]:
    """Closest known names: typo matches plus names sharing words with the query.
    A known name that is the query minus a modifier ranks first
    ("nepalese restaurant" -> "restaurant")."""
    q = norm(query)
    q_tokens = set(q.split())
    if not q_tokens:
        return []
    scores = {k: SequenceMatcher(None, q, k).ratio() for k in get_close_matches(q, keys, n=8, cutoff=0.75)}
    for key in set().union(*(index.get(t, set()) for t in q_tokens)):
        k_tokens = set(key.split())
        score = len(q_tokens & k_tokens) / len(q_tokens | k_tokens)
        if k_tokens <= q_tokens:
            score += 0.3
        if key.split()[-1] == q.split()[-1]:
            score += 0.2
        scores[key] = max(scores.get(key, 0.0), score)
    ranked = sorted(scores, key=lambda k: (-scores[k], len(k)))
    return [display[k] for k in ranked[:n]]


_UNIT_WORDS = {"yard", "yards", "meter", "meters", "metre", "metres", "mile", "miles", "foot", "feet"}
_NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                 "hundred", "thousand", "few", "several", "many"}


# Spatial terms too common to point out on their own ("at least", "within 2 km distance").
_SKIP_SPATIAL = {"at", "distance"}

BRAND_HINT = ("If these words are a brand, a chain or the name of one particular place, use "
              "'brand:<name>' as the entity name instead, in the user's spelling and script "
              "(never translate it).")


@dataclass
class Scan:
    """SPOT vocabulary found in a text: (canonical form, what the user wrote)."""
    objects: list[tuple[str, str]]
    properties: list[tuple[str, str]]
    spatial: list[tuple[str, str, str]]          # (term, distance, what the user wrote)


def _contains_run(longer: list[str], shorter: list[str]) -> bool:
    n = len(shorter)
    return any(longer[i:i + n] == shorter for i in range(len(longer) - n + 1))


def _singular_forms(phrase: str) -> list[str]:
    """The phrase plus plausible singulars of its last word ("bakeries" -> "bakery")."""
    words = phrase.split()
    last, forms = words[-1], [phrase]
    for old, new in (("ies", "y"), ("ches", "ch"), ("shes", "sh"), ("xes", "x"), ("ses", "s"), ("s", "")):
        if last.endswith(old) and not last.endswith("ss") and len(last) > len(old) + 2:
            forms.append(" ".join(words[:-1] + [last[: -len(old)] + new]))
    return forms


class Descriptors:
    def __init__(self, entities, entity_display, properties, colours, colour_display):
        self.entities: dict[str, set[int]] = entities
        self.entity_display: dict[str, str] = entity_display
        self.properties: dict[str, PropertySpec] = properties
        self.colours: dict[str, set[int]] = colours
        self.colour_display: dict[str, str] = colour_display
        self._entity_keys, self._entity_index = list(entities), _token_index(entities)
        self._prop_keys, self._prop_index = list(properties), _token_index(properties)
        self._prop_display = {k: k for k in properties}
        self._colour_keys, self._colour_index = list(colour_display), _token_index(colour_display)
        self._vocab = sorted({t for k in [*entities, *properties, *SPATIAL_TERMS, *SPATIAL_TERM_ALIASES]
                              for t in k.split()})
        self._vocab_set = set(self._vocab)
        self._corrections: dict[str, str] = {}

    # ----- loading --------------------------------------------------------- #

    @classmethod
    def load(cls, bundles_path: Path, colours_path: Path | None = None) -> "Descriptors":
        from openpyxl import load_workbook

        workbook = load_workbook(bundles_path, read_only=True, data_only=True)
        rows = workbook.worksheets[0].iter_rows(values_only=True)
        column = {str(h).strip(): i for i, h in enumerate(next(rows)) if h}
        entities, entity_display, properties = {}, {}, {}
        for bundle, row in enumerate(rows):
            names = [d.strip() for d in str(row[column["descriptors"]] or "").split("|") if d.strip()]
            kind = str(row[column["core/prop"]] or "core")
            if not names:
                continue
            if "core" in kind:
                for name in names:
                    entities.setdefault(norm(name), set()).add(bundle)
                    entity_display.setdefault(norm(name), _display(name))
            if "prop" in kind:
                spec = _property_spec(bundle, names, row[column["tags"]])
                for name in names:
                    properties.setdefault(norm(name), spec)
        workbook.close()

        colours, colour_display = {}, {}
        if colours_path and Path(colours_path).exists():
            with open(colours_path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    bundle = int(row["bundle_id"])
                    for field, shown in (("colour_descriptors", True), ("colour_values", False)):
                        for colour in row[field].split(","):
                            key = norm(colour)
                            if key and not key.startswith("#"):
                                colours.setdefault(key, set()).add(bundle)
                                if shown:
                                    colour_display.setdefault(key, colour.strip())
        return cls(entities, entity_display, properties, colours, colour_display)

    # ----- lookups --------------------------------------------------------- #

    def knows_entity(self, name: str) -> bool:
        key = norm(name)
        return key.startswith("brand:") or key in self.entities

    def entity_bundles(self, name: str) -> frozenset:
        """Bundle ids of a name (the unknown name itself if it is not a descriptor)."""
        key = norm(name)
        return frozenset(self.entities.get(key, ())) or frozenset({key})

    def property(self, name: str) -> PropertySpec | None:
        return self.properties.get(norm(name))

    def colour_bundles(self, value: str) -> frozenset:
        """Colour bundle ids; unknown shades fall back to their last word ('pale blue' -> blue)."""
        key = norm(value)
        found = self.colours.get(key) or self.colours.get(key.split()[-1] if key else "", set())
        return frozenset(found)

    def suggest_entities(self, name: str, n: int = 5) -> list[str]:
        return _suggest(name, self._entity_keys, self._entity_index, self.entity_display, n)

    def suggest_properties(self, name: str, n: int = 4) -> list[str]:
        return _suggest(name, self._prop_keys, self._prop_index, self._prop_display, n)

    def suggest_colours(self, value: str, n: int = 4) -> list[str]:
        return _suggest(value, self._colour_keys, self._colour_index, self.colour_display, n)

    # ----- scanning the user's message --------------------------------------- #

    def _correct(self, token: str) -> str:
        """Fix a typo if the token is close to a word used in SPOT's vocabulary."""
        if token in self._vocab_set or not token.isalpha() or len(token) < 4:
            return token
        if token not in self._corrections:
            match = get_close_matches(token, self._vocab, n=1, cutoff=0.8)
            if len(self._corrections) < 50_000:
                self._corrections[token] = match[0] if match else token
            return match[0] if match else token
        return self._corrections[token]

    def scan(self, text: str, max_words: int = 6) -> Scan:
        """SPOT object descriptors, property names and spatial terms in the text, typos corrected.
        Only the longest matches count ("irish bar", not also "bar"); plurals count ("bakeries");
        words right after a number are left alone (they are units: "300 miter")."""
        raw = re.findall(r"[^\W_]+(?:'[^\W_]+)?", norm(text).replace("\u2019", "'"))

        def after_number(i: int) -> bool:
            return i > 0 and (raw[i - 1][0].isdigit() or raw[i - 1] in _NUMBER_WORDS)

        tokens = [t if after_number(i) or t[0].isdigit() else self._correct(t) for i, t in enumerate(raw)]
        spans = []
        for i in range(len(tokens)):
            for j in range(min(len(tokens), i + max_words), i, -1):
                phrase, wrote = " ".join(tokens[i:j]), " ".join(raw[i:j])
                if j - i == 1 and phrase in _UNIT_WORDS and after_number(i):
                    continue
                hit = None
                for form in _singular_forms(phrase):
                    if form.startswith(("a ", "an ", "the ")):   # "a building" is not a name
                        continue
                    if form in self.properties:
                        hit = ("property", form, wrote)
                        break
                    if form in self.entities:
                        hit = ("entity", self.entity_display[form], wrote)
                        break
                if hit is None and (phrase in SPATIAL_TERMS or phrase in SPATIAL_TERM_ALIASES) \
                        and phrase not in _SKIP_SPATIAL and not (j < len(tokens) and tokens[j][0].isdigit()):
                    term, distance = normalize_spatial_term(phrase)
                    hit = ("spatial", term, wrote, distance)
                if hit:
                    spans.append((i, j, *hit))
        maximal = [s for s in spans
                   if not any(o[0] <= s[0] and s[1] <= o[1] and o[:2] != s[:2] for o in spans)]

        def unique(items):                        # first spelling per descriptor
            return list({item[0]: item for item in items}.values())
        return Scan(
            objects=unique((s[3], s[4]) for s in maximal if s[2] == "entity"),
            properties=unique((s[3], s[4]) for s in maximal if s[2] == "property"),
            spatial=unique((s[3], s[5], s[4]) for s in maximal if s[2] == "spatial"),
        )

    def find_in_text(self, text: str) -> tuple[list[str], list[str]]:
        """Object and property descriptors in the text (see scan)."""
        found = self.scan(text)
        return ([o for o, _ in found.objects], [p for p, _ in found.properties])

    def generalization_notes(self, query: dict, found: Scan) -> list[str]:
        """Entities whose name is a cut-down version of a descriptor the user wrote
        ("station" when the user wrote "petrol station", "teahouse" + cuisine = japanese
        when the user wrote "japanese teahouse")."""
        notes = []
        in_query = {norm(e["name"]) for e in query["entities"]}
        written = [(norm(name), name, wrote) for name, wrote in found.objects]
        written_keys = {k for k, _, _ in written}
        for entity in query["entities"]:
            key = norm(entity["name"])
            if key.startswith("brand:"):
                continue
            count = len(notes)
            for full_key, full_name, wrote in written:
                full_tokens = full_key.split()
                if full_key in in_query or len(full_tokens) <= len(key.split()) \
                        or not _contains_run(full_tokens, key.split()):
                    continue
                repeats = [f"{p['name']} = {p['value']}" for p in entity.get("properties", [])
                           if p.get("value") and set(norm(p["value"]).split()) <= set(full_tokens)]
                notes.append(
                    f"entity {entity['id']} name '{entity['name']}' is more general than "
                    f"'{full_name}', which the user wrote"
                    + (f" (as '{wrote}')" if norm(wrote) != full_key else "")
                    + f" and which is a SPOT descriptor. Rename it with update_entity(entity_id="
                    f"{entity['id']}, name='{full_name}')"
                    + (f" and remove the properties that only repeat its words: {'; '.join(repeats)}"
                       if repeats else "") + "."
                )
                break
            if len(notes) > count or key in written_keys:
                continue
            # A synonym from the same bundle while the user's own descriptor is unused
            # ("fire hydrant" when the user wrote "city fire plug").
            for full_key, full_name, wrote in written:
                if full_key not in in_query and self.entities.get(full_key, set()) & self.entities.get(key, set()):
                    notes.append(
                        f"entity {entity['id']} name '{entity['name']}' is a synonym of '{full_name}', "
                        "which the user wrote and which is a SPOT descriptor. Keep the user's word: "
                        f"update_entity(entity_id={entity['id']}, name='{full_name}')."
                    )
                    break
        return notes

    # ----- notes for the model (local table) ------------------------------ #

    def entity_note(self, name: str, label: str) -> str | None:
        """Note for a name that is not a descriptor, with local suggestions."""
        if self.knows_entity(name):
            return None
        close = self.suggest_entities(name)
        if not close:
            return (f"{label} name '{name}' is not a SPOT descriptor and nothing similar exists. "
                    + BRAND_HINT + " Otherwise keep it, but tell the user that SPOT may not know "
                    "this object.")
        return (f"{label} name '{name}' is not a SPOT descriptor. Closest descriptors: "
                f"{', '.join(close)}. Switch to one that means the same thing; move extra details "
                "(cuisine, religion, material, ...) into properties. " + BRAND_HINT + " If nothing "
                "fits, keep the name and tell the user that SPOT may not know this object.")

    def property_notes(self, entity: dict, label: str) -> list[str]:
        notes = []
        for prop in entity.get("properties", []):
            spec = self.property(prop["name"])
            if spec is None:
                close = self.suggest_properties(prop["name"])
                notes.append(
                    f"{label} property '{prop['name']}' is not a SPOT property"
                    + (f" (closest: {', '.join(close)}). Use one of those if it means the same thing."
                       if close else ". Keep it only if the user really asked for it.")
                )
            elif spec.is_colour and prop.get("value") and norm(prop["value"]) not in self.colours:
                close = self.suggest_colours(prop["value"])
                notes.append(
                    f"{label} colour '{prop['value']}' is not in SPOT's colour list"
                    + (f" (closest: {', '.join(close)}). Use the listed colour that fits best."
                       if close else ".")
                )
        return notes

    # ----- prompt ---------------------------------------------------------- #

    def property_prompt_lines(self) -> str:
        groups = {"flag": [], "numeric": [], "text=": [], "text~": []}
        seen = set()
        for spec in self.properties.values():
            if spec.bundle in seen:
                continue
            seen.add(spec.bundle)
            names = list(dict.fromkeys(n.lower() for n in spec.names))
            label = names[0] + (f" ({', '.join(names[1:4])})" if len(names) > 1 else "")
            groups[spec.kind if spec.kind != "text" else f"text{spec.operator}"].append(label)
        return "\n".join([
            "- Features without a value (write only the name): " + "; ".join(groups["flag"]),
            '- Numbers (operator "=", "<" or ">"): ' + "; ".join(groups["numeric"]),
            '- Exact values (operator "="): ' + "; ".join(groups["text="]),
            '- Text search (operator "~"): ' + "; ".join(groups["text~"]),
        ])


def remote_entity_note(name: str, label: str, matches) -> str | None:
    """Note based on SPOT's tag search results (what the pipeline will search for)."""
    key = norm(name)
    if any(key in {norm(d) for d in m.descriptors} for m in matches):
        return None                                   # exact descriptor (maybe newer than our table)
    if not matches:
        return (f"{label} name '{name}' is not a SPOT descriptor and SPOT's tag search finds "
                "nothing for it. " + BRAND_HINT + " Otherwise keep it, but tell the user that "
                "SPOT may not know this object.")
    readings = "; ".join(
        f"'{m.name}' (same as: {', '.join(d for d in m.descriptors[:4] if norm(d) != norm(m.name))})"
        for m in matches[:3]
    )
    return (f"{label} name '{name}' is not an exact SPOT descriptor. SPOT's tag search reads it as: "
            f"{readings}. SPOT will search for the first one. If one of them is what the user means, "
            "switch to that descriptor (move extra details such as cuisine or material into "
            "properties). " + BRAND_HINT + " If nothing fits, keep the name and tell the user that "
            "SPOT may not know this object.")


def notes_for(query: dict, message: str | None = None) -> list[str]:
    """Notes about names SPOT may not understand: entity names are checked with SPOT's
    tag search API when it is reachable, otherwise with the local table; properties
    and colours are checked with the local tables. With the user's message, entities
    that are cut-down versions of a descriptor the user wrote are pointed out too."""
    from .tag_search import get_tag_search

    local, remote = get_descriptors(), get_tag_search()
    notes = local.generalization_notes(query, local.scan(message)) if local and message else []
    for entity in query["entities"]:
        label, name = f"entity {entity['id']}", entity["name"]
        if name.lower().startswith("brand:") or (local and local.knows_entity(name)):
            note = None
        else:
            matches = remote.search(name) if remote else None
            if matches is not None:
                note = remote_entity_note(name, label, matches)
            else:
                note = local.entity_note(name, label) if local else None
        if note:
            notes.append(note)
        if local:
            notes += local.property_notes(entity, label)
    return notes


@lru_cache(maxsize=1)
def get_descriptors() -> Descriptors | None:
    bundles = Path(os.getenv("SPOT_BUNDLES_PATH", DATA_DIR / "SPOT_OSM-tag-bundles.xlsx"))
    colours = Path(os.getenv("SPOT_COLOURS_PATH", DATA_DIR / "spot_colours.csv"))
    if not bundles.exists():
        print(f"Descriptor table not found at {bundles}; name checks are disabled.")
        return None
    return Descriptors.load(bundles, colours)
