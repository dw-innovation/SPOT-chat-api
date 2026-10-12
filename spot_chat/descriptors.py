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
import unicodedata
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


# Ordinary words that are one letter away from a word in SPOT's vocabulary. They are never
# read as typos ("have" is not "shave", "least" is not "east").
_COMMON_WORDS = frozenset("""
    about above after again also another apart around assigned away back been before being below
    best both called came come contain containing contains could county curious does done down
    each eight either else ending ends even ever every feet find fine five found four from give
    goes gone good have having here high inside into just keep kind know last least less lies
    like line list live locate located long look looking made make many maybe more most much
    must name named near need next nice nine none once only other over paid part past payed
    place please radius regarding requirements same searching seen send serves seven should show
    side some sorry starting such take tell than thank thanks that their them then there these
    they thing this those three time told took trying very want well went were what when where
    which while whose will with within would your
""".split())


def _without_accents(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", word) if not unicodedata.combining(c))


def _plural_forms(word: str) -> list[str]:
    if word.endswith("y"):
        return [word[:-1] + "ies"]
    return [word + "es" if word.endswith(("s", "x", "ch", "sh")) else word + "s"]


def _is_swap(a: str, b: str) -> bool:
    """Two neighbouring letters swapped ("sprots" for "sports"), the most common slip."""
    differ = [i for i in range(min(len(a), len(b))) if a[i] != b[i]]
    return len(a) == len(b) and len(differ) == 2 and differ[1] == differ[0] + 1 \
        and a[differ[0]] == b[differ[1]] and a[differ[1]] == b[differ[0]]


def _same_start(a: str, b: str) -> bool:
    """Typos rarely change the first letter, except for letters that sound alike and a letter
    stuck to the front of a long word ("aphysiotherapy")."""
    return a[0] == b[0] or {a[0], b[0]} in ({"c", "k"}, {"f", "p"}, {"s", "c"}, {"s", "z"}) \
        or (len(b) >= 8 and a[1:] == b)


def _edit_distance(a: str, b: str) -> int:
    """Number of single-letter edits between two words; swapping two neighbours counts as one."""
    rows = [list(range(len(b) + 1))]
    for i in range(1, len(a) + 1):
        row = [i]
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            best = min(rows[i - 1][j] + 1, row[j - 1] + 1, rows[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                best = min(best, rows[i - 2][j - 2] + 1)
            row.append(best)
        rows.append(row)
    return rows[-1][-1]


_SENTENCE_START_RE = re.compile(r"(?:^|[.!?:]\s+)[\"'\u201c(]*$")


def _proper_nouns(text: str) -> set[str]:
    """Words written with a capital inside a sentence ("Poland", "Olt"), in the form the scan
    uses. Their spelling is the user's, so they are not corrected."""
    found = set()
    for match in re.finditer(r"[^\W\d_]+", str(text)):
        word = match.group(0)
        if word[0].isupper() and not word.isupper() \
                and not _SENTENCE_START_RE.search(text[:match.start()][-4:]):
            found.add(norm(word))
    return found


# Spatial terms too common to point out on their own ("at least", "within 2 km distance").
_SKIP_SPATIAL = {"at", "distance"}

# The pipeline looks every name up in the tag search anyway, so a name the user chose is not
# replaced by the search result: it reads the same later and stays the user's word.
KEEP_HINT = ("If '{name}' is the user's own word for the object (typos fixed, singular), keep it "
             "as it is. Change it only to fix a typo, or if the user did not write it; a cuisine, "
             "religion, material or colour in front of the object becomes a property.")

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
        """Fix a typo if the token is close to a word used in SPOT's vocabulary. Ordinary
        English words are left alone ("county" is not "country"), and so are words that
        would need more than a slip of the finger ("brazil" is not "rail")."""
        if token in self._vocab_set or token in _COMMON_WORDS or token in _UNIT_WORDS or not token.isalpha() or len(token) < 4 \
                or token in self.colours or any(f in self._vocab_set for f in _singular_forms(token)):
            return token
        if token not in self._corrections:
            fixed = token
            # "shuld" is a misspelt "should", not a "shul".
            if not any(len(w) > 4 and abs(len(w) - len(token)) <= 1 and _edit_distance(token, w) <= 1
                       for w in _COMMON_WORDS):
                allowed = 1 if len(token) < 7 else 2
                scored = []
                for rank, match in enumerate(get_close_matches(token, self._vocab, n=4, cutoff=0.8)):
                    # the word itself or its plural ("caffes" is "cafes", so "cafe")
                    distance, form = min((_edit_distance(token, form), form)
                                         for form in (match, *_plural_forms(match)))
                    if distance <= allowed and _same_start(token, match) and len(form) >= len(token) - 1:
                        scored.append((distance, not _is_swap(token, match), rank, match))
                if scored:
                    fixed = min(scored)[-1]
            if len(self._corrections) >= 50_000:
                return fixed
            self._corrections[token] = fixed
        return self._corrections[token]

    def _split(self, token: str) -> tuple[str, str] | None:
        """Two words written as one, if together they are a descriptor ("firehose", "busstop")."""
        if token in self._vocab_set or not token.isalpha():
            return None
        for cut in range(3, len(token) - 2):
            first, second = token[:cut], token[cut:]
            if first in self._vocab_set and second in self._vocab_set \
                    and (f"{first} {second}" in self.entities or f"{first} {second}" in self.properties):
                return first, second
        return None

    def scan(self, text: str, max_words: int = 6) -> Scan:
        """SPOT object descriptors, property names and spatial terms in the text, typos corrected.
        Only the longest matches count ("irish bar", not also "bar"); plurals count ("bakeries");
        words right after a number are left alone (they are units: "300 miter")."""
        raw, tokens = [], []
        names = _proper_nouns(text)
        for word in re.findall(r"[^\W_]+(?:'[^\W_]+)?", norm(text).replace("\u2019", "'")):
            # A unit after a number ("300 miter") and a name ("Poland") are not typos.
            after_number = bool(raw) and (raw[-1][:1].isdigit() or raw[-1] in _NUMBER_WORDS)
            if after_number or word[0].isdigit():
                parts = (word,)
            else:
                parts = self._split(word) or ((word,) if word in names else (self._correct(word),))
            raw += [word] + [""] * (len(parts) - 1)        # what the user wrote, once
            tokens += parts

        def after_number(i: int) -> bool:
            return i > 0 and (raw[i - 1][:1].isdigit() or raw[i - 1] in _NUMBER_WORDS)

        spans = []
        for i in range(len(tokens)):
            for j in range(min(len(tokens), i + max_words), i, -1):
                phrase, wrote = " ".join(tokens[i:j]), " ".join(filter(None, raw[i:j]))
                if j - i == 1 and phrase in _UNIT_WORDS and after_number(i):
                    continue
                hit = None
                forms = _singular_forms(phrase)
                plain = _without_accents(phrase)                   # "cafés" is "cafes"
                for form in forms + ([f for f in _singular_forms(plain) if f not in forms] if plain != phrase else []):
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
        # "at" is too common to list on its own, but between two objects it is the spatial
        # term ("parking spaces at a gas station"), which is easily taken for "inside".
        starts = {s[0] for s in maximal if s[2] == "entity"}
        ends = {s[1] for s in maximal if s[2] == "entity"}
        for i, token in enumerate(tokens):
            after = i + 2 if i + 1 < len(tokens) and tokens[i + 1] in ("a", "an", "the") else i + 1
            if token == "at" and i in ends and after in starts:
                maximal.append((i, i + 1, "spatial", "at", "at", SPATIAL_TERMS["at"]))

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
                f"{', '.join(close)}. " + KEEP_HINT.format(name=name) + " " + BRAND_HINT)

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
            elif spec.is_colour and prop.get("value") and not self.colour_bundles(prop["value"]):
                # a shade of a listed colour ("dark orange") is read as that colour and kept
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
            f"{readings}. SPOT will search for the first one. " + KEEP_HINT.format(name=name)
            + " " + BRAND_HINT)


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
