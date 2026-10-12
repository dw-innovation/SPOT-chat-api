"""Tests for SPOT's tag search API integration and the transcription-guideline rules.
The API is mocked with a real response captured from tags.findthatspot.io."""

import json
from types import SimpleNamespace

import pytest

import spot_chat.descriptors as descriptors_module
from spot_chat import tag_search
from spot_chat.query import empty_query
from spot_chat.tools import execute_tool
from spot_chat.tag_search import Match, TagSearch, parse_results
from test_spot_chat import cafe_query, needs_tables, run

REAL_RESPONSE_FOR_A = json.loads("""[
 {"imr":[{"or":[{"key":"emergency","operator":"=","value":"defibrillator"}]}],"cluster_id":"4b73","name":"aed","score":8.27,
  "descriptors":["emergency cardiac aid","heart starter","aed","pulse generator","cardiac resuscitator","cardiac device","defibrillator"]},
 {"imr":[{"or":[{"key":"building","operator":"=","value":"***any***"}]}],"cluster_id":"cf0b","name":"a building","score":6.77,
  "descriptors":["some kind of building","building","facility","a building"]},
 {"imr":[],"cluster_id":"de7a","name":"is in a cutting","score":4.97,"descriptors":["is in a cutting","running through a cutting"]},
 {"imr":[],"cluster_id":"de7a","name":"running through a cutting","score":4.97,"descriptors":["is in a cutting","running through a cutting"]}
]""")


class FakeTagSearch:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def search(self, word):
        self.calls.append(word)
        return self.answers.get(word, ())


def with_remote(monkeypatch, answers):
    fake = FakeTagSearch(answers)
    monkeypatch.setattr(tag_search, "get_tag_search", lambda: fake)
    return fake


# --------------------------------------------------------------------------- #
# Tag search API
# --------------------------------------------------------------------------- #

def test_parse_results_dedupes_bundles():
    assert [m.name for m in parse_results(REAL_RESPONSE_FOR_A)] == ["aed", "a building", "is in a cutting"]


def test_remote_note_says_what_spot_will_search(monkeypatch):
    with_remote(monkeypatch, {"a": parse_results(REAL_RESPONSE_FOR_A)})
    notes = descriptors_module.notes_for({"entities": [{"id": 0, "name": "a", "type": "nwr"}]})
    assert len(notes) == 1
    assert "reads it as: 'aed'" in notes[0] and "search for the first one" in notes[0]


def test_remote_exact_descriptor_and_empty_results(monkeypatch):
    with_remote(monkeypatch, {"boozer": (Match("pub", ("pub", "bar", "boozer"), "7", 9.0),), "zzqx": ()})
    notes = descriptors_module.notes_for({"entities": [
        {"id": 0, "name": "boozer", "type": "nwr"}, {"id": 1, "name": "zzqx", "type": "nwr"}]})
    assert len(notes) == 1 and "finds nothing" in notes[0] and "tell the user" in notes[0]


def test_known_names_and_brands_skip_the_api(monkeypatch):
    fake = with_remote(monkeypatch, {})
    descriptors_module.notes_for({"entities": [
        {"id": 0, "name": "bar", "type": "nwr"}, {"id": 1, "name": "brand:KFC", "type": "nwr"}]})
    assert fake.calls == []


@needs_tables
def test_unreachable_api_falls_back_to_local(monkeypatch):
    monkeypatch.setattr(tag_search, "get_tag_search", lambda: SimpleNamespace(search=lambda word: None))
    notes = descriptors_module.notes_for({"entities": [{"id": 0, "name": "backery", "type": "nwr"}]})
    assert "Closest descriptors: bakery" in notes[0]


def test_tag_search_client_caches_and_backs_off(monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(REAL_RESPONSE_FOR_A).encode()

    def fake_urlopen(url, timeout):
        calls.append(url)
        if "fail" in url:
            raise OSError("network down")
        return Response()

    monkeypatch.setattr(tag_search.urllib.request, "urlopen", fake_urlopen)
    client = TagSearch("https://example.org/api/search")
    assert client.search("A")[0].name == "aed" and client.search("a")[0].name == "aed"
    assert len(calls) == 1 and "word=a" in calls[0] and "limit=5" in calls[0]
    assert client.search("fail") is None and client.search("other") is None   # backing off
    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# Transcription guideline rules
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("max_distance, expected", [
    ("next to each other", "50 m"), ("close to one another", "100 m"),
    ("side by side", "100 m"), ("250 meters", "250 m"),
])
def test_cluster_distance_accepts_spatial_terms(max_distance, expected):
    query, result = run(empty_query(), "new_query", area={"type": "bbox"}, relations=[], entities=[
        {"id": 0, "name": "bench", "type": "cluster", "minPoints": 3, "maxDistance": max_distance}])
    assert result.ok, result.message
    assert query["entities"][0]["maxDistance"] == expected


def test_cluster_distance_defaults_follow_guidelines():
    # only a contains relation -> default 50 m
    query, _ = run(empty_query(), "new_query", area={"type": "bbox"}, entities=[
        {"id": 0, "name": "park", "type": "nwr"}, {"id": 1, "name": "bench", "type": "cluster", "minPoints": 3}],
        relations=[{"source": 0, "target": 1, "type": "contains"}])
    assert query["entities"][1]["maxDistance"] == "50 m"
    # distance relation -> that relation's value ("three benches 200 m from a kiosk")
    query, _ = run(empty_query(), "new_query", area={"type": "bbox"}, entities=[
        {"id": 0, "name": "bench", "type": "cluster", "minPoints": 3}, {"id": 1, "name": "kiosk", "type": "nwr"}],
        relations=[{"source": 0, "target": 1, "type": "distance", "value": "200 m"}])
    assert query["entities"][0]["maxDistance"] == "200 m"


def test_size_limits_warn():
    entities = [{"id": i, "name": n, "type": "nwr"} for i, n in enumerate(["bar", "park", "kiosk", "bench"])]
    _, result = run(empty_query(), "new_query", area={"type": "bbox"}, entities=entities, relations=[])
    assert result.ok and any("at most 3" in w for w in result.warnings)
    props = [{"name": n} for n in ["outdoor seating", "vegan", "halal", "wi-fi"]]
    _, result = run(cafe_query(), "add_properties", entity_id=0, properties=props)
    assert result.ok and any("at most 3 per object" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# Fixes from the first benchmark run (chat-01)
# --------------------------------------------------------------------------- #

@needs_tables
def test_message_scan_finds_the_users_own_descriptors():
    d = descriptors_module.get_descriptors()
    objects, properties = d.find_in_text(
        "find a lockable bike parking 5 yards from the ski trail and 2 irish bars with outdoor seating")
    assert objects == ["lockable bike parking", "ski trail", "irish bar"]   # longest match, singular
    assert properties == ["outdoor seating"]
    assert "yard" not in objects                                            # unit after a number


@needs_tables
def test_user_turn_lists_descriptors_from_the_message():
    from spot_chat.prompts import user_turn
    text = user_turn("a pv farm next to a river", empty_query())
    assert "- objects: PV farm, river" in text   # acronyms keep their capitals


def test_table_terms_always_get_their_table_distance():
    query, result = run(empty_query(), "new_query", area={"type": "bbox"}, entities=[
        {"id": 0, "name": "bar", "type": "nwr"}, {"id": 1, "name": "river", "type": "nwr"}],
        relations=[{"source": 0, "target": 1, "type": "distance", "spatial_term": "near", "value": "100 m"}])
    assert result.ok and query["relations"][0]["value"] == "250 m"


def test_spatial_term_sets_the_distance_but_is_not_exported():
    from spot_chat.query import to_export, to_yaml
    query, _ = run(empty_query(), "new_query", area={"type": "bbox"}, entities=[
        {"id": 0, "name": "bar", "type": "nwr"}, {"id": 1, "name": "river", "type": "nwr"}],
        relations=[{"source": 0, "target": 1, "type": "distance", "spatial_term": "nearby"}])
    assert query["relations"][0]["spatial_term"] == "nearby"                 # used internally
    assert to_export(query)["relations"][0] == {"source": 0, "target": 1, "type": "distance", "value": "250 m"}
    assert "spatial_term" not in to_yaml(query)


# --------------------------------------------------------------------------- #
# Fixes from the second benchmark run: typos, split descriptors, clusters
# --------------------------------------------------------------------------- #

@needs_tables
def test_scan_corrects_typos_but_not_units():
    found = descriptors_module.get_descriptors().scan(
        "an air twer 300 miter from a pertrol statsion, the ptrl station is a block away, around 1 km")
    assert ("air tower", "air twer") in found.objects
    assert [o for o, _ in found.objects].count("petrol station") == 1
    assert "minster" not in [o for o, _ in found.objects]                 # "300 miter" is a unit
    assert [(t, d) for t, d, _ in found.spatial] == [("a block away", "250 m")]   # not "around 1 km"


@needs_tables
def test_split_descriptor_gets_a_rename_note():
    message = "find me a japanese teahouse within 150 meters from a frozen yoghurt shop"
    args = {"area": {"type": "bbox"}, "relations": [{"source": 0, "target": 1, "type": "distance", "value": "150 m"}],
            "entities": [{"id": 0, "name": "teahouse", "type": "nwr",
                          "properties": [{"name": "cuisine", "operator": "=", "value": "japanese"}]},
                         {"id": 1, "name": "frozen yoghurt shop", "type": "nwr"}]}
    _, result = execute_tool(empty_query(), "new_query", json.dumps(args), message)
    assert len(result.notes) == 1
    assert "name='japanese teahouse'" in result.notes[0] and "cuisine = japanese" in result.notes[0]
    _, without_message = execute_tool(empty_query(), "new_query", json.dumps(args))
    assert without_message.notes == []


@needs_tables
def test_agent_passes_the_user_message_to_the_checks():
    from test_spot_chat import FakeLLM, tool_call_message
    from spot_chat.agent import ToolEvent, run_turn
    from spot_chat.config import Settings
    llm = FakeLLM([
        tool_call_message(("new_query", {"area": {"type": "bbox"}, "relations": [],
                                         "entities": [{"id": 0, "name": "station", "type": "nwr"}]})),
        SimpleNamespace(content="Done.", tool_calls=None),
    ])
    events = list(run_turn(llm, Settings(api_key="x", base_url=None, model="fake"), [],
                           "find a pertrol statsion", empty_query()))
    tool_event = next(e for e in events if isinstance(e, ToolEvent))
    assert any("name='petrol station'" in n for n in tool_event.result.notes)


@needs_tables
def test_synonym_of_the_users_descriptor_gets_a_note():
    message = "find me a city fire plug and an irish bar within 2 miles of each other"
    args = {"area": {"type": "bbox"}, "relations": [{"source": 0, "target": 1, "type": "distance", "value": "2 mi"}],
            "entities": [{"id": 0, "name": "fire hydrant", "type": "nwr"}, {"id": 1, "name": "irish bar", "type": "nwr"}]}
    _, result = execute_tool(empty_query(), "new_query", json.dumps(args), message)
    assert len(result.notes) == 1 and "name='city fire plug'" in result.notes[0]


@needs_tables
def test_scan_ignores_descriptors_starting_with_an_article():
    objects, _ = descriptors_module.get_descriptors().find_in_text("show me a building next to a park")
    assert "a building" not in objects and "building" in objects
