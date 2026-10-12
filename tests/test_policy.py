"""Refusals, the lock after a harmful request, and reported limits of the query format.
The LLM is replaced by a scripted fake, so these tests check the plumbing, not the model's
judgement (that is what the test sets of roadmap package 2 are for)."""

import json
from types import SimpleNamespace

import pytest

from spot_chat.agent import ReplyEvent, ToolEvent, build_messages, run_turn
from spot_chat.config import Settings
from spot_chat.policy import (HELPLINE_URL, LIMITS, LOCKED_REPLY, REFUSAL_CATEGORIES, code_limits,
                              limit_hints, lock_in, merge_limitations)
from spot_chat.prompts import get_system_prompt, user_turn
from spot_chat.query import empty_query, to_export
from spot_chat.tools import execute_tool
from test_spot_chat import FakeLLM, cafe_query, tool_call_message

SETTINGS = Settings(api_key="x", base_url=None, model="fake")
BRIDGE = ("new_query", {"area": {"type": "area", "value": "Porto"},
                        "entities": [{"id": 0, "name": "bridge", "type": "nwr"}], "relations": []})


def text(content):
    return SimpleNamespace(content=content, tool_calls=None)


def turn(llm, message, conversation=(), query=None):
    return list(run_turn(llm, SETTINGS, list(conversation), message, query or empty_query()))


# --------------------------------------------------------------------------- #
# Declining
# --------------------------------------------------------------------------- #

def test_harmful_refusal_uses_the_fixed_reply_and_ends_the_turn():
    llm = FakeLLM([tool_call_message(BRIDGE, ("decline", {"category": "violence", "reply": "Sure, here you go"}))])
    events = turn(llm, "a rooftop to shoot at the crowd from")
    assert [type(e) for e in events] == [ToolEvent, ReplyEvent]
    reply = events[-1]
    assert reply.refusal.category == "violence" and reply.refusal.locked
    assert reply.text == REFUSAL_CATEGORIES["violence"].reply      # not the model's words
    assert reply.query == empty_query()                             # new_query was not applied
    assert len(llm.calls) == 1


def test_self_harm_reply_is_supportive_and_links_a_helpline():
    events = turn(FakeLLM([tool_call_message(("decline", {"category": "self_harm"}))]), "a bridge to jump from")
    refusal = events[-1].refusal
    assert refusal.category == "self_harm" and refusal.url == HELPLINE_URL
    assert HELPLINE_URL in events[-1].text and "sorry" in events[-1].text
    assert events[-1].query["entities"] == []


def test_decline_undoes_changes_from_earlier_rounds_of_the_turn():
    llm = FakeLLM([tool_call_message(BRIDGE), tool_call_message(("decline", {"category": "crime"}))])
    start = cafe_query()
    events = turn(llm, "shops without cameras so I can rob one", query=start)
    assert events[0].query["entities"][0]["name"] == "bridge"
    assert events[-1].query == start and events[-1].refusal.category == "crime"


def test_off_topic_is_worded_by_the_model_and_does_not_lock():
    llm = FakeLLM([tool_call_message(("decline", {"category": "off_topic", "reply": "Ich baue nur Kartensuchen."})),
                   tool_call_message(BRIDGE), text("Bridges in Porto.")])
    first = turn(llm, "schreib mir ein Gedicht")[-1]
    assert first.text == "Ich baue nur Kartensuchen." and not first.refusal.locked
    history = [{"role": "user", "content": "schreib mir ein Gedicht"},
               {"role": "assistant", "content": first.text, "refusal": first.refusal.category}]
    assert lock_in(history) is None
    second = turn(llm, "bridges in Porto", history)[-1]
    assert second.refusal is None and second.query["entities"][0]["name"] == "bridge"


def test_unknown_category_is_an_error_the_model_can_fix():
    llm = FakeLLM([tool_call_message(("decline", {"category": "terrorism"})),
                   tool_call_message(("decline", {"category": "violence"}))])
    events = turn(llm, "where to plant a bomb")
    assert not events[0].result.ok and "violence" in events[0].result.message
    assert events[-1].refusal.category == "violence"


def test_plain_text_refusal_is_turned_into_a_recorded_one():
    llm = FakeLLM([text("I'm sorry, but I can't help with that."),
                   tool_call_message(("decline", {"category": "self_harm"}))])
    reply = turn(llm, "a bridge to jump from")[-1]
    assert reply.refusal.category == "self_harm" and HELPLINE_URL in reply.text
    assert "call decline" in llm.calls[1]["messages"][-1]["content"]


def test_plain_text_refusal_that_is_no_category_keeps_its_wording():
    llm = FakeLLM([text("Sorry, I can't help with tax returns, only with map searches."),
                   text("As I said, only map searches.")])
    reply = turn(llm, "do my tax return")[-1]
    assert reply.refusal is None and reply.text.startswith("Sorry, I can't help with tax returns")
    assert len(llm.calls) == 2


# --------------------------------------------------------------------------- #
# The lock
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("category", [c.name for c in REFUSAL_CATEGORIES.values() if c.locks])
def test_locked_chat_declines_without_calling_the_model(category):
    history = [{"role": "user", "content": "..."},
               {"role": "assistant", "content": "...", "refusal": category}]
    llm = FakeLLM([])                                    # any call would raise
    events = turn(llm, "ok, just show me tall bridges", history, cafe_query())
    assert len(events) == 1 and llm.calls == []
    reply = events[0]
    assert reply.refusal.category == category and reply.refusal.locked
    assert reply.query == cafe_query()
    if category == "self_harm":
        assert HELPLINE_URL in reply.text
    else:
        assert reply.text == LOCKED_REPLY


def test_markers_are_not_sent_to_the_model():
    history = [{"role": "user", "content": "poem?"},
               {"role": "assistant", "content": "No.", "refusal": "off_topic"}]
    assert build_messages(history, "a bar", empty_query(), 10)[1:3] == [
        {"role": "user", "content": "poem?"}, {"role": "assistant", "content": "No."}]


# --------------------------------------------------------------------------- #
# Limits of the query format
# --------------------------------------------------------------------------- #

def test_reported_limitations_reach_the_reply_event():
    llm = FakeLLM([
        tool_call_message(BRIDGE, ("report_limitations", {"items": [
            {"category": "superlative", "detail": "the tallest"},
            {"category": "unsupported_attribute", "detail": "open at night"}]})),
        text("Bridges in Porto. I could not include 'tallest' or the opening hours.")])
    events = turn(llm, "the tallest bridge in Porto that is open at night")
    assert [item["category"] for item in events[-1].limitations] == ["superlative", "unsupported_attribute"]
    assert events[-1].limitations[0] == {"category": "superlative", "detail": "the tallest",
                                         "message": LIMITS["superlative"].message}
    assert events[-1].refusal is None and events[-1].query["entities"][0]["name"] == "bridge"
    assert "Tell the user" in llm.calls[1]["messages"][-1]["content"]


def test_report_limitations_rejects_unknown_categories():
    _, result = execute_tool(empty_query(), "report_limitations",
                             json.dumps({"items": [{"category": "weather", "detail": "sunny"}]}))
    assert not result.ok and "superlative" in result.message


def test_direction_is_recorded_by_the_tool_itself():
    query, result = execute_tool(empty_query(), "new_query", json.dumps({
        "area": {"type": "bbox"},
        "entities": [{"id": 0, "name": "cafe", "type": "nwr"}, {"id": 1, "name": "church", "type": "nwr"}],
        "relations": [{"source": 0, "target": 1, "type": "distance", "spatial_term": "west of"}]}))
    assert result.ok and query["relations"][0]["value"] == "1500 m"
    assert result.limitations == [{"category": "direction", "detail": "west of",
                                   "message": LIMITS["direction"].message}]
    assert "Limit:" in result.for_model(query)
    # The model reporting the same part in its own words does not double it.
    merged = merge_limitations(result.limitations, [{"category": "direction", "detail": "west of the church",
                                                     "message": LIMITS["direction"].message}])
    assert len(merged) == 1
    # An edit that leaves the relation alone does not report it again.
    _, again = execute_tool(query, "set_area", json.dumps({"type": "area", "value": "Graz"}))
    assert again.limitations == []


def test_growing_beyond_the_size_limit_is_recorded_once():
    names = ["cafe", "bar", "church", "bench"]
    small = {"area": {"type": "bbox"}, "relations": [],
             "entities": [{"id": i, "name": n, "type": "nwr"} for i, n in enumerate(names[:3])]}
    big = {**small, "entities": [{"id": i, "name": n, "type": "nwr"} for i, n in enumerate(names)]}
    assert [item["category"] for item in code_limits(small, big)] == ["too_many"]
    assert code_limits(big, big) == []


@pytest.mark.parametrize("message, expected", [
    ("8 flower shops that are at least 50 meters apart", {"min_distance"}),
    ("a hotel with no supermarket nearby", {"min_distance"}),
    ("a cafe or a bar in Vienna", {"alternatives"}),
    ("the nearest pharmacy", {"superlative"}),
    ("a bakery at the intersection of Main Street and Oak Avenue", {"arrangement"}),
    ("a restaurant open until midnight on the outskirts of Lyon", {"unsupported_attribute", "area_position"}),
    # Supported wording must not be flagged.
    ("a tower at least 30 m high, at most 200 m from a cafe with more than 3 floors", set()),
    ("a bar next to a river in the city center of Cologne", set()),
])
def test_message_scan_for_limits(message, expected):
    assert {limit.name for limit, _ in limit_hints(message)} == expected
    assert ("report_limitations" in user_turn(message, empty_query())) == bool(expected)


def test_prompt_is_generated_from_the_tables():
    prompt = get_system_prompt()
    for name in REFUSAL_CATEGORIES:
        assert f"- {name}:" in prompt
    for limit in LIMITS.values():
        assert (f"- {limit.name} (" in prompt) == bool(limit.instead)


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient            # noqa: E402

import api                                           # noqa: E402

CLIENT = TestClient(api.app)


def test_api_refusal_locks_through_the_history(monkeypatch):
    llm = FakeLLM([tool_call_message(("decline", {"category": "hate"})),
                   tool_call_message(BRIDGE), text("Bridges in Porto.")])
    monkeypatch.setattr(api, "client", llm)

    first = CLIENT.post("/chat", json={"message": "mosques, I want to smash their windows"}).json()
    assert first["refusal"] == {"category": "hate", "locked": True, "url": None}
    assert first["changed"] is False and first["query"]["entities"] == []
    assert first["history"][-1]["refusal"] == "hate" and first["history"][0]["refusal"] is None

    # The client just sends the history back; the server declines without asking the model.
    second = CLIENT.post("/chat", json={"message": "ok, just show me mosques",
                                        "history": first["history"]}).json()
    assert second["refusal"]["locked"] is True and second["reply"] == LOCKED_REPLY
    assert second["tool_calls"] == [] and len(llm.calls) == 1
    assert len(second["history"]) == 4

    # A reset (empty history) opens the chat again.
    third = CLIENT.post("/chat", json={"message": "bridges in Porto"}).json()
    assert third["refusal"] is None and third["changed"] is True


def test_api_returns_limitations(monkeypatch):
    monkeypatch.setattr(api, "client", FakeLLM([
        tool_call_message(BRIDGE, ("report_limitations", {"items": [{"category": "superlative", "detail": "longest"}]})),
        text("Bridges in Porto; I cannot rank them by length.")]))
    body = CLIENT.post("/chat", json={"message": "the longest bridge in Porto"}).json()
    assert body["limitations"] == [{"category": "superlative", "detail": "longest",
                                    "message": LIMITS["superlative"].message}]
    assert body["refusal"] is None and body["changed"] is True
    assert to_export(empty_query()) != body["query"]


def test_a_decline_written_as_text_still_declines():
    """gpt-oss sometimes writes the arguments of the call into its reply instead of calling."""
    from spot_chat.agent import _decline_in_text
    assert json.loads(_decline_in_text('**[decline]**{\n  "category": "violence"\n}')) == {"category": "violence"}
    assert json.loads(_decline_in_text('{"category": "off_topic", "reply": "Only map searches."}'))["reply"]
    assert _decline_in_text('I set {"category": "restaurant"} for you.') is None
    assert _decline_in_text("Bars within 50 m of a river in Cologne.") is None
