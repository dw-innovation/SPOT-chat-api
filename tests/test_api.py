"""Tests for api.py (the HTTP API). The LLM is replaced by a scripted fake."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient            # noqa: E402

import api                                           # noqa: E402
from test_spot_chat import FakeLLM, tool_call_message  # noqa: E402

CLIENT = TestClient(api.app)

NEW_QUERY = tool_call_message(("new_query", {
    "area": {"type": "area", "value": "Cologne"},
    "entities": [{"id": 0, "name": "bar", "type": "nwr"}, {"id": 1, "name": "river", "type": "nwr"}],
    "relations": [{"source": 0, "target": 1, "type": "distance", "spatial_term": "next to"}]}))
REPLY = SimpleNamespace(content="Bars within 50 m of a river in Cologne.", tool_calls=None)


def use_llm(monkeypatch, responses):
    llm = FakeLLM(responses)
    monkeypatch.setattr(api, "client", llm)
    return llm


def test_health():
    body = CLIENT.get("/health").json()
    assert body["status"] == "ok" and "model" in body


def test_chat_first_turn(monkeypatch):
    use_llm(monkeypatch, [NEW_QUERY, REPLY])
    body = CLIENT.post("/chat", json={"message": "show me all bars next to rivers in cologne"}).json()
    assert body["reply"].startswith("Bars within 50 m")
    assert body["query"]["area"] == {"type": "area", "value": "Cologne"}
    assert [e["name"] for e in body["query"]["entities"]] == ["bar", "river"]
    assert body["query"]["relations"][0]["value"] == "50 m"
    assert body["changed"] is True
    assert [m["role"] for m in body["history"]] == ["user", "assistant"]
    assert body["tool_calls"][0]["name"] == "new_query" and body["tool_calls"][0]["ok"] is True
    assert "value: Cologne" in body["query_yaml"] and "spatial_term" not in body["query_yaml"]


def test_chat_follow_up_keeps_the_query(monkeypatch):
    llm = use_llm(monkeypatch, [
        tool_call_message(("add_properties", {"entity_id": 0, "properties": [{"name": "outdoor seating"}]})),
        SimpleNamespace(content="Added outdoor seating.", tool_calls=None)])
    first = {"area": {"type": "area", "value": "Cologne"},
             "entities": [{"id": 0, "name": "bar", "type": "nwr"}, {"id": 1, "name": "river", "type": "nwr"}],
             "relations": [{"source": 0, "target": 1, "type": "distance", "value": "50 m"}]}
    body = CLIENT.post("/chat", json={
        "message": "only ones with outdoor seating", "query": first,
        "history": [{"role": "user", "content": "bars next to rivers in cologne"},
                    {"role": "assistant", "content": "Here you go."}]}).json()
    assert body["query"]["entities"][0]["properties"] == [{"name": "outdoor seating"}]
    assert len(body["history"]) == 4
    sent = llm.calls[0]["messages"]                    # history and current query reach the model
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert "name: bar" in sent[-1]["content"]


def test_chat_accepts_yaml_and_reports_no_change(monkeypatch):
    use_llm(monkeypatch, [SimpleNamespace(content="A cluster means several of the same object.", tool_calls=None)])
    body = CLIENT.post("/chat", json={
        "message": "what does cluster mean?",
        "query_yaml": "area:\n  type: bbox\nentities:\n- id: 0\n  name: bar\n  type: nwr\n"}).json()
    assert body["changed"] is False and body["tool_calls"] == []
    assert body["query"]["entities"][0]["name"] == "bar"


def test_chat_errors(monkeypatch):
    assert CLIENT.post("/chat", json={"message": "   "}).status_code == 400
    assert CLIENT.post("/chat", json={"message": "hi", "query_yaml": "entities: [oops"}).status_code == 400

    class Broken:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            import openai
            raise openai.APIConnectionError(request=None)

    monkeypatch.setattr(api, "client", Broken())
    assert CLIENT.post("/chat", json={"message": "a bar"}).status_code == 502


def test_validate():
    ok = CLIENT.post("/validate", json={"query_yaml": "area:\n  type: bbox\nentities:\n- name: bars\n  type: nwr\n"}).json()
    assert ok["valid"] and ok["query"]["entities"][0]["id"] == 0
    bad = CLIENT.post("/validate", json={"query": {"area": {"type": "area"}, "entities": [{"name": "bar"}]}}).json()
    assert bad["valid"] is False and "value" in bad["error"]


def test_api_key_is_enforced_when_set(monkeypatch):
    monkeypatch.setattr(api, "API_KEY", "secret")
    assert CLIENT.post("/validate", json={"query_yaml": "area:\n  type: bbox\n"}).status_code == 401
    assert CLIENT.post("/validate", json={"query_yaml": "area:\n  type: bbox\n"},
                       headers={"X-API-Key": "secret"}).status_code == 200
