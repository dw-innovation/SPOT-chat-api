"""HTTP API for the SPOT query builder.

The server keeps no state: every request carries the current query and the chat
history, and gets the updated ones back. Run it with

    uvicorn api:app --host 0.0.0.0 --port 8000

See API.md for the request and response format.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from spot_chat.agent import ErrorEvent, ReplyEvent, ToolEvent, run_turn
from spot_chat.config import load_settings, make_client
from spot_chat.descriptors import get_descriptors
from spot_chat.query import QueryError, build_query, empty_query, from_yaml, to_export, to_yaml
from spot_chat.tag_search import get_tag_search

settings = load_settings()
client = make_client(settings)
API_KEY = os.getenv("SPOT_API_KEY", "").strip()          # optional; unset means open
CORS_ORIGINS = [o.strip() for o in os.getenv("SPOT_CORS_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(
    title="SPOT query builder",
    version="1.0",
    summary="Turns a chat message into a SPOT search query (area, entities, relations).",
)
app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request and response models
# --------------------------------------------------------------------------- #

class Message(BaseModel):
    role: str = Field(description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(description="What the user typed.")
    query: dict | None = Field(default=None, description="The current query; omit to start fresh.")
    query_yaml: str | None = Field(default=None, description="The current query as YAML (alternative to query).")
    history: list[Message] = Field(default_factory=list, description="Earlier turns, oldest first.")
    model: str | None = Field(default=None, description="Override the configured model.")


class ToolCall(BaseModel):
    name: str
    ok: bool
    message: str
    arguments: dict = {}
    warnings: list[str] = []
    notes: list[str] = []


class ChatResponse(BaseModel):
    reply: str
    query: dict
    query_yaml: str
    changed: bool
    history: list[Message]
    tool_calls: list[ToolCall]


class ValidateRequest(BaseModel):
    query: dict | None = None
    query_yaml: str | None = None


class ValidateResponse(BaseModel):
    valid: bool
    error: str | None = None
    query: dict | None = None
    query_yaml: str | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def check_key(given: str | None) -> None:
    if API_KEY and given != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header.")


def read_query(query: dict | None, query_yaml: str | None) -> dict:
    """Accept a query object or YAML, and validate/normalise it the same way the tools do."""
    try:
        if query_yaml:
            return from_yaml(query_yaml)
        if query:
            return build_query(query.get("area"), query.get("entities") or [], query.get("relations") or [])
    except QueryError as exc:
        raise HTTPException(status_code=400, detail=f"The query is not valid: {exc}") from None
    return empty_query()


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health")
def health() -> dict:
    descriptors = get_descriptors()
    tag_search = get_tag_search()
    return {
        "status": "ok",
        "model": settings.model,
        "models": settings.models,
        "llm_endpoint_configured": bool(settings.base_url),
        "llm_api_key_configured": bool(settings.api_key),
        "descriptors_loaded": len(descriptors.entities) if descriptors else 0,
        "tag_search_url": tag_search.url if tag_search else None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, x_api_key: str | None = Header(default=None)) -> ChatResponse:
    """One chat turn: runs the tools and returns the updated query and reply."""
    check_key(x_api_key)
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty.")

    query = read_query(request.query, request.query_yaml)
    before = to_export(query)
    conversation = [m.model_dump() for m in request.history]
    reply, tool_calls = "", []

    for event in run_turn(client, settings, conversation, request.message, query, model=request.model):
        query = event.query
        if isinstance(event, ToolEvent):
            result = event.result
            tool_calls.append(ToolCall(
                name=result.name, ok=result.ok, message=result.message,
                arguments=result.arguments, warnings=result.warnings, notes=result.notes,
            ))
        elif isinstance(event, ReplyEvent):
            reply = event.text
        elif isinstance(event, ErrorEvent):
            raise HTTPException(status_code=502, detail=event.text)

    history = request.history + [Message(role="user", content=request.message),
                                 Message(role="assistant", content=reply)]
    return ChatResponse(
        reply=reply,
        query=to_export(query),
        query_yaml=to_yaml(query),
        changed=to_export(query) != before,
        history=history,
        tool_calls=tool_calls,
    )


@app.post("/validate", response_model=ValidateResponse)
def validate(request: ValidateRequest, x_api_key: str | None = Header(default=None)) -> ValidateResponse:
    """Check and normalise a query the user edited by hand (or in a graph editor)."""
    check_key(x_api_key)
    try:
        query = read_query(request.query, request.query_yaml)
    except HTTPException as exc:
        return ValidateResponse(valid=False, error=str(exc.detail))
    return ValidateResponse(valid=True, query=to_export(query), query_yaml=to_yaml(query))
