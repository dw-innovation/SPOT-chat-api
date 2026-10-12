"""HTTP API for the SPOT query builder.

The server keeps no state: every request carries the current query and the chat
history, and gets the updated ones back. Run it with

    uvicorn api:app --host 0.0.0.0 --port 8000

See API.md for the request and response format.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from collections import deque
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from spot_chat.agent import ErrorEvent, ReplyEvent, ToolEvent, run_turn
from spot_chat.config import load_settings, make_client
from spot_chat.descriptors import get_descriptors
from spot_chat.query import QueryError, build_query, empty_query, from_yaml, to_export, to_yaml
from spot_chat.tag_search import get_tag_search

settings = load_settings()
client = make_client(settings)
API_KEY = os.getenv("SPOT_API_KEY", "").strip()          # optional; unset means open
CORS_ORIGINS = [o.strip() for o in os.getenv("SPOT_CORS_ORIGINS", "*").split(",") if o.strip()]
# Debug mode shows what a public deployment keeps to itself: /docs, the tool calls of a turn
# and the configuration in /health.
DEBUG = os.getenv("SPOT_DEBUG", "").strip().lower() in ("1", "true", "yes")
RATE_LIMIT = int(os.getenv("SPOT_RATE_LIMIT", "30"))     # /chat calls per minute and user; 0 = off

MAX_BODY_BYTES = 256_000        # one request, including history and query
MAX_MESSAGE_CHARS = 2_000
MAX_HISTORY_MESSAGES = 200      # the model only sees the latest ones (Settings.history_turns)
MAX_USER_CHARS = 80             # of the X-Spot-User header, see check_rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("spot.api")
if not API_KEY:
    log.warning("SPOT_API_KEY is not set: anyone who can reach this server can use it.")
if "*" in CORS_ORIGINS:
    log.warning("SPOT_CORS_ORIGINS is '*': any website can call this server from a browser.")

app = FastAPI(
    title="SPOT query builder",
    version="1.0",
    summary="Turns a chat message into a SPOT search query (area, entities, relations).",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
    openapi_url="/openapi.json" if DEBUG else None,
)


def error_response(status_code: int, message: str, detail=None, headers=None) -> JSONResponse:
    """Every error has the same body. `status` and `message` are what the other SPOT services
    send and what the frontend's proxy reads; `detail` is FastAPI's own field."""
    body = {"status": "error", "message": message, "detail": message if detail is None else detail}
    return JSONResponse(body, status_code=status_code, headers=headers)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return error_response(exc.status_code, str(exc.detail), headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422: `detail` stays FastAPI's list of fields, `message` says the same in one line."""
    errors = exc.errors()
    message = "; ".join(
        f"{'.'.join(str(part) for part in error['loc'][1:]) or 'body'}: {error['msg']}" for error in errors)
    return error_response(422, message, detail=jsonable_encoder(errors))


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Refuse oversized requests before they are read. A request without a length cannot be
    checked, so it is refused too (browsers and HTTP libraries always send one)."""
    if request.method == "POST":
        length = request.headers.get("content-length", "")
        if not length.isdigit():
            return error_response(411, "A Content-Length header is required.")
        if int(length) > MAX_BODY_BYTES:
            return error_response(413, f"The request is larger than {MAX_BODY_BYTES} bytes.")
    return await call_next(request)


# Added last, so it is the outermost layer and its headers are also on the errors above.
app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request and response models
# --------------------------------------------------------------------------- #

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    refusal: str | None = Field(
        default=None, max_length=40,
        description="Set by the server on the reply of a declined turn. Send it back unchanged: "
                    "it is what keeps a chat closed after a harmful request.",
    )


class ChatRequest(BaseModel):
    message: str = Field(max_length=MAX_MESSAGE_CHARS, description="What the user typed.")
    query: dict | None = Field(default=None, description="The current query; omit to start fresh.")
    query_yaml: str | None = Field(default=None, description="The current query as YAML (alternative to query).")
    history: list[Message] = Field(
        default_factory=list, max_length=MAX_HISTORY_MESSAGES, description="Earlier turns, oldest first.")
    model: str | None = Field(default=None, description="One of the models the server offers (LLM_MODELS).")


class ToolCall(BaseModel):
    name: str
    ok: bool
    message: str
    arguments: dict = {}
    warnings: list[str] = []
    notes: list[str] = []


class Refusal(BaseModel):
    category: str = Field(description="self_harm, violence, targeting_people, crime, hate or off_topic")
    locked: bool = Field(description="True: the chat is closed until it is reset (empty history).")
    url: str | None = Field(default=None, description="A link to show with the reply (self_harm: helpline directory).")


class Limitation(BaseModel):
    category: str
    detail: str = Field(description="The user's words for the part that could not be expressed.")
    message: str = Field(description="A fixed explanation that can be shown to the user.")


class ChatResponse(BaseModel):
    reply: str
    query: dict
    query_yaml: str
    changed: bool
    history: list[Message]
    tool_calls: list[ToolCall]
    refusal: Refusal | None = Field(default=None, description="Set when the message was declined.")
    limitations: list[Limitation] = Field(
        default_factory=list, description="Parts of the request the query could not express.")


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
    if API_KEY and not hmac.compare_digest((given or "").encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header.")


class RateLimiter:
    """At most `limit` calls per `window` seconds and client, counted in memory. Every worker
    process counts on its own, so this stops a runaway client, not a determined one."""

    def __init__(self, window: float = 60.0, max_clients: int = 10_000):
        self.window, self.max_clients = window, max_clients
        self.calls: dict[str, deque] = {}
        self.lock = threading.Lock()

    def wait_time(self, client_id: str, limit: int) -> int:
        """Record a call. Returns 0 if it is allowed, otherwise the seconds until it would be."""
        now = time.monotonic()
        with self.lock:
            if len(self.calls) > self.max_clients:        # forget clients that have gone quiet
                self.calls = {c: t for c, t in self.calls.items() if now - t[-1] < self.window}
            times = self.calls.setdefault(client_id, deque())
            while times and now - times[0] >= self.window:
                times.popleft()
            if len(times) >= limit:
                return int(self.window - (now - times[0])) + 1
            times.append(now)
            return 0


rate_limiter = RateLimiter()


def check_rate(request: Request, user: str | None = None) -> None:
    """Counts per user if the caller names one, otherwise per client address. The name is only
    believed from a caller that has shown the API key: behind the SPOT frontend all requests
    come from the frontend's server, so the address says nothing about the user."""
    if RATE_LIMIT <= 0:
        return
    if API_KEY and user and user.strip():
        client_id = "user:" + user.strip()[:MAX_USER_CHARS]
    else:
        client_id = request.client.host if request.client else "unknown"
    wait = rate_limiter.wait_time(client_id, RATE_LIMIT)
    if wait:
        log.warning("chat outcome=rate_limited")
        raise HTTPException(status_code=429, detail="Too many messages. Please wait a moment.",
                            headers={"Retry-After": str(wait)})


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
    if not DEBUG:
        return {"status": "ok"}
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
def chat(request: ChatRequest, http: Request, x_api_key: str | None = Header(default=None),
         x_spot_user: str | None = Header(default=None)) -> ChatResponse:
    """One chat turn: runs the tools and returns the updated query and reply."""
    check_key(x_api_key)
    check_rate(http, x_spot_user)
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty.")
    if request.model and request.model not in settings.models:
        raise HTTPException(status_code=400, detail="This model is not available.")
    started = time.monotonic()

    query = read_query(request.query, request.query_yaml)
    before = to_export(query)
    conversation = [m.model_dump() for m in request.history]
    reply, tool_calls, refusal, limitations = "", [], None, []

    for event in run_turn(client, settings, conversation, request.message, query, model=request.model):
        query = event.query
        if isinstance(event, ToolEvent):
            result = event.result
            tool_calls.append(ToolCall(
                name=result.name, ok=result.ok, message=result.message,
                arguments=result.arguments, warnings=result.warnings, notes=result.notes,
            ))
        elif isinstance(event, ReplyEvent):
            reply, refusal, limitations = event.text, event.refusal, event.limitations
        elif isinstance(event, ErrorEvent):
            log.error("chat outcome=%s detail=%r", "timeout" if event.timed_out else "llm_error", event.detail)
            raise HTTPException(status_code=504 if event.timed_out else 502, detail=event.text)

    # Counts and categories only: what users write is not logged.
    log.info("chat outcome=%s locked=%s limitations=%s tools=%d history=%d chars=%d seconds=%.1f",
             refusal.category if refusal else "ok", bool(refusal and refusal.locked),
             ",".join(item["category"] for item in limitations) or "-", len(tool_calls),
             len(request.history), len(request.message), time.monotonic() - started)

    # The refusal marker on the reply is how a lock travels: run_turn finds it in the history.
    history = request.history + [
        Message(role="user", content=request.message),
        Message(role="assistant", content=reply, refusal=refusal.category if refusal else None)]
    return ChatResponse(
        reply=reply,
        query=to_export(query),
        query_yaml=to_yaml(query),
        changed=to_export(query) != before,
        history=history,
        tool_calls=tool_calls if DEBUG else [],
        refusal=Refusal(category=refusal.category, locked=refusal.locked, url=refusal.url) if refusal else None,
        limitations=limitations,
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
