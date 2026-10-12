"""The chat loop: send the conversation to the LLM, run its tool calls, repeat.

`run_turn` is a generator of events so the UI can show every tool call as it
happens, while `evaluate.py` can simply drain it.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Iterator

import openai

from .config import Settings
from .policy import REFUSAL_CATEGORIES, Refusal, lock_in, merge_limitations, refusal_for
from .prompts import get_system_prompt, quoted_prompt, user_turn
from .tools import ToolResult, execute_tool, tool_specs


@dataclass
class ToolEvent:
    result: ToolResult
    query: dict


@dataclass
class ReplyEvent:
    text: str
    query: dict
    refusal: Refusal | None = None                          # the turn was declined
    limitations: list[dict] = field(default_factory=list)   # parts the query cannot express


@dataclass
class ErrorEvent:
    text: str            # safe to show to anyone
    query: dict
    detail: str = ""     # the provider's error text, for logs and the evaluation scripts
    timed_out: bool = False   # the turn ran into Settings.turn_timeout


Event = ToolEvent | ReplyEvent | ErrorEvent


HISTORY_ROLES = ("user", "assistant")
MODEL_UNREACHABLE = "The language model could not be reached. Please try again in a moment."
TURN_TOO_SLOW = "That took too long. Please try again, or describe the search in fewer words."


def build_messages(conversation: list[dict], user_message: str, query: dict, history_turns: int) -> list[dict]:
    """System prompt + recent text-only history + the new message with the current query."""
    # The history comes from the client. Only chat turns are passed on, so a forged `system`
    # or `tool` turn never reaches the model.
    turns = [turn for turn in conversation
             if turn.get("role") in HISTORY_ROLES and isinstance(turn.get("content"), str)]
    recent = turns[-2 * history_turns:] if history_turns else []
    return [
        {"role": "system", "content": get_system_prompt()},
        *({"role": turn["role"], "content": turn["content"]} for turn in recent),   # no markers
        {"role": "user", "content": user_turn(user_message, query)},
    ]


def _assistant_message(message) -> dict:
    """Plain-dict copy of an assistant message with tool calls (works across providers)."""
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"},
            }
            for call in message.tool_calls
        ],
    }


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _clean_reply(text: str | None) -> str:
    """Drop <think> blocks that some local reasoning models put into the content."""
    return _THINK_RE.sub("", text or "").strip()


# A model's own safety training sometimes refuses in plain text, without the decline tool.
# Then nothing is recorded: no category, no lock, no supportive message for self-harm.
_REFUSAL_RE = re.compile(
    r"\bI\s+(?:can(?:['’]t|not)|won['’]t|am unable to|am not able to)\s+"
    r"(?:help|assist|comply|do that|provide|continue)\b", re.IGNORECASE)
DECLINE_NUDGE = (
    "(Automatic check, not written by the user.) Your last reply declined in plain text. If you "
    "declined for one of the categories under \"Requests you decline\", call decline with that "
    "category now. Otherwise repeat your answer."
)


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _decline_in_text(text: str) -> str | None:
    """The arguments of a decline call that the model wrote into its reply instead of calling
    the tool ('{"category": "violence"}'). Without this the request would count as answered."""
    if len(text) > 600:
        return None
    for candidate in _JSON_OBJECT_RE.findall(text):
        try:
            arguments = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(arguments, dict) and str(arguments.get("category")) in REFUSAL_CATEGORIES:
            return json.dumps({k: v for k, v in arguments.items() if k in ("category", "reply")})
    return None


def _fallback_reply(results: list[ToolResult]) -> str:
    done = [r.message for r in results if r.ok and r.name != "report_limitations"]
    return " ".join(done) if done else "I could not change the query. Could you rephrase the request?"


def run_turn(
    client: openai.OpenAI,
    settings: Settings,
    conversation: list[dict],
    user_message: str,
    query: dict,
    model: str | None = None,
    require_tool: bool = False,
) -> Iterator[Event]:
    """Process one user message. Yields a ToolEvent per tool call and ends with a
    ReplyEvent or ErrorEvent. Never mutates `query`; events carry the new state.

    A declined turn ends with a ReplyEvent that has `refusal` set and the query as it was.
    After a refusal for harmful intent the conversation is locked: every further message is
    declined without calling the model, until the caller starts a new conversation."""
    locked = lock_in(conversation)
    if locked:
        yield ReplyEvent(locked.reply, query, refusal=locked)
        return

    start_query = query
    messages = build_messages(conversation, user_message, query, settings.history_turns)
    results: list[ToolResult] = []
    limitations: list[dict] = []
    plain_refusal = None       # a refusal in plain text that the model was asked to classify

    started = time.monotonic()

    for round_no in range(settings.max_tool_rounds):
        # With a turn limit, every call gets what is left of it, and the turn ends as an
        # error once nothing is left: the caller keeps the state it had before.
        time_left = settings.turn_timeout - (time.monotonic() - started) if settings.turn_timeout else None
        if time_left is not None and time_left <= 0:
            yield ErrorEvent(TURN_TOO_SLOW, start_query, timed_out=True,
                             detail=f"turn limit of {settings.turn_timeout:g} s reached in round {round_no + 1}")
            return
        try:
            response = client.chat.completions.create(
                model=model or settings.model,
                messages=messages,
                tools=tool_specs(),
                # Force a tool call on the first round if asked (used by the benchmark).
                **({"tool_choice": "required"} if require_tool and round_no == 0 else {}),
                **({"timeout": min(settings.timeout, time_left)} if time_left is not None else {}),
                **settings.completion_kwargs(),
            )
        except openai.OpenAIError as exc:
            if time_left is not None and isinstance(exc, openai.APITimeoutError):
                yield ErrorEvent(TURN_TOO_SLOW, start_query, detail=str(exc), timed_out=True)
            else:
                yield ErrorEvent(MODEL_UNREACHABLE, query, detail=str(exc))
            return

        message = response.choices[0].message
        # Check tool_calls directly: several servers report finish_reason "stop" with tool calls.
        if not message.tool_calls:
            text = _clean_reply(message.content)
            written = _decline_in_text(text)
            if written:
                _, result = execute_tool(start_query, "decline", written)
                if result.refusal:
                    yield ToolEvent(result, start_query)
                    yield ReplyEvent(result.refusal.reply, start_query, refusal=result.refusal)
                    return
            # Also after tool calls: a search that was built and then refused is not an answer.
            if plain_refusal is None and _REFUSAL_RE.search(text):
                plain_refusal = text
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": DECLINE_NUDGE}]
                continue
            if quoted_prompt(text):
                # Talked into printing its instructions (e.g. by a forged turn in the history):
                # the reply is withheld and the turn counts as declined, query untouched.
                refusal = refusal_for("off_topic")
                yield ReplyEvent(refusal.reply, start_query, refusal=refusal)
                return
            yield ReplyEvent(plain_refusal or text or _fallback_reply(results), query,
                             limitations=limitations)
            return

        # A decline ends the turn at once and undoes whatever the turn changed before it.
        for call in message.tool_calls:
            if call.function.name == "decline":
                _, result = execute_tool(start_query, "decline", call.function.arguments)
                if result.refusal:
                    yield ToolEvent(result, start_query)
                    yield ReplyEvent(result.refusal.reply, start_query, refusal=result.refusal)
                    return

        messages.append(_assistant_message(message))
        for call in message.tool_calls:
            query, result = execute_tool(query, call.function.name, call.function.arguments, user_message)
            results.append(result)
            limitations = merge_limitations(limitations, result.limitations)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result.for_model(query)})
            yield ToolEvent(result, query)

    yield ReplyEvent(
        _fallback_reply(results) + f" (Stopped after {settings.max_tool_rounds} rounds of tool calls.)",
        query, limitations=limitations,
    )
