"""The chat loop: send the conversation to the LLM, run its tool calls, repeat.

`run_turn` is a generator of events so the UI can show every tool call as it
happens, while `evaluate.py` can simply drain it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

import openai

from .config import Settings
from .prompts import get_system_prompt, user_turn
from .tools import ToolResult, execute_tool, tool_specs


@dataclass
class ToolEvent:
    result: ToolResult
    query: dict


@dataclass
class ReplyEvent:
    text: str
    query: dict


@dataclass
class ErrorEvent:
    text: str
    query: dict


Event = ToolEvent | ReplyEvent | ErrorEvent


def build_messages(conversation: list[dict], user_message: str, query: dict, history_turns: int) -> list[dict]:
    """System prompt + recent text-only history + the new message with the current query."""
    recent = conversation[-2 * history_turns:] if history_turns else []
    return [
        {"role": "system", "content": get_system_prompt()},
        *recent,
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


def _fallback_reply(results: list[ToolResult]) -> str:
    done = [r.message for r in results if r.ok]
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
    ReplyEvent or ErrorEvent. Never mutates `query`; events carry the new state."""
    messages = build_messages(conversation, user_message, query, settings.history_turns)
    results: list[ToolResult] = []

    for round_no in range(settings.max_tool_rounds):
        try:
            response = client.chat.completions.create(
                model=model or settings.model,
                messages=messages,
                tools=tool_specs(),
                # Force a tool call on the first round if asked (used by the benchmark).
                **({"tool_choice": "required"} if require_tool and round_no == 0 else {}),
                **settings.completion_kwargs(),
            )
        except openai.OpenAIError as exc:
            yield ErrorEvent(f"The language model could not be reached: {exc}", query)
            return

        message = response.choices[0].message
        # Check tool_calls directly: several servers report finish_reason "stop" with tool calls.
        if not message.tool_calls:
            yield ReplyEvent(_clean_reply(message.content) or _fallback_reply(results), query)
            return

        messages.append(_assistant_message(message))
        for call in message.tool_calls:
            query, result = execute_tool(query, call.function.name, call.function.arguments, user_message)
            results.append(result)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result.for_model(query)})
            yield ToolEvent(result, query)

    yield ReplyEvent(
        _fallback_reply(results) + f" (Stopped after {settings.max_tool_rounds} rounds of tool calls.)",
        query,
    )
