"""Settings from environment variables (HF Spaces secrets or a local .env file).

LLM_API_KEY           API key (use any string, e.g. "ollama", for a local Ollama server)
LLM_BASE_URL          OpenAI-compatible endpoint, e.g. http://localhost:11434/v1 for Ollama
LLM_MODEL             default model name
LLM_MODELS            optional comma-separated list; shows a model picker in the UI
LLM_TEMPERATURE       optional, e.g. 0 for more deterministic output
LLM_REASONING_EFFORT  optional, "low" / "medium" / "high" for reasoning models (gpt-oss, GPT-5)
LLM_MAX_TOOL_ROUNDS   optional, default 8
LLM_TIMEOUT           optional request timeout in seconds, default 120
LLM_TURN_TIMEOUT      optional limit in seconds for a whole chat turn (all tool rounds), default off
LLM_NO_CACHE          optional, 1 asks a LiteLLM proxy for a fresh answer instead of a cached one
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"   # next to app.py and evaluate.py


@dataclass
class Settings:
    api_key: str | None
    base_url: str | None
    model: str
    models: list[str] = field(default_factory=list)
    temperature: float | None = None
    reasoning_effort: str | None = None
    max_tool_rounds: int = 8
    history_turns: int = 10
    timeout: float = 120.0
    turn_timeout: float = 0.0      # whole turn, all tool rounds; 0 = no limit
    no_cache: bool = False         # skip the answer cache of a LiteLLM proxy (test runs)

    def completion_kwargs(self) -> dict:
        """Optional parameters, only sent when set (some servers reject unknown ones)."""
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.no_cache:
            # A LiteLLM proxy returns the stored answer for an identical request, which hides
            # the model's run-to-run variation. Other servers may reject the field.
            kwargs["extra_body"] = {"cache": {"no-cache": True}}
        return kwargs


def load_settings() -> Settings:
    # Variables already set in the shell win; then .env in the project folder,
    # then a .env in the folder you run the command from.
    load_dotenv(ENV_FILE)
    load_dotenv(find_dotenv(usecwd=True))
    model = os.getenv("LLM_MODEL", "GPT-oss-120b")
    models = [m.strip() for m in os.getenv("LLM_MODELS", "").split(",") if m.strip()]
    if model not in models:
        models.insert(0, model)
    temperature = os.getenv("LLM_TEMPERATURE")
    return Settings(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL") or None,
        model=model,
        models=models,
        temperature=float(temperature) if temperature not in (None, "") else None,
        reasoning_effort=os.getenv("LLM_REASONING_EFFORT") or None,
        max_tool_rounds=int(os.getenv("LLM_MAX_TOOL_ROUNDS", "8")),
        timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        turn_timeout=float(os.getenv("LLM_TURN_TIMEOUT") or 0),
        no_cache=os.getenv("LLM_NO_CACHE", "").strip().lower() in ("1", "true", "yes"),
    )


def make_client(settings: Settings) -> OpenAI:
    return OpenAI(
        api_key=settings.api_key or "missing-api-key",
        base_url=settings.base_url,
        timeout=settings.timeout,
        # The library repeats a failed call on its own, which would run past a turn limit.
        max_retries=0 if settings.turn_timeout else 2,
    )
