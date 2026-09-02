"""Provider-neutral LLM call helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from core.llm import call_llm

from .config import ConfigurationError


def parse_json_output(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ConfigurationError("The model response did not contain a JSON object.")
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ConfigurationError("The model returned invalid JSON.") from exc


def _llm_call_oneshot(
    provider: str, client: Any, model: str, prompt: str, temperature: float, max_tokens: int
) -> str:
    try:
        return call_llm(
            provider,
            client,
            model,
            prompt,
            temperature,
            max_tokens,
            json_mode=True,
            json_object=True,
            gemini_chat=True,
        ) or "{}"
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc


def _llm_chat_send(
    provider: str, client: Any, model: str, prompt: str, temperature: float, max_tokens: int
) -> dict[str, Any]:
    """Send one clarification turn; history is explicitly embedded in the prompt."""
    return parse_json_output(
        _llm_call_oneshot(provider, client, model, prompt, temperature, max_tokens)
    )
