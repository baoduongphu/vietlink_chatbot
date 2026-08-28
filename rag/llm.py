"""Provider-neutral LLM call helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from google.genai import types as genai_types
from openai import BadRequestError

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
    if provider == "gpt":
        request = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            response = client.chat.completions.create(**request, temperature=temperature)
        except BadRequestError:
            # Some reasoning models reject explicit temperature values.
            response = client.chat.completions.create(**request)
        return response.choices[0].message.content or "{}"
    if provider == "claude":
        request = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = client.messages.create(**request, temperature=temperature)
        except TypeError as exc:
            if "temperature" not in str(exc):
                raise
            response = client.messages.create(**request)
        return "".join(block.text for block in response.content if block.type == "text")
    if provider == "gemini":
        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
        chat = client.chats.create(model=model, config=config)
        response = chat.send_message(prompt)
        return response.text or "{}"
    raise ConfigurationError(f"Unsupported provider: {provider!r}.")


def _llm_chat_send(
    provider: str, client: Any, model: str, prompt: str, temperature: float, max_tokens: int
) -> dict[str, Any]:
    """Send one clarification turn; history is explicitly embedded in the prompt."""
    return parse_json_output(
        _llm_call_oneshot(provider, client, model, prompt, temperature, max_tokens)
    )
