"""Provider-neutral one-shot text generation."""

from __future__ import annotations

from typing import Any

from google.genai import types as genai_types
from openai import BadRequestError


def call_llm(
    provider: str,
    client: Any,
    model: str,
    prompt: str,
    temperature: float = 0,
    max_tokens: int = 8192,
    *,
    json_mode: bool = False,
    json_object: bool = False,
    gemini_chat: bool = False,
) -> str:
    """Call one provider while preserving each SDK's existing behavior."""
    if provider == "gpt":
        request: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
        }
        if json_object:
            request["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(
                **request, temperature=temperature
            )
        except BadRequestError:
            # Some reasoning models reject explicit temperature values.
            response = client.chat.completions.create(**request)
        return response.choices[0].message.content or ""

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
        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        )

    if provider == "gemini":
        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else None,
        )
        if gemini_chat:
            response = client.chats.create(model=model, config=config).send_message(prompt)
        else:
            response = client.models.generate_content(
                model=model, contents=prompt, config=config
            )
        return response.text or ""

    raise ValueError(f"Unsupported provider: {provider!r}.")
