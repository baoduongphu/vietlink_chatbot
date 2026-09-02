"""Provider client lifecycle management shared across services."""

from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic
from google import genai
from openai import OpenAI


PROVIDER_API_KEYS = {
    "gpt": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


class MissingAPIKeyError(RuntimeError):
    """Raised when the selected provider has no configured API key."""


class ClientManager:
    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}

    def get(self, provider: str) -> Any:
        if provider not in PROVIDER_API_KEYS:
            raise ValueError(f"Unsupported provider: {provider!r}.")
        if provider in self._clients:
            return self._clients[provider]

        env_name = PROVIDER_API_KEYS[provider]
        api_key = os.getenv(env_name)
        if not api_key:
            raise MissingAPIKeyError(f"Missing environment variable {env_name}.")
        if provider == "gpt":
            client = OpenAI(api_key=api_key)
        elif provider == "claude":
            client = Anthropic(api_key=api_key)
        else:
            # Avoid the Gemini SDK warning when both supported names are set.
            os.environ.pop("GEMINI_API_KEY", None)
            client = genai.Client(api_key=api_key)
        self._clients[provider] = client
        return client
