"""Provider client lifecycle management."""

from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic
from google import genai
from openai import OpenAI

from .config import PROVIDER_REGISTRY, SecretNotFoundError


class ClientManager:
    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}

    def get(self, provider: str) -> Any:
        if provider in self._clients:
            return self._clients[provider]
        api_key = os.getenv(PROVIDER_REGISTRY[provider]["secret_key"])
        if not api_key:
            raise SecretNotFoundError(
                f"Missing environment variable {PROVIDER_REGISTRY[provider]['secret_key']}."
            )
        if provider == "gpt":
            client = OpenAI(api_key=api_key)
        elif provider == "claude":
            client = Anthropic(api_key=api_key)
        else:
            # The Gemini SDK warns when both supported environment names exist.
            # This application deliberately uses GOOGLE_API_KEY as its public contract.
            os.environ.pop("GEMINI_API_KEY", None)
            client = genai.Client(api_key=api_key)
        self._clients[provider] = client
        return client
