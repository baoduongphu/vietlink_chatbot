"""Provider client lifecycle management."""

from __future__ import annotations

from typing import Any

from core.clients import ClientManager as CoreClientManager
from core.clients import MissingAPIKeyError

from .config import ConfigurationError, PROVIDER_REGISTRY, SecretNotFoundError


class ClientManager:
    def __init__(self) -> None:
        self._manager = CoreClientManager()

    def get(self, provider: str) -> Any:
        if provider not in PROVIDER_REGISTRY:
            raise ConfigurationError(f"Unsupported provider: {provider!r}.")
        try:
            return self._manager.get(provider)
        except MissingAPIKeyError as exc:
            raise SecretNotFoundError(
                f"Missing environment variable {PROVIDER_REGISTRY[provider]['secret_key']}."
            ) from exc
