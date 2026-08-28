"""Runtime configuration and provider metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXED_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MAX_CLARIFICATION_ROUNDS = 3


class ConfigurationError(RuntimeError):
    """Raised for an invalid or incomplete application configuration."""


class SecretNotFoundError(ConfigurationError):
    """Raised when a selected provider has no API key."""


PROVIDER_REGISTRY = {
    "gpt": {
        "label": "GPT",
        "generation_model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        "light_model": os.getenv("OPENAI_LIGHT_MODEL", "gpt-5-mini"),
        "kb_path": PROJECT_ROOT / "data" / "gpt" / "kbs.json",
        "chroma_dir": PROJECT_ROOT / ".chroma" / "gpt",
        "chroma_collection": "kb_gpt",
        "secret_key": "OPENAI_API_KEY",
    },
    "claude": {
        "label": "Claude",
        "generation_model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        "light_model": os.getenv("ANTHROPIC_LIGHT_MODEL", "claude-haiku-4-5"),
        "kb_path": PROJECT_ROOT / "data" / "claude" / "kbs.json",
        "chroma_dir": PROJECT_ROOT / ".chroma" / "claude",
        "chroma_collection": "kb_claude",
        "secret_key": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "label": "Gemini",
        "generation_model": os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
        "light_model": os.getenv("GOOGLE_LIGHT_MODEL", "gemini-2.5-flash-lite"),
        "kb_path": PROJECT_ROOT / "data" / "gemini" / "kbs.json",
        "chroma_dir": PROJECT_ROOT / ".chroma" / "gemini",
        "chroma_collection": "kb_gemini",
        "secret_key": "GOOGLE_API_KEY",
    },
}


@dataclass(frozen=True)
class RAGConfig:
    provider: str
    temperature: float = 0.0
    max_tokens: int = 8192
    top_k: int = 5
    similarity_threshold: float = 0.0
    max_question_rounds: int = MAX_CLARIFICATION_ROUNDS

    def validate(self) -> None:
        if self.provider not in PROVIDER_REGISTRY:
            raise ConfigurationError(f"Unsupported provider: {self.provider!r}.")
        if not PROVIDER_REGISTRY[self.provider]["kb_path"].is_file():
            path = PROVIDER_REGISTRY[self.provider]["kb_path"]
            raise ConfigurationError(f"Knowledge base file was not found: {path}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigurationError("Temperature must be between 0.0 and 2.0.")
        if self.max_tokens <= 0:
            raise ConfigurationError("Max tokens must be positive.")
        if self.top_k <= 0:
            raise ConfigurationError("Top K must be positive.")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ConfigurationError("Similarity threshold must be between 0.0 and 1.0.")
        if self.max_question_rounds <= 0:
            raise ConfigurationError("Max question rounds must be positive.")
