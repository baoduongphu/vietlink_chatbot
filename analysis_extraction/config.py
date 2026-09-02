"""Configuration for the analysis extraction service."""

from __future__ import annotations

import os

NO_CASE_FOUND_MARKER = "No sufficiently clear ambiguous case found"
SUPPORTED_EXTENSIONS = {".txt", ".md"}

MODEL_DEFAULTS = {
    "gemini": {"analysis": "gemini-pro-latest", "extraction": "gemini-flash-lite-latest"},
    "gpt": {"analysis": "gpt-5", "extraction": "gpt-5-mini"},
    "claude": {"analysis": "claude-opus-4-8", "extraction": "claude-haiku-4-5-20251001"},
}
MODEL_ENV_VARS = {
    "gemini": ("GOOGLE_ANALYSIS_MODEL", "GOOGLE_EXTRACTION_MODEL"),
    "gpt": ("OPENAI_ANALYSIS_MODEL", "OPENAI_EXTRACTION_MODEL"),
    "claude": ("ANTHROPIC_ANALYSIS_MODEL", "ANTHROPIC_EXTRACTION_MODEL"),
}


def resolve_models(provider: str) -> tuple[str, str]:
    if provider not in MODEL_DEFAULTS:
        raise ValueError(f"Unsupported provider: {provider!r}.")
    analysis_env, extraction_env = MODEL_ENV_VARS[provider]
    defaults = MODEL_DEFAULTS[provider]
    return os.getenv(analysis_env, defaults["analysis"]), os.getenv(extraction_env, defaults["extraction"])
