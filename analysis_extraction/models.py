"""Data models returned by analysis extraction workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnalysisExtractionResult:
    conversation_id: str
    analysis_report: str
    entries: list[dict[str, Any]]
