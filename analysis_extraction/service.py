"""Application service for the two-stage analysis extraction workflow."""

from __future__ import annotations

from typing import Any

from core.clients import ClientManager
from core.llm import call_llm

from .config import MODEL_DEFAULTS, NO_CASE_FOUND_MARKER, resolve_models
from .models import AnalysisExtractionResult
from .parser import parse_json_array
from .prompts import ANALYSIS_PROMPT, EXTRACTION_PROMPT


class AnalysisExtractionService:
    """Analyze one conversation and extract reusable KB entries from it."""

    def __init__(
        self,
        provider: str,
        max_tokens: int = 8192,
        analysis_model: str | None = None,
        extraction_model: str | None = None,
        client: Any | None = None,
    ) -> None:
        if provider not in MODEL_DEFAULTS:
            raise ValueError(f"Unsupported provider: {provider!r}.")
        if max_tokens <= 0:
            raise ValueError("Maximum output tokens must be greater than zero.")
        default_analysis_model, default_extraction_model = resolve_models(provider)
        self.provider = provider
        self.max_tokens = max_tokens
        self.analysis_model = analysis_model or default_analysis_model
        self.extraction_model = extraction_model or default_extraction_model
        self._client = client or ClientManager().get(provider)

    def _call(self, prompt: str, model: str, temperature: float, json_mode: bool) -> str:
        return call_llm(
            self.provider,
            self._client,
            model,
            prompt,
            temperature,
            self.max_tokens,
            json_mode=json_mode,
        )

    def analyze(self, conversation: str) -> str:
        conversation = conversation.strip()
        if not conversation:
            raise ValueError("Enter a conversation before running extraction.")
        analysis_prompt = ANALYSIS_PROMPT.format(
            no_case_marker=NO_CASE_FOUND_MARKER,
            conversation=conversation,
        )
        return self._call(analysis_prompt, self.analysis_model, 0.3, False)

    def extract(self, analysis_report: str, conversation_id: str) -> list[dict[str, Any]]:
        conversation_id = conversation_id.strip()
        if not conversation_id:
            raise ValueError("Enter a conversation ID before running extraction.")
        extraction_prompt = EXTRACTION_PROMPT.format(
            analysis_result=analysis_report,
            conversation_id=conversation_id,
        )
        raw_entries = self._call(extraction_prompt, self.extraction_model, 0, True)
        entries = parse_json_array(raw_entries)
        for index, entry in enumerate(entries, start=1):
            entry["knowledge_id"] = f"KB-{index}"
        return entries

    def run(self, conversation: str, conversation_id: str) -> AnalysisExtractionResult:
        if not conversation_id.strip():
            raise ValueError("Enter a conversation ID before running extraction.")
        report = self.analyze(conversation)
        entries = []
        if NO_CASE_FOUND_MARKER not in report:
            entries = self.extract(report, conversation_id)
        return AnalysisExtractionResult(conversation_id, report, entries)
