"""Three-strategy KB retrieval based on request length."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .client_manager import ClientManager
from .config import PROVIDER_REGISTRY, RAGConfig
from .embedding import FixedEmbeddingManager
from .llm import _llm_call_oneshot, parse_json_output
from .prompts import NORMALIZATION_PROMPT_TEMPLATE
from .vector_db import VectorDBManager


SHORT_INPUT_TOKEN_THRESHOLD = 40
LONG_INPUT_TOKEN_THRESHOLD = 150


@dataclass(frozen=True)
class NormalizedRequest:
    core_request: str
    stated_info: list[str]
    background_context: str


@dataclass(frozen=True)
class RetrievalResult:
    candidates: list[dict[str, Any]]
    query_strategy: str
    normalized: NormalizedRequest | None = None


def _token_count(text: str) -> int:
    return len(text.split())


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part.strip()]


class Retriever:
    def __init__(
        self, embedding_manager: FixedEmbeddingManager, vector_db: VectorDBManager,
        client_manager: ClientManager,
    ) -> None:
        self._embedding_manager = embedding_manager
        self._vector_db = vector_db
        self._client_manager = client_manager

    def _entries(self, config: RAGConfig, scores) -> list[dict[str, Any]]:
        kb_data = self._vector_db.get_kb_data(config.provider)
        entries = []
        for result in scores:
            if result.knowledge_id in kb_data:
                entry = dict(kb_data[result.knowledge_id])
                entry["retrieval_score"] = round(result.score, 3)
                entries.append(entry)
        return entries

    def _normalize(self, provider: str, request: str) -> NormalizedRequest:
        provider_config = PROVIDER_REGISTRY[provider]
        raw = _llm_call_oneshot(
            provider, self._client_manager.get(provider), provider_config["light_model"],
            NORMALIZATION_PROMPT_TEMPLATE.format(raw_request=request), 0, 600,
        )
        payload = parse_json_output(raw)
        return NormalizedRequest(
            core_request=str(payload.get("core_request", "")),
            stated_info=[str(item) for item in payload.get("stated_info", [])],
            background_context=str(payload.get("background_context", "")),
        )

    def retrieve(self, config: RAGConfig, request: str) -> RetrievalResult:
        count = _token_count(request)
        if count <= SHORT_INPUT_TOKEN_THRESHOLD:
            scores = self._vector_db.search(
                config.provider, self._embedding_manager.embed(request), config.top_k,
                config.similarity_threshold,
            )
            return RetrievalResult(self._entries(config, scores), "direct")
        if count <= LONG_INPUT_TOKEN_THRESHOLD:
            best: dict[str, Any] = {}
            for sentence in _sentences(request):
                for score in self._vector_db.search(
                    config.provider, self._embedding_manager.embed(sentence), config.top_k,
                    config.similarity_threshold,
                ):
                    previous = best.get(score.knowledge_id)
                    if previous is None or score.score > previous.score:
                        best[score.knowledge_id] = score
            scores = sorted(best.values(), key=lambda item: item.score, reverse=True)[:config.top_k]
            return RetrievalResult(self._entries(config, scores), "sentence_chunk")
        normalized = self._normalize(config.provider, request)
        query = normalized.core_request or request
        scores = self._vector_db.search(
            config.provider, self._embedding_manager.embed(query), config.top_k,
            config.similarity_threshold,
        )
        return RetrievalResult(self._entries(config, scores), "normalized", normalized)
