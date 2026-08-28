"""Knowledge-base loading and per-provider Chroma vector indexes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import chromadb

from .config import ConfigurationError, PROVIDER_REGISTRY
from .embedding import FixedEmbeddingManager


@dataclass(frozen=True)
class SearchResult:
    knowledge_id: str
    score: float


def load_kb_data(path) -> dict[str, dict[str, Any]]:
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in knowledge base: {path}") from exc
    if not isinstance(entries, list) or not entries:
        raise ConfigurationError(f"Knowledge base must be a non-empty JSON array: {path}")
    required = {"knowledge_id", "trigger"}
    invalid = [str(entry.get("knowledge_id", "unknown")) for entry in entries
               if not isinstance(entry, dict) or not required.issubset(entry)]
    if invalid:
        raise ConfigurationError("KB entries require knowledge_id and trigger: " + ", ".join(invalid))
    return {str(entry["knowledge_id"]): entry for entry in entries}


def index_all_validated(
    kb_data: dict[str, dict[str, Any]],
    embedding_manager: FixedEmbeddingManager,
    collection: Any,
) -> None:
    entries = list(kb_data.values())
    vectors = embedding_manager.embed_batch([str(entry["trigger"]) for entry in entries])
    collection.add(
        ids=[str(entry["knowledge_id"]) for entry in entries],
        embeddings=vectors,
        metadatas=[{"status": str(entry.get("status", "Validated"))} for entry in entries],
    )


class VectorDBManager:
    def __init__(self, embedding_manager: FixedEmbeddingManager) -> None:
        self._embedding_manager = embedding_manager
        self._collections: dict[str, Any] = {}
        self._kb_data: dict[str, dict[str, dict[str, Any]]] = {}

    def switch_provider(self, provider: str) -> None:
        if provider not in PROVIDER_REGISTRY:
            raise ConfigurationError(f"Unsupported provider: {provider!r}.")
        config = PROVIDER_REGISTRY[provider]
        kb_data = load_kb_data(config["kb_path"])
        config["chroma_dir"].mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(config["chroma_dir"]))
        try:
            client.delete_collection(config["chroma_collection"])
        except Exception:
            pass
        collection = client.get_or_create_collection(
            name=config["chroma_collection"], metadata={"hnsw:space": "cosine"}
        )
        index_all_validated(kb_data, self._embedding_manager, collection)
        self._collections[provider] = collection
        self._kb_data[provider] = kb_data

    def search(
        self, provider: str, vector: list[float], top_k: int, similarity_threshold: float
    ) -> list[SearchResult]:
        if provider not in self._collections:
            raise ConfigurationError("Vector index has not been initialized.")
        result_count = min(top_k, len(self._kb_data[provider]))
        raw = self._collections[provider].query(query_embeddings=[vector], n_results=result_count)
        ids = raw.get("ids", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        return [
            SearchResult(knowledge_id, 1.0 - float(distance))
            for knowledge_id, distance in zip(ids, distances)
            if 1.0 - float(distance) >= similarity_threshold
        ]

    def get_kb_data(self, provider: str) -> dict[str, dict[str, Any]]:
        if provider not in self._kb_data:
            raise ConfigurationError("Knowledge base has not been loaded.")
        return self._kb_data[provider]
