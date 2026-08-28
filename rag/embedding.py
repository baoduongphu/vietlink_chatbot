"""Shared multilingual embedding model."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from .config import ConfigurationError


class FixedEmbeddingManager:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        try:
            # Avoid a network check on every Streamlit restart when the model is cached.
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except OSError:
            try:
                self._model = SentenceTransformer(model_name)
            except OSError as exc:
                raise ConfigurationError(
                    "The embedding model is unavailable locally and could not be downloaded. "
                    "Check the Hugging Face network connection or pre-download the model."
                ) from exc

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()
