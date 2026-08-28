"""RAG orchestration for retrieval and multi-turn clarification."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .client_manager import ClientManager
from .config import FIXED_EMBEDDING_MODEL, PROVIDER_REGISTRY, RAGConfig
from .embedding import FixedEmbeddingManager
from .llm import _llm_chat_send
from .prompts import CLARIFICATION_PROMPT_TEMPLATE
from .retriever import Retriever
from .vector_db import VectorDBManager


@dataclass
class ClarificationTurn:
    question: str
    answer: str


@dataclass
class ClarificationSession:
    original_request: str
    candidates: list[dict[str, Any]]
    retrieval_strategy: str
    turns: list[ClarificationTurn] = field(default_factory=list)
    pending_question: str | None = None
    clarified_request: str | None = None
    assumptions_made: list[str] = field(default_factory=list)


class RAGSystem:
    def __init__(self, config: RAGConfig) -> None:
        config.validate()
        self.config = config
        self._clients = ClientManager()
        self._embeddings = FixedEmbeddingManager(FIXED_EMBEDDING_MODEL)
        self._vector_db = VectorDBManager(self._embeddings)
        self._vector_db.switch_provider(config.provider)
        self._retriever = Retriever(self._embeddings, self._vector_db, self._clients)

    def _advance(self, session: ClarificationSession) -> ClarificationSession:
        history = [{"question": turn.question, "answer": turn.answer} for turn in session.turns]
        prompt = CLARIFICATION_PROMPT_TEMPLATE.format(
            original_request=session.original_request,
            conversation_history=json.dumps(history, ensure_ascii=False),
            candidate_kb_entries=json.dumps(session.candidates, ensure_ascii=False),
            round_number=len(session.turns),
            max_rounds=self.config.max_question_rounds,
        )
        provider_config = PROVIDER_REGISTRY[self.config.provider]
        payload = _llm_chat_send(
            self.config.provider, self._clients.get(self.config.provider),
            provider_config["generation_model"], prompt, self.config.temperature, self.config.max_tokens,
        )
        question = str(payload.get("next_question", "")).strip()
        if (
            payload.get("status") == "ask"
            and question
            and len(session.turns) < self.config.max_question_rounds
        ):
            session.pending_question = question
            return session
        session.pending_question = None
        session.clarified_request = str(payload.get("clarified_request", "")).strip() or session.original_request
        assumptions = payload.get("assumptions_made", [])
        session.assumptions_made = [str(item) for item in assumptions] if isinstance(assumptions, list) else []
        return session

    def start(self, request: str) -> ClarificationSession:
        request = request.strip()
        if not request:
            raise ValueError("Enter a request before starting a clarification session.")
        retrieved = self._retriever.retrieve(self.config, request)
        return self._advance(ClarificationSession(
            original_request=request,
            candidates=retrieved.candidates,
            retrieval_strategy=retrieved.query_strategy,
        ))

    def answer(self, session: ClarificationSession, answer: str) -> ClarificationSession:
        if not session.pending_question:
            raise ValueError("There is no pending clarification question.")
        answer = answer.strip()
        if not answer:
            raise ValueError("Enter an answer before continuing.")
        session.turns.append(ClarificationTurn(session.pending_question, answer))
        session.pending_question = None
        return self._advance(session)
