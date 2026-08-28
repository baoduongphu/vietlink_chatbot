"""Streamlit interface for the VietLink requirement clarification workflow."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def load_streamlit_secrets() -> None:
    """Expose Streamlit Cloud secrets to provider SDKs as environment variables."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        if not os.getenv(name):
            secret = st.secrets.get(name)
            if secret:
                os.environ[name] = str(secret)


load_streamlit_secrets()

from rag.config import ConfigurationError, PROVIDER_REGISTRY, RAGConfig

if TYPE_CHECKING:
    from rag.rag_system import RAGSystem


st.set_page_config(page_title="VietLink RAG", page_icon="VL", layout="centered")
st.title("VietLink Requirement Clarifier")
st.caption("Retrieve relevant knowledge, clarify missing details, and produce an implementation-ready request.")

with st.sidebar:
    st.header("Configuration")
    provider = st.selectbox("Provider", list(PROVIDER_REGISTRY), format_func=lambda value: PROVIDER_REGISTRY[value]["label"])
    top_k = st.slider("Candidate KB entries", 1, 15, 5)
    similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.0, 0.05)
    max_tokens = st.number_input("Maximum output tokens", min_value=256, max_value=32768, value=8192, step=256)
    max_question_rounds = st.slider("Maximum clarification rounds", 1, 10, 3)
    st.caption(f"Generation model: {PROVIDER_REGISTRY[provider]['generation_model']}")
    if st.button("New session", use_container_width=True):
        st.session_state.pop("session", None)
        st.session_state.pop("configuration_key", None)
        st.rerun()

configuration_key = (provider, top_k, similarity_threshold, max_tokens, max_question_rounds)
if st.session_state.get("configuration_key") != configuration_key:
    st.session_state.pop("session", None)
    st.session_state.configuration_key = configuration_key


@st.cache_resource(show_spinner=False)
def get_system(
    provider_name: str, result_count: int, threshold: float, output_token_limit: int, question_round_limit: int
) -> "RAGSystem":
    from rag.rag_system import RAGSystem

    return RAGSystem(RAGConfig(
        provider=provider_name,
        top_k=result_count,
        similarity_threshold=threshold,
        max_tokens=output_token_limit,
        max_question_rounds=question_round_limit,
    ))


def current_system() -> "RAGSystem":
    return get_system(provider, top_k, similarity_threshold, max_tokens, max_question_rounds)


session = st.session_state.get("session")
if session is None:
    request = st.text_area("Initial request", height=160, placeholder="Describe what you need to accomplish...")
    if st.button("Analyze request", type="primary"):
        try:
            with st.spinner("Retrieving knowledge and analyzing the request..."):
                st.session_state.session = current_system().start(request)
            st.rerun()
        except (ConfigurationError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unable to start the session: {type(exc).__name__}: {exc}")
else:
    st.subheader("Original request")
    st.write(session.original_request)
    if session.pending_question:
        st.subheader(f"Clarification question ({len(session.turns) + 1}/{max_question_rounds})")
        st.write(session.pending_question)
        answer = st.text_area("Your answer", key=f"answer_{len(session.turns)}")
        if st.button("Continue", type="primary"):
            try:
                with st.spinner("Updating the clarified request..."):
                    st.session_state.session = current_system().answer(session, answer)
                st.rerun()
            except (ConfigurationError, ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Unable to process the answer: {type(exc).__name__}: {exc}")
    else:
        st.subheader("Clarified request")
        st.success(session.clarified_request)
        if session.assumptions_made:
            st.caption("Assumptions: " + "; ".join(session.assumptions_made))
    with st.expander("Retrieved knowledge-base entries"):
        st.caption(f"Retrieval strategy: {session.retrieval_strategy}")
        if not session.candidates:
            st.write("No candidate entry exceeded the similarity threshold.")
        for entry in session.candidates:
            title = entry.get("title") or entry.get("trigger")
            st.write(f"**{entry['knowledge_id']}** - {title} (score: {entry['retrieval_score']})")
