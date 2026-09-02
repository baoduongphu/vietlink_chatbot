"""Streamlit entrypoint for VietLink's chatbot services."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def load_streamlit_secrets() -> None:
    """Expose Streamlit Cloud secrets to provider SDKs as environment variables."""
    names = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_LIGHT_MODEL",
        "OPENAI_ANALYSIS_MODEL",
        "OPENAI_EXTRACTION_MODEL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_LIGHT_MODEL",
        "ANTHROPIC_ANALYSIS_MODEL",
        "ANTHROPIC_EXTRACTION_MODEL",
        "GOOGLE_MODEL",
        "GOOGLE_LIGHT_MODEL",
        "GOOGLE_ANALYSIS_MODEL",
        "GOOGLE_EXTRACTION_MODEL",
    )
    for name in names:
        if not os.getenv(name):
            secret = st.secrets.get(name)
            if secret:
                os.environ[name] = str(secret)


load_streamlit_secrets()

from analysis_extraction import resolve_models
from rag.config import PROVIDER_REGISTRY, RAGConfig
from ui.analysis_extraction_page import render as render_analysis_extraction
from ui.clarification_page import render as render_clarification

if TYPE_CHECKING:
    from rag.rag_system import RAGSystem


st.set_page_config(page_title="VietLink RAG", page_icon="VL", layout="centered")
st.title("VietLink Requirement Clarifier")

with st.sidebar:
    st.header("Configuration")
    service_name = st.radio("Service", ("Requirement clarification", "Analysis extraction"))
    provider = st.selectbox(
        "Provider",
        list(PROVIDER_REGISTRY),
        format_func=lambda value: PROVIDER_REGISTRY[value]["label"],
    )
    top_k = st.slider("Candidate KB entries", 1, 15, 5)
    similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.0, 0.05)
    max_tokens = st.number_input("Maximum output tokens", min_value=256, max_value=32768, value=8192, step=256)
    max_question_rounds = st.slider("Maximum clarification rounds", 1, 10, 3)
    if service_name == "Analysis extraction":
        analysis_model, extraction_model = resolve_models(provider)
        st.caption(f"Analysis model: {analysis_model}")
        st.caption(f"Extraction model: {extraction_model}")
    else:
        st.caption(f"Generation model: {PROVIDER_REGISTRY[provider]['generation_model']}")
    if st.button("New session", use_container_width=True):
        st.session_state.pop("session", None)
        st.session_state.pop("extraction_result", None)
        st.session_state.pop("configuration_key", None)
        st.rerun()

configuration_key = (service_name, provider, top_k, similarity_threshold, max_tokens, max_question_rounds)
if st.session_state.get("configuration_key") != configuration_key:
    st.session_state.pop("session", None)
    st.session_state.pop("extraction_result", None)
    st.session_state.configuration_key = configuration_key


@st.cache_resource(show_spinner=False)
def get_system(
    provider_name: str,
    result_count: int,
    threshold: float,
    output_token_limit: int,
    question_round_limit: int,
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


if service_name == "Analysis extraction":
    render_analysis_extraction(provider, int(max_tokens))
else:
    render_clarification(current_system, max_question_rounds)
