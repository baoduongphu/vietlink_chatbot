"""Streamlit view for conversation analysis and KB extraction."""

from __future__ import annotations

import json

import streamlit as st

from analysis_extraction import AnalysisExtractionService
from rag.config import ConfigurationError


def render(provider: str, max_tokens: int) -> None:
    st.caption("Analyze a conversation and convert reusable ambiguity cases into KB entries.")
    conversation_id = st.text_input("Conversation ID", value="conversation_1")
    uploaded_file = st.file_uploader("Upload conversation (.txt or .md)", type=("txt", "md"))
    uploaded_text = ""
    if uploaded_file is not None:
        try:
            uploaded_text = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error("The uploaded file must use UTF-8 encoding.")
    conversation = st.text_area(
        "Conversation",
        value=uploaded_text,
        height=280,
        placeholder="Paste the full conversation here...",
    )
    if st.button("Run analysis extraction", type="primary"):
        try:
            with st.spinner("Analyzing ambiguity and extracting knowledge..."):
                service = AnalysisExtractionService(provider, int(max_tokens))
                st.session_state.extraction_result = service.run(conversation, conversation_id)
        except (ConfigurationError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unable to run analysis extraction: {type(exc).__name__}: {exc}")

    result = st.session_state.get("extraction_result")
    if result is None:
        return
    st.subheader("Analysis report")
    st.markdown(result.analysis_report)
    st.download_button(
        "Download analysis report",
        result.analysis_report,
        file_name=f"{result.conversation_id}.analysis.md",
        mime="text/markdown",
    )
    st.subheader(f"Knowledge-base entries ({len(result.entries)})")
    st.json(result.entries)
    st.download_button(
        "Download KB JSON",
        json.dumps(result.entries, ensure_ascii=False, indent=2),
        file_name=f"{result.conversation_id}.kbs.json",
        mime="application/json",
    )
