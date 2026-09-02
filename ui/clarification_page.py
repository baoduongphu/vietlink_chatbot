"""Streamlit view for the requirement clarification workflow."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from rag.config import ConfigurationError


def render(current_system: Callable[[], Any], max_question_rounds: int) -> None:
    st.caption("Retrieve relevant knowledge, clarify missing details, and produce an implementation-ready request.")
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
        return

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
