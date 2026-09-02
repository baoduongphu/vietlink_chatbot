"""Streamlit smoke test for the analysis extraction service."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class FakeAnalysisExtractionService:
    def __init__(self, provider: str, max_tokens: int) -> None:
        self.provider = provider
        self.max_tokens = max_tokens

    def run(self, conversation: str, conversation_id: str) -> SimpleNamespace:
        assert conversation == "User: Build a report."
        assert conversation_id == "ui-smoke-test"
        return SimpleNamespace(
            conversation_id=conversation_id,
            analysis_report="### Case 1\n\nMissing report scope.",
            entries=[{"knowledge_id": "KB-1", "title": "Missing report scope"}],
        )


class AnalysisExtractionStreamlitTest(TestCase):
    def test_analysis_extraction_service_renders_results(self) -> None:
        with patch(
            "analysis_extraction.AnalysisExtractionService",
            FakeAnalysisExtractionService,
        ):
            app_path = Path(__file__).resolve().parents[1] / "app.py"
            app = AppTest.from_file(app_path).run(timeout=30)
            app.radio(key=None).set_value("Analysis extraction").run(timeout=30)
            app.text_input(key=None).set_value("ui-smoke-test")
            app.text_area(key=None).set_value("User: Build a report.")
            app.button(key=None).click().run(timeout=30)

        self.assertFalse(app.exception)
        self.assertTrue(any(item.value == "Analysis report" for item in app.subheader))
        self.assertTrue(
            any("Knowledge-base entries (1)" == item.value for item in app.subheader)
        )
