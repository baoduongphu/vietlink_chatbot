"""Regression tests for analysis extraction domain behavior."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from analysis_extraction.batch import BatchExtractor
from analysis_extraction.config import NO_CASE_FOUND_MARKER, resolve_models
from analysis_extraction.service import AnalysisExtractionService


class StubService(AnalysisExtractionService):
    def __init__(self, outputs: list[str]) -> None:
        self.provider = "gpt"
        self.max_tokens = 8192
        self.analysis_model = "analysis-model"
        self.extraction_model = "extraction-model"
        self.outputs = iter(outputs)
        self.calls: list[tuple[str, str, float, bool]] = []

    def _call(self, prompt: str, model: str, temperature: float, json_mode: bool) -> str:
        self.calls.append((prompt, model, temperature, json_mode))
        return next(self.outputs)


class AnalysisExtractionServiceTest(TestCase):
    def test_no_case_skips_extraction(self) -> None:
        service = StubService([f"{NO_CASE_FOUND_MARKER}."])
        result = service.run("User: Clear request", "conversation-1")
        self.assertEqual(result.entries, [])
        self.assertEqual(len(service.calls), 1)

    def test_extraction_parses_json_and_assigns_ids(self) -> None:
        service = StubService([
            "### Case 1\nAmbiguous scope",
            '[{"knowledge_id":"temporary","title":"Scope"}]',
        ])
        result = service.run("User: Make it better", "conversation-2")
        self.assertEqual(result.entries[0]["knowledge_id"], "KB-1")
        self.assertEqual(len(service.calls), 2)
        self.assertFalse(service.calls[0][3])
        self.assertTrue(service.calls[1][3])

    def test_models_are_resolved_from_environment(self) -> None:
        values = {
            "OPENAI_ANALYSIS_MODEL": "env-analysis",
            "OPENAI_EXTRACTION_MODEL": "env-extraction",
        }
        with patch.dict("os.environ", values):
            self.assertEqual(resolve_models("gpt"), ("env-analysis", "env-extraction"))


class FakeBatchService:
    def analyze(self, conversation: str) -> str:
        return "analysis: " + conversation

    def extract(self, analysis_report: str, conversation_id: str) -> list[dict[str, str]]:
        return [{"knowledge_id": "KB-1", "title": conversation_id}]


class BatchExtractorTest(TestCase):
    def test_batch_preserves_reports_and_assigns_global_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "a.txt").write_text("first", encoding="utf-8")
            (input_dir / "b.md").write_text("second", encoding="utf-8")

            count = BatchExtractor(FakeBatchService()).run(input_dir, output_dir)  # type: ignore[arg-type]
            entries = json.loads((output_dir / "kbs.json").read_text(encoding="utf-8"))

            self.assertEqual(count, 2)
            self.assertEqual([entry["knowledge_id"] for entry in entries], ["KB-1", "KB-2"])
            self.assertTrue((output_dir / "analysis_reports" / "a.analysis.md").is_file())
            self.assertTrue((output_dir / "analysis_reports" / "b.analysis.md").is_file())
