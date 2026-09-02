"""Folder-based batch workflow for analysis extraction."""

from __future__ import annotations

import json
from pathlib import Path

from .config import NO_CASE_FOUND_MARKER, SUPPORTED_EXTENSIONS
from .service import AnalysisExtractionService


class BatchExtractor:
    def __init__(self, service: AnalysisExtractionService) -> None:
        self.service = service

    def run(self, input_dir: Path, output_dir: Path) -> int:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        sample_files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not sample_files:
            raise FileNotFoundError(f"No .txt/.md sample files found in {input_dir}")

        reports_dir = output_dir / "analysis_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        all_entries: list[dict[str, object]] = []
        for sample_file in sample_files:
            conversation_id = sample_file.stem
            print(f"[{conversation_id}] Running Prompt 1 (Analysis)...")
            report = self.service.analyze(sample_file.read_text(encoding="utf-8"))
            report_path = reports_dir / f"{conversation_id}.analysis.md"
            report_path.write_text(report, encoding="utf-8")
            print(f"[{conversation_id}] Analysis report saved -> {report_path}")
            if NO_CASE_FOUND_MARKER in report:
                print(f"[{conversation_id}] No ambiguous case found; skipping extraction.")
                continue
            print(f"[{conversation_id}] Running Prompt 2 (Extraction)...")
            entries = self.service.extract(report, conversation_id)
            print(f"[{conversation_id}] Extracted {len(entries)} entries.")
            all_entries.extend(entries)

        for index, entry in enumerate(all_entries, start=1):
            entry["knowledge_id"] = f"KB-{index}"
        output_path = output_dir / "kbs.json"
        output_path.write_text(json.dumps(all_entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Done. {len(all_entries)} KB entries written -> {output_path}")
        print(f"Per-file analysis reports saved under -> {reports_dir}")
        return len(all_entries)
