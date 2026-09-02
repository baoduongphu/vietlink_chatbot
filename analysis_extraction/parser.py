"""Parsing helpers for model-produced knowledge entries."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_array(raw_output: str) -> list[dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip()).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            raise ValueError("Model response did not contain a JSON array.") from exc
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as nested_exc:
            raise ValueError("Model response contained an invalid JSON array.") from nested_exc
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("Model response must be a JSON array of objects.")
    return data
