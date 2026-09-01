"""JSONL persistence for generated questions. Append-only; never truncates existing rows."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from roleready.generation.schemas import GeneratedQuestion

logger = logging.getLogger(__name__)


def load_valid_questions(path: Path) -> list[GeneratedQuestion]:
    """Load valid questions from JSONL. Invalid or blank lines are skipped, not rewritten."""
    if not path.exists():
        return []
    questions: list[GeneratedQuestion] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
                questions.append(GeneratedQuestion.model_validate(payload))
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
                logger.warning("Skipping invalid JSONL line %s in %s", line_number, path.name)
    return questions


def append_questions(path: Path, questions: list[GeneratedQuestion]) -> None:
    """Append questions immediately. Does not modify earlier lines."""
    if not questions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for question in questions:
            handle.write(question.model_dump_json() + "\n")
        handle.flush()


def remaining_to_generate(existing_count: int, target_count: int) -> int:
    if target_count < 1:
        raise ValueError("target_count must be at least 1.")
    if existing_count < 0:
        raise ValueError("existing_count cannot be negative.")
    return max(0, target_count - existing_count)
