"""Validate generated question JSONL. No OpenAI, Pinecone, or SQLite."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roleready.generation.schemas import (
    CATEGORIES,
    COMPANIES,
    MAX_DIFFICULTY,
    MIN_DIFFICULTY,
    ROLES,
    SENIORITIES,
)

REQUIRED_FIELDS = (
    "id",
    "company",
    "role",
    "seniority",
    "category",
    "difficulty",
    "question_text",
    "rubric",
    "follow_up_hints",
)

MIN_QUESTION_TEXT_LENGTH = 40
MIN_RUBRIC_LENGTH = 80

_MALFORMED_PATTERNS = (
    re.compile(r"```"),
    re.compile(r"\bas an ai\b", re.I),
    re.compile(r"\bas a language model\b", re.I),
    re.compile(r"\bhere is (the |a )?(json|output)\b", re.I),
    re.compile(r"\bthe following json\b", re.I),
    re.compile(r"\bhope this helps\b", re.I),
    re.compile(r"\blet me know if\b", re.I),
    re.compile(r"\bno_follow_up\b", re.I),
    re.compile(r"^\s*(sure|certainly|of course|absolutely)[,!.]", re.I),
    re.compile(r"^\s*\{"),
    re.compile(r"</?(script|system|assistant)\b", re.I),
)


@dataclass(frozen=True)
class ValidationSummary:
    total: int
    valid: int
    rejected: int
    duplicate_ids: int


def _as_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value


def _malformed_llm_output(text: str) -> bool:
    stripped = text.strip()
    if stripped.count("{") >= 2 and stripped.count("}") >= 2 and '"rubric"' in stripped:
        return True
    return any(pattern.search(stripped) for pattern in _MALFORMED_PATTERNS)


def normalize_question_text(text: str) -> str:
    """Lowercase, trim, collapse whitespace, strip wrapping punctuation. Not semantic."""
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed.strip(" \t.,;:!?\"'`“”‘’")


def validate_record(payload: object, *, seen_ids: set[str]) -> list[str]:
    reasons: list[str] = []
    if not isinstance(payload, dict):
        return ["record is not a JSON object"]

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        reasons.append("missing fields: " + ", ".join(missing))

    question_id = payload.get("id")
    if "id" in payload:
        if not isinstance(question_id, str):
            reasons.append("id must be a string")
        elif not question_id.strip():
            reasons.append("id is empty")
        else:
            normalized = question_id.strip()
            if normalized in seen_ids:
                reasons.append("duplicate id")
            else:
                seen_ids.add(normalized)

    company = payload.get("company")
    if "company" in payload and company not in COMPANIES:
        reasons.append("company is not an allowed value")

    role = payload.get("role")
    if "role" in payload and role not in ROLES:
        reasons.append("role is not an allowed value")

    seniority = payload.get("seniority")
    if "seniority" in payload and seniority not in SENIORITIES:
        reasons.append("seniority must be junior, mid, or senior")

    category = payload.get("category")
    if "category" in payload and category not in CATEGORIES:
        reasons.append("category must be technical, system_design, behavioral, or coding")

    difficulty = payload.get("difficulty")
    if "difficulty" in payload:
        if isinstance(difficulty, bool) or not isinstance(difficulty, int):
            reasons.append("difficulty must be an integer")
        elif not MIN_DIFFICULTY <= difficulty <= MAX_DIFFICULTY:
            reasons.append(f"difficulty must be between {MIN_DIFFICULTY} and {MAX_DIFFICULTY}")

    question_text = payload.get("question_text")
    if "question_text" in payload:
        text = _as_text(question_text)
        if text is None:
            reasons.append("question_text must be a string")
        elif not text.strip():
            reasons.append("question_text is empty")
        elif len(text.strip()) < MIN_QUESTION_TEXT_LENGTH:
            reasons.append(f"question_text is shorter than {MIN_QUESTION_TEXT_LENGTH} characters")
        elif _malformed_llm_output(text):
            reasons.append("question_text looks like malformed LLM output")

    rubric = payload.get("rubric")
    if "rubric" in payload:
        text = _as_text(rubric)
        if text is None:
            reasons.append("rubric must be a string")
        elif not text.strip():
            reasons.append("rubric is empty")
        elif len(text.strip()) < MIN_RUBRIC_LENGTH:
            reasons.append(f"rubric is shorter than {MIN_RUBRIC_LENGTH} characters")

    if "follow_up_hints" in payload:
        hints = payload.get("follow_up_hints")
        if hints is not None and not isinstance(hints, str):
            reasons.append("follow_up_hints must be a string or empty")

    return reasons


def load_jsonl_records(path: Path) -> list[tuple[int, Any]]:
    """Read JSONL without rewriting the file. Blank lines are ignored."""
    records: list[tuple[int, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                records.append((line_number, json.loads(text)))
            except json.JSONDecodeError:
                records.append((line_number, {"_raw": text, "_parse_error": True}))
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_corpus(
    input_path: Path,
    valid_path: Path,
    rejected_path: Path,
) -> ValidationSummary:
    if input_path.resolve() in {valid_path.resolve(), rejected_path.resolve()}:
        raise ValueError("Output paths must not be the input generated_questions.jsonl file.")

    records = load_jsonl_records(input_path)
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    id_counts: dict[str, int] = {}
    valid_rows: list[dict] = []
    rejected_rows: list[dict] = []

    for _line_number, payload in records:
        if isinstance(payload, dict) and isinstance(payload.get("id"), str) and payload["id"].strip():
            key = payload["id"].strip()
            id_counts[key] = id_counts.get(key, 0) + 1

    seen_ids.clear()
    for line_number, payload in records:
        if isinstance(payload, dict) and payload.get("_parse_error"):
            reasons = ["line is not valid JSON"]
            rejected_rows.append(
                {"line_number": line_number, "record": payload, "reasons": reasons}
            )
            continue
        reasons = validate_record(payload, seen_ids=seen_ids)
        if not reasons and isinstance(payload, dict):
            question_text = payload.get("question_text")
            if isinstance(question_text, str):
                text_key = normalize_question_text(question_text)
                if text_key:
                    if text_key in seen_texts:
                        reasons = ["duplicate_question_text"]
                    else:
                        seen_texts.add(text_key)
        if reasons:
            rejected_rows.append(
                {"line_number": line_number, "record": payload, "reasons": reasons}
            )
        else:
            assert isinstance(payload, dict)
            valid_rows.append(payload)

    write_jsonl(valid_path, valid_rows)
    write_jsonl(rejected_path, rejected_rows)

    duplicate_ids = sum(1 for count in id_counts.values() if count > 1)
    return ValidationSummary(
        total=len(records),
        valid=len(valid_rows),
        rejected=len(rejected_rows),
        duplicate_ids=duplicate_ids,
    )


def format_summary(summary: ValidationSummary) -> str:
    return (
        f"total: {summary.total}\n"
        f"valid: {summary.valid}\n"
        f"rejected: {summary.rejected}\n"
        f"duplicate IDs: {summary.duplicate_ids}"
    )
