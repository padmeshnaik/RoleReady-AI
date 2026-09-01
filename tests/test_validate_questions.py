"""Question corpus validation tests. No OpenAI, Pinecone, or SQLite."""

import json
from pathlib import Path

from roleready.generation.validate import (
    format_summary,
    normalize_question_text,
    validate_corpus,
    validate_record,
)


def _record(**overrides: object) -> dict:
    data: dict = {
        "id": "gq-0001",
        "company": "Generic",
        "role": "software_engineer",
        "seniority": "mid",
        "category": "technical",
        "difficulty": 3,
        "question_text": "How would you design retries for an idempotent payments API under load?",
        "rubric": (
            "A strong answer covers idempotency keys, at-least-once delivery, "
            "backoff, jitter, poison messages, and observing retry storms in production."
        ),
        "follow_up_hints": "What happens if the client retries a non-idempotent POST?",
    }
    data.update(overrides)
    return data


def test_validate_record_accepts_empty_follow_up_hints() -> None:
    reasons = validate_record(_record(follow_up_hints=""), seen_ids=set())
    assert reasons == []


def test_validate_record_rejects_non_string_follow_up_hints() -> None:
    reasons = validate_record(_record(follow_up_hints=["a"]), seen_ids=set())
    assert any("follow_up_hints" in reason for reason in reasons)


def test_validate_record_rejects_duplicate_ids() -> None:
    seen: set[str] = set()
    first = validate_record(_record(), seen_ids=seen)
    second = validate_record(_record(), seen_ids=seen)
    assert first == []
    assert "duplicate id" in second


def test_validate_record_rejects_malformed_question_text() -> None:
    reasons = validate_record(
        _record(question_text="Sure, here is the JSON output you asked for in a paragraph that is long enough."),
        seen_ids=set(),
    )
    assert any("malformed" in reason for reason in reasons)


def test_normalize_question_text_collapses_whitespace_and_wrapping_punctuation() -> None:
    assert normalize_question_text("Explain REST APIs.") == normalize_question_text(
        "  Explain REST APIs.  "
    )
    assert normalize_question_text("Explain REST APIs.") == "explain rest apis"


def test_validate_corpus_keeps_first_valid_normalized_duplicate(tmp_path: Path) -> None:
    source = tmp_path / "generated_questions.jsonl"
    valid_path = tmp_path / "questions_valid.jsonl"
    rejected_path = tmp_path / "questions_rejected.jsonl"
    first = _record(id="gq-0001", question_text="Explain REST APIs in a production service.")
    second = _record(id="gq-0002", question_text="  Explain REST APIs in a production service.  ")
    source.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    original = source.read_text(encoding="utf-8")

    summary = validate_corpus(source, valid_path, rejected_path)

    assert source.read_text(encoding="utf-8") == original
    assert summary.valid == 1
    assert summary.rejected == 1
    assert json.loads(valid_path.read_text(encoding="utf-8").splitlines()[0])["id"] == "gq-0001"
    rejected = json.loads(rejected_path.read_text(encoding="utf-8").splitlines()[0])
    assert rejected["reasons"] == ["duplicate_question_text"]
    assert rejected["record"]["id"] == "gq-0002"


def test_invalid_first_record_does_not_block_later_normalized_text(tmp_path: Path) -> None:
    source = tmp_path / "generated_questions.jsonl"
    valid_path = tmp_path / "questions_valid.jsonl"
    rejected_path = tmp_path / "questions_rejected.jsonl"
    invalid = _record(id="gq-0001", company="Netflix", question_text="Explain REST APIs in a production service.")
    valid = _record(id="gq-0002", question_text="  Explain REST APIs in a production service.  ")
    source.write_text(json.dumps(invalid) + "\n" + json.dumps(valid) + "\n", encoding="utf-8")

    summary = validate_corpus(source, valid_path, rejected_path)
    assert summary.valid == 1
    assert json.loads(valid_path.read_text(encoding="utf-8").splitlines()[0])["id"] == "gq-0002"


def test_validate_corpus_writes_outputs_without_touching_input(tmp_path: Path) -> None:
    source = tmp_path / "generated_questions.jsonl"
    valid_path = tmp_path / "questions_valid.jsonl"
    rejected_path = tmp_path / "questions_rejected.jsonl"
    good = _record()
    bad = _record(id="gq-0002", company="Netflix", question_text="Too short")
    source.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
    original = source.read_text(encoding="utf-8")

    summary = validate_corpus(source, valid_path, rejected_path)

    assert source.read_text(encoding="utf-8") == original
    assert summary.total == 2
    assert summary.valid == 1
    assert summary.rejected == 1
    assert summary.duplicate_ids == 0
    assert "total: 2" in format_summary(summary)
    assert json.loads(valid_path.read_text(encoding="utf-8").splitlines()[0])["id"] == "gq-0001"
    rejected = json.loads(rejected_path.read_text(encoding="utf-8").splitlines()[0])
    assert "company is not an allowed value" in rejected["reasons"]


def test_validation_modules_do_not_import_openai_pinecone_or_sqlite() -> None:
    paths = [
        Path("src/roleready/generation/validate.py"),
        Path("scripts/validate_questions.py"),
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "import openai" not in source
        assert "from openai" not in source
        assert "from pinecone" not in source
        assert "import pinecone" not in source
        assert "roleready.db" not in source
        assert "import sqlite3" not in source
