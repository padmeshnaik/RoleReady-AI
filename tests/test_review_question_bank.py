"""Corpus review helper tests. No LLM, Pinecone, or SQLite."""

import json
from pathlib import Path

from roleready.generation.review import format_review_stats, sample_for_review, write_review_sample


def _record(qid: str, **overrides: object) -> dict:
    data = {
        "id": qid,
        "company": "Generic",
        "role": "software_engineer",
        "seniority": "mid",
        "category": "technical",
        "difficulty": 3,
        "question_text": f"Question {qid}?",
        "rubric": "Strong answers cover trade-offs.",
        "follow_up_hints": "Ask why.",
    }
    data.update(overrides)
    return data


def test_format_review_stats_includes_totals_and_breakdowns() -> None:
    records = [
        _record("1", company="Generic", role="software_engineer", seniority="junior", category="coding", difficulty=1),
        _record("2", company="Google", role="ai_engineer", seniority="senior", category="technical", difficulty=5),
        _record("3", company="Generic", role="software_engineer", seniority="junior", category="coding", difficulty=1),
    ]
    text = format_review_stats(records)
    assert "total questions: 3" in text
    assert "Generic: 2" in text
    assert "software_engineer: 2" in text
    assert "junior: 2" in text
    assert "coding: 2" in text
    assert "1: 2" in text


def test_sample_is_deterministic_and_omits_id() -> None:
    records = [_record(f"gq-{index:02d}") for index in range(30)]
    first = sample_for_review(records, size=25, seed=42)
    second = sample_for_review(records, size=25, seed=42)
    assert first == second
    assert len(first) == 25
    assert "id" not in first[0]
    assert set(first[0]) == {
        "company",
        "role",
        "seniority",
        "category",
        "difficulty",
        "question_text",
        "rubric",
        "follow_up_hints",
    }


def test_write_review_sample_does_not_require_llm(tmp_path: Path) -> None:
    path = tmp_path / "question_review_sample.json"
    write_review_sample(path, sample_for_review([_record("a"), _record("b")], size=25, seed=42))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload) == 2
