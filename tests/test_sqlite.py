"""SQLite question-bank tests. Always use a temporary database, never data/roleready.db."""

import json
from pathlib import Path

import pytest

from roleready.db.models import GENERIC_COMPANY, Question
from roleready.db.sqlite import (
    connect,
    create_tables,
    filter_by_category,
    filter_by_company,
    filter_by_role,
    filter_by_seniority,
    filter_with_generic_fallback,
    get_question_by_id,
    insert_questions,
    seed_from_jsonl,
    unused_questions_with_generic_fallback,
)

REAL_DB = Path(__file__).resolve().parents[1] / "data" / "roleready.db"


def _question(**kwargs: object) -> Question:
    data: dict = {
        "id": "q-test-1",
        "company": GENERIC_COMPANY,
        "role": "software_engineer",
        "seniority": "mid",
        "category": "technical",
        "difficulty": 3,
        "question_text": "What is an API?",
        "rubric": "Mentions interface, contract, and consumers.",
        "follow_up_hints": "Ask for an example.",
    }
    data.update(kwargs)
    return Question(**data)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test_questions.db"
    assert path.resolve() != REAL_DB.resolve()
    return path


@pytest.fixture
def conn(db_path: Path):
    connection = connect(db_path)
    create_tables(connection)
    yield connection
    connection.close()


def test_database_initialization(conn) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'questions';"
    ).fetchone()
    assert row is not None
    assert row["name"] == "questions"

    columns = {r["name"] for r in conn.execute("PRAGMA table_info(questions);").fetchall()}
    assert columns == {
        "id",
        "company",
        "role",
        "seniority",
        "category",
        "difficulty",
        "question_text",
        "rubric",
        "follow_up_hints",
    }


def test_question_insertion(conn) -> None:
    count = insert_questions(
        conn,
        [
            _question(id="ins-1"),
            _question(id="ins-2", company="Google"),
        ],
    )
    assert count == 2
    stored = conn.execute("SELECT COUNT(*) AS n FROM questions;").fetchone()["n"]
    assert stored == 2


def test_get_question_by_id(conn) -> None:
    insert_questions(conn, [_question(id="lookup-1", question_text="Unique prompt")])

    found = get_question_by_id(conn, "lookup-1")
    assert found is not None
    assert found.id == "lookup-1"
    assert found.question_text == "Unique prompt"
    assert found.company == GENERIC_COMPANY
    assert get_question_by_id(conn, "does-not-exist") is None


def test_filter_by_company(conn) -> None:
    insert_questions(
        conn,
        [
            _question(id="c-google", company="Google"),
            _question(id="c-generic", company=GENERIC_COMPANY),
            _question(id="c-amazon", company="Amazon"),
        ],
    )
    google = filter_by_company(conn, "Google")
    assert [q.id for q in google] == ["c-google"]
    assert all(q.company == "Google" for q in google)


def test_filter_by_role(conn) -> None:
    insert_questions(
        conn,
        [
            _question(id="r-se", role="software_engineer"),
            _question(id="r-de", role="data_engineer"),
            _question(id="r-ai", role="ai_engineer"),
        ],
    )
    engineers = filter_by_role(conn, "software_engineer")
    assert [q.id for q in engineers] == ["r-se"]
    assert all(q.role == "software_engineer" for q in engineers)


def test_filter_by_seniority(conn) -> None:
    insert_questions(
        conn,
        [
            _question(id="s-junior", seniority="junior"),
            _question(id="s-mid", seniority="mid"),
            _question(id="s-senior", seniority="senior"),
        ],
    )
    seniors = filter_by_seniority(conn, "senior")
    assert [q.id for q in seniors] == ["s-senior"]
    assert all(q.seniority == "senior" for q in seniors)


def test_filter_by_category(conn) -> None:
    insert_questions(
        conn,
        [
            _question(id="cat-tech", category="technical"),
            _question(id="cat-sd", category="system_design"),
            _question(id="cat-beh", category="behavioral"),
        ],
    )
    behavioral = filter_by_category(conn, "behavioral")
    assert [q.id for q in behavioral] == ["cat-beh"]
    assert all(q.category == "behavioral" for q in behavioral)


def test_generic_company_fallback_prefers_company_specific(conn) -> None:
    insert_questions(
        conn,
        [
            _question(
                id="google-se",
                company="Google",
                role="software_engineer",
                seniority="mid",
                category="technical",
            ),
            _question(
                id="generic-se",
                company=GENERIC_COMPANY,
                role="software_engineer",
                seniority="mid",
                category="technical",
            ),
        ],
    )
    results = filter_with_generic_fallback(
        conn,
        "Google",
        role="software_engineer",
        seniority="mid",
        category="technical",
    )
    assert [q.id for q in results] == ["google-se"]
    assert GENERIC_COMPANY not in {q.company for q in results}


def test_generic_company_fallback_when_no_company_match(conn) -> None:
    insert_questions(
        conn,
        [
            _question(
                id="generic-se",
                company=GENERIC_COMPANY,
                role="software_engineer",
                seniority="mid",
                category="technical",
            ),
            _question(
                id="google-de",
                company="Google",
                role="data_engineer",
                seniority="mid",
                category="technical",
            ),
        ],
    )
    results = filter_with_generic_fallback(
        conn,
        "Netflix",
        role="software_engineer",
        seniority="mid",
        category="technical",
    )
    assert [q.id for q in results] == ["generic-se"]
    assert all(q.company == GENERIC_COMPANY for q in results)


def test_generic_company_selection_does_not_return_named_companies(conn) -> None:
    insert_questions(
        conn,
        [
            _question(id="google-se", company="Google", role="software_engineer"),
            _question(id="generic-se", company=GENERIC_COMPANY, role="software_engineer"),
        ],
    )
    results = filter_with_generic_fallback(conn, GENERIC_COMPANY, role="software_engineer")
    assert [q.id for q in results] == ["generic-se"]
    assert all(q.company == GENERIC_COMPANY for q in results)


def test_unused_questions_relax_filters_when_exact_combo_missing(conn) -> None:
    insert_questions(
        conn,
        [
            _question(
                id="google-se",
                company="Google",
                role="software_engineer",
                seniority="mid",
                category="technical",
            ),
        ],
    )
    results = unused_questions_with_generic_fallback(
        conn,
        "Google",
        role="data_engineer",
        seniority="junior",
        category="behavioral",
        used_ids=[],
    )
    assert [q.id for q in results] == ["google-se"]


def _jsonl_record(**overrides: object) -> dict:
    record: dict = {
        "id": "gq-seed-1",
        "company": "Generic",
        "role": "software_engineer",
        "seniority": "mid",
        "category": "technical",
        "difficulty": 3,
        "question_text": "Describe how you would design a cache for a high-traffic API.",
        "rubric": (
            "A strong answer names eviction policy, TTL, stampede control, "
            "and consistency with the source of truth under failure."
        ),
        "follow_up_hints": "What happens if the cache is unavailable?",
    }
    record.update(overrides)
    return record


def _write_jsonl(path: Path, records: list[object]) -> None:
    path.write_text(
        "".join(
            (row if isinstance(row, str) else json.dumps(row)) + "\n" for row in records
        ),
        encoding="utf-8",
    )


def test_seed_from_jsonl_inserts_validated_fields(conn, tmp_path: Path) -> None:
    jsonl_path = tmp_path / "questions_clean.jsonl"
    record = _jsonl_record(category="coding", role="backend_engineer")
    _write_jsonl(jsonl_path, [record])

    summary = seed_from_jsonl(conn, jsonl_path)

    assert summary.records_read == 1
    assert summary.inserted == 1
    assert summary.skipped == 0
    assert summary.failed == 0
    stored = get_question_by_id(conn, "gq-seed-1")
    assert stored is not None
    assert stored.company == "Generic"
    assert stored.role == "backend_engineer"
    assert stored.seniority == "mid"
    assert stored.category == "coding"
    assert stored.difficulty == 3
    assert stored.question_text == record["question_text"]
    assert stored.rubric == record["rubric"]
    assert stored.follow_up_hints == record["follow_up_hints"]


def test_seed_from_jsonl_is_idempotent_and_skips_duplicate_ids(conn, tmp_path: Path) -> None:
    jsonl_path = tmp_path / "questions_clean.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            _jsonl_record(id="gq-dup-1"),
            _jsonl_record(id="gq-dup-1", question_text="A different prompt that is still long enough."),
        ],
    )

    first = seed_from_jsonl(conn, jsonl_path)
    assert first.inserted == 1
    assert first.skipped == 1
    assert first.failed == 0

    second = seed_from_jsonl(conn, jsonl_path)
    assert second.records_read == 2
    assert second.inserted == 0
    assert second.skipped == 2
    assert second.failed == 0
    count = conn.execute("SELECT COUNT(*) AS n FROM questions;").fetchone()["n"]
    assert count == 1
    stored = get_question_by_id(conn, "gq-dup-1")
    assert stored is not None
    assert "cache" in stored.question_text


def test_seed_from_jsonl_counts_invalid_records_as_failed(conn, tmp_path: Path) -> None:
    jsonl_path = tmp_path / "questions_clean.jsonl"
    _write_jsonl(
        jsonl_path,
        [
            _jsonl_record(id="gq-ok"),
            "{not json",
            _jsonl_record(id="gq-bad", difficulty=9),
        ],
    )

    summary = seed_from_jsonl(conn, jsonl_path)
    assert summary.records_read == 3
    assert summary.inserted == 1
    assert summary.skipped == 0
    assert summary.failed == 2
    assert get_question_by_id(conn, "gq-ok") is not None
    assert get_question_by_id(conn, "gq-bad") is None
