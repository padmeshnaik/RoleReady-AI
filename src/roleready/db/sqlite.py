"""SQLite access for the interview question bank. Keep this layer out of the UI."""

from __future__ import annotations

import csv
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from roleready.db.models import GENERIC_COMPANY, MAX_DIFFICULTY, MIN_DIFFICULTY, Question
from roleready.generation.validate import load_jsonl_records, validate_record

CREATE_QUESTIONS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    company TEXT NOT NULL,
    role TEXT NOT NULL,
    seniority TEXT NOT NULL,
    category TEXT NOT NULL,
    difficulty INTEGER NOT NULL CHECK (
        difficulty >= {MIN_DIFFICULTY} AND difficulty <= {MAX_DIFFICULTY}
    ),
    question_text TEXT NOT NULL,
    rubric TEXT NOT NULL,
    follow_up_hints TEXT
);
"""

CREATE_INDEXES_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_questions_company ON questions (company);",
    "CREATE INDEX IF NOT EXISTS idx_questions_role ON questions (role);",
    "CREATE INDEX IF NOT EXISTS idx_questions_seniority ON questions (seniority);",
    "CREATE INDEX IF NOT EXISTS idx_questions_category ON questions (category);",
)

INSERT_QUESTION_SQL = """
INSERT OR REPLACE INTO questions (
    id, company, role, seniority, category, difficulty,
    question_text, rubric, follow_up_hints
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

INSERT_QUESTION_IF_ABSENT_SQL = """
INSERT OR IGNORE INTO questions (
    id, company, role, seniority, category, difficulty,
    question_text, rubric, follow_up_hints
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


@dataclass(frozen=True)
class SeedSummary:
    records_read: int
    inserted: int
    skipped: int
    failed: int


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_QUESTIONS_TABLE_SQL)
    for statement in CREATE_INDEXES_SQL:
        conn.execute(statement)
    conn.commit()


def insert_questions(conn: sqlite3.Connection, questions: Iterable[Question]) -> int:
    rows = [_question_to_row(question) for question in questions]
    conn.executemany(INSERT_QUESTION_SQL, rows)
    conn.commit()
    return len(rows)


def get_question_by_id(conn: sqlite3.Connection, question_id: str) -> Question | None:
    row = conn.execute(
        "SELECT * FROM questions WHERE id = ?;",
        (question_id,),
    ).fetchone()
    return _row_to_question(row) if row is not None else None


def filter_by_company(conn: sqlite3.Connection, company: str) -> list[Question]:
    return _filter(conn, "company", company)


def filter_by_role(conn: sqlite3.Connection, role: str) -> list[Question]:
    return _filter(conn, "role", role)


def filter_by_seniority(conn: sqlite3.Connection, seniority: str) -> list[Question]:
    return _filter(conn, "seniority", seniority)


def filter_by_category(conn: sqlite3.Connection, category: str) -> list[Question]:
    return _filter(conn, "category", category)


def list_questions(conn: sqlite3.Connection) -> list[Question]:
    """Return every question. SQLite remains the source of truth."""
    return _query_questions(conn)


def filter_with_generic_fallback(
    conn: sqlite3.Connection,
    company: str,
    *,
    role: str | None = None,
    seniority: str | None = None,
    category: str | None = None,
) -> list[Question]:
    """Prefer company-specific questions; use Generic rows if none match.

    If ``company`` is already Generic, only Generic questions are returned.
    Optional role, seniority, and category filters apply to both stages.
    """
    if company == GENERIC_COMPANY:
        return _query_questions(
            conn,
            company=GENERIC_COMPANY,
            role=role,
            seniority=seniority,
            category=category,
        )

    specific = _query_questions(
        conn,
        company=company,
        role=role,
        seniority=seniority,
        category=category,
    )
    if specific:
        return specific
    return _query_questions(
        conn,
        company=GENERIC_COMPANY,
        role=role,
        seniority=seniority,
        category=category,
    )


def unused_questions_with_generic_fallback(
    conn: sqlite3.Connection,
    company: str,
    *,
    role: str,
    seniority: str,
    category: str,
    used_ids: Iterable[str],
) -> list[Question]:
    """Prefer exact metadata matches, then relax filters so a sparse bank still works.

    Order: selected company, then Generic. Other named companies are last resort only.
    """
    used = set(used_ids)
    companies = [company]
    if company != GENERIC_COMPANY:
        companies.append(GENERIC_COMPANY)

    filter_stages: list[dict[str, str]] = [
        {"role": role, "seniority": seniority, "category": category},
        {"role": role, "category": category},
        {"role": role, "seniority": seniority},
        {"role": role},
        {"category": category},
        {},
    ]

    for filters in filter_stages:
        found = _unused_for_companies(conn, companies, used, filters)
        if found:
            return found

    leftovers: list[Question] = []
    seen: set[str] = set()
    for question in _query_questions(conn):
        if question.id in used or question.id in seen:
            continue
        seen.add(question.id)
        leftovers.append(question)
    return leftovers


def _unused_for_companies(
    conn: sqlite3.Connection,
    companies: list[str],
    used: set[str],
    filters: dict[str, str],
) -> list[Question]:
    results: list[Question] = []
    seen: set[str] = set()
    for stage_company in companies:
        for question in _query_questions(conn, company=stage_company, **filters):
            if question.id in used or question.id in seen:
                continue
            seen.add(question.id)
            results.append(question)
    return results


def load_questions_from_csv(csv_path: str | Path) -> list[Question]:
    path = Path(csv_path)
    questions: list[Question] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            hints = (row.get("follow_up_hints") or "").strip()
            questions.append(
                Question(
                    id=row["id"].strip(),
                    company=row["company"].strip(),
                    role=row["role"].strip(),
                    seniority=row["seniority"].strip(),
                    category=row["category"].strip(),
                    difficulty=int(row["difficulty"]),
                    question_text=row["question_text"].strip(),
                    rubric=row["rubric"].strip(),
                    follow_up_hints=hints or None,
                )
            )
    return questions


def seed_from_csv(conn: sqlite3.Connection, csv_path: str | Path) -> int:
    create_tables(conn)
    return insert_questions(conn, load_questions_from_csv(csv_path))


def seed_from_jsonl(conn: sqlite3.Connection, jsonl_path: str | Path) -> SeedSummary:
    """Validate cleaned JSONL and insert new question IDs only. Safe to rerun."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Question seed JSONL not found: {path}")

    create_tables(conn)
    records = load_jsonl_records(path)
    existing_ids = {
        str(row["id"]) for row in conn.execute("SELECT id FROM questions;").fetchall()
    }
    seen_ids: set[str] = set()
    inserted = 0
    skipped = 0
    failed = 0

    for _line_number, payload in records:
        if isinstance(payload, dict) and payload.get("_parse_error"):
            failed += 1
            continue
        reasons = validate_record(payload, seen_ids=set())
        if reasons or not isinstance(payload, dict):
            failed += 1
            continue
        try:
            question = _question_from_jsonl_payload(payload)
        except (KeyError, TypeError, ValueError):
            failed += 1
            continue
        if question.id in existing_ids or question.id in seen_ids:
            skipped += 1
            continue
        cursor = conn.execute(INSERT_QUESTION_IF_ABSENT_SQL, _question_to_row(question))
        if cursor.rowcount == 1:
            inserted += 1
            existing_ids.add(question.id)
            seen_ids.add(question.id)
        else:
            skipped += 1
            seen_ids.add(question.id)

    conn.commit()
    return SeedSummary(
        records_read=len(records),
        inserted=inserted,
        skipped=skipped,
        failed=failed,
    )


def format_seed_summary(summary: SeedSummary) -> str:
    return (
        f"records read: {summary.records_read}\n"
        f"inserted: {summary.inserted}\n"
        f"skipped: {summary.skipped}\n"
        f"failed: {summary.failed}"
    )


def _question_from_jsonl_payload(payload: dict) -> Question:
    hints = payload.get("follow_up_hints")
    hint_text = hints.strip() if isinstance(hints, str) else None
    return Question(
        id=str(payload["id"]).strip(),
        company=str(payload["company"]).strip(),
        role=str(payload["role"]).strip(),
        seniority=str(payload["seniority"]).strip(),
        category=str(payload["category"]).strip(),
        difficulty=int(payload["difficulty"]),
        question_text=str(payload["question_text"]).strip(),
        rubric=str(payload["rubric"]).strip(),
        follow_up_hints=hint_text or None,
    )


def _filter(conn: sqlite3.Connection, column: str, value: str) -> list[Question]:
    allowed = {"company", "role", "seniority", "category"}
    if column not in allowed:
        raise ValueError(f"Unsupported filter column: {column}")
    return _query_questions(conn, **{column: value})


def _query_questions(
    conn: sqlite3.Connection,
    *,
    company: str | None = None,
    role: str | None = None,
    seniority: str | None = None,
    category: str | None = None,
) -> list[Question]:
    clauses: list[str] = []
    params: list[str] = []
    filters = {
        "company": company,
        "role": role,
        "seniority": seniority,
        "category": category,
    }
    for column, value in filters.items():
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM questions{where} ORDER BY id;",
        params,
    ).fetchall()
    return [_row_to_question(row) for row in rows]


def _question_to_row(question: Question) -> tuple:
    return (
        question.id,
        question.company,
        question.role,
        question.seniority,
        question.category,
        question.difficulty,
        question.question_text,
        question.rubric,
        question.follow_up_hints,
    )


def _row_to_question(row: sqlite3.Row) -> Question:
    hints = row["follow_up_hints"]
    return Question(
        id=row["id"],
        company=row["company"],
        role=row["role"],
        seniority=row["seniority"],
        category=row["category"],
        difficulty=row["difficulty"],
        question_text=row["question_text"],
        rubric=row["rubric"],
        follow_up_hints=hints if hints else None,
    )
