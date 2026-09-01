"""SQLite question bank."""

from roleready.db.models import GENERIC_COMPANY, Question
from roleready.db.sqlite import (
    SeedSummary,
    connect,
    create_tables,
    filter_by_category,
    filter_by_company,
    filter_by_role,
    filter_by_seniority,
    filter_with_generic_fallback,
    format_seed_summary,
    unused_questions_with_generic_fallback,
    get_question_by_id,
    insert_questions,
    list_questions,
    load_questions_from_csv,
    seed_from_csv,
    seed_from_jsonl,
)

__all__ = [
    "GENERIC_COMPANY",
    "Question",
    "connect",
    "create_tables",
    "filter_by_category",
    "filter_by_company",
    "filter_by_role",
    "filter_by_seniority",
    "filter_with_generic_fallback",
    "get_question_by_id",
    "unused_questions_with_generic_fallback",
    "SeedSummary",
    "format_seed_summary",
    "insert_questions",
    "list_questions",
    "load_questions_from_csv",
    "seed_from_csv",
    "seed_from_jsonl",
]
