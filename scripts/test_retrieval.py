"""Dev smoke test for RetrievalService. Does not call the interviewer LLM.

    python scripts/test_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from roleready.config.settings import get_settings  # noqa: E402
from roleready.db.models import GENERIC_COMPANY  # noqa: E402
from roleready.db.sqlite import connect, get_question_by_id  # noqa: E402
from roleready.rag.retriever import RetrievalService  # noqa: E402
from roleready.session.models import InterviewSession, SessionStatus  # noqa: E402

LIMIT = 5


def main() -> None:
    settings = get_settings()
    db_path = Path(settings.sqlite_path)
    if not db_path.is_absolute():
        db_path = ROOT / db_path

    session = InterviewSession(
        session_id="dev-retrieval",
        company=GENERIC_COMPANY,
        role="software_engineer",
        seniority="senior",
        category="system_design",
        skills=["Python", "APIs", "distributed systems"],
        status=SessionStatus.IN_PROGRESS,
    )

    service = RetrievalService(settings=settings, db_path=db_path, top_k=LIMIT)
    ranked = service.retrieve_ranked(session, limit=LIMIT)

    conn = connect(db_path)
    try:
        printed = 0
        for question_id, similarity in ranked:
            question = get_question_by_id(conn, question_id)
            if question is None:
                continue
            print(f"id: {question.id}")
            print(f"question: {question.question_text}")
            if similarity is None:
                print("similarity: n/a (SQLite fallback)")
            else:
                print(f"similarity: {similarity:.4f}")
            print()
            printed += 1
            if printed >= LIMIT:
                break
    finally:
        conn.close()

    if printed == 0:
        raise SystemExit("No questions retrieved. Confirm SQLite is seeded and Pinecone ingest has run.")


if __name__ == "__main__":
    main()
