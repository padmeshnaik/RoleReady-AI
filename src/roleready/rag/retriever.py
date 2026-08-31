"""Hybrid retrieval: Pinecone ranks, SQLite hydrates. No Streamlit."""

from __future__ import annotations

import logging
from pathlib import Path

from roleready.config.settings import Settings, get_settings
from roleready.db.models import GENERIC_COMPANY, Question
from roleready.db.sqlite import connect, get_question_by_id, unused_questions_with_generic_fallback
from roleready.rag.embeddings import EmbeddingClient
from roleready.rag.pinecone_store import PineconeQuestionStore
from roleready.session.models import InterviewSession

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """No unused question could be retrieved from Pinecone or SQLite."""


def build_query_text(session: InterviewSession) -> str:
    skills = ", ".join(session.skills) if session.skills else "(none listed)"
    parts = [
        f"Interview at {session.company} for a {session.seniority} {session.role}.",
        f"Focus category: {session.category}.",
        f"Candidate skills: {skills}.",
    ]
    previous = _previous_topic(session)
    if previous:
        parts.append(f"Previous answer or topic: {previous}")
    return " ".join(parts)


def _previous_topic(session: InterviewSession) -> str | None:
    if not session.turns:
        return None
    last = session.turns[-1]
    if last.follow_up_answer:
        return last.follow_up_answer
    if last.user_answer:
        return last.user_answer
    if last.question_text:
        return last.question_text
    return None


def _metadata_filter(
    session: InterviewSession,
    company: str,
    *,
    include_role: bool = True,
    include_seniority: bool = True,
    include_category: bool = True,
) -> dict:
    metadata_filter: dict = {"company": {"$eq": company}}
    if include_role:
        metadata_filter["role"] = {"$eq": session.role}
    if include_seniority:
        metadata_filter["seniority"] = {"$eq": session.seniority}
    if include_category:
        metadata_filter["category"] = {"$eq": session.category}
    return metadata_filter


class RetrievalService:
    """Semantic search in Pinecone, then load the full question from SQLite."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: EmbeddingClient | None = None,
        store: PineconeQuestionStore | None = None,
        db_path: str | Path | None = None,
        top_k: int = 15,
    ) -> None:
        self._settings = settings or get_settings()
        self._store = store
        self._embedder = embedder
        self._db_path = Path(db_path or self._settings.sqlite_path)
        self._top_k = top_k

    def _ensure_store(self) -> PineconeQuestionStore:
        if self._store is None:
            self._store = PineconeQuestionStore(settings=self._settings)
        return self._store

    def _ensure_embedder(self) -> EmbeddingClient:
        if self._embedder is None:
            store = self._ensure_store()
            try:
                index_dim = store.describe_dimension()
            except Exception:
                logger.exception("Could not read Pinecone index dimension; using settings default.")
                index_dim = None
            self._embedder = EmbeddingClient(settings=self._settings, dimensions=index_dim)
        return self._embedder

    def next_question(self, session: InterviewSession) -> Question:
        ids = self.retrieve_question_ids(session)
        conn = connect(self._db_path)
        try:
            for question_id in ids:
                question = get_question_by_id(conn, question_id)
                if question is not None:
                    return question
            fallback = unused_questions_with_generic_fallback(
                conn,
                session.company,
                role=session.role,
                seniority=session.seniority,
                category=session.category,
                used_ids=session.used_question_ids,
            )
            if fallback:
                return fallback[0]
        finally:
            conn.close()
        raise RetrievalError(
            "No unused interview question found for this company, role, seniority, and category."
        )

    def retrieve_question_ids(self, session: InterviewSession) -> list[str]:
        return [question_id for question_id, _score in self.retrieve_ranked(session)]

    def retrieve_ranked(
        self,
        session: InterviewSession,
        limit: int | None = None,
    ) -> list[tuple[str, float | None]]:
        """Unused question IDs with Pinecone similarity when available."""
        used = set(session.used_question_ids)
        ranked = self._pinecone_exact_ranked(session)
        unused = [(qid, score) for qid, score in ranked if qid not in used]
        if unused:
            return unused[: limit or len(unused)]
        fallback = [(qid, None) for qid in self._sqlite_fallback_ids(session)]
        return fallback[: limit or len(fallback)]

    def _pinecone_exact_ranked(self, session: InterviewSession) -> list[tuple[str, float | None]]:
        """One embedding plus at most company + Generic Pinecone queries. No extra filter stages."""
        try:
            vector = self._ensure_embedder().embed_query(build_query_text(session))
        except Exception:
            logger.exception("Embedding query failed; falling back to SQLite.")
            return []

        companies = [session.company]
        if session.company != GENERIC_COMPANY:
            companies.append(GENERIC_COMPANY)

        ordered: list[tuple[str, float | None]] = []
        seen: set[str] = set()
        store = self._ensure_store()
        for company in companies:
            try:
                matches = store.query_matches(
                    vector,
                    metadata_filter=_metadata_filter(session, company),
                    top_k=self._top_k,
                )
            except Exception:
                logger.exception("Pinecone query failed for company %s; stopping vector search.", company)
                return ordered
            for question_id, score in matches:
                if question_id in seen:
                    continue
                seen.add(question_id)
                ordered.append((question_id, score))
            unused = [qid for qid, _score in ordered if qid not in session.used_question_ids]
            if unused and company == session.company:
                return ordered
        return ordered

    def _sqlite_fallback_ids(self, session: InterviewSession) -> list[str]:
        conn = connect(self._db_path)
        try:
            questions = unused_questions_with_generic_fallback(
                conn,
                session.company,
                role=session.role,
                seniority=session.seniority,
                category=session.category,
                used_ids=session.used_question_ids,
            )
        finally:
            conn.close()
        return [question.id for question in questions]
