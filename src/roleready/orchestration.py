"""Application facade. The UI talks to InterviewApp, not OpenAI or Pinecone clients."""

from __future__ import annotations

from pathlib import Path

from roleready.config.settings import Settings, get_settings
from roleready.db.models import Question
from roleready.db.sqlite import connect, list_questions
from roleready.llm.interviewer import OpenAIInterviewer
from roleready.llm.scorer import OpenAIScorer
from roleready.rag.retriever import RetrievalService
from roleready.session.interfaces import InterviewerService, QuestionRetriever, ScoringService
from roleready.session.manager import SessionManager
from roleready.session.models import InterviewSession


def resolve_sqlite_path(settings: Settings, project_root: str | Path | None = None) -> Path:
    db_path = Path(settings.sqlite_path)
    if db_path.is_absolute():
        return db_path
    root = Path(project_root) if project_root is not None else Path.cwd()
    return (root / db_path).resolve()


class InterviewApp:
    """Wires settings, SQLite, retrieval, interviewer, scorer, and session management."""

    def __init__(
        self,
        *,
        settings: Settings,
        db_path: str | Path,
        retriever: QuestionRetriever | None = None,
        interviewer: InterviewerService | None = None,
        scorer: ScoringService | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.settings = settings
        self.db_path = Path(db_path)
        self._retriever = retriever
        self._interviewer = interviewer
        self._scorer = scorer
        self._session_manager = session_manager
        if (
            session_manager is None
            and retriever is not None
            and interviewer is not None
            and scorer is not None
        ):
            self._session_manager = SessionManager(
                retriever=retriever,
                scorer=scorer,
                interviewer=interviewer,
                question_count=settings.interview_question_count,
            )

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        project_root: str | Path | None = None,
        *,
        live: bool = True,
    ) -> InterviewApp:
        """Wire the app from settings. live=False loads config and SQLite only (no OpenAI/Pinecone)."""
        settings = settings or get_settings()
        db_path = resolve_sqlite_path(settings, project_root)
        if not live:
            return cls(settings=settings, db_path=db_path)
        return cls(
            settings=settings,
            db_path=db_path,
            retriever=RetrievalService(settings=settings, db_path=db_path),
            interviewer=OpenAIInterviewer(
                settings=settings,
                question_count=settings.interview_question_count,
            ),
            scorer=OpenAIScorer(
                settings=settings,
                question_count=settings.interview_question_count,
            ),
        )

    def _ensure_live(self) -> None:
        """Build OpenAI and Pinecone clients on first interview action, not on Streamlit import."""
        if self._session_manager is not None:
            return
        if self._retriever is None:
            self._retriever = RetrievalService(settings=self.settings, db_path=self.db_path)
        if self._interviewer is None:
            self._interviewer = OpenAIInterviewer(
                settings=self.settings,
                question_count=self.settings.interview_question_count,
            )
        if self._scorer is None:
            self._scorer = OpenAIScorer(
                settings=self.settings,
                question_count=self.settings.interview_question_count,
            )
        self._session_manager = SessionManager(
            retriever=self._retriever,
            scorer=self._scorer,
            interviewer=self._interviewer,
            question_count=self.settings.interview_question_count,
        )

    @property
    def retriever(self) -> QuestionRetriever:
        self._ensure_live()
        assert self._retriever is not None
        return self._retriever

    @property
    def interviewer(self) -> InterviewerService:
        self._ensure_live()
        assert self._interviewer is not None
        return self._interviewer

    @property
    def scorer(self) -> ScoringService:
        self._ensure_live()
        assert self._scorer is not None
        return self._scorer

    @property
    def session_manager(self) -> SessionManager:
        self._ensure_live()
        assert self._session_manager is not None
        return self._session_manager

    def create_session(
        self,
        *,
        company: str,
        role: str,
        seniority: str,
        category: str,
        skills: list[str] | None = None,
        session_id: str | None = None,
    ) -> InterviewSession:
        return self.session_manager.create_session(
            company=company,
            role=role,
            seniority=seniority,
            category=category,
            skills=skills,
            session_id=session_id,
        )

    def start_interview(self, session: InterviewSession) -> InterviewSession:
        return self.session_manager.start_interview(session)

    def submit_answer(self, session: InterviewSession, answer: str) -> InterviewSession:
        return self.session_manager.submit_answer(session, answer)

    def list_bank_questions(self) -> list[Question]:
        """Read the SQLite question bank (source of truth). Does not call OpenAI or Pinecone."""
        conn = connect(self.db_path)
        try:
            return list_questions(conn)
        finally:
            conn.close()


def create_app(
    *,
    settings: Settings | None = None,
    retriever: QuestionRetriever | None = None,
    interviewer: InterviewerService | None = None,
    scorer: ScoringService | None = None,
    db_path: str | Path | None = None,
    session_manager: SessionManager | None = None,
    project_root: str | Path | None = None,
    live: bool = True,
) -> InterviewApp:
    """Build the app. Pass fake retriever/interviewer/scorer in tests; omit them for live services."""
    if retriever is not None and interviewer is not None and scorer is not None:
        resolved_settings = settings or get_settings()
        return InterviewApp(
            settings=resolved_settings,
            retriever=retriever,
            interviewer=interviewer,
            scorer=scorer,
            db_path=db_path or resolved_settings.sqlite_path,
            session_manager=session_manager,
        )
    if not live:
        settings = settings or get_settings()
        return InterviewApp(
            settings=settings,
            db_path=db_path or resolve_sqlite_path(settings, project_root),
            retriever=retriever,
            interviewer=interviewer,
            scorer=scorer,
            session_manager=session_manager,
        )
    settings = settings or get_settings()
    resolved_db = Path(db_path) if db_path is not None else resolve_sqlite_path(settings, project_root)
    return InterviewApp(
        settings=settings,
        db_path=resolved_db,
        retriever=retriever or RetrievalService(settings=settings, db_path=resolved_db),
        interviewer=interviewer
        or OpenAIInterviewer(settings=settings, question_count=settings.interview_question_count),
        scorer=scorer or OpenAIScorer(settings=settings, question_count=settings.interview_question_count),
        session_manager=session_manager,
    )
