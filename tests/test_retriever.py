"""RetrievalService tests with fake Pinecone and fake embeddings. No live APIs."""

from pathlib import Path

from roleready.config.settings import Settings
from roleready.db.models import GENERIC_COMPANY
from roleready.db.sqlite import connect, create_tables, insert_questions
from roleready.rag.embeddings import EmbeddingClient
from roleready.rag.pinecone_store import PineconeQuestionStore
from roleready.rag.retriever import RetrievalService, build_query_text
from roleready.session.models import InterviewSession, InterviewTurn, SessionStatus
from tests.fakes import make_question


def _settings(db_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-openai",
        pinecone_api_key="test-pinecone",
        sqlite_path=str(db_path),
        openai_embedding_model="text-embedding-3-small",
    )


def _session(**overrides: object) -> InterviewSession:
    data: dict = {
        "session_id": "s1",
        "company": "Google",
        "role": "software_engineer",
        "seniority": "mid",
        "category": "technical",
        "skills": ["python", "sql"],
        "status": SessionStatus.IN_PROGRESS,
    }
    data.update(overrides)
    return InterviewSession(**data)


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


class _FakeIndex:
    def __init__(self, hits_by_company: dict[str, list[str]]) -> None:
        self.hits_by_company = hits_by_company
        self.filters: list[dict] = []

    def query(self, vector, top_k, filter, include_metadata):
        self.filters.append(filter)
        company = filter["company"]["$eq"]
        ids = self.hits_by_company.get(company, [])
        return {
            "matches": [
                {"id": qid, "metadata": {"question_id": qid, "company": company}}
                for qid in ids[:top_k]
            ]
        }


def _service(
    tmp_path: Path,
    hits_by_company: dict[str, list[str]],
    questions: list | None = None,
) -> tuple[RetrievalService, _FakeEmbeddings]:
    db_path = tmp_path / "questions.db"
    conn = connect(db_path)
    create_tables(conn)
    insert_questions(
        conn,
        questions
        or [
            make_question("google-1", company="Google"),
            make_question("generic-1", company=GENERIC_COMPANY),
            make_question("google-2", company="Google"),
        ],
    )
    conn.close()

    embeddings = _FakeEmbeddings()
    settings = _settings(db_path)
    embedder = EmbeddingClient(settings=settings, embeddings=embeddings)
    store = PineconeQuestionStore(
        settings=settings,
        client=object(),
        index=_FakeIndex(hits_by_company),
    )
    service = RetrievalService(
        settings=settings,
        embedder=embedder,
        store=store,
        db_path=db_path,
    )
    return service, embeddings


def test_query_includes_session_fields_and_optional_previous_answer() -> None:
    session = _session()
    session.turns.append(
        InterviewTurn(
            question_id="google-1",
            question_number=1,
            question_text="What is REST?",
            user_answer="HTTP resources and verbs.",
        )
    )
    text = build_query_text(session)
    assert "Google" in text
    assert "software_engineer" in text
    assert "mid" in text
    assert "technical" in text
    assert "python" in text
    assert "HTTP resources and verbs." in text


def test_prefers_company_specific(tmp_path: Path) -> None:
    service, embeddings = _service(
        tmp_path,
        {"Google": ["google-1", "google-2"], GENERIC_COMPANY: ["generic-1"]},
    )
    session = _session()
    ids = service.retrieve_question_ids(session)
    assert ids[0] == "google-1"
    assert "generic-1" not in ids
    question = service.next_question(session)
    assert question.id == "google-1"
    assert question.company == "Google"
    assert question.rubric
    assert embeddings.queries


def test_generic_fallback_when_company_hits_exhausted(tmp_path: Path) -> None:
    service, _ = _service(
        tmp_path,
        {"Google": ["google-1"], GENERIC_COMPANY: ["generic-1"]},
    )
    session = _session(used_question_ids=["google-1"])
    ids = service.retrieve_question_ids(session)
    assert ids[0] == "generic-1"
    question = service.next_question(session)
    assert question.company == GENERIC_COMPANY
    assert question.rubric


def test_used_question_ids_are_removed(tmp_path: Path) -> None:
    service, _ = _service(
        tmp_path,
        {"Google": ["google-1", "google-2"], GENERIC_COMPANY: ["generic-1"]},
    )
    session = _session(used_question_ids=["google-1"])
    ids = service.retrieve_question_ids(session)
    assert "google-1" not in ids
    assert ids[0] == "google-2"


def test_generic_company_does_not_search_named_companies(tmp_path: Path) -> None:
    service, _ = _service(
        tmp_path,
        {"Google": ["google-1"], GENERIC_COMPANY: ["generic-1"]},
    )
    session = _session(company=GENERIC_COMPANY)
    ids = service.retrieve_question_ids(session)
    assert ids == ["generic-1"]
    assert len(service._store._index.filters) == 1



def test_sqlite_fallback_when_pinecone_returns_nothing(tmp_path: Path) -> None:
    service, _ = _service(tmp_path, {})
    session = _session()
    ids = service.retrieve_question_ids(session)
    assert ids[0] == "google-1"
    question = service.next_question(session)
    assert question.id == "google-1"
    assert "Key concepts" in question.rubric


def test_sqlite_fallback_when_exact_metadata_combo_is_missing(tmp_path: Path) -> None:
    service, _ = _service(tmp_path, {})
    session = _session(role="data_engineer", seniority="junior", category="behavioral")
    question = service.next_question(session)
    assert question.id in {"google-1", "generic-1", "google-2"}


def test_does_not_run_relaxed_pinecone_filter_stages(tmp_path: Path) -> None:
    service, _ = _service(tmp_path, {})
    session = _session()
    service.retrieve_question_ids(session)
    filters = service._store._index.filters
    assert len(filters) == 2
    companies = [item["company"]["$eq"] for item in filters]
    assert companies == ["Google", GENERIC_COMPANY]
    for item in filters:
        assert item["role"]["$eq"] == session.role
        assert item["seniority"]["$eq"] == session.seniority
        assert item["category"]["$eq"] == session.category

