"""InterviewApp facade tests. Uses fakes — no OpenAI, Pinecone, or Streamlit."""

from pathlib import Path

from roleready.config.settings import Settings
from roleready.db.sqlite import connect, create_tables, insert_questions
from roleready.orchestration import InterviewApp, create_app
from roleready.session.models import SessionStatus
from tests.fakes import FakeRetriever, FakeScorer, OneFollowUpInterviewer, make_question


def _settings(db_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-openai",
        pinecone_api_key="test-pinecone",
        sqlite_path=str(db_path),
        interview_question_count=10,
    )


def test_create_app_accepts_fake_services(tmp_path: Path) -> None:
    db_path = tmp_path / "bank.db"
    conn = connect(db_path)
    create_tables(conn)
    insert_questions(conn, [make_question("q01"), make_question("q02")])
    conn.close()

    questions = [make_question(f"q{i:02d}") for i in range(1, 13)]
    app = create_app(
        settings=_settings(db_path),
        retriever=FakeRetriever(questions),
        interviewer=OneFollowUpInterviewer(),
        scorer=FakeScorer(),
        db_path=db_path,
    )

    bank = app.list_bank_questions()
    assert {q.id for q in bank} == {"q01", "q02"}

    session = app.create_session(
        company="Generic",
        role="software_engineer",
        seniority="mid",
        category="technical",
        skills=["python"],
    )
    assert session.status is SessionStatus.SETUP

    session = app.start_interview(session)
    assert session.status is SessionStatus.AWAITING_ANSWER
    assert session.turns[0].question_number == 1

    session = app.submit_answer(session, "First answer")
    assert session.question_index == 0
    session = app.submit_answer(session, "Follow-up answer")
    assert session.question_index == 1


def test_interview_app_constructor_does_not_require_live_clients(tmp_path: Path) -> None:
    import inspect

    db_path = tmp_path / "bank.db"
    app = InterviewApp(
        settings=_settings(db_path),
        retriever=FakeRetriever([make_question("q01")]),
        interviewer=OneFollowUpInterviewer(),
        scorer=FakeScorer(),
        db_path=db_path,
    )
    assert app.session_manager is not None
    source = Path(inspect.getfile(InterviewApp)).read_text(encoding="utf-8")
    assert "import streamlit" not in source
    assert "from streamlit" not in source


def test_from_settings_live_false_does_not_construct_openai_or_pinecone(
    tmp_path: Path, monkeypatch
) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("live OpenAI or Pinecone client should not be constructed")

    monkeypatch.setattr("roleready.orchestration.RetrievalService", boom)
    monkeypatch.setattr("roleready.orchestration.OpenAIInterviewer", boom)
    monkeypatch.setattr("roleready.orchestration.OpenAIScorer", boom)

    db_path = tmp_path / "bank.db"
    conn = connect(db_path)
    create_tables(conn)
    conn.close()

    app = InterviewApp.from_settings(settings=_settings(db_path), live=False)
    assert app.list_bank_questions() == []
    assert app._retriever is None
    assert app._interviewer is None
    assert app._scorer is None

