"""SessionManager tests with fake retrieval and fake scoring. No OpenAI or Pinecone."""

import inspect
from pathlib import Path

import pytest

from roleready.session.manager import DEFAULT_QUESTION_COUNT, SessionError, SessionManager
from roleready.session.models import SessionStatus

from tests.fakes import (
    DuplicateIdRetriever,
    FakeRetriever,
    FakeScorer,
    OneFollowUpInterviewer,
    make_question,
)


def _manager(interviewer=None) -> tuple[SessionManager, FakeRetriever, FakeScorer]:
    questions = [make_question(f"q{i:02d}") for i in range(1, DEFAULT_QUESTION_COUNT + 5)]
    retriever = FakeRetriever(questions)
    scorer = FakeScorer()
    manager = SessionManager(
        retriever=retriever,
        scorer=scorer,
        interviewer=interviewer,
        question_count=DEFAULT_QUESTION_COUNT,
    )
    return manager, retriever, scorer


def _new_session(manager: SessionManager):
    return manager.create_session(
        company="Google",
        role="software_engineer",
        seniority="mid",
        category="technical",
        skills=["python", "sql"],
    )


def test_interview_starts_at_question_1() -> None:
    manager, retriever, scorer = _manager()
    session = manager.start_interview(_new_session(manager))

    assert session.status is SessionStatus.AWAITING_ANSWER
    assert session.question_index == 0
    assert session.turns[0].question_number == 1
    assert session.turns[0].question_id == "q01"
    assert len(retriever.calls) == 1
    assert scorer.scored_turns == []
    assert scorer.final_eval_calls == 0


def test_main_questions_increment_the_counter() -> None:
    manager, _, _ = _manager()
    session = manager.start_interview(_new_session(manager))
    assert session.question_index == 0

    manager.submit_answer(session, "Answer one")
    assert session.question_index == 1
    assert session.turns[0].is_scored
    assert session.turns[1].question_number == 2

    manager.submit_answer(session, "Answer two")
    assert session.question_index == 2
    assert session.turns[1].is_scored
    assert session.turns[2].question_number == 3


def test_follow_up_questions_do_not_increment_the_counter() -> None:
    manager, retriever, scorer = _manager(interviewer=OneFollowUpInterviewer())
    session = manager.start_interview(_new_session(manager))
    index_before = session.question_index
    used_before = list(session.used_question_ids)
    retriever_calls_before = len(retriever.calls)

    session = manager.submit_answer(session, "I would use an index.")

    assert session.question_index == index_before
    assert session.status is SessionStatus.AWAITING_ANSWER
    assert session.current_turn is not None
    assert session.current_turn.interviewer_followup
    assert session.current_turn.score is None
    assert session.used_question_ids == used_before
    assert len(retriever.calls) == retriever_calls_before
    assert scorer.scored_turns == []


def test_used_question_ids_are_not_repeated() -> None:
    manager, retriever, _ = _manager()
    session = manager.start_interview(_new_session(manager))
    for i in range(4):
        manager.submit_answer(session, f"Answer {i}")

    assert session.used_question_ids == ["q01", "q02", "q03", "q04", "q05"]
    assert len(session.used_question_ids) == len(set(session.used_question_ids))
    for seen in retriever.calls:
        assert len(seen) == len(set(seen))


def test_duplicate_question_id_from_retriever_is_rejected() -> None:
    question = make_question("dup")
    manager = SessionManager(
        retriever=DuplicateIdRetriever(question),
        scorer=FakeScorer(),
        question_count=DEFAULT_QUESTION_COUNT,
    )
    session = manager.start_interview(_new_session(manager))
    with pytest.raises(SessionError, match="cannot be reused"):
        manager.submit_answer(session, "First answer")


def test_after_10_scored_answers_interview_is_complete() -> None:
    manager, retriever, scorer = _manager()
    session = manager.start_interview(_new_session(manager))
    for i in range(DEFAULT_QUESTION_COUNT):
        assert session.status is SessionStatus.AWAITING_ANSWER
        manager.submit_answer(session, f"Answer {i}")

    assert session.question_index == DEFAULT_QUESTION_COUNT
    assert session.status is SessionStatus.COMPLETE
    assert len([turn for turn in session.turns if turn.is_scored]) == DEFAULT_QUESTION_COUNT
    assert len(session.turns) == DEFAULT_QUESTION_COUNT
    assert len(retriever.calls) == DEFAULT_QUESTION_COUNT


def test_final_report_is_requested_once() -> None:
    manager, _, scorer = _manager()
    session = manager.start_interview(_new_session(manager))
    assert scorer.final_eval_calls == 0
    for i in range(DEFAULT_QUESTION_COUNT):
        manager.submit_answer(session, f"Answer {i}")

    assert scorer.final_eval_calls == 1
    assert session.final_evaluation is not None
    with pytest.raises(SessionError):
        manager.submit_answer(session, "Another answer")
    assert scorer.final_eval_calls == 1


def test_eleventh_main_question_cannot_be_generated() -> None:
    manager, retriever, _ = _manager()
    session = manager.start_interview(_new_session(manager))
    for i in range(DEFAULT_QUESTION_COUNT):
        manager.submit_answer(session, f"Answer {i}")

    assert session.status is SessionStatus.COMPLETE
    assert len(session.turns) == DEFAULT_QUESTION_COUNT
    assert len(retriever.calls) == DEFAULT_QUESTION_COUNT

    with pytest.raises(SessionError, match="11th main question"):
        manager._pose_next_question(session)

    assert len(session.turns) == DEFAULT_QUESTION_COUNT
    assert len(retriever.calls) == DEFAULT_QUESTION_COUNT
    assert session.status is SessionStatus.COMPLETE


def test_manager_does_not_import_streamlit_or_live_clients() -> None:
    source = Path(inspect.getfile(SessionManager)).read_text(encoding="utf-8")
    assert "import streamlit" not in source
    assert "from streamlit" not in source
    assert "import openai" not in source
    assert "import pinecone" not in source
