from roleready.db.models import GENERIC_COMPANY
from roleready.llm.schemas import FinalEvaluationResult
from roleready.session.models import ChatMessage, InterviewSession, InterviewTurn, Score
from roleready.ui.components import (
    parse_final_report,
    score_dimension_averages,
    setup_choices,
    visible_messages,
)
from tests.fakes import make_question


def test_setup_choices_includes_generic_and_bank_values() -> None:
    choices = setup_choices(
        [
            make_question("a", company="Google", role="software_engineer", seniority="mid", category="technical"),
            make_question("b", company=GENERIC_COMPANY, role="data_engineer", seniority="senior", category="system_design"),
        ]
    )
    assert GENERIC_COMPANY in choices["companies"]
    assert "Google" in choices["companies"]
    assert "data_engineer" in choices["roles"]
    assert "system_design" in choices["categories"]


def test_parse_final_report_json() -> None:
    raw = FinalEvaluationResult(
        overall_score=7.5,
        recommendation="lean hire",
        summary="Solid fundamentals.",
        strengths=["clear communication"],
        gaps=["limited scale discussion"],
        practice_next=["practice hot keys"],
    ).model_dump_json()
    parsed = parse_final_report(raw)
    assert parsed is not None
    assert parsed.overall_score == 7.5
    assert "clear communication" in parsed.strengths


def test_parse_final_report_invalid_returns_none() -> None:
    assert parse_final_report("not-json") is None
    assert parse_final_report(None) is None


def test_score_dimension_averages() -> None:
    session = InterviewSession(
        session_id="s",
        company="Generic",
        role="software_engineer",
        seniority="mid",
        category="technical",
    )
    session.turns.append(
        InterviewTurn(
            question_id="q1",
            question_number=1,
            question_text="Q1",
            score=Score(score=8, technical_accuracy=8, communication=6, structure=7),
        )
    )
    session.turns.append(
        InterviewTurn(
            question_id="q2",
            question_number=2,
            question_text="Q2",
            score=Score(score=6, technical_accuracy=4, communication=8, structure=5),
        )
    )
    averages = score_dimension_averages(session)
    assert averages["Overall"] == 7.0
    assert averages["Technical accuracy"] == 6.0
    assert averages["Communication"] == 7.0
    assert averages["Structure"] == 6.0


def test_visible_messages_hide_next_question_during_score_review() -> None:
    session = InterviewSession(
        session_id="s",
        company="Generic",
        role="software_engineer",
        seniority="mid",
        category="technical",
    )
    session.messages = [
        ChatMessage(role="interviewer", content="Q1?"),
        ChatMessage(role="candidate", content="A1"),
        ChatMessage(role="interviewer", content="Q2?"),
        ChatMessage(role="candidate", content="A2"),
        ChatMessage(role="interviewer", content="Q3?"),
    ]
    session.turns = [
        InterviewTurn(
            question_id="q1",
            question_number=1,
            question_text="Q1?",
            user_answer="A1",
            score=Score(score=5),
        ),
        InterviewTurn(
            question_id="q2",
            question_number=2,
            question_text="Q2?",
            user_answer="A2",
            score=Score(score=0),
        ),
        InterviewTurn(question_id="q3", question_number=3, question_text="Q3?"),
    ]
    visible = visible_messages(session, review_turn_index=1)
    assert [m.content for m in visible] == ["Q1?", "A1", "Q2?", "A2"]
    assert visible_messages(session, None)[-1].content == "Q3?"
