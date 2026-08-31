"""Structured scorer parsing tests. Fake model payloads only — no OpenAI calls."""

import pytest

from roleready.config.settings import Settings
from roleready.llm.scorer import OpenAIScorer, ScoringError, parse_turn_score
from roleready.session.models import InterviewSession, InterviewTurn, SessionStatus


def _valid_payload(**overrides: object) -> dict:
    payload: dict = {
        "score": 7,
        "technical_accuracy": 7,
        "communication": 8,
        "structure": 6,
        "feedback": "Mention caching and uniqueness.",
        "missed_points": ["hot-key handling"],
    }
    payload.update(overrides)
    return payload


class _FakeStructured:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def invoke(self, messages: list) -> object:
        return self.payload


class _FakeScorerLLM:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def with_structured_output(self, schema: type) -> _FakeStructured:
        return _FakeStructured(self.payload)


def _scorer_with_fake(payload: object) -> OpenAIScorer:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-openai",
        pinecone_api_key="test-pinecone",
    )
    return OpenAIScorer(settings=settings, llm=_FakeScorerLLM(payload))


def _session() -> InterviewSession:
    return InterviewSession(
        session_id="s1",
        company="Generic",
        role="software_engineer",
        seniority="mid",
        category="technical",
        status=SessionStatus.AWAITING_ANSWER,
    )


def _turn() -> InterviewTurn:
    return InterviewTurn(
        question_id="q01",
        question_number=1,
        question_text="Design a URL shortener.",
        rubric="API, uniqueness, scale.",
        user_answer="Hash the URL and store it.",
    )


def test_valid_score_accepted() -> None:
    score = parse_turn_score(_valid_payload())
    assert score.score == 7
    assert score.technical_accuracy == 7
    assert score.communication == 8
    assert score.structure == 6
    assert "caching" in score.feedback
    assert score.missed_points == ("hot-key handling",)

    from_service = _scorer_with_fake(_valid_payload()).score_turn(_session(), _turn())
    assert from_service.score == 7


def test_score_below_0_rejected() -> None:
    with pytest.raises(ScoringError, match="malformed or invalid"):
        parse_turn_score(_valid_payload(score=-1))
    with pytest.raises(ScoringError):
        _scorer_with_fake(_valid_payload(score=-1)).score_turn(_session(), _turn())


def test_score_above_10_rejected() -> None:
    with pytest.raises(ScoringError, match="malformed or invalid"):
        parse_turn_score(_valid_payload(score=11))
    with pytest.raises(ScoringError):
        _scorer_with_fake(_valid_payload(score=11)).score_turn(_session(), _turn())


def test_missing_feedback_handled() -> None:
    payload = _valid_payload()
    del payload["feedback"]
    score = parse_turn_score(payload)
    assert score.feedback == ""
    assert score.score == 7


def test_malformed_output_results_in_controlled_error() -> None:
    with pytest.raises(ScoringError, match="malformed or invalid"):
        parse_turn_score("this is not json {")
    with pytest.raises(ScoringError, match="malformed or invalid"):
        parse_turn_score({"not": "a score object"})
    with pytest.raises(ScoringError, match="malformed or invalid"):
        parse_turn_score(None)
    with pytest.raises(ScoringError):
        _scorer_with_fake("<<<broken>>>").score_turn(_session(), _turn())
