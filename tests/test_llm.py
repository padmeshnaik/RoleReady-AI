"""LLM service tests with injected fakes. Does not call OpenAI, Pinecone, or Streamlit."""

from langchain_core.messages import HumanMessage, SystemMessage

from roleready.config.settings import Settings
from roleready.llm.interviewer import OpenAIInterviewer
from roleready.llm.schemas import FinalEvaluationResult, TurnScoreResult
from roleready.llm.scorer import OpenAIScorer
from roleready.session.models import InterviewSession, InterviewTurn, Score, SessionStatus
from tests.fakes import make_question


def _settings() -> Settings:
    return Settings(_env_file=None, openai_api_key="test-openai", pinecone_api_key="test-pinecone")


def _session() -> InterviewSession:
    return InterviewSession(
        session_id="s1",
        company="Google",
        role="software_engineer",
        seniority="mid",
        category="technical",
        skills=["python"],
        status=SessionStatus.AWAITING_ANSWER,
    )


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list = []

    def invoke(self, messages: list) -> object:
        self.calls.append(messages)
        return type("Response", (), {"content": self.content})()


class _FakeStructured:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list = []

    def invoke(self, messages: list) -> object:
        self.calls.append(messages)
        return self.payload


class _FakeScorerLLM:
    def __init__(self, turn: TurnScoreResult, final: FinalEvaluationResult) -> None:
        self.turn = _FakeStructured(turn)
        self.final = _FakeStructured(final)

    def with_structured_output(self, schema: type) -> _FakeStructured:
        if schema is TurnScoreResult:
            return self.turn
        return self.final


def test_interviewer_returns_text_only_and_uses_system_prompt() -> None:
    llm = _FakeChat("How would you design a URL shortener?")
    interviewer = OpenAIInterviewer(settings=_settings(), llm=llm)
    question = make_question("q01", question_text="Design a URL shortener.")
    text = interviewer.present_question(_session(), question)
    assert text == "How would you design a URL shortener?"
    assert llm.calls
    assert isinstance(llm.calls[0][0], SystemMessage)
    assert isinstance(llm.calls[0][1], HumanMessage)
    assert "rubric" not in llm.calls[0][1].content.lower()


def test_interviewer_follow_up_sentinel_returns_none() -> None:
    llm = _FakeChat("NO_FOLLOW_UP")
    interviewer = OpenAIInterviewer(settings=_settings(), llm=llm)
    turn = InterviewTurn(question_id="q01", question_number=1, question_text="What is REST?")
    assert interviewer.maybe_follow_up(_session(), turn, "It is an API style.") is None


def test_scorer_returns_structured_score() -> None:
    turn_result = TurnScoreResult(
        score=7,
        technical_accuracy=7,
        communication=8,
        structure=6,
        feedback="Mention latency and cache invalidation.",
        missed_points=["hot-key handling"],
    )
    final_result = FinalEvaluationResult(
        overall_score=7.0,
        recommendation="lean hire",
        summary="Competent with gaps in scale.",
        strengths=["clear communication"],
        gaps=["limited scale discussion"],
        practice_next=["practice hot-key designs"],
    )
    llm = _FakeScorerLLM(turn_result, final_result)
    scorer = OpenAIScorer(settings=_settings(), llm=llm)
    turn = InterviewTurn(
        question_id="q01",
        question_number=1,
        question_text="Design a URL shortener.",
        rubric="API, uniqueness, scale, caching.",
        user_answer="I would hash the URL and store it in a database.",
    )
    score = scorer.score_turn(_session(), turn)
    assert isinstance(score, Score)
    assert score.score == 7
    assert score.technical_accuracy == 7
    assert score.missed_points == ("hot-key handling",)
    assert llm.turn.calls
    human = llm.turn.calls[0][1].content
    assert "API, uniqueness, scale, caching." in human


def test_final_evaluation_is_json_from_pydantic() -> None:
    turn_result = TurnScoreResult(
        score=8,
        technical_accuracy=8,
        communication=8,
        structure=8,
        feedback="Good.",
        missed_points=[],
    )
    final_result = FinalEvaluationResult(
        overall_score=8.0,
        recommendation="hire",
        summary="Strong interview.",
        strengths=["structure"],
        gaps=[],
        practice_next=["system design drills"],
    )
    scorer = OpenAIScorer(settings=_settings(), llm=_FakeScorerLLM(turn_result, final_result))
    session = _session()
    session.turns.append(
        InterviewTurn(
            question_id="q01",
            question_number=1,
            question_text="Q",
            rubric="R",
            user_answer="A",
            score=Score(score=8, feedback="Good."),
        )
    )
    report = scorer.final_evaluation(session)
    assert '"recommendation":"hire"' in report.replace(" ", "")
    parsed = FinalEvaluationResult.model_validate_json(report)
    assert parsed.overall_score == 8.0
