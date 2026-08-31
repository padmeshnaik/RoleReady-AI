"""OpenAI scoring via LangChain structured output. No Pinecone or Streamlit."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from pydantic import ValidationError

from roleready.config.settings import Settings, get_settings
from roleready.llm.schemas import FinalEvaluationResult, TurnScoreResult
from roleready.prompts.scorer import (
    build_final_evaluation_prompt,
    build_final_evaluation_system_prompt,
    build_score_turn_prompt,
    build_scorer_system_prompt,
)
from roleready.session.manager import DEFAULT_QUESTION_COUNT
from roleready.session.models import InterviewSession, InterviewTurn, Score

SCORER_TEMPERATURE = 0


class ScoringError(Exception):
    """Controlled failure when structured scorer output is missing or invalid."""


def parse_turn_score(payload: object) -> Score:
    """Validate a fake or real model payload into a Score. Does not call OpenAI."""
    try:
        if isinstance(payload, TurnScoreResult):
            parsed = payload
        elif isinstance(payload, str):
            parsed = TurnScoreResult.model_validate_json(payload)
        else:
            parsed = TurnScoreResult.model_validate(payload)
    except (ValidationError, ValueError, TypeError) as exc:
        raise ScoringError(
            "Scorer returned malformed or invalid output. "
            "Expected JSON with score 0-10, dimension scores, feedback, and missed_points."
        ) from exc

    return Score(
        score=parsed.score,
        technical_accuracy=parsed.technical_accuracy,
        communication=parsed.communication,
        structure=parsed.structure,
        feedback=parsed.feedback,
        missed_points=tuple(parsed.missed_points),
    )


class OpenAIScorer:
    """Turn scoring and final evaluation. Uses OPENAI_MODEL_SCORER at temperature 0."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: ChatOpenAI | None = None,
        question_count: int = DEFAULT_QUESTION_COUNT,
    ) -> None:
        self._settings = settings or get_settings()
        self._question_count = question_count
        self._llm = llm or ChatOpenAI(
            model=self._settings.openai_model_scorer,
            api_key=self._settings.openai_api_key,
            temperature=SCORER_TEMPERATURE,
        )
        self._turn_llm = self._llm.with_structured_output(TurnScoreResult)
        self._final_llm = self._llm.with_structured_output(FinalEvaluationResult)

    def score_turn(self, session: InterviewSession, turn: InterviewTurn) -> Score:
        messages = [
            SystemMessage(content=build_scorer_system_prompt()),
            HumanMessage(
                content=build_score_turn_prompt(
                    company=session.company,
                    role=session.role,
                    seniority=session.seniority,
                    category=session.category,
                    skills=session.skills,
                    question_text=turn.question_text,
                    rubric=turn.rubric,
                    user_answer=turn.user_answer or "",
                    interviewer_followup=turn.interviewer_followup,
                    follow_up_answer=turn.follow_up_answer,
                )
            ),
        ]
        try:
            result = self._turn_llm.invoke(messages)
        except Exception as exc:
            raise ScoringError("Scorer model call failed.") from exc
        return parse_turn_score(result)

    def final_evaluation(self, session: InterviewSession) -> str:
        summaries = []
        for turn in session.turns:
            score = turn.score
            score_text = "unscored" if score is None else str(score.score)
            feedback = "" if score is None else score.feedback
            missed = "" if score is None else "; ".join(score.missed_points)
            summaries.append(
                f"Q{turn.question_number} ({turn.question_id}): score={score_text}. "
                f"Feedback: {feedback}. Missed: {missed or '(none)'}."
            )
        messages = [
            SystemMessage(
                content=build_final_evaluation_system_prompt(question_count=self._question_count)
            ),
            HumanMessage(
                content=build_final_evaluation_prompt(
                    company=session.company,
                    role=session.role,
                    seniority=session.seniority,
                    category=session.category,
                    skills=session.skills,
                    turn_summaries="\n".join(summaries) if summaries else "(no turns)",
                )
            ),
        ]
        result = self._final_llm.invoke(messages)
        parsed = (
            result
            if isinstance(result, FinalEvaluationResult)
            else FinalEvaluationResult.model_validate(result)
        )
        return parsed.model_dump_json()
