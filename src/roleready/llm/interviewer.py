"""OpenAI interviewer via LangChain. No Pinecone or Streamlit."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from roleready.config.settings import Settings, get_settings
from roleready.db.models import Question
from roleready.prompts.interviewer import (
    build_follow_up_prompt,
    build_interviewer_system_prompt,
    build_present_question_prompt,
)
from roleready.session.manager import DEFAULT_QUESTION_COUNT
from roleready.session.models import InterviewSession, InterviewTurn

NO_FOLLOW_UP_SENTINEL = "NO_FOLLOW_UP"
INTERVIEWER_TEMPERATURE = 0.5


class OpenAIInterviewer:
    """Interview dialogue. Uses OPENAI_MODEL_INTERVIEWER. Returns interviewer text only."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: ChatOpenAI | None = None,
        question_count: int = DEFAULT_QUESTION_COUNT,
    ) -> None:
        self._settings = settings or get_settings()
        self._question_count = question_count
        self._llm = llm or ChatOpenAI(
            model=self._settings.openai_model_interviewer,
            api_key=self._settings.openai_api_key,
            temperature=INTERVIEWER_TEMPERATURE,
        )

    def present_question(self, session: InterviewSession, question: Question) -> str:
        previous = [
            turn.question_text
            for turn in session.turns
            if turn.question_id != question.id
        ]
        messages = [
            SystemMessage(content=self._system_prompt(session)),
            HumanMessage(
                content=build_present_question_prompt(
                    question_text=question.question_text,
                    question_number=session.question_index + 1,
                    question_count=self._question_count,
                    previous_questions=previous,
                )
            ),
        ]
        return self._invoke_text(messages)

    def maybe_follow_up(
        self,
        session: InterviewSession,
        turn: InterviewTurn,
        latest_answer: str,
    ) -> str | None:
        messages = [
            SystemMessage(content=self._system_prompt(session)),
            HumanMessage(
                content=build_follow_up_prompt(
                    question_text=turn.question_text,
                    latest_answer=latest_answer,
                )
            ),
        ]
        text = self._invoke_text(messages)
        normalized = "".join(ch for ch in text.upper() if ch.isalnum() or ch == "_")
        if normalized == NO_FOLLOW_UP_SENTINEL or normalized.startswith(NO_FOLLOW_UP_SENTINEL):
            return None
        return text

    def _system_prompt(self, session: InterviewSession) -> str:
        return build_interviewer_system_prompt(
            company=session.company,
            role=session.role,
            seniority=session.seniority,
            category=session.category,
            skills=session.skills,
        )

    def _invoke_text(self, messages: list) -> str:
        response = self._llm.invoke(messages)
        content = getattr(response, "content", response)
        if not isinstance(content, str):
            content = str(content)
        return content.strip()
