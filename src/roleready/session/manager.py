"""Interview session state machine. Independent of Streamlit, OpenAI, and Pinecone."""

from __future__ import annotations

import uuid

from roleready.db.models import Question
from roleready.session.interfaces import InterviewerService, QuestionRetriever, ScoringService
from roleready.session.models import ChatMessage, InterviewSession, InterviewTurn, SessionStatus

DEFAULT_QUESTION_COUNT = 10


class SessionError(Exception):
    """Invalid session transition or invariant violation."""


class _NoFollowUpInterviewer:
    """Default interviewer: pose the question text, never ask a follow-up."""

    def present_question(self, session: InterviewSession, question: Question) -> str:
        return question.question_text

    def maybe_follow_up(
        self,
        session: InterviewSession,
        turn: InterviewTurn,
        latest_answer: str,
    ) -> str | None:
        return None


class SessionManager:
    def __init__(
        self,
        retriever: QuestionRetriever,
        scorer: ScoringService,
        interviewer: InterviewerService | None = None,
        question_count: int = DEFAULT_QUESTION_COUNT,
    ) -> None:
        if question_count < 1:
            raise ValueError("question_count must be at least 1.")
        self._retriever = retriever
        self._scorer = scorer
        self._interviewer = interviewer or _NoFollowUpInterviewer()
        self._question_count = question_count

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
        return InterviewSession(
            session_id=session_id or str(uuid.uuid4()),
            company=company,
            role=role,
            seniority=seniority,
            category=category,
            skills=list(skills or []),
            status=SessionStatus.SETUP,
        )

    def start_interview(self, session: InterviewSession) -> InterviewSession:
        if session.status is not SessionStatus.SETUP:
            raise SessionError("Interview can only be started from setup.")
        session.status = SessionStatus.IN_PROGRESS
        self._pose_next_question(session)
        return session

    def submit_answer(self, session: InterviewSession, answer: str) -> InterviewSession:
        if session.status is not SessionStatus.AWAITING_ANSWER:
            raise SessionError("An answer can only be submitted while awaiting_answer.")
        text = answer.strip()
        if not text:
            raise SessionError("Answer cannot be empty.")

        turn = session.current_turn
        if turn is None or turn.is_scored:
            raise SessionError("No active question to answer.")

        session.messages.append(ChatMessage(role="candidate", content=text, kind="answer"))

        if turn.user_answer is None:
            turn.user_answer = text
            follow_up = self._interviewer.maybe_follow_up(session, turn, text)
            if follow_up:
                turn.interviewer_followup = follow_up
                session.messages.append(
                    ChatMessage(role="interviewer", content=follow_up, kind="follow_up")
                )
                session.status = SessionStatus.AWAITING_ANSWER
                return session
        else:
            turn.follow_up_answer = text

        self._score_current_turn(session, turn)
        return session

    def _score_current_turn(self, session: InterviewSession, turn: InterviewTurn) -> None:
        session.status = SessionStatus.IN_PROGRESS
        turn.score = self._scorer.score_turn(session, turn)
        session.question_index += 1

        if session.question_index >= self._question_count:
            session.status = SessionStatus.COMPLETE
            session.final_evaluation = self._scorer.final_evaluation(session)
            return

        self._pose_next_question(session)

    def _pose_next_question(self, session: InterviewSession) -> None:
        if session.status is SessionStatus.COMPLETE or session.question_index >= self._question_count:
            raise SessionError(
                "An 11th main question cannot be generated; the interview is complete."
            )
        question = self._retriever.next_question(session)
        if question.id in session.used_question_ids:
            raise SessionError(
                f"Question ID {question.id!r} cannot be reused in the same interview."
            )

        session.used_question_ids.append(question.id)
        utterance = self._interviewer.present_question(session, question)
        session.messages.append(
            ChatMessage(role="interviewer", content=utterance, kind="question")
        )
        session.turns.append(
            InterviewTurn(
                question_id=question.id,
                question_number=session.question_index + 1,
                question_text=question.question_text,
                rubric=question.rubric,
            )
        )
        session.status = SessionStatus.AWAITING_ANSWER
