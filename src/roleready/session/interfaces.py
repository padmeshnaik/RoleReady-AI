"""Collaborator protocols for SessionManager. Implementations may be fakes or real services."""

from __future__ import annotations

from typing import Protocol

from roleready.db.models import Question
from roleready.session.models import InterviewSession, InterviewTurn, Score


class QuestionRetriever(Protocol):
    def next_question(self, session: InterviewSession) -> Question:
        """Return the next unused question for this session."""


class ScoringService(Protocol):
    def score_turn(self, session: InterviewSession, turn: InterviewTurn) -> Score:
        """Score one completed turn. Must not depend on Streamlit."""

    def final_evaluation(self, session: InterviewSession) -> str:
        """Produce the end-of-interview evaluation after 10 scored questions."""


class InterviewerService(Protocol):
    def present_question(self, session: InterviewSession, question: Question) -> str:
        """Interviewer utterance that poses the retrieved question."""

    def maybe_follow_up(
        self,
        session: InterviewSession,
        turn: InterviewTurn,
        latest_answer: str,
    ) -> str | None:
        """Return one clarifying follow-up, or None to score the turn."""
