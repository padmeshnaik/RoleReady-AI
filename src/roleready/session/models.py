"""Interview session domain models. No UI or LLM client code here."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SessionStatus(str, Enum):
    SETUP = "setup"
    IN_PROGRESS = "in_progress"
    AWAITING_ANSWER = "awaiting_answer"
    COMPLETE = "complete"


@dataclass(frozen=True)
class Score:
    score: int
    technical_accuracy: int | None = None
    communication: int | None = None
    structure: int | None = None
    feedback: str = ""
    missed_points: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 10:
            raise ValueError("Score.score must be between 0 and 10.")


@dataclass
class ChatMessage:
    role: str
    content: str
    kind: str = "message"


@dataclass
class InterviewTurn:
    question_id: str
    question_number: int
    question_text: str
    rubric: str = ""
    user_answer: str | None = None
    follow_up_answer: str | None = None
    interviewer_followup: str | None = None
    score: Score | None = None

    @property
    def is_scored(self) -> bool:
        return self.score is not None


@dataclass
class InterviewSession:
    session_id: str
    company: str
    role: str
    seniority: str
    category: str
    skills: list[str] = field(default_factory=list)
    question_index: int = 0
    used_question_ids: list[str] = field(default_factory=list)
    messages: list[ChatMessage] = field(default_factory=list)
    turns: list[InterviewTurn] = field(default_factory=list)
    status: SessionStatus = SessionStatus.SETUP
    final_evaluation: str | None = None

    @property
    def current_turn(self) -> InterviewTurn | None:
        return self.turns[-1] if self.turns else None
