"""In-memory fakes for SessionManager tests. No OpenAI or Pinecone."""

from __future__ import annotations

from roleready.db.models import Question
from roleready.session.models import InterviewSession, InterviewTurn, Score


def make_question(question_id: str, **overrides: object) -> Question:
    data: dict = {
        "id": question_id,
        "company": "Generic",
        "role": "software_engineer",
        "seniority": "mid",
        "category": "technical",
        "difficulty": 3,
        "question_text": f"Question {question_id}?",
        "rubric": "Key concepts for a strong answer.",
        "follow_up_hints": None,
    }
    data.update(overrides)
    return Question(**data)


class FakeRetriever:
    def __init__(self, questions: list[Question]) -> None:
        self.questions = questions
        self.calls: list[list[str]] = []

    def next_question(self, session: InterviewSession) -> Question:
        self.calls.append(list(session.used_question_ids))
        for question in self.questions:
            if question.id not in session.used_question_ids:
                return question
        raise LookupError("FakeRetriever has no unused questions left.")


class DuplicateIdRetriever:
    """Always returns the same question id (used to test reuse rejection)."""

    def __init__(self, question: Question) -> None:
        self.question = question

    def next_question(self, session: InterviewSession) -> Question:
        return self.question


class FakeScorer:
    def __init__(self, overall: int = 7) -> None:
        self.overall = overall
        self.scored_turns: list[str] = []
        self.final_eval_calls = 0

    def score_turn(self, session: InterviewSession, turn: InterviewTurn) -> Score:
        self.scored_turns.append(turn.question_id)
        return Score(score=self.overall, feedback="Solid answer.")

    def final_evaluation(self, session: InterviewSession) -> str:
        self.final_eval_calls += 1
        return "Final evaluation: candidate is hireable."


class OneFollowUpInterviewer:
    """Asks a single follow-up on the first answer of a turn, then stops."""

    def present_question(self, session: InterviewSession, question: Question) -> str:
        return question.question_text

    def maybe_follow_up(
        self,
        session: InterviewSession,
        turn: InterviewTurn,
        latest_answer: str,
    ) -> str | None:
        if turn.interviewer_followup is None:
            return "Can you give a concrete example?"
        return None
