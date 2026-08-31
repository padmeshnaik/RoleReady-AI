"""Question-bank row types. SQLite is the source of truth for interview questions."""

from dataclasses import dataclass

GENERIC_COMPANY = "Generic"

SENIORITY_VALUES = ("junior", "mid", "senior")
CATEGORY_VALUES = ("technical", "system_design", "behavioral")
ROLE_EXAMPLES = ("software_engineer", "data_engineer", "ai_engineer")

MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 5


@dataclass(frozen=True)
class Question:
    id: str
    company: str
    role: str
    seniority: str
    category: str
    difficulty: int
    question_text: str
    rubric: str
    follow_up_hints: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Question id must uniquely identify each question and cannot be blank.")
        if not self.company.strip():
            raise ValueError("company is required. Use a company name or 'Generic'.")
        if not MIN_DIFFICULTY <= self.difficulty <= MAX_DIFFICULTY:
            raise ValueError(
                f"difficulty must be between {MIN_DIFFICULTY} and {MAX_DIFFICULTY}, got {self.difficulty}."
            )
        if not self.question_text.strip():
            raise ValueError("question_text cannot be blank.")
        if not self.rubric.strip():
            raise ValueError("rubric cannot be blank.")
