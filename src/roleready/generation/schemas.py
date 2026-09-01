"""Allowed values and Pydantic schemas for offline question generation. No LLM calls."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

COMPANIES = (
    "Generic",
    "Google",
    "Amazon",
    "Microsoft",
    "Meta",
    "FinTech",
)
ROLES = (
    "software_engineer",
    "backend_engineer",
    "data_engineer",
    "machine_learning_engineer",
    "ai_engineer",
)
SENIORITIES = ("junior", "mid", "senior")
CATEGORIES = ("technical", "system_design", "behavioral", "coding")
MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 5

CompanyLiteral = Literal["Generic", "Google", "Amazon", "Microsoft", "Meta", "FinTech"]
RoleLiteral = Literal[
    "software_engineer",
    "backend_engineer",
    "data_engineer",
    "machine_learning_engineer",
    "ai_engineer",
]
SeniorityLiteral = Literal["junior", "mid", "senior"]
CategoryLiteral = Literal["technical", "system_design", "behavioral", "coding"]


class GeneratedQuestion(BaseModel):
    id: str
    company: CompanyLiteral
    role: RoleLiteral
    seniority: SeniorityLiteral
    category: CategoryLiteral
    difficulty: int = Field(ge=MIN_DIFFICULTY, le=MAX_DIFFICULTY)
    question_text: str
    rubric: str
    follow_up_hints: str

    @field_validator("id", "question_text", "rubric", "follow_up_hints", mode="after")
    @classmethod
    def nonempty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class GeneratedQuestionBatch(BaseModel):
    questions: list[GeneratedQuestion]
