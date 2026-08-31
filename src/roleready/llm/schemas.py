"""Pydantic schemas for scorer structured output."""

from pydantic import BaseModel, Field


class TurnScoreResult(BaseModel):
    score: int = Field(ge=0, le=10)
    technical_accuracy: int = Field(ge=0, le=10)
    communication: int = Field(ge=0, le=10)
    structure: int = Field(ge=0, le=10)
    feedback: str = ""
    missed_points: list[str] = Field(default_factory=list)


class FinalEvaluationResult(BaseModel):
    overall_score: float = Field(ge=0, le=10)
    recommendation: str
    summary: str
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    practice_next: list[str] = Field(default_factory=list)
