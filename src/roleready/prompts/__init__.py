"""Versionable prompt templates. No LLM calls live here."""

from roleready.prompts.common import format_skills
from roleready.prompts.interviewer import (
    INTERVIEWER_SYSTEM_PROMPT,
    build_follow_up_prompt,
    build_interviewer_system_prompt,
    build_present_question_prompt,
)
from roleready.prompts.scorer import (
    FINAL_EVALUATION_SYSTEM_PROMPT,
    SCORER_SYSTEM_PROMPT,
    build_final_evaluation_prompt,
    build_final_evaluation_system_prompt,
    build_score_turn_prompt,
    build_scorer_system_prompt,
)

__all__ = [
    "format_skills",
    "FINAL_EVALUATION_SYSTEM_PROMPT",
    "INTERVIEWER_SYSTEM_PROMPT",
    "SCORER_SYSTEM_PROMPT",
    "build_final_evaluation_prompt",
    "build_final_evaluation_system_prompt",
    "build_follow_up_prompt",
    "build_interviewer_system_prompt",
    "build_present_question_prompt",
    "build_score_turn_prompt",
    "build_scorer_system_prompt",
]
