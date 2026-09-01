"""Offline question-bank generation. No SQLite or Pinecone."""

from roleready.generation.generator import (
    GenerationSummary,
    QuestionBankGenerator,
    format_summary,
    run_generation,
)
from roleready.generation.plan import (
    DEFAULT_CORPUS_DISTRIBUTION,
    build_generation_plan,
    planned_batches,
)
from roleready.generation.schemas import GeneratedQuestion, GeneratedQuestionBatch

__all__ = [
    "DEFAULT_CORPUS_DISTRIBUTION",
    "GeneratedQuestion",
    "GeneratedQuestionBatch",
    "GenerationSummary",
    "QuestionBankGenerator",
    "build_generation_plan",
    "format_summary",
    "planned_batches",
    "run_generation",
]
