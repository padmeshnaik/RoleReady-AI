"""Offline question-bank generation. Uses OpenAI only. No SQLite or Pinecone."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from roleready.config.settings import Settings, get_settings
from roleready.generation.jsonl import append_questions, load_valid_questions, remaining_to_generate
from roleready.generation.plan import (
    CorpusDistribution,
    DEFAULT_CORPUS_DISTRIBUTION,
    build_generation_plan,
    plan_counts,
    planned_batches,
    remaining_plan,
)
from roleready.generation.prompts import QUESTION_GENERATOR_SYSTEM_PROMPT, build_batch_prompt
from roleready.generation.schemas import GeneratedQuestion, GeneratedQuestionBatch

logger = logging.getLogger(__name__)

DEFAULT_TARGET_COUNT = 1000
DEFAULT_BATCH_SIZE = 10
DEFAULT_RETRY_COUNT = 3
DEFAULT_MAX_FAILED_BATCHES = 50
RETRY_BASE_SECONDS = 1.5

_TRANSIENT = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


class GenerationError(Exception):
    """Controlled failure for the offline generation pipeline."""


@dataclass(frozen=True)
class GenerationSummary:
    target_count: int
    existing_count: int
    newly_generated_count: int
    total_count: int
    failed_batches: int


def _log_plan(plan: list[dict], batch_size: int, pending: list[dict]) -> None:
    batches = planned_batches(pending, batch_size) if pending else []
    logger.info(
        "Generation plan: %s slots; pending=%s; batches=%s; "
        "roles=%s; companies=%s; categories=%s; seniorities=%s",
        len(plan),
        len(pending),
        len(batches),
        plan_counts(plan, "role"),
        plan_counts(plan, "company"),
        plan_counts(plan, "category"),
        plan_counts(plan, "seniority"),
    )


def next_question_id(used_ids: set[str], sequence: int) -> tuple[str, int]:
    while True:
        candidate = f"gq-{sequence:04d}"
        sequence += 1
        if candidate not in used_ids:
            return candidate, sequence


def assign_unique_ids(
    questions: list[GeneratedQuestion],
    used_ids: set[str],
    sequence_start: int,
) -> list[GeneratedQuestion]:
    sequence = sequence_start
    assigned: list[GeneratedQuestion] = []
    for question in questions:
        qid = question.id
        if qid in used_ids:
            qid, sequence = next_question_id(used_ids, sequence)
        used_ids.add(qid)
        assigned.append(question.model_copy(update={"id": qid}))
    return assigned


def is_transient_api_error(exc: BaseException) -> bool:
    if isinstance(exc, _TRANSIENT):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status >= 500


def invoke_with_retry(
    operation: Callable[[], GeneratedQuestionBatch],
    *,
    retry_count: int = DEFAULT_RETRY_COUNT,
    sleep: Callable[[float], None] = time.sleep,
) -> GeneratedQuestionBatch:
    last: BaseException | None = None
    attempts = retry_count + 1
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last = exc
            if not is_transient_api_error(exc) or attempt == attempts:
                raise
            logger.warning(
                "Transient API error %s (attempt %s/%s); retrying. No secrets logged.",
                type(exc).__name__,
                attempt,
                attempts,
            )
            sleep(RETRY_BASE_SECONDS * attempt)
    raise GenerationError("Question generation failed after retries.") from last


class QuestionBankGenerator:
    """Calls OPENAI_MODEL_QUESTION_GENERATOR. Does not touch SQLite or Pinecone."""

    def __init__(
        self,
        settings: Settings | None = None,
        structured_llm=None,
        retry_count: int = DEFAULT_RETRY_COUNT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings or get_settings()
        model = self._settings.openai_model_question_generator
        if not model:
            raise GenerationError(
                "OPENAI_MODEL_QUESTION_GENERATOR is missing or empty. "
                "Set it in .env before running the generation script."
            )
        self._model_name = model
        if structured_llm is not None:
            self._structured = structured_llm
        else:
            llm = ChatOpenAI(
                model=model,
                api_key=self._settings.openai_api_key,
                temperature=0.6,
            )
            self._structured = llm.with_structured_output(GeneratedQuestionBatch)
        self._retry_count = retry_count
        self._sleep = sleep

    def generate_batch(
        self,
        slots: list[dict],
        *,
        existing_ids: list[str],
        existing_question_previews: list[str],
    ) -> list[GeneratedQuestion]:
        messages = [
            SystemMessage(content=QUESTION_GENERATOR_SYSTEM_PROMPT),
            HumanMessage(
                content=build_batch_prompt(
                    slots=slots,
                    existing_ids=existing_ids,
                    existing_question_previews=existing_question_previews,
                )
            ),
        ]

        def _invoke() -> GeneratedQuestionBatch:
            result = self._structured.invoke(messages)
            if isinstance(result, GeneratedQuestionBatch):
                return result
            return GeneratedQuestionBatch.model_validate(result)

        logger.info(
            "Requesting %s questions from model %s (no API keys logged).",
            len(slots),
            self._model_name,
        )
        batch = invoke_with_retry(_invoke, retry_count=self._retry_count, sleep=self._sleep)
        questions = list(batch.questions[: len(slots)])
        aligned: list[GeneratedQuestion] = []
        for question, slot in zip(questions, slots, strict=False):
            aligned.append(question.model_copy(update=slot))
        return aligned


def run_generation(
    *,
    output_path: Path,
    target_count: int = DEFAULT_TARGET_COUNT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    generator: QuestionBankGenerator | None = None,
    max_failed_batches: int = DEFAULT_MAX_FAILED_BATCHES,
    distribution: CorpusDistribution | None = None,
    avoid_corpus_path: Path | None = None,
) -> GenerationSummary:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    existing = load_valid_questions(output_path)
    existing_count = len(existing)
    remaining = remaining_to_generate(existing_count, target_count)
    newly = 0
    failed_batches = 0
    used_ids = {question.id for question in existing}
    seen_text = {question.question_text.casefold() for question in existing}
    sequence = existing_count + 1
    previews = [question.question_text[:160] for question in existing]
    if avoid_corpus_path is not None and avoid_corpus_path.exists():
        avoided = load_valid_questions(avoid_corpus_path)
        for question in avoided:
            used_ids.add(question.id)
            seen_text.add(question.question_text.casefold())
            previews.append(question.question_text[:160])
        logger.info(
            "Avoiding %s questions from %s (topics/ids only; not written to output).",
            len(avoided),
            avoid_corpus_path.name,
        )
    generator = generator or QuestionBankGenerator()

    plan = build_generation_plan(target_count, distribution or DEFAULT_CORPUS_DISTRIBUTION)
    queue = remaining_plan(plan, existing_count)
    _log_plan(plan, batch_size, queue)

    logger.info(
        "Generation start: target=%s existing=%s remaining=%s output=%s",
        target_count,
        existing_count,
        remaining,
        output_path.name,
    )

    while remaining > 0 and queue and failed_batches < max_failed_batches:
        slots = queue[: min(batch_size, remaining)]
        try:
            produced = generator.generate_batch(
                slots,
                existing_ids=sorted(used_ids),
                existing_question_previews=previews,
            )
        except Exception as exc:
            failed_batches += 1
            logger.error(
                "Batch failed (%s/%s): %s. No secrets logged. Continuing if retries remain.",
                failed_batches,
                max_failed_batches,
                type(exc).__name__,
            )
            continue

        unique: list[GeneratedQuestion] = []
        for question in produced:
            key = question.question_text.casefold()
            if key in seen_text:
                logger.info("Skipping duplicate question_text in batch (id was %s).", question.id)
                continue
            unique.append(question)
            seen_text.add(key)
        unique = assign_unique_ids(unique, used_ids, sequence)
        sequence = existing_count + newly + len(unique) + 1
        if not unique:
            failed_batches += 1
            logger.warning("Batch produced no new unique questions.")
            queue = queue[len(slots) :]
            continue
        append_questions(output_path, unique)
        queue = queue[len(unique) :]
        newly += len(unique)
        remaining = remaining_to_generate(existing_count + newly, target_count)
        previews.extend(question.question_text[:160] for question in unique)
        logger.info(
            "Wrote %s questions. Progress %s/%s. Planned batches left: %s.",
            len(unique),
            existing_count + newly,
            target_count,
            len(planned_batches(queue, batch_size)),
        )

    total = existing_count + newly
    summary = GenerationSummary(
        target_count=target_count,
        existing_count=existing_count,
        newly_generated_count=newly,
        total_count=total,
        failed_batches=failed_batches,
    )
    logger.info(
        "Generation finished: target=%s existing=%s newly=%s total=%s failed_batches=%s",
        summary.target_count,
        summary.existing_count,
        summary.newly_generated_count,
        summary.total_count,
        summary.failed_batches,
    )
    return summary


def format_summary(summary: GenerationSummary) -> str:
    return (
        f"target count: {summary.target_count}\n"
        f"existing count: {summary.existing_count}\n"
        f"newly generated count: {summary.newly_generated_count}\n"
        f"total count: {summary.total_count}\n"
        f"failed batches: {summary.failed_batches}"
    )
