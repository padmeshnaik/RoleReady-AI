"""Offline question generation tests. No OpenAI, SQLite writes, or Pinecone."""

from pathlib import Path

import pytest

from roleready.config.settings import Settings
from roleready.generation.generator import (
    QuestionBankGenerator,
    assign_unique_ids,
    format_summary,
    invoke_with_retry,
    is_transient_api_error,
    remaining_to_generate,
    run_generation,
)
from roleready.generation.jsonl import append_questions, load_valid_questions
from roleready.generation.plan import (
    DEFAULT_CORPUS_DISTRIBUTION,
    allocate_counts,
    build_generation_plan,
    plan_counts,
    planned_batches,
    remaining_plan,
)
from roleready.generation.prompts import QUESTION_GENERATOR_SYSTEM_PROMPT, build_batch_prompt
from roleready.generation.schemas import GeneratedQuestion, GeneratedQuestionBatch


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-openai",
        pinecone_api_key="test-pinecone",
        openai_model_question_generator="test-question-model",
    )


def _question(qid: str, text: str = "What is REST?", **overrides: object) -> GeneratedQuestion:
    data: dict = {
        "id": qid,
        "company": "Generic",
        "role": "software_engineer",
        "seniority": "mid",
        "category": "technical",
        "difficulty": 3,
        "question_text": text,
        "rubric": "Cover HTTP resources, statelessness, and status codes.",
        "follow_up_hints": "Ask about idempotency.",
    }
    data.update(overrides)
    return GeneratedQuestion.model_validate(data)


def test_remaining_to_generate_resumes_from_existing() -> None:
    assert remaining_to_generate(40, 100) == 60
    assert remaining_to_generate(100, 100) == 0
    assert remaining_to_generate(0, 100) == 100


def test_jsonl_appends_without_overwriting_existing(tmp_path: Path) -> None:
    path = tmp_path / "generated_questions.jsonl"
    append_questions(path, [_question("gq-0001", "First question?")])
    original = path.read_text(encoding="utf-8")
    append_questions(path, [_question("gq-0002", "Second question?")])
    text = path.read_text(encoding="utf-8")
    assert text.startswith(original)
    loaded = load_valid_questions(path)
    assert [item.id for item in loaded] == ["gq-0001", "gq-0002"]


def test_load_valid_questions_skips_invalid_lines(tmp_path: Path) -> None:
    path = tmp_path / "generated_questions.jsonl"
    path.write_text(
        _question("gq-0001").model_dump_json() + "\nnot-json\n",
        encoding="utf-8",
    )
    loaded = load_valid_questions(path)
    assert len(loaded) == 1
    assert loaded[0].id == "gq-0001"


def test_generated_question_rejects_disallowed_company() -> None:
    with pytest.raises(Exception):
        _question("gq-0001", company="Netflix")


def test_generator_prompt_requires_professional_corpus_quality() -> None:
    prompt = QUESTION_GENERATOR_SYSTEM_PROMPT
    assert "RoleReady AI" in prompt
    assert "Never ask definition-only" in prompt
    assert "4–6 concrete check items" in prompt or "4-6 concrete check items" in prompt
    assert "palindrome" in prompt.lower()
    assert "mid + difficulty 3" in prompt
    batch = build_batch_prompt(
        slots=[
            {
                "company": "Generic",
                "role": "software_engineer",
                "seniority": "senior",
                "category": "system_design",
                "difficulty": 5,
            }
        ],
        existing_ids=[],
        existing_question_previews=[],
    )
    assert "distinct concept" in batch
    assert "Banned overused topics" in batch
    assert "URL shortener" in batch


def test_plan_slots_uses_only_allowed_values() -> None:
    plan = build_generation_plan(10)
    assert len(plan) == 10
    assert all(slot["difficulty"] in range(1, 6) for slot in plan)
    assert all(slot["seniority"] in {"junior", "mid", "senior"} for slot in plan)


def test_generation_plan_for_100_matches_configured_distribution() -> None:
    plan = build_generation_plan(100, DEFAULT_CORPUS_DISTRIBUTION)
    roles = plan_counts(plan, "role")
    assert roles == {
        "software_engineer": 25,
        "backend_engineer": 20,
        "data_engineer": 20,
        "machine_learning_engineer": 15,
        "ai_engineer": 20,
    }
    companies = plan_counts(plan, "company")
    assert companies["Generic"] == 65
    named = {key: value for key, value in companies.items() if key != "Generic"}
    assert named == {"Google": 7, "Amazon": 7, "Microsoft": 6, "Meta": 6, "FinTech": 9}
    categories = plan_counts(plan, "category")
    assert set(categories) == {"technical", "system_design", "behavioral", "coding"}
    seniorities = plan_counts(plan, "seniority")
    assert set(seniorities) == {"junior", "mid", "senior"}
    assert seniorities["mid"] >= seniorities["junior"]
    for slot in plan:
        if slot["seniority"] == "junior":
            assert slot["difficulty"] in (1, 2)
        elif slot["seniority"] == "mid":
            assert slot["difficulty"] in (3, 4)
        else:
            assert slot["difficulty"] in (4, 5)
    combos = {
        (slot["company"], slot["role"], slot["seniority"], slot["category"]) for slot in plan
    }
    assert len(combos) < 6 * 5 * 3 * 4
    batches = planned_batches(plan, 10)
    assert len(batches) == 10
    assert all(len(batch) == 10 for batch in batches)


def test_generation_plan_scales_roles_for_1000() -> None:
    plan = build_generation_plan(1000, DEFAULT_CORPUS_DISTRIBUTION)
    roles = plan_counts(plan, "role")
    assert roles["software_engineer"] == 250
    assert roles["backend_engineer"] == 200
    assert roles["data_engineer"] == 200
    assert roles["machine_learning_engineer"] == 150
    assert roles["ai_engineer"] == 200
    assert plan_counts(plan, "company")["Generic"] == 650
    companies = plan_counts(plan, "company")
    assert companies["Google"] == 70
    assert companies["Amazon"] == 70
    assert companies["Microsoft"] == 60
    assert companies["Meta"] == 60
    assert companies["FinTech"] == 90
    senior_se_sd = [
        slot
        for slot in plan
        if slot["role"] == "software_engineer" and slot["seniority"] == "senior"
    ]
    junior_se_sd = [
        slot
        for slot in plan
        if slot["role"] == "software_engineer" and slot["seniority"] == "junior"
    ]
    senior_sd = sum(1 for slot in senior_se_sd if slot["category"] == "system_design") / len(senior_se_sd)
    junior_sd = sum(1 for slot in junior_se_sd if slot["category"] == "system_design") / len(junior_se_sd)
    assert senior_sd > junior_sd
    ml = [slot for slot in plan if slot["role"] == "machine_learning_engineer"]
    se = [slot for slot in plan if slot["role"] == "software_engineer"]
    ml_tech = sum(1 for slot in ml if slot["category"] == "technical") / len(ml)
    se_tech = sum(1 for slot in se if slot["category"] == "technical") / len(se)
    assert ml_tech > se_tech


def test_remaining_plan_skips_already_generated_slots() -> None:
    plan = build_generation_plan(100)
    pending = remaining_plan(plan, 40)
    assert len(pending) == 60
    assert pending == plan[40:]
    assert allocate_counts({"a": 1, "b": 1}, 3)["a"] + allocate_counts({"a": 1, "b": 1}, 3)["b"] == 3


def test_assign_unique_ids_never_reuses_existing() -> None:
    used = {"gq-0001"}
    questions = [_question("gq-0001", "New text?")]
    assigned = assign_unique_ids(questions, used, sequence_start=1)
    assert assigned[0].id == "gq-0002"


class _FakeStructured:
    def __init__(self, batches: list[GeneratedQuestionBatch]) -> None:
        self.batches = list(batches)
        self.calls = 0

    def invoke(self, messages: list) -> GeneratedQuestionBatch:
        self.calls += 1
        if not self.batches:
            raise AssertionError("unexpected extra LLM call")
        return self.batches.pop(0)


def test_run_generation_writes_batches_and_stops_at_target(tmp_path: Path) -> None:
    path = tmp_path / "generated_questions.jsonl"
    existing = [_question(f"gq-{i:04d}", f"Existing {i}?") for i in range(1, 3)]
    append_questions(path, existing)

    def _batch(start: int, size: int) -> GeneratedQuestionBatch:
        return GeneratedQuestionBatch(
            questions=[
                _question(f"tmp-{start + i}", f"Generated {start + i}?")
                for i in range(size)
            ]
        )

    fake = _FakeStructured([_batch(3, 2), _batch(5, 2)])
    generator = QuestionBankGenerator(settings=_settings(), structured_llm=fake)
    summary = run_generation(
        output_path=path,
        target_count=5,
        batch_size=2,
        generator=generator,
    )
    assert summary.existing_count == 2
    assert summary.newly_generated_count == 3
    assert summary.total_count == 5
    assert summary.failed_batches == 0
    assert fake.calls == 2
    loaded = load_valid_questions(path)
    assert [item.id for item in loaded[:2]] == ["gq-0001", "gq-0002"]
    assert len(loaded) == 5


def test_invoke_with_retry_retries_transient_then_succeeds() -> None:
    calls = {"n": 0}

    def _flaky() -> GeneratedQuestionBatch:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("network")
        return GeneratedQuestionBatch(questions=[_question("gq-0001")])

    slept: list[float] = []
    result = invoke_with_retry(_flaky, retry_count=2, sleep=slept.append)
    assert len(result.questions) == 1
    assert calls["n"] == 2
    assert slept


def test_transient_errors_include_timeouts_and_http_5xx() -> None:
    class _ServerError(Exception):
        status_code = 503

    assert is_transient_api_error(TimeoutError())
    assert is_transient_api_error(_ServerError())
    assert is_transient_api_error(ValueError("bad schema")) is False


def test_missing_generator_model_raises() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-openai",
        pinecone_api_key="test-pinecone",
        openai_model_question_generator=None,
    )
    with pytest.raises(Exception, match="OPENAI_MODEL_QUESTION_GENERATOR"):
        QuestionBankGenerator(settings=settings, structured_llm=object())


def test_format_summary_includes_required_fields() -> None:
    from roleready.generation.generator import GenerationSummary

    text = format_summary(
        GenerationSummary(
            target_count=100,
            existing_count=40,
            newly_generated_count=60,
            total_count=100,
            failed_batches=1,
        )
    )
    assert "target count: 100" in text
    assert "existing count: 40" in text
    assert "newly generated count: 60" in text
    assert "total count: 100" in text
    assert "failed batches: 1" in text


def test_generation_modules_do_not_import_sqlite_or_pinecone() -> None:
    from pathlib import Path as P

    paths = [
        P("src/roleready/generation/generator.py"),
        P("src/roleready/generation/prompts.py"),
        P("src/roleready/generation/plan.py"),
        P("src/roleready/generation/jsonl.py"),
        P("src/roleready/generation/schemas.py"),
        P("scripts/generate_question_bank.py"),
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "roleready.db" not in source
        assert "from pinecone" not in source
        assert "import pinecone" not in source
        assert "roleready.rag" not in source
