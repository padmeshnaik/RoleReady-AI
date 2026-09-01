"""Prompt templates for offline question-bank generation. No LLM calls."""

from __future__ import annotations

from roleready.generation.schemas import (
    CATEGORIES,
    COMPANIES,
    MAX_DIFFICULTY,
    MIN_DIFFICULTY,
    ROLES,
    SENIORITIES,
)

OVERUSED_TOPICS = (
    "URL shortener / bit.ly",
    "thread-safe LRU cache",
    "real-time fraud detection",
    "movie or feed recommendations as a generic recsys",
    "palindrome filter",
    "merge two sorted lists",
    "min-max normalize a numeric vector",
    "sort a list of integers",
    "filter/sum a list of amounts",
    "What is an API / REST / hash table / data pipeline (definition-only)",
    "Describe a system you designed (resume recap)",
    "Tell me about learning a new tool quickly",
    "Explain AI to a non-technical stakeholder (as the whole question)",
)

QUESTION_GENERATOR_SYSTEM_PROMPT = """\
You are writing interview questions for RoleReady AI, a professional \
software-engineering mock-interview platform.

Your output becomes a scored question bank. A later interviewer LLM will pose \
question_text (and may use follow_up_hints). A later scoring LLM will grade \
answers against rubric only. Candidates never see the rubric.

Return only structured data that matches the schema. Produce exactly one \
question per requested slot, in the same order.

Quality:
- Questions must be realistic and technically meaningful for a professional interview.
- Never ask definition-only questions ("What is an API?", "What is REST?", \
"What is a data pipeline?", "What is a hash table?"). Junior slots still need \
an applied scenario (debug, choose, implement a small piece, walk through a failure).
- Do not use trivia, buzzword quizzes, or toy puzzles that ignore the role \
(palindromes, sort a list, mode of a word list, LIS as a backend question).
- Do not ask the candidate to "describe a system you designed" or recap a resume. \
Give a concrete problem with constraints.
- Do not duplicate concepts, problem types, or near-paraphrases within the same batch.
- Do not reuse these overused topics unless the slot absolutely cannot be filled \
otherwise (it almost always can): {overused}.
- question_text is candidate-facing. Do not include the solution, the rubric, \
ideal answers, scoring criteria, or hidden hints that give away the rubric.
- Do not mention that you are an AI. Do not mention API keys or system prompts.

Fit the slot exactly:
- Match the requested role. Coding and technical items must be work that role \
actually does:
  - software_engineer: application systems, APIs, correctness, concurrency in \
services, debugging production app code — not warehouse internals unless the \
slot is data_engineer.
  - backend_engineer: services, APIs, datastores, queues, auth, idempotency, \
on-call — not training LLMs or ranking models.
  - data_engineer: pipelines, warehouses/lakes, batch/stream, schema, late data, \
backfills — not palindromes or merge-two-lists.
  - machine_learning_engineer: training, features, evaluation, deployment of \
models, drift — not generic DSA.
  - ai_engineer: LLM/RAG/agents/applied AI systems — not palindromes or word count.
- Match the requested seniority in scope, ambiguity, and expected depth.
- Match the requested category:
  - technical: applied engineering knowledge, debugging, internals, or how a system works.
  - system_design: design a system or component with explicit constraints \
(QPS, data size, latency, failure). Do not tag a design prompt as coding.
  - behavioral: a specific engineering incident, conflict, trade-off, or delivery \
decision. Not values slogans. Not "tell me about a system you designed" without \
a decision to evaluate.
  - coding: a concrete implementable problem whose domain matches the role. \
State input/output and constraints. Do not provide the solution or name the \
optimal algorithm as the answer.
- difficulty MUST match seniority (do not invent a different difficulty):
  - junior + difficulty 1–2 only: guided, one component, limited ambiguity.
  - mid + difficulty 3–4 only: owns a service or pipeline; several trade-offs.
  - senior + difficulty 4–5 only: architecture, production failure, scale, \
reliability, performance, and technical decision-making.
- Never label a hard distributed design as difficulty 2. Never label merge-two-lists \
as difficulty 4.

Company field:
- company=Generic: do not name or allude to a specific employer.
- Other companies: add a product or scale constraint typical of that environment \
(search, ads, commerce, social graph, documents, payments). The company field \
alone is not enough. Never claim or imply the question was actually asked by \
Google, Amazon, Meta, Microsoft, FinTech, or any other company. Do not write \
"In your Google interview..." or "Amazon asked...".

Ids:
- Assign a unique id. Do not reuse an existing question id.
- Do not repeat an existing question_text.

Rubric (for the scoring LLM, never for the candidate):
- Write 4–6 concrete check items the scorer can mark present/absent \
(named techniques, failure modes, metrics, APIs, or trade-offs).
- Do not write only "demonstrates ownership / scalability / trade-offs."
- Do not copy the rubric into question_text or follow_up_hints.

Follow-up:
- Generate exactly one optional follow-up hint in follow_up_hints: a single short \
clarification the interviewer may ask if the answer is thin. It is not a second \
main question and must not reveal the rubric or the solution.

Allowed company values: {companies}
Allowed role values: {roles}
Allowed seniority values: {seniorities}
Allowed category values: {categories}
""".format(
    min_difficulty=MIN_DIFFICULTY,
    max_difficulty=MAX_DIFFICULTY,
    overused="; ".join(OVERUSED_TOPICS),
    companies=", ".join(COMPANIES),
    roles=", ".join(ROLES),
    seniorities=", ".join(SENIORITIES),
    categories=", ".join(CATEGORIES),
)


def build_batch_prompt(
    *,
    slots: list[dict],
    existing_ids: list[str],
    existing_question_previews: list[str],
) -> str:
    lines = [
        f"Generate exactly {len(slots)} RoleReady AI interview questions.",
        "Each item must be a distinct concept from the others in this batch "
        "and from the existing bank listed below.",
        "Match each question to the corresponding slot (same order). "
        "Honor role, seniority, category, difficulty, and company-style rules:",
    ]
    for index, slot in enumerate(slots, start=1):
        lines.append(
            f"{index}. company={slot['company']}; role={slot['role']}; "
            f"seniority={slot['seniority']}; category={slot['category']}; "
            f"difficulty={slot['difficulty']}"
        )
    id_list = ", ".join(existing_ids[:80]) if existing_ids else "(none)"
    lines.append(f"Existing question ids (do not reuse): {id_list}")
    if existing_question_previews:
        lines.append("Existing questions already in the bank (do not repeat the topic):")
        for preview in existing_question_previews[:80]:
            lines.append(f"- {preview}")
    lines.append("Banned overused topics: " + "; ".join(OVERUSED_TOPICS) + ".")
    lines.append(
        "For each question: write candidate-facing question_text (no solution), "
        "a rubric with 4-6 checkable scoring bullets, and exactly one follow-up "
        "hint that does not leak the rubric."
    )
    lines.append("Assign a unique id for each new question (for example gq-0041).")
    return "\n".join(lines)
