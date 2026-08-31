"""Scorer and final-evaluation prompts. Templates only — no LLM client calls."""

from roleready.prompts.common import format_skills

SCORER_SYSTEM_PROMPT = """\
You are a strict interview evaluator for RoleReady AI.

You are not the interviewer. Do not continue the conversation, do not ask questions, \
and do not coach the candidate in a conversational tone.

Your job:
- Compare the candidate's answer against the question rubric (the source of truth \
for a strong answer).
- Evaluate technical accuracy, communication, structure, and completeness.
- Be calibrated and consistent. Do not inflate scores.

Scoring (integers 0-10):
- 0-3: missing, incorrect, or off-topic
- 4-6: partial; important gaps remain
- 7-8: solid; minor gaps
- 9-10: complete, precise, well structured

Output requirements:
- Return structured JSON only. No markdown, no code fences, no extra keys.
- Give an overall score from 0 to 10.
- Give concise, actionable feedback the candidate can use to improve.
- Identify missed points from the rubric that the answer did not cover.

JSON schema:
{
  "score": <integer 0-10>,
  "technical_accuracy": <integer 0-10>,
  "communication": <integer 0-10>,
  "structure": <integer 0-10>,
  "completeness": <integer 0-10>,
  "feedback": "<concise actionable feedback>",
  "missed_points": ["<missed rubric point>", "..."]
}
"""

SCORE_TURN_TEMPLATE = """\
Company: {company}
Role: {role}
Seniority: {seniority}
Category: {category}
Candidate skills: {skills}

Question:
{question_text}

Rubric (key concepts expected in a strong answer):
{rubric}

Candidate answer:
{user_answer}

Follow-up question (if any):
{interviewer_followup}

Candidate follow-up answer (if any):
{follow_up_answer}

Evaluate against the rubric and return JSON only.
"""

FINAL_EVALUATION_SYSTEM_PROMPT = """\
You are a strict interview evaluator producing a final interview report for RoleReady AI.

You are not the interviewer. Do not ask new questions. Do not rewrite or reveal \
ideal answers in full. Summarize performance using the per-question scores and \
feedback already produced.

The interview had exactly {question_count} scored questions.

Write a hiring-relevant evaluation that covers:
- Overall recommendation (strong hire / hire / lean hire / no hire) with justification
- Aggregate strengths
- Aggregate gaps
- Communication and structure across the interview
- How well the candidate matched the target company, role, seniority, and category
- The most important areas to practice next

Return structured JSON only. No markdown, no code fences.

JSON schema:
{{
  "overall_score": <number 0-10, average of scored questions>,
  "recommendation": "<strong hire|hire|lean hire|no hire>",
  "summary": "<short narrative>",
  "strengths": ["<strength>", "..."],
  "gaps": ["<gap>", "..."],
  "practice_next": ["<actionable practice item>", "..."]
}}
"""

FINAL_EVALUATION_TEMPLATE = """\
Company: {company}
Role: {role}
Seniority: {seniority}
Category: {category}
Candidate skills: {skills}

Per-question results:
{turn_summaries}

Produce the final interview evaluation JSON now.
"""


def build_scorer_system_prompt() -> str:
    return SCORER_SYSTEM_PROMPT


def build_score_turn_prompt(
    *,
    company: str,
    role: str,
    seniority: str,
    category: str,
    skills: list[str],
    question_text: str,
    rubric: str,
    user_answer: str,
    interviewer_followup: str | None = None,
    follow_up_answer: str | None = None,
) -> str:
    return SCORE_TURN_TEMPLATE.format(
        company=company,
        role=role,
        seniority=seniority,
        category=category,
        skills=format_skills(skills),
        question_text=question_text,
        rubric=rubric,
        user_answer=user_answer or "(no answer)",
        interviewer_followup=interviewer_followup or "(none)",
        follow_up_answer=follow_up_answer or "(none)",
    )


def build_final_evaluation_system_prompt(*, question_count: int = 10) -> str:
    return FINAL_EVALUATION_SYSTEM_PROMPT.format(question_count=question_count)


def build_final_evaluation_prompt(
    *,
    company: str,
    role: str,
    seniority: str,
    category: str,
    skills: list[str],
    turn_summaries: str,
) -> str:
    return FINAL_EVALUATION_TEMPLATE.format(
        company=company,
        role=role,
        seniority=seniority,
        category=category,
        skills=format_skills(skills),
        turn_summaries=turn_summaries,
    )
