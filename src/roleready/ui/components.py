"""Pure presentation helpers and Streamlit widgets. No interview state machine here."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

import streamlit as st

from roleready.db.models import GENERIC_COMPANY, Question
from roleready.llm.schemas import FinalEvaluationResult
from roleready.session.models import InterviewSession, InterviewTurn

logger = logging.getLogger(__name__)

DEFAULT_SKILLS = [
    "Python",
    "SQL",
    "APIs",
    "distributed systems",
    "system design",
    "machine learning",
]

_DISPLAY_LABELS = {
    "ai_engineer": "AI Engineer",
    "machine_learning_engineer": "Machine Learning Engineer",
    "software_engineer": "Software Engineer",
    "backend_engineer": "Backend Engineer",
    "data_engineer": "Data Engineer",
    "system_design": "System Design",
    "technical": "Technical",
    "behavioral": "Behavioral",
    "coding": "Coding",
    "junior": "Junior",
    "mid": "Mid",
    "senior": "Senior",
}

_TOKEN_LABELS = {
    "ai": "AI",
    "ml": "ML",
}


def display_label(value: str) -> str:
    """Show Title Case labels in the UI. Stored values stay snake_case for retrieval."""
    if not value:
        return value
    known = _DISPLAY_LABELS.get(value)
    if known is not None:
        return known
    if "_" not in value and "-" not in value:
        return value
    parts = value.replace("-", "_").split("_")
    return " ".join(_TOKEN_LABELS.get(part.lower(), part.capitalize()) for part in parts if part)


def unique_sorted(values: Sequence[str]) -> list[str]:
    return sorted({value for value in values if value})


def setup_choices(questions: Sequence[Question]) -> dict[str, list[str]]:
    companies = unique_sorted([q.company for q in questions] + [GENERIC_COMPANY])
    roles = unique_sorted([q.role for q in questions]) or ["software_engineer"]
    seniorities = unique_sorted([q.seniority for q in questions]) or ["junior", "mid", "senior"]
    categories = unique_sorted([q.category for q in questions]) or [
        "technical",
        "system_design",
        "behavioral",
    ]
    return {
        "companies": companies,
        "roles": roles,
        "seniorities": seniorities,
        "categories": categories,
    }


def parse_final_report(raw: str | None) -> FinalEvaluationResult | None:
    if not raw:
        return None
    try:
        return FinalEvaluationResult.model_validate_json(raw)
    except Exception:
        logger.exception("Could not parse final evaluation JSON.")
        try:
            return FinalEvaluationResult.model_validate(json.loads(raw))
        except Exception:
            logger.exception("Could not parse final evaluation object.")
            return None


def score_dimension_averages(session: InterviewSession) -> dict[str, float]:
    scored = [turn.score for turn in session.turns if turn.score is not None]
    if not scored:
        return {}

    def avg(getter) -> float:
        values = [getter(item) for item in scored if getter(item) is not None]
        return sum(values) / len(values) if values else 0.0

    return {
        "Overall": avg(lambda s: s.score),
        "Technical accuracy": avg(lambda s: s.technical_accuracy),
        "Communication": avg(lambda s: s.communication),
        "Structure": avg(lambda s: s.structure),
    }


def current_question_number(
    session: InterviewSession,
    review_turn_index: int | None = None,
    question_count: int = 10,
) -> int:
    if review_turn_index is not None and 0 <= review_turn_index < len(session.turns):
        return session.turns[review_turn_index].question_number
    turn = session.current_turn
    if turn is not None:
        return turn.question_number
    return min(session.question_index + 1, question_count)


def visible_messages(session: InterviewSession, review_turn_index: int | None = None) -> list:
    """Hide the next interviewer question while a score review is on screen."""
    messages = list(session.messages)
    if review_turn_index is None:
        return messages
    if review_turn_index < 0 or review_turn_index >= len(session.turns):
        return messages
    keep = 0
    for turn in session.turns[: review_turn_index + 1]:
        keep += 1
        if turn.user_answer:
            keep += 1
        if turn.interviewer_followup:
            keep += 1
        if turn.follow_up_answer:
            keep += 1
    return messages[:keep]


def render_setup_form(choices: dict[str, list[str]]) -> None:
    st.title("RoleReady AI")
    st.caption("Mock interview practice with retrieval-backed questions.")
    st.text_input("Candidate Name", key="candidate_name")
    st.selectbox("Company", choices["companies"], key="setup_company")
    st.selectbox("Role / Position", choices["roles"], format_func=display_label, key="setup_role")
    st.selectbox("Seniority", choices["seniorities"], format_func=display_label, key="setup_seniority")
    st.multiselect("Skills", DEFAULT_SKILLS, default=["Python"], key="setup_skills")
    extra = st.text_input("Additional skills (comma-separated)", key="setup_skills_extra")
    if extra.strip():
        st.caption("Extra skills will be included when you start.")
    st.selectbox(
        "Interview Category",
        choices["categories"],
        format_func=display_label,
        key="setup_category",
    )


def collected_skills() -> list[str]:
    selected = list(st.session_state.get("setup_skills") or [])
    extra = st.session_state.get("setup_skills_extra") or ""
    for token in extra.split(","):
        skill = token.strip()
        if skill and skill not in selected:
            selected.append(skill)
    return selected


def render_chat(
    session: InterviewSession,
    review_turn_index: int | None = None,
    question_count: int = 10,
) -> None:
    for message in visible_messages(session, review_turn_index):
        role = "assistant" if message.role == "interviewer" else "user"
        with st.chat_message(role):
            if message.kind == "follow_up":
                st.caption(
                    "Follow-up — still the same scored question, "
                    f"not the next one of {question_count}"
                )
            st.markdown(message.content)


def render_turn_score(turn: InterviewTurn, question_count: int = 10) -> None:
    score = turn.score
    if score is None:
        return
    st.subheader(f"Score for question {turn.question_number} of {question_count}")
    if turn.interviewer_followup or turn.follow_up_answer:
        st.caption(
            "The previous interviewer prompt was a clarification, not question "
            f"{turn.question_number + 1}. Both of your answers were scored together."
        )
    st.metric("Score", f"{score.score} / 10")
    cols = st.columns(3)
    cols[0].metric("Technical", score.technical_accuracy if score.technical_accuracy is not None else "—")
    cols[1].metric("Communication", score.communication if score.communication is not None else "—")
    cols[2].metric("Structure", score.structure if score.structure is not None else "—")
    if score.feedback:
        st.markdown("**Feedback**")
        st.write(score.feedback)
    if score.missed_points:
        st.markdown("**Missed points**")
        for point in score.missed_points:
            st.write(f"- {point}")


def render_answer_form(disabled: bool = False) -> str | None:
    with st.form("answer_form", clear_on_submit=True):
        answer = st.text_area("Your answer", height=160, disabled=disabled)
        submitted = st.form_submit_button("Submit answer", disabled=disabled)
    if submitted:
        return answer
    return None


def render_final_report(session: InterviewSession, question_count: int) -> None:
    st.title("Interview results")
    report = parse_final_report(session.final_evaluation)
    if report is not None:
        st.metric("Overall score", f"{report.overall_score:.1f} / 10")
        st.caption(report.recommendation)
        if report.summary:
            st.write(report.summary)
        st.subheader("Strengths")
        for item in report.strengths or ["(none recorded)"]:
            st.write(f"- {item}")
        st.subheader("Improvement areas")
        gaps = list(report.gaps) + list(report.practice_next)
        for item in gaps or ["(none recorded)"]:
            st.write(f"- {item}")
    elif session.final_evaluation:
        st.write(session.final_evaluation)
    else:
        st.info("Final evaluation was not produced.")

    averages = score_dimension_averages(session)
    if averages:
        st.subheader("Category breakdown")
        cols = st.columns(len(averages))
        for column, (label, value) in zip(cols, averages.items(), strict=True):
            column.metric(label, f"{value:.1f}")

    st.subheader("Per-question feedback")
    for turn in session.turns:
        title = f"Question {turn.question_number} / {question_count}"
        if turn.score is not None:
            title += f" — {turn.score.score}/10"
        with st.expander(title):
            st.markdown("**Question**")
            st.write(turn.question_text)
            st.markdown("**Your answer**")
            st.write(turn.user_answer or "(none)")
            if turn.follow_up_answer:
                st.markdown("**Follow-up answer**")
                st.write(turn.follow_up_answer)
            render_turn_score(turn, question_count)
