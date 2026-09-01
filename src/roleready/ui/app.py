"""Streamlit UI. Holds view state only; InterviewApp owns the interview lifecycle."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from pydantic import ValidationError
from roleready.config.settings import apply_env_overrides, clear_settings_cache, missing_env_names
from roleready.db.sqlite import connect, seed_from_jsonl
from roleready.llm.scorer import ScoringError
from roleready.orchestration import InterviewApp
from roleready.rag.retriever import RetrievalError
from roleready.session.models import SessionStatus
from roleready.ui.components import (
    collected_skills,
    current_question_number,
    render_answer_form,
    render_chat,
    render_final_report,
    render_setup_form,
    render_turn_score,
    setup_choices,
)

logger = logging.getLogger("roleready.ui")

st.set_page_config(page_title="RoleReady AI", layout="centered")

_USER_FAILURE = "Something went wrong. Check the terminal logs and try again."


def _public_error(exc: BaseException, fallback: str) -> str:
    if isinstance(exc, (RetrievalError, ScoringError)):
        return str(exc)
    logger.exception(fallback)
    return fallback


def _streamlit_secrets_env() -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        secrets = st.secrets
    except Exception:
        return values
    try:
        keys = list(secrets.keys())
    except Exception:
        return values
    for key in keys:
        try:
            raw = secrets[key]
        except Exception:
            continue
        if isinstance(raw, (str, int, float)):
            values[str(key)] = str(raw)
    return values


def _ensure_question_bank(app: InterviewApp) -> None:
    jsonl = REPO_ROOT / "data" / "questions_clean.jsonl"
    if not jsonl.exists():
        return
    try:
        existing = app.list_bank_questions()
    except Exception:
        existing = []
    if existing:
        return
    conn = connect(app.db_path)
    try:
        seed_from_jsonl(conn, jsonl)
    finally:
        conn.close()


def _ensure_app() -> InterviewApp | None:
    if "app" in st.session_state:
        return st.session_state.app
    apply_env_overrides(_streamlit_secrets_env())
    clear_settings_cache()
    try:
        app = InterviewApp.from_settings(project_root=REPO_ROOT, live=False)
        _ensure_question_bank(app)
    except ValidationError as exc:
        logger.exception("Could not load RoleReady AI settings.")
        missing = missing_env_names(exc)
        detail = f" Missing: {', '.join(missing)}." if missing else ""
        st.error(
            "Could not start RoleReady AI. "
            "On Streamlit Cloud, set these names in App settings → Secrets "
            "(GitHub does not include your local `.env`)."
            + detail
        )
        return None
    except Exception:
        logger.exception("Could not load RoleReady AI settings.")
        st.error(
            "Could not start RoleReady AI. "
            "Check Streamlit secrets (or local `.env`) and that SQLite is seeded."
        )
        return None
    st.session_state.app = app
    return app


def _reset_interview() -> None:
    for key in ("interview", "review_turn_index", "ui_error"):
        st.session_state.pop(key, None)


def _render_setup(app: InterviewApp) -> None:
    try:
        choices = setup_choices(app.list_bank_questions())
    except Exception:
        logger.exception("Could not load the question bank.")
        st.error("Could not load the question bank. Run `python scripts/init_db.py`.")
        return

    render_setup_form(choices)
    if st.session_state.get("ui_error"):
        st.error(st.session_state.ui_error)

    if st.button("Start Interview", type="primary"):
        name = (st.session_state.get("candidate_name") or "").strip()
        skills = collected_skills()
        if not name:
            st.session_state.ui_error = "Enter a candidate name to start."
            st.rerun()
        if not skills:
            st.session_state.ui_error = "Select or enter at least one skill."
            st.rerun()
        try:
            session = app.create_session(
                company=st.session_state.setup_company,
                role=st.session_state.setup_role,
                seniority=st.session_state.setup_seniority,
                category=st.session_state.setup_category,
                skills=skills,
            )
            session = app.start_interview(session)
            st.session_state.interview = session
            st.session_state.ui_error = None
            st.session_state.review_turn_index = None
        except Exception as exc:
            st.session_state.ui_error = _public_error(
                exc,
                "Could not start the interview. Check `.env`, Pinecone ingest, and the question bank.",
            )
        st.rerun()


def _render_interview(app: InterviewApp) -> None:
    session = st.session_state.interview
    question_count = app.settings.interview_question_count
    name = st.session_state.get("candidate_name") or "Candidate"

    st.title("RoleReady AI")
    st.caption(
        f"{name} · {session.company} · {session.role} · {session.seniority} · {session.category}"
    )
    review_index = st.session_state.get("review_turn_index")
    number = current_question_number(session, review_index, question_count)
    st.progress(min(session.question_index / question_count, 1.0))
    st.subheader(f"Question {number} / {question_count}")
    turn = session.current_turn
    if (
        review_index is None
        and turn is not None
        and not turn.is_scored
        and turn.interviewer_followup
        and turn.follow_up_answer is None
    ):
        st.info(
            "This is a follow-up on the same question. "
            f"It does not count as the next question of {question_count}."
        )

    if st.session_state.get("ui_error"):
        st.error(st.session_state.ui_error)

    render_chat(session, review_index, question_count)

    if review_index is not None and 0 <= review_index < len(session.turns):
        reviewed = session.turns[review_index]
        if reviewed.is_scored:
            render_turn_score(reviewed, question_count)
            label = "View final report" if session.status is SessionStatus.COMPLETE else "Next question"
            if st.button(label, type="primary"):
                st.session_state.review_turn_index = None
                st.session_state.ui_error = None
                st.rerun()
            return

    if session.status is SessionStatus.COMPLETE:
        _render_final(app)
        return

    answer = render_answer_form(disabled=session.status is not SessionStatus.AWAITING_ANSWER)
    if answer is None:
        return
    if not answer.strip():
        st.session_state.ui_error = "Enter an answer before submitting."
        st.rerun()

    scored_before = {id(turn) for turn in session.turns if turn.is_scored}
    try:
        session = app.submit_answer(session, answer)
        st.session_state.interview = session
        st.session_state.ui_error = None
        newly_scored = [
            index
            for index, turn in enumerate(session.turns)
            if turn.is_scored and id(turn) not in scored_before
        ]
        if newly_scored:
            st.session_state.review_turn_index = newly_scored[-1]
    except Exception as exc:
        st.session_state.ui_error = _public_error(exc, _USER_FAILURE)
    st.rerun()


def _render_final(app: InterviewApp) -> None:
    session = st.session_state.interview
    render_final_report(session, app.settings.interview_question_count)
    if st.button("Restart interview", type="primary"):
        _reset_interview()
        st.rerun()


def main() -> None:
    app = _ensure_app()
    if app is None:
        return

    interview = st.session_state.get("interview")
    if interview is None or interview.status is SessionStatus.SETUP:
        _render_setup(app)
        return
    if (
        interview.status is SessionStatus.COMPLETE
        and st.session_state.get("review_turn_index") is None
    ):
        _render_final(app)
        return
    _render_interview(app)


main()
