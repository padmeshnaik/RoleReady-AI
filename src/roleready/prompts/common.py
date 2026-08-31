"""Shared prompt helpers. No LLM client calls."""


def format_skills(skills: list[str]) -> str:
    return ", ".join(skills) if skills else "(none listed)"
