"""Corpus quality-review helpers. No LLM, SQLite, or Pinecone."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from roleready.generation.validate import load_jsonl_records

SAMPLE_FIELDS = (
    "company",
    "role",
    "seniority",
    "category",
    "difficulty",
    "question_text",
    "rubric",
    "follow_up_hints",
)

DEFAULT_SAMPLE_SIZE = 25
DEFAULT_RANDOM_SEED = 42


def load_questions(path: Path) -> list[dict]:
    records: list[dict] = []
    for _line, payload in load_jsonl_records(path):
        if isinstance(payload, dict):
            records.append(payload)
    return records


def counts_by(records: list[dict], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter(str(record.get(field, "(missing)")) for record in records)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def format_counts(title: str, counts: dict[str, int]) -> str:
    lines = [title]
    for key, value in counts.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def format_review_stats(records: list[dict]) -> str:
    parts = [
        f"total questions: {len(records)}",
        format_counts("questions by company:", counts_by(records, "company")),
        format_counts("questions by role:", counts_by(records, "role")),
        format_counts("questions by seniority:", counts_by(records, "seniority")),
        format_counts("questions by category:", counts_by(records, "category")),
        format_counts("questions by difficulty:", counts_by(records, "difficulty")),
    ]
    return "\n".join(parts)


def sample_for_review(
    records: list[dict],
    *,
    size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict]:
    if size < 1:
        raise ValueError("sample size must be at least 1.")
    count = min(size, len(records))
    if count == 0:
        return []
    chosen = random.Random(seed).sample(records, count)
    sampled: list[dict] = []
    for record in chosen:
        sampled.append({field: record.get(field) for field in SAMPLE_FIELDS})
    return sampled


def write_review_sample(path: Path, sample: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
