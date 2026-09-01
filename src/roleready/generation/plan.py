"""Configurable corpus mix for offline question generation.

Company counts below are exact at target_count=1000 and scale at other targets.
Category is assigned from role + seniority (not a full cartesian product).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from roleready.generation.schemas import MAX_DIFFICULTY, MIN_DIFFICULTY

# Scaled exactly at 1000. At 100 this is 25/20/20/15/20.
DEFAULT_ROLE_WEIGHTS: dict[str, int] = {
    "software_engineer": 250,
    "backend_engineer": 200,
    "data_engineer": 200,
    "machine_learning_engineer": 150,
    "ai_engineer": 200,
}

DEFAULT_SENIORITY_WEIGHTS: dict[str, int] = {
    "junior": 300,
    "mid": 400,
    "senior": 300,
}

DEFAULT_GENERIC_RATIO = 0.65

# Exact named-company counts at 1000 (remainder after 650 Generic).
DEFAULT_NAMED_COMPANY_WEIGHTS: dict[str, int] = {
    "Google": 70,
    "Amazon": 70,
    "Microsoft": 60,
    "Meta": 60,
    "FinTech": 90,
}

# Fallback only; live assignment uses ROLE_SENIORITY_CATEGORY_WEIGHTS.
DEFAULT_CATEGORY_WEIGHTS: dict[str, int] = {
    "technical": 1,
    "system_design": 1,
    "behavioral": 1,
    "coding": 1,
}

SENIORITY_DIFFICULTY: dict[str, tuple[int, int]] = {
    "junior": (1, 2),
    "mid": (3, 4),
    "senior": (4, 5),
}

# Higher system_design for senior software/backend. Higher technical for ML/AI.
ROLE_SENIORITY_CATEGORY_WEIGHTS: dict[str, dict[str, dict[str, int]]] = {
    "software_engineer": {
        "junior": {"coding": 4, "technical": 3, "behavioral": 2, "system_design": 1},
        "mid": {"technical": 3, "system_design": 3, "coding": 2, "behavioral": 2},
        "senior": {"system_design": 5, "technical": 3, "behavioral": 2, "coding": 1},
    },
    "backend_engineer": {
        "junior": {"coding": 4, "technical": 3, "behavioral": 2, "system_design": 1},
        "mid": {"technical": 3, "system_design": 3, "coding": 2, "behavioral": 2},
        "senior": {"system_design": 5, "technical": 3, "behavioral": 2, "coding": 1},
    },
    "data_engineer": {
        "junior": {"technical": 4, "coding": 3, "behavioral": 2, "system_design": 1},
        "mid": {"technical": 3, "system_design": 3, "coding": 2, "behavioral": 2},
        "senior": {"system_design": 4, "technical": 3, "behavioral": 2, "coding": 1},
    },
    "machine_learning_engineer": {
        "junior": {"technical": 4, "coding": 3, "behavioral": 2, "system_design": 1},
        "mid": {"technical": 4, "coding": 2, "system_design": 2, "behavioral": 2},
        "senior": {"technical": 4, "system_design": 3, "behavioral": 2, "coding": 1},
    },
    "ai_engineer": {
        "junior": {"technical": 4, "coding": 3, "behavioral": 2, "system_design": 1},
        "mid": {"technical": 4, "coding": 2, "system_design": 2, "behavioral": 2},
        "senior": {"technical": 4, "system_design": 3, "behavioral": 2, "coding": 1},
    },
}


@dataclass(frozen=True)
class CorpusDistribution:
    """Company/role/seniority marginals plus role-dependent category weights."""

    role_weights: dict[str, int]
    category_weights: dict[str, int]
    seniority_weights: dict[str, int]
    generic_ratio: float
    named_company_weights: dict[str, int]
    role_seniority_category_weights: dict[str, dict[str, dict[str, int]]]


DEFAULT_CORPUS_DISTRIBUTION = CorpusDistribution(
    role_weights=DEFAULT_ROLE_WEIGHTS,
    category_weights=DEFAULT_CATEGORY_WEIGHTS,
    seniority_weights=DEFAULT_SENIORITY_WEIGHTS,
    generic_ratio=DEFAULT_GENERIC_RATIO,
    named_company_weights=DEFAULT_NAMED_COMPANY_WEIGHTS,
    role_seniority_category_weights=ROLE_SENIORITY_CATEGORY_WEIGHTS,
)


def allocate_counts(weights: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation so counts sum to total and scale with target_count."""
    if total < 0:
        raise ValueError("total cannot be negative.")
    if not weights:
        raise ValueError("weights cannot be empty.")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("weights must be positive.")
    if total == 0:
        return {key: 0 for key in weights}

    exact = {key: total * (value / weight_sum) for key, value in weights.items()}
    counts = {key: int(exact[key]) for key in weights}
    leftover = total - sum(counts.values())
    order = sorted(weights, key=lambda key: (exact[key] - counts[key], key), reverse=True)
    for key in order[:leftover]:
        counts[key] += 1
    return counts


def _round_robin_expand(counts: dict[str, int]) -> list[str]:
    remaining = dict(counts)
    items: list[str] = []
    while any(remaining.values()):
        for key in remaining:
            if remaining[key] > 0:
                items.append(key)
                remaining[key] -= 1
    return items


def _stride_permute(items: list[str], stride: int) -> list[str]:
    n = len(items)
    if n <= 1:
        return list(items)
    step = stride
    while math.gcd(step, n) != 1:
        step += 1
    return [items[(index * step) % n] for index in range(n)]


def _company_sequence(n: int, distribution: CorpusDistribution) -> list[str]:
    generic_count = int(round(n * distribution.generic_ratio))
    generic_count = min(max(generic_count, 0), n)
    named_count = n - generic_count
    named_counts = allocate_counts(distribution.named_company_weights, named_count)
    labels = ["Generic"] * generic_count + _round_robin_expand(named_counts)
    return _stride_permute(labels, stride=3)


def difficulty_for(seniority: str, index: int) -> int:
    low, high = SENIORITY_DIFFICULTY.get(seniority, (MIN_DIFFICULTY, MAX_DIFFICULTY))
    span = high - low + 1
    return low + (index % span)


def _category_for(role: str, seniority: str, index: int, distribution: CorpusDistribution) -> str:
    by_role = distribution.role_seniority_category_weights.get(role) or ROLE_SENIORITY_CATEGORY_WEIGHTS["software_engineer"]
    weights = by_role.get(seniority) or by_role[next(iter(by_role))]
    cycle = _round_robin_expand(weights)
    return cycle[index % len(cycle)]


def build_generation_plan(
    target_count: int,
    distribution: CorpusDistribution = DEFAULT_CORPUS_DISTRIBUTION,
) -> list[dict]:
    """Deterministic slot list of length target_count. Index 0 is the first question to generate."""
    if target_count < 1:
        raise ValueError("target_count must be at least 1.")

    roles = _round_robin_expand(allocate_counts(distribution.role_weights, target_count))
    seniorities = _stride_permute(
        _round_robin_expand(allocate_counts(distribution.seniority_weights, target_count)),
        stride=7,
    )
    companies = _company_sequence(target_count, distribution)
    group_index: dict[tuple[str, str], int] = defaultdict(int)

    plan: list[dict] = []
    for index in range(target_count):
        role = roles[index]
        seniority = seniorities[index]
        key = (role, seniority)
        category = _category_for(role, seniority, group_index[key], distribution)
        group_index[key] += 1
        plan.append(
            {
                "company": companies[index],
                "role": role,
                "seniority": seniority,
                "category": category,
                "difficulty": difficulty_for(seniority, index),
            }
        )
    return plan


def remaining_plan(plan: list[dict], existing_count: int) -> list[dict]:
    if existing_count < 0:
        raise ValueError("existing_count cannot be negative.")
    return list(plan[existing_count:])


def planned_batches(slots: list[dict], batch_size: int) -> list[list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    return [slots[index : index + batch_size] for index in range(0, len(slots), batch_size)]


def plan_counts(plan: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for slot in plan:
        value = str(slot[field])
        counts[value] = counts.get(value, 0) + 1
    return counts


def _share(plan: list[dict], *, role: str, seniority: str | None, category: str) -> float:
    matching = [
        slot
        for slot in plan
        if slot["role"] == role and (seniority is None or slot["seniority"] == seniority)
    ]
    if not matching:
        return 0.0
    hits = sum(1 for slot in matching if slot["category"] == category)
    return hits / len(matching)


def format_plan_report(
    plan: list[dict],
    *,
    existing_count: int,
    batch_size: int,
) -> str:
    pending = remaining_plan(plan, existing_count)
    batches = planned_batches(pending, batch_size) if pending else []
    lines = [
        f"target slots: {len(plan)}",
        f"existing records (resume offset): {existing_count}",
        f"slots still to generate: {len(pending)}",
        f"batch size: {batch_size}",
        f"planned API batches remaining: {len(batches)}",
    ]
    for field in ("company", "role", "seniority", "category", "difficulty"):
        lines.append(f"full plan by {field}:")
        counts = plan_counts(plan, field)
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"  {key}: {value}")
    if existing_count:
        lines.append("remaining slots by company:")
        for key, value in sorted(
            plan_counts(pending, "company").items(), key=lambda item: (-item[1], item[0])
        ):
            lines.append(f"  {key}: {value}")
        lines.append("remaining slots by role:")
        for key, value in sorted(
            plan_counts(pending, "role").items(), key=lambda item: (-item[1], item[0])
        ):
            lines.append(f"  {key}: {value}")
    lines.append("role x category:")
    roles = sorted({slot["role"] for slot in plan})
    categories = ["technical", "system_design", "behavioral", "coding"]
    for role in roles:
        parts = []
        for category in categories:
            count = sum(1 for slot in plan if slot["role"] == role and slot["category"] == category)
            parts.append(f"{category}={count}")
        lines.append(f"  {role}: " + ", ".join(parts))
    lines.append(
        "senior software_engineer system_design share: "
        f"{_share(plan, role='software_engineer', seniority='senior', category='system_design'):.2f}"
    )
    lines.append(
        "junior software_engineer system_design share: "
        f"{_share(plan, role='software_engineer', seniority='junior', category='system_design'):.2f}"
    )
    lines.append(
        "machine_learning_engineer technical share: "
        f"{_share(plan, role='machine_learning_engineer', seniority=None, category='technical'):.2f}"
    )
    lines.append(
        "ai_engineer technical share: "
        f"{_share(plan, role='ai_engineer', seniority=None, category='technical'):.2f}"
    )
    return "\n".join(lines)
