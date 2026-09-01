"""Semantic duplicate detection via embeddings. No SQLite or Pinecone writes."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from roleready.config.settings import Settings, get_settings
from roleready.generation.validate import load_jsonl_records, write_jsonl
from roleready.rag.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.92


@dataclass(frozen=True)
class DedupeSummary:
    input_questions: int
    retained_questions: int
    semantic_duplicates: int
    threshold: float


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding lengths must match for cosine similarity.")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_embedding_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Embedding cache unreadable; regenerating missing vectors.")
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def save_embedding_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")


def cache_entry_is_current(entry: object, *, model: str, dimensions: int, fingerprint: str) -> bool:
    if not isinstance(entry, dict):
        return False
    vector = entry.get("vector")
    return (
        entry.get("model") == model
        and entry.get("dimensions") == dimensions
        and entry.get("text_sha256") == fingerprint
        and isinstance(vector, list)
        and bool(vector)
        and all(isinstance(value, (int, float)) for value in vector)
    )


def embeddings_for_questions(
    records: list[dict],
    client: EmbeddingClient,
    cache: dict,
) -> dict[str, list[float]]:
    model = client.model_name
    dimensions = client.dimension()
    vectors: dict[str, list[float]] = {}
    missing_ids: list[str] = []
    missing_texts: list[str] = []

    for record in records:
        question_id = str(record["id"])
        text = str(record["question_text"])
        fingerprint = text_fingerprint(text)
        entry = cache.get(question_id)
        if cache_entry_is_current(entry, model=model, dimensions=dimensions, fingerprint=fingerprint):
            vectors[question_id] = [float(value) for value in entry["vector"]]
            continue
        missing_ids.append(question_id)
        missing_texts.append(text)

    if missing_texts:
        logger.info(
            "Embedding %s question texts with %s (cached %s). No API keys logged.",
            len(missing_texts),
            model,
            len(vectors),
        )
        computed = client.embed_texts(missing_texts)
        if len(computed) != len(missing_ids):
            raise ValueError("Embedding client returned the wrong number of vectors.")
        for question_id, text, vector in zip(missing_ids, missing_texts, computed, strict=True):
            values = [float(value) for value in vector]
            vectors[question_id] = values
            cache[question_id] = {
                "model": model,
                "dimensions": dimensions,
                "text_sha256": text_fingerprint(text),
                "vector": values,
            }
    else:
        logger.info("All %s question embeddings loaded from cache.", len(vectors))
    return vectors


def find_semantic_duplicates(
    records: list[dict],
    vectors: dict[str, list[float]],
    threshold: float,
) -> tuple[list[dict], list[dict]]:
    retained: list[dict] = []
    retained_ids: list[str] = []
    duplicates: list[dict] = []

    for record in records:
        question_id = str(record["id"])
        vector = vectors[question_id]
        best_id: str | None = None
        best_score = -1.0
        for kept_id in retained_ids:
            score = cosine_similarity(vector, vectors[kept_id])
            if score > best_score:
                best_score = score
                best_id = kept_id
        if best_id is not None and best_score >= threshold:
            kept = next(item for item in retained if str(item["id"]) == best_id)
            duplicates.append(
                {
                    "rejected_question_id": question_id,
                    "retained_question_id": best_id,
                    "rejected_question_text": record["question_text"],
                    "retained_question_text": kept["question_text"],
                    "cosine_similarity": round(best_score, 6),
                    "threshold": threshold,
                }
            )
            continue
        retained.append(record)
        retained_ids.append(question_id)
    return retained, duplicates


def deduplicate_questions(
    *,
    input_path: Path,
    clean_path: Path,
    duplicates_path: Path,
    cache_path: Path,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    client: EmbeddingClient | None = None,
    settings: Settings | None = None,
) -> DedupeSummary:
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1.")
    resolved_input = input_path.resolve()
    if resolved_input in {clean_path.resolve(), duplicates_path.resolve(), cache_path.resolve()}:
        raise ValueError("Output or cache paths must not overwrite the input file.")

    loaded = load_jsonl_records(input_path)
    records = [payload for _line, payload in loaded if isinstance(payload, dict) and "id" in payload]
    settings = settings or get_settings()
    embedder = client or EmbeddingClient(settings=settings)
    cache = load_embedding_cache(cache_path)
    vectors = embeddings_for_questions(records, embedder, cache)
    save_embedding_cache(cache_path, cache)

    retained, duplicates = find_semantic_duplicates(records, vectors, threshold)
    write_jsonl(clean_path, retained)
    write_jsonl(duplicates_path, duplicates)

    summary = DedupeSummary(
        input_questions=len(records),
        retained_questions=len(retained),
        semantic_duplicates=len(duplicates),
        threshold=threshold,
    )
    logger.info(
        "Semantic dedupe finished: input=%s retained=%s duplicates=%s threshold=%s",
        summary.input_questions,
        summary.retained_questions,
        summary.semantic_duplicates,
        summary.threshold,
    )
    return summary


def format_summary(summary: DedupeSummary) -> str:
    return (
        f"input questions: {summary.input_questions}\n"
        f"retained questions: {summary.retained_questions}\n"
        f"semantic duplicates: {summary.semantic_duplicates}\n"
        f"threshold used: {summary.threshold}"
    )
