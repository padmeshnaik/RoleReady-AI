"""Batch SQLite questions into Pinecone. Does not change retrieval."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from roleready.db.models import Question
from roleready.rag.embeddings import EmbeddingClient, embedding_text
from roleready.rag.pinecone_store import PineconeQuestionStore

ProgressFn = Callable[[str], None]

DEFAULT_INGEST_BATCH_SIZE = 32


def ingest_questions(
    questions: Sequence[Question],
    embedder: EmbeddingClient,
    store: PineconeQuestionStore,
    *,
    batch_size: int = DEFAULT_INGEST_BATCH_SIZE,
    progress: ProgressFn | None = None,
) -> int:
    """Embed question_text + rubric and upsert by question id. Safe to rerun."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    log = progress or (lambda _msg: None)
    total = len(questions)
    log(
        f"Ingesting {total} questions from SQLite into index {store.index_name} "
        f"using {embedder.model_name}."
    )
    upserted = 0
    for start in range(0, total, batch_size):
        batch = list(questions[start : start + batch_size])
        end = start + len(batch)
        log(f"Embedding {start + 1}-{end}/{total}")
        vectors = embedder.embed_texts([embedding_text(question) for question in batch])
        log(f"Upserting {start + 1}-{end}/{total}")
        upserted += store.upsert_vectors(
            batch,
            vectors,
            batch_size=len(batch),
        )
        log(f"Progress {upserted}/{total}")
    log(
        f"Done. Upserted {upserted} vectors. "
        "Vector IDs match SQLite question ids, so reruns overwrite instead of duplicating."
    )
    return upserted
