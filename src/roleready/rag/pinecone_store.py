"""Pinecone vector index for questions. SQLite remains the source of truth."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pinecone import Pinecone, ServerlessSpec

from roleready.config.settings import Settings, get_settings
from roleready.db.models import Question

ProgressFn = Callable[[str], None]


def question_metadata(question: Question) -> dict:
    return {
        "company": question.company,
        "role": question.role,
        "seniority": question.seniority,
        "category": question.category,
        "difficulty": question.difficulty,
        "question_id": question.id,
    }


class PineconeQuestionStore:
    def __init__(
        self,
        settings: Settings | None = None,
        client: Pinecone | None = None,
        index=None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._index = index

    def _pinecone_client(self) -> Pinecone:
        if self._client is None:
            self._client = Pinecone(api_key=self._settings.pinecone_api_key)
        return self._client

    @property
    def index_name(self) -> str:
        return self._settings.pinecone_index

    def describe_dimension(self) -> int | None:
        """Return the existing index dimension, or None if the index does not exist."""
        client = self._pinecone_client()
        if not client.has_index(self.index_name):
            return None
        info = client.describe_index(self.index_name)
        dimension = getattr(info, "dimension", None)
        if dimension is None and isinstance(info, dict):
            dimension = info.get("dimension")
        return int(dimension) if dimension is not None else None

    def ensure_index(self, dimension: int, progress: ProgressFn | None = None) -> int:
        log = progress or (lambda _msg: None)
        existing = self.describe_dimension()
        if existing is not None:
            log(f"Pinecone index already exists: {self.index_name} (dimension={existing})")
            if existing != dimension:
                raise ValueError(
                    f"Pinecone index {self.index_name!r} is {existing}-d but embeddings "
                    f"are {dimension}-d. Ingest now matches an existing index automatically "
                    "for text-embedding-3-* models. If you still see this error, recreate "
                    f"the index at {dimension} or set OPENAI_EMBEDDING_DIMENSIONS={existing}."
                )
            return existing
        log(
            f"Creating Pinecone index {self.index_name} "
            f"(dimension={dimension}, metric=cosine, "
            f"cloud={self._settings.pinecone_cloud}, region={self._settings.pinecone_region})"
        )
        self._pinecone_client().create_index(
            name=self.index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=self._settings.pinecone_cloud,
                region=self._settings.pinecone_region,
            ),
        )
        log(f"Created Pinecone index {self.index_name}")
        return dimension

    def upsert_vectors(
        self,
        questions: Sequence[Question],
        vectors: Sequence[Sequence[float]],
        *,
        batch_size: int = 100,
        progress: ProgressFn | None = None,
    ) -> int:
        if len(questions) != len(vectors):
            raise ValueError("questions and vectors must be the same length.")
        if vectors:
            vector_dim = len(vectors[0])
            index_dim = self.describe_dimension()
            if index_dim is not None and vector_dim != index_dim:
                raise ValueError(
                    f"Vector dimension {vector_dim} does not match the dimension of the "
                    f"index {index_dim}. Use text-embedding-3-small with matching "
                    f"OPENAI_EMBEDDING_DIMENSIONS={index_dim}, or recreate the index."
                )
        log = progress or (lambda _msg: None)
        index = self._get_index()
        total = 0
        for start in range(0, len(questions), batch_size):
            batch_questions = questions[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            records = [
                {
                    "id": question.id,
                    "values": list(values),
                    "metadata": question_metadata(question),
                }
                for question, values in zip(batch_questions, batch_vectors, strict=True)
            ]
            index.upsert(vectors=records)
            total += len(records)
            log(f"Upserted {total}/{len(questions)} vectors")
        return total

    def query_matches(
        self,
        vector: Sequence[float],
        *,
        metadata_filter: dict,
        top_k: int = 15,
    ) -> list[tuple[str, float | None]]:
        """Semantic search. Returns (question_id, similarity) pairs; hydrate from SQLite."""
        result = self._get_index().query(
            vector=list(vector),
            top_k=top_k,
            filter=metadata_filter,
            include_metadata=True,
        )
        matches = getattr(result, "matches", None)
        if matches is None and isinstance(result, dict):
            matches = result.get("matches", [])
        ranked: list[tuple[str, float | None]] = []
        for match in matches or []:
            metadata = getattr(match, "metadata", None)
            if metadata is None and isinstance(match, dict):
                metadata = match.get("metadata") or {}
            question_id = None
            if isinstance(metadata, dict):
                question_id = metadata.get("question_id")
            if not question_id:
                question_id = getattr(match, "id", None) or (
                    match.get("id") if isinstance(match, dict) else None
                )
            raw_score = getattr(match, "score", None)
            if raw_score is None and isinstance(match, dict):
                raw_score = match.get("score")
            score = float(raw_score) if raw_score is not None else None
            if question_id:
                ranked.append((str(question_id), score))
        return ranked

    def query_question_ids(
        self,
        vector: Sequence[float],
        *,
        metadata_filter: dict,
        top_k: int = 15,
    ) -> list[str]:
        return [question_id for question_id, _score in self.query_matches(
            vector, metadata_filter=metadata_filter, top_k=top_k
        )]

    def _get_index(self):
        if self._index is None:
            self._index = self._pinecone_client().Index(self.index_name)
        return self._index
