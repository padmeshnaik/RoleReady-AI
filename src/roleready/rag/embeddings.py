"""OpenAI embeddings for question-bank ingest. Keys come from settings, never literals."""

from __future__ import annotations

from typing import Protocol

from langchain_openai import OpenAIEmbeddings

from roleready.config.settings import Settings, get_settings
from roleready.db.models import Question

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

VARIABLE_DIMENSION_MODELS = frozenset(
    {"text-embedding-3-small", "text-embedding-3-large"}
)


def embedding_text(question: Question) -> str:
    """SQLite question_text + rubric is what gets embedded. Full text stays in SQLite."""
    return f"{question.question_text.strip()}\n\n{question.rubric.strip()}"


def native_dimension(model_name: str) -> int:
    known = EMBEDDING_DIMENSIONS.get(model_name)
    if known is None:
        raise ValueError(
            f"Unknown embedding model {model_name!r}. "
            "Add its dimension to EMBEDDING_DIMENSIONS or use a supported model."
        )
    return known


class EmbeddingsLike(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class EmbeddingClient:
    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: EmbeddingsLike | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._output_dim = self._resolve_output_dimension(dimensions)
        if embeddings is not None:
            self._embeddings = embeddings
        else:
            kwargs: dict = {
                "model": self._settings.openai_embedding_model,
                "api_key": self._settings.openai_api_key,
            }
            native = native_dimension(self.model_name)
            if self._output_dim != native:
                kwargs["dimensions"] = self._output_dim
            self._embeddings = OpenAIEmbeddings(**kwargs)

    @property
    def model_name(self) -> str:
        return self._settings.openai_embedding_model

    def dimension(self) -> int:
        return self._output_dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    def _resolve_output_dimension(self, override: int | None) -> int:
        native = native_dimension(self.model_name)
        requested = override if override is not None else self._settings.openai_embedding_dimensions
        if requested is None:
            return native
        if requested == native:
            return native
        if self.model_name not in VARIABLE_DIMENSION_MODELS:
            raise ValueError(
                f"Model {self.model_name!r} always produces {native}-d vectors. "
                f"The Pinecone index is {requested}-d. Recreate the index at {native} "
                "or switch OPENAI_EMBEDDING_MODEL to text-embedding-3-small."
            )
        if requested > native:
            raise ValueError(
                f"Cannot request {requested} dimensions from {self.model_name!r} "
                f"(native maximum is {native})."
            )
        return requested
