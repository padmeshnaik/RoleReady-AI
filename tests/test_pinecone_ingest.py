"""Pinecone ingest helpers. No live OpenAI or Pinecone calls."""

import pytest

from roleready.config.settings import Settings
from roleready.rag.embeddings import EmbeddingClient, embedding_text
from roleready.rag.pinecone_store import PineconeQuestionStore, question_metadata
from tests.fakes import make_question


def test_embedding_text_is_question_plus_rubric() -> None:
    question = make_question(
        "q01",
        question_text="What is REST?",
        rubric="HTTP, resources, statelessness.",
    )
    text = embedding_text(question)
    assert "What is REST?" in text
    assert "HTTP, resources, statelessness." in text


def test_metadata_fields() -> None:
    question = make_question("q01", company="Google", difficulty=3)
    meta = question_metadata(question)
    assert meta == {
        "company": "Google",
        "role": "software_engineer",
        "seniority": "mid",
        "category": "technical",
        "difficulty": 3,
        "question_id": "q01",
    }


class _FakeIndex:
    def __init__(self) -> None:
        self.upserts: list[list[dict]] = []

    def upsert(self, vectors: list[dict]) -> None:
        self.upserts.append(vectors)


class _FakePinecone:
    def __init__(self, dimension: int | None = None) -> None:
        self.created: list[tuple] = []
        self._dimension = dimension
        self._has = dimension is not None

    def has_index(self, name: str) -> bool:
        return self._has

    def describe_index(self, name: str) -> object:
        return type("IndexInfo", (), {"dimension": self._dimension})()

    def create_index(self, **kwargs) -> None:
        self.created.append(kwargs)
        self._has = True
        self._dimension = kwargs.get("dimension")

    def Index(self, name: str) -> _FakeIndex:
        return _FakeIndex()


def test_upsert_uses_question_id_so_reruns_do_not_duplicate() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-openai",
        pinecone_api_key="test-pinecone",
        pinecone_index="roleready-questions",
    )
    index = _FakeIndex()
    store = PineconeQuestionStore(settings=settings, client=_FakePinecone(), index=index)
    q1 = make_question("q01")
    q2 = make_question("q02")
    store.upsert_vectors([q1, q2], [[0.1, 0.2], [0.3, 0.4]], batch_size=10)
    store.upsert_vectors([q1], [[0.5, 0.6]], batch_size=10)

    first_ids = [row["id"] for row in index.upserts[0]]
    second_ids = [row["id"] for row in index.upserts[1]]
    assert first_ids == ["q01", "q02"]
    assert second_ids == ["q01"]
    assert index.upserts[1][0]["metadata"]["question_id"] == "q01"


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-openai",
        pinecone_api_key="test-pinecone",
        pinecone_index="roleready-questions",
        openai_embedding_model="text-embedding-3-small",
    )


def test_embedding_client_can_match_1024_index() -> None:
    client = EmbeddingClient(settings=_settings(), embeddings=object(), dimensions=1024)
    assert client.dimension() == 1024


def test_ensure_index_rejects_mismatched_vectors_on_existing_index() -> None:
    settings = _settings()
    store = PineconeQuestionStore(
        settings=settings,
        client=_FakePinecone(dimension=1024),
        index=_FakeIndex(),
    )
    with pytest.raises(ValueError, match="1024"):
        store.ensure_index(1536)
