"""Semantic dedupe tests. No live OpenAI, Pinecone, or SQLite."""

import json
from pathlib import Path

from roleready.config.settings import Settings
from roleready.generation.dedupe import (
    cosine_similarity,
    deduplicate_questions,
    embeddings_for_questions,
    find_semantic_duplicates,
    format_summary,
)
from roleready.rag.embeddings import EmbeddingClient


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-openai",
        pinecone_api_key="test-pinecone",
        openai_embedding_model="text-embedding-3-small",
    )


def _record(qid: str, text: str) -> dict:
    return {
        "id": qid,
        "company": "Generic",
        "role": "software_engineer",
        "seniority": "mid",
        "category": "technical",
        "difficulty": 3,
        "question_text": text,
        "rubric": "A strong answer covers trade-offs, failure modes, and production operations in enough detail.",
        "follow_up_hints": "Ask about a concrete incident.",
    }


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.embed_documents_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls += 1
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "rest" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "cache" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def test_cosine_similarity_identical_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_find_semantic_duplicates_keeps_first_above_threshold() -> None:
    records = [
        _record("a", "Explain REST APIs in production."),
        _record("b", "Please explain REST APIs used in production."),
        _record("c", "How do you shard a key-value store?"),
    ]
    vectors = {
        "a": [1.0, 0.0],
        "b": [0.99, 0.01],
        "c": [0.0, 1.0],
    }
    retained, duplicates = find_semantic_duplicates(records, vectors, threshold=0.92)
    assert [item["id"] for item in retained] == ["a", "c"]
    assert len(duplicates) == 1
    assert duplicates[0]["rejected_question_id"] == "b"
    assert duplicates[0]["retained_question_id"] == "a"
    assert duplicates[0]["cosine_similarity"] >= 0.92
    assert "REST" in duplicates[0]["rejected_question_text"]
    assert "REST" in duplicates[0]["retained_question_text"]


def test_embeddings_are_reused_from_cache(tmp_path: Path) -> None:
    fake = _FakeEmbeddings()
    client = EmbeddingClient(settings=_settings(), embeddings=fake, dimensions=1536)
    records = [_record("gq-1", "Explain REST APIs.")]
    cache: dict = {}
    first = embeddings_for_questions(records, client, cache)
    assert fake.embed_documents_calls == 1
    second = embeddings_for_questions(records, client, cache)
    assert fake.embed_documents_calls == 1
    assert first["gq-1"] == second["gq-1"]


def test_deduplicate_questions_writes_outputs_without_touching_input(tmp_path: Path) -> None:
    source = tmp_path / "questions_valid.jsonl"
    clean = tmp_path / "questions_clean.jsonl"
    dups = tmp_path / "questions_duplicates.jsonl"
    cache = tmp_path / "cache.json"
    rows = [
        _record("gq-1", "Explain REST APIs in production systems."),
        _record("gq-2", "Explain REST APIs in production systems please."),
        _record("gq-3", "How would you design a distributed cache?"),
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    original = source.read_text(encoding="utf-8")
    fake = _FakeEmbeddings()
    client = EmbeddingClient(settings=_settings(), embeddings=fake, dimensions=1536)

    summary = deduplicate_questions(
        input_path=source,
        clean_path=clean,
        duplicates_path=dups,
        cache_path=cache,
        threshold=0.92,
        client=client,
        settings=_settings(),
    )

    assert source.read_text(encoding="utf-8") == original
    assert summary.input_questions == 3
    assert summary.retained_questions == 2
    assert summary.semantic_duplicates == 1
    assert summary.threshold == 0.92
    assert "threshold used: 0.92" in format_summary(summary)
    kept_ids = [json.loads(line)["id"] for line in clean.read_text(encoding="utf-8").splitlines()]
    assert kept_ids == ["gq-1", "gq-3"]
    dup = json.loads(dups.read_text(encoding="utf-8").splitlines()[0])
    assert dup["rejected_question_id"] == "gq-2"
    assert dup["retained_question_id"] == "gq-1"


def test_dedupe_modules_do_not_write_sqlite_or_pinecone() -> None:
    for path in (
        Path("src/roleready/generation/dedupe.py"),
        Path("scripts/deduplicate_questions.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "roleready.db" not in source
        assert "from pinecone" not in source
        assert "import pinecone" not in source
        assert "upsert" not in source
