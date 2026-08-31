"""Load SQLite questions into Pinecone. Re-running upserts the same question IDs.

    python scripts/ingest_pinecone.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from roleready.config.settings import get_settings  # noqa: E402
from roleready.db.sqlite import connect, list_questions  # noqa: E402
from roleready.rag.embeddings import EmbeddingClient, embedding_text  # noqa: E402
from roleready.rag.pinecone_store import PineconeQuestionStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed SQLite questions and upsert into Pinecone.")
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite path (default: SQLITE_PATH from settings)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embed/upsert batch size")
    return parser.parse_args()


def _resolve_db_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def main() -> None:
    args = parse_args()
    settings = get_settings()
    db_path = _resolve_db_path(args.db_path or settings.sqlite_path)

    print("RoleReady AI Pinecone ingest")
    print(f"  SQLite (source of truth): {db_path}")
    print(f"  Embedding model: {settings.openai_embedding_model}")
    print(f"  Pinecone index: {settings.pinecone_index}")
    print(f"  Pinecone cloud/region: {settings.pinecone_cloud}/{settings.pinecone_region}")
    print("  OPENAI_API_KEY: set" if settings.openai_api_key else "  OPENAI_API_KEY: missing")
    print("  PINECONE_API_KEY: set" if settings.pinecone_api_key else "  PINECONE_API_KEY: missing")

    if not db_path.exists():
        raise SystemExit(f"SQLite file not found: {db_path}. Run python scripts/init_db.py first.")

    conn = connect(db_path)
    try:
        questions = list_questions(conn)
    finally:
        conn.close()

    if not questions:
        raise SystemExit("No questions in SQLite. Run python scripts/init_db.py first.")

    print(f"Loaded {len(questions)} questions from SQLite.")

    store = PineconeQuestionStore(settings=settings)
    existing_dim = store.describe_dimension()
    if existing_dim is not None:
        print(f"Existing Pinecone index dimension: {existing_dim}")
        embedder = EmbeddingClient(settings=settings, dimensions=existing_dim)
        print(f"Embedding output dimension: {embedder.dimension()} (matched to index)")
    else:
        embedder = EmbeddingClient(settings=settings)
        print(f"Embedding output dimension: {embedder.dimension()}")
    store.ensure_index(embedder.dimension(), progress=print)

    texts = [embedding_text(question) for question in questions]
    vectors: list[list[float]] = []
    batch_size = max(1, args.batch_size)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        ids = [q.id for q in questions[start : start + batch_size]]
        print(f"Embedding {start + 1}-{start + len(chunk)}/{len(texts)} (ids: {', '.join(ids)})")
        vectors.extend(embedder.embed_texts(chunk))

    count = store.upsert_vectors(
        questions,
        vectors,
        batch_size=batch_size,
        progress=print,
    )
    print(
        f"Done. Upserted {count} vectors. IDs match SQLite question ids, "
        "so re-running this script overwrites vectors instead of duplicating them."
    )


if __name__ == "__main__":
    main()
