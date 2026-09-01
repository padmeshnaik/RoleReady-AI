"""Load every SQLite question into Pinecone. Re-running upserts the same IDs.

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
from roleready.rag.embeddings import EmbeddingClient  # noqa: E402
from roleready.rag.ingest import DEFAULT_INGEST_BATCH_SIZE, ingest_questions  # noqa: E402
from roleready.rag.pinecone_store import PineconeQuestionStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed SQLite questions and upsert into Pinecone.")
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite path (default: SQLITE_PATH from settings)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_INGEST_BATCH_SIZE,
        help=f"Embed/upsert batch size (default: {DEFAULT_INGEST_BATCH_SIZE})",
    )
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

    count = ingest_questions(
        questions,
        embedder,
        store,
        batch_size=max(1, args.batch_size),
        progress=print,
    )
    print(f"Final count: {count}")


if __name__ == "__main__":
    main()
