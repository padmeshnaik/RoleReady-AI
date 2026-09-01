"""Create the SQLite question bank and load data/questions_clean.jsonl.

Does not require OpenAI or Pinecone. Run from the repository root:

    python scripts/init_db.py
    python scripts/init_db.py --db-path data/roleready.db
    python scripts/init_db.py --jsonl-path data/questions_clean.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from roleready.db.sqlite import connect, format_seed_summary, seed_from_jsonl  # noqa: E402

DEFAULT_DB_PATH = os.environ.get("SQLITE_PATH", "data/roleready.db")
DEFAULT_JSONL_PATH = ROOT / "data" / "questions_clean.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the RoleReady AI SQLite question bank.")
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="SQLite file path (default: SQLITE_PATH or data/roleready.db)",
    )
    parser.add_argument(
        "--jsonl-path",
        default=str(DEFAULT_JSONL_PATH),
        help="Cleaned question JSONL path (default: data/questions_clean.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.is_absolute():
        jsonl_path = ROOT / jsonl_path

    conn = connect(db_path)
    try:
        summary = seed_from_jsonl(conn, jsonl_path)
    finally:
        conn.close()

    print(f"Seeded SQLite question bank: {db_path}")
    print(format_seed_summary(summary))


if __name__ == "__main__":
    main()
