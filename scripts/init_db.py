"""Create the SQLite question bank and load data/questions_seed.csv.

Does not require OpenAI or Pinecone. Run from the repository root:

    python scripts/init_db.py
    python scripts/init_db.py --db-path data/roleready.db
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

from roleready.db.sqlite import connect, seed_from_csv  # noqa: E402

DEFAULT_DB_PATH = os.environ.get("SQLITE_PATH", "data/roleready.db")
DEFAULT_CSV_PATH = ROOT / "data" / "questions_seed.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the RoleReady AI SQLite question bank.")
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="SQLite file path (default: SQLITE_PATH or data/roleready.db)",
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_CSV_PATH),
        help="Seed CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path

    conn = connect(db_path)
    try:
        count = seed_from_csv(conn, csv_path)
    finally:
        conn.close()

    print(f"Seeded {count} questions into {db_path}")


if __name__ == "__main__":
    main()
