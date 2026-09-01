"""Flag semantically similar questions using OpenAI embeddings. No SQLite or Pinecone.

Run from the repository root:

    python scripts/deduplicate_questions.py
    python scripts/deduplicate_questions.py --threshold 0.92
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from roleready.config.settings import get_settings  # noqa: E402
from roleready.generation.dedupe import (  # noqa: E402
    DEFAULT_SIMILARITY_THRESHOLD,
    deduplicate_questions,
    format_summary,
)

DEFAULT_INPUT = ROOT / "data" / "questions_valid.jsonl"
DEFAULT_CLEAN = ROOT / "data" / "questions_clean.jsonl"
DEFAULT_DUPLICATES = ROOT / "data" / "questions_duplicates.jsonl"
DEFAULT_CACHE = ROOT / "data" / "question_text_embeddings.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect semantic duplicate questions via cosine similarity of embeddings."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Validated JSONL input")
    parser.add_argument("--clean-output", default=str(DEFAULT_CLEAN), help="Retained questions JSONL")
    parser.add_argument(
        "--duplicates-output",
        default=str(DEFAULT_DUPLICATES),
        help="Semantic duplicate report JSONL",
    )
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_CACHE),
        help="Local embedding cache (not Pinecone)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Cosine similarity at or above this value flags a later question as a duplicate (default: 0.92).",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    configure_logging()
    args = parse_args()
    input_path = _resolve(Path(args.input))
    clean_path = _resolve(Path(args.clean_output))
    duplicates_path = _resolve(Path(args.duplicates_output))
    cache_path = _resolve(Path(args.cache))

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        raise SystemExit(1)
    if input_path.resolve() in {clean_path.resolve(), duplicates_path.resolve(), cache_path.resolve()}:
        print("Refusing to overwrite the input file.", file=sys.stderr)
        raise SystemExit(1)

    settings = get_settings()
    summary = deduplicate_questions(
        input_path=input_path,
        clean_path=clean_path,
        duplicates_path=duplicates_path,
        cache_path=cache_path,
        threshold=args.threshold,
        settings=settings,
    )
    print(format_summary(summary))


if __name__ == "__main__":
    main()
