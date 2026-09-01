"""Validate generated RoleReady AI questions. Does not call OpenAI, Pinecone, or SQLite.

Run from the repository root:

    python scripts/validate_questions.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from roleready.generation.validate import format_summary, validate_corpus  # noqa: E402

DEFAULT_INPUT = ROOT / "data" / "generated_questions.jsonl"
DEFAULT_VALID = ROOT / "data" / "questions_valid.jsonl"
DEFAULT_REJECTED = ROOT / "data" / "questions_rejected.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate generated interview questions. Does not modify the input file."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input JSONL path")
    parser.add_argument("--valid-output", default=str(DEFAULT_VALID), help="Valid JSONL path")
    parser.add_argument(
        "--rejected-output",
        default=str(DEFAULT_REJECTED),
        help="Rejected JSONL path",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> None:
    args = parse_args()
    input_path = _resolve(Path(args.input))
    valid_path = _resolve(Path(args.valid_output))
    rejected_path = _resolve(Path(args.rejected_output))

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        raise SystemExit(1)

    if valid_path.resolve() == input_path.resolve() or rejected_path.resolve() == input_path.resolve():
        print("Refusing to overwrite the input generated questions file.", file=sys.stderr)
        raise SystemExit(1)

    summary = validate_corpus(input_path, valid_path, rejected_path)
    print(format_summary(summary))


if __name__ == "__main__":
    main()
