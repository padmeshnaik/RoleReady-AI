"""Print corpus stats and a fixed-seed sample for manual quality review. No LLM.

Run from the repository root:

    python scripts/review_question_bank.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from roleready.generation.review import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_SIZE,
    format_review_stats,
    load_questions,
    sample_for_review,
    write_review_sample,
)

DEFAULT_INPUT = ROOT / "data" / "questions_clean.jsonl"
DEFAULT_SAMPLE = ROOT / "data" / "question_review_sample.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the clean question corpus and write a random review sample."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Clean JSONL input")
    parser.add_argument(
        "--sample-output",
        default=str(DEFAULT_SAMPLE),
        help="JSON sample path for manual review",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    input_path = _resolve(Path(args.input))
    sample_path = _resolve(Path(args.sample_output))

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        raise SystemExit(1)
    if sample_path.resolve() == input_path.resolve():
        print("Refusing to overwrite the input corpus file.", file=sys.stderr)
        raise SystemExit(1)

    records = load_questions(input_path)
    print(format_review_stats(records))
    sample = sample_for_review(records, size=args.sample_size, seed=args.seed)
    write_review_sample(sample_path, sample)
    print(f"wrote review sample: {sample_path} ({len(sample)} questions, seed={args.seed})")


if __name__ == "__main__":
    main()
