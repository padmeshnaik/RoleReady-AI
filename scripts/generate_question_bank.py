"""Generate interview questions to JSONL. Does not write SQLite or call Pinecone.

Run from the repository root:

    python scripts/generate_question_bank.py --plan-only
    python scripts/generate_question_bank.py --target-count 1000 --batch-size 10
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
from roleready.generation.generator import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_TARGET_COUNT,
    GenerationError,
    QuestionBankGenerator,
    format_summary,
    run_generation,
)
from roleready.generation.jsonl import load_valid_questions  # noqa: E402
from roleready.generation.plan import (  # noqa: E402
    DEFAULT_CORPUS_DISTRIBUTION,
    build_generation_plan,
    format_plan_report,
)

DEFAULT_OUTPUT = ROOT / "data" / "generated_questions.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RoleReady AI interview questions to JSONL (OpenAI only)."
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help="Total valid questions to have on disk (default: 1000). Resumes if the file already has some.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Questions per LLM call (default: 10).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSONL output path (default: data/generated_questions.jsonl)",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=3,
        help="Retries for transient API errors (default: 3).",
    )
    parser.add_argument(
        "--avoid-corpus",
        default="",
        help="Optional JSONL of existing questions whose topics/ids must not be repeated.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the generation plan and resume offset; do not call OpenAI.",
    )
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    configure_logging()
    args = parse_args()
    output_path = _resolve(ROOT, Path(args.output))

    if args.plan_only:
        existing = load_valid_questions(output_path) if output_path.exists() else []
        plan = build_generation_plan(args.target_count, DEFAULT_CORPUS_DISTRIBUTION)
        print(
            format_plan_report(
                plan,
                existing_count=len(existing),
                batch_size=args.batch_size,
            )
        )
        return

    try:
        settings = get_settings()
        avoid = Path(args.avoid_corpus) if args.avoid_corpus else None
        if avoid is not None:
            avoid = _resolve(ROOT, avoid)
        generator = QuestionBankGenerator(settings=settings, retry_count=args.retry_count)
        summary = run_generation(
            output_path=output_path,
            target_count=args.target_count,
            batch_size=args.batch_size,
            generator=generator,
            avoid_corpus_path=avoid,
        )
    except GenerationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    print(format_summary(summary))


if __name__ == "__main__":
    main()
