"""Run the unchanged application pipeline for evaluation questions."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_utils import (  # noqa: E402
    RESULTS_DIR,
    append_jsonl,
    confirm_api_run,
    load_jsonl,
    selected_questions,
)
from src.rag import answer_question  # noqa: E402


def is_transient(error: str | None) -> bool:
    """Identify common retryable provider/network failures."""
    text = (error or "").lower()
    return any(marker in text for marker in ("429", "rate limit", "timeout", "temporar", "503", "connection"))


def generation_record(item: dict, result) -> dict:
    """Convert the app result to a durable, self-contained generation record."""
    metrics = result.metrics
    chunks = []
    for rank, document in enumerate(result.documents, 1):
        page_index = document.metadata.get("page")
        chunks.append(
            {
                "source_file": document.metadata.get("source", "Unknown source"),
                "page": page_index + 1 if isinstance(page_index, int) else page_index,
                "score": document.metadata.get("score"),
                "rank": rank,
                "text": document.page_content,
            }
        )
    timings = metrics.get("timings_ms", {})
    return {
        "id": item["id"],
        "question": item["question"],
        "type": item["type"],
        "answerable": item["answerable"],
        "answer": result.answer,
        "rewritten_query": metrics.get("rewritten_query"),
        "retrieved_chunks": chunks,
        "rewrite_ms": timings.get("rewrite_ms", 0.0),
        "retrieval_ms": timings.get("retrieval_ms", 0.0),
        "llm_ms": timings.get("llm_ms", 0.0),
        "total_ms": timings.get("total_ms", 0.0),
        "input_tokens": metrics.get("input_tokens", 0),
        "output_tokens": metrics.get("output_tokens", 0),
        "estimated_cost_usd": metrics.get("estimated_cost_usd", 0.0),
        "cache_hit": metrics.get("cache_hit", False),
        "error": metrics.get("error"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-tag", required=True)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, help="Override the app's retrieval result count for this run.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between questions.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the API-cost confirmation prompt.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.sleep < 0:
        parser.error("--sleep must be non-negative")

    try:
        questions = selected_questions(args.split)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.limit:
        questions = questions[:args.limit]
    output_path = RESULTS_DIR / f"{args.config_tag}_generation.jsonl"
    if output_path.exists() and not (args.resume or args.force):
        print(f"Refusing to overwrite {output_path}; use --resume or --force.", file=sys.stderr)
        return 2
    if args.force and output_path.exists():
        output_path.unlink()
    completed_ids = {str(record.get("id")) for record in load_jsonl(output_path)} if args.resume else set()
    pending = [item for item in questions if str(item["id"]) not in completed_ids]
    if not pending:
        print("No pending questions.")
        return 0
    # The normal pipeline makes a rewrite call and, if chunks exist, an answer call.
    if not confirm_api_run(f"generation eval for split '{args.split}'", 2 * len(pending), args.yes):
        print("Cancelled.")
        return 0

    session_id = f"eval-{uuid4()}"
    for position, item in enumerate(pending, 1):
        result = None
        for attempt in range(3):
            result = answer_question(
                item["question"], session_id=session_id, source="eval", top_k=args.top_k
            )
            if not is_transient(result.metrics.get("error")):
                break
            if attempt < 2:
                time.sleep(2**attempt)
        append_jsonl(output_path, generation_record(item, result))
        print(f"Completed {position}/{len(pending)}: {item['id']}")
        if args.sleep and position < len(pending):
            time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
