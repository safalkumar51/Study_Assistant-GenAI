"""Run a retrieval-only evaluation against verified evaluation questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag import retrieve_documents, retrieve_with_rewrite, rewrite_question  # noqa: E402

EVAL_PATH = ROOT / "eval" / "questions.json"


def page_matches(metadata: dict, item: dict) -> bool:
    """Compare one zero-based retriever page with the 1-based eval schema."""
    return (
        metadata.get("source") == item.get("source_file")
        and metadata.get("page") is not None
        and int(metadata["page"]) + 1 in item.get("source_pages", [])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--config-tag", default="baseline")
    parser.add_argument(
        "--use-rewrite",
        action="store_true",
        help="Use the app's Gemini query rewrite before retrieval (this is no longer LLM-free).",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not EVAL_PATH.exists():
        print(f"Evaluation set not found: {EVAL_PATH}", file=sys.stderr)
        return 2
    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    questions = [item for item in items if item.get("answerable") is True]
    if not questions:
        print("No answerable questions found.")
        return 2

    results = []
    reciprocal_ranks = []
    hits = 0
    for item in questions:
        query = item["question"]
        if args.use_rewrite:
            rewritten = rewrite_question(query)
            docs = retrieve_with_rewrite(query, rewritten, args.top_k)
        else:
            rewritten = None
            docs = retrieve_documents(query, args.top_k)
        rank = next((position for position, doc in enumerate(docs, 1) if page_matches(doc.metadata, item)), None)
        hit = rank is not None
        hits += int(hit)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        results.append({
            "id": item.get("id"),
            "question": query,
            "source_file": item.get("source_file"),
            "source_pages": item.get("source_pages"),
            "rewritten_query": rewritten,
            "hit": hit,
            "rank": rank,
            "retrieved": [
                {
                    "source_file": doc.metadata.get("source"),
                    "page": (int(doc.metadata["page"]) + 1 if doc.metadata.get("page") is not None else None),
                    "score": doc.metadata.get("score"),
                }
                for doc in docs
            ],
        })

    hit_rate = hits / len(questions)
    mrr = sum(reciprocal_ranks) / len(questions)
    payload = {
        "config_tag": args.config_tag,
        "top_k": args.top_k,
        "use_rewrite": args.use_rewrite,
        "question_count": len(questions),
        "hit_rate_at_k": hit_rate,
        "mrr": mrr,
        "results": results,
    }
    output_dir = ROOT / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.config_tag}_retrieval.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Questions: {len(questions)}")
    print(f"hit_rate@{args.top_k}: {hit_rate:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"Saved per-question results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
