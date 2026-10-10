"""Judge generation-evaluation outputs with separate, strict rubric prompts."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from src.eval_utils import (  # noqa: E402
    RESULTS_DIR,
    append_jsonl,
    confirm_api_run,
    load_json,
    load_jsonl,
    selected_question_ids,
)
from src.judge_prompts import (  # noqa: E402
    JUDGE_PROMPT_VERSION,
    abstention_prompt,
    answer_relevance_prompt,
    context_relevance_prompt,
    correctness_prompt,
    faithfulness_prompt,
)
from src.rag import message_content_to_text  # noqa: E402


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a strict object response while tolerating Markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("judge output is not a JSON object")
    return value


def judge_call(
    llm: ChatGoogleGenerativeAI,
    prompt: str,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
    attempts: int = 3,
) -> tuple[dict[str, Any], str]:
    """Call and validate a judge response, retrying transient/model-format errors."""
    last_error: Exception | None = None
    raw = ""
    for attempt in range(attempts):
        try:
            response = llm.invoke(prompt)
            raw = message_content_to_text(response.content)
            return validator(parse_json_object(raw)), raw
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"judge failed after {attempts} attempts: {last_error}")


def require_verdict(values: set[str]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def validate(value: dict[str, Any]) -> dict[str, Any]:
        if value.get("verdict") not in values or not isinstance(value.get("reason"), str):
            raise ValueError("invalid verdict JSON")
        return value
    return validate


def validate_faithfulness(value: dict[str, Any]) -> dict[str, Any]:
    claims = value.get("claims")
    if not isinstance(claims, list) or not isinstance(value.get("faithful"), bool):
        raise ValueError("invalid faithfulness JSON")
    normalized = []
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("claim"), str) or not isinstance(claim.get("supported"), bool):
            raise ValueError("invalid claim")
        normalized.append({
            "claim": claim["claim"], "supported": claim["supported"], "evidence": str(claim.get("evidence", "")),
        })
    supported = sum(claim["supported"] for claim in normalized)
    return {"claims": normalized, "number_supported": supported, "number_total": len(normalized), "faithful": not any(not claim["supported"] for claim in normalized)}


def validate_relevance(value: dict[str, Any]) -> dict[str, Any]:
    score = value.get("score")
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError("relevance score must be an integer 1-5")
    return {"score": score, "reason": str(value.get("reason", ""))}


def validate_context_relevance(chunk_count: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def validate(value: dict[str, Any]) -> dict[str, Any]:
        indices = value.get("relevant_indices")
        if not isinstance(indices, list) or any(not isinstance(index, int) or not 0 <= index < chunk_count for index in indices):
            raise ValueError("invalid relevant chunk indices")
        unique_indices = sorted(set(indices))
        return {"relevant_indices": unique_indices, "relevant_count": len(unique_indices), "total_count": chunk_count, "score": len(unique_indices) / chunk_count if chunk_count else 0.0, "evidence": value.get("evidence", [])}
    return validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-tag", required=True)
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")))
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the API-cost confirmation prompt.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    generation_path = RESULTS_DIR / f"{args.config_tag}_generation.jsonl"
    output_path = RESULTS_DIR / f"{args.config_tag}_judgments.jsonl"
    if not generation_path.exists():
        print(f"Generation results not found: {generation_path}", file=sys.stderr)
        return 2
    try:
        allowed_ids = selected_question_ids(args.split)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    records_by_id = {str(record.get("id")): record for record in load_jsonl(generation_path)}
    records = [record for question_id, record in records_by_id.items() if question_id in allowed_ids]
    random.Random(args.seed).shuffle(records)
    if args.limit:
        records = records[:args.limit]
    done_ids = {str(record.get("id")) for record in load_jsonl(output_path)} if args.resume else set()
    pending = [record for record in records if str(record.get("id")) not in done_ids]
    if not pending:
        print("No pending generations to judge.")
        return 0
    calls = sum(4 if record.get("answerable") else 2 for record in pending)
    if not confirm_api_run(f"LLM judging with model {args.judge_model}", calls, args.yes):
        print("Cancelled.")
        return 0
    llm = ChatGoogleGenerativeAI(model=args.judge_model, temperature=0)
    questions = {str(item["id"]): item for item in load_json(ROOT / "eval" / "questions.json", [])}
    for position, generated in enumerate(pending, 1):
        question = questions.get(str(generated["id"]), generated)
        chunks = generated.get("retrieved_chunks", [])
        context = "\n\n".join(chunk.get("text", "") for chunk in chunks)
        raw_outputs: dict[str, str] = {}
        judgment: dict[str, Any] = {}
        error = None
        try:
            faith, raw_outputs["faithfulness"] = judge_call(
                llm, faithfulness_prompt(generated.get("answer", ""), context), validate_faithfulness
            )
            judgment["faithfulness"] = faith
            if generated.get("answerable"):
                correctness, raw_outputs["correctness"] = judge_call(
                    llm,
                    correctness_prompt(question.get("question", ""), question.get("reference_answer", ""), generated.get("answer", "")),
                    require_verdict({"correct", "partially_correct", "incorrect", "abstained"}),
                )
                answer_relevance, raw_outputs["answer_relevance"] = judge_call(
                    llm, answer_relevance_prompt(question.get("question", ""), generated.get("answer", "")), validate_relevance
                )
                context_relevance, raw_outputs["context_relevance"] = judge_call(
                    llm, context_relevance_prompt(question.get("question", ""), chunks), validate_context_relevance(len(chunks))
                )
                judgment.update({"correctness": correctness, "answer_relevance": answer_relevance, "context_relevance": context_relevance})
            else:
                abstention, raw_outputs["abstention"] = judge_call(
                    llm, abstention_prompt(question.get("question", ""), generated.get("answer", "")),
                    require_verdict({"properly_abstained", "hallucinated_answer", "partial_hallucination"}),
                )
                judgment["abstention"] = abstention
        except Exception as exc:
            error = str(exc)
        append_jsonl(output_path, {
            "id": generated["id"], "type": generated.get("type"), "answerable": generated.get("answerable"),
            "judge_model": args.judge_model, "judge_prompt_version": os.getenv("JUDGE_PROMPT_VERSION", JUDGE_PROMPT_VERSION),
            "raw_judge_output": raw_outputs, "judgment": judgment, "error": error,
        })
        print(f"Judged {position}/{len(pending)}: {generated['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
