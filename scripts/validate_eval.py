"""Validate the hand-verified evaluation set and its PDF citations."""

from __future__ import annotations

import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag import DATA_DIR  # noqa: E402

EVAL_PATH = ROOT / "eval" / "questions.json"
REQUIRED = {"id", "question", "type", "answerable", "reference_answer", "source_file", "source_pages", "supporting_text", "verified", "notes"}
VALID_TYPES = {"factual", "conceptual", "multi_chunk", "out_of_scope"}


def pdf_page_count(path: Path) -> int:
    """Count physical PDF pages, including pages that contain no extractable text."""
    try:
        import fitz

        with fitz.open(str(path)) as pdf:
            return len(pdf)
    except ImportError:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)


def main() -> int:
    if not EVAL_PATH.exists():
        print(f"Evaluation set not found: {EVAL_PATH}")
        return 1
    try:
        items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON: {exc}")
        return 1
    if not isinstance(items, list):
        print("questions.json must contain a JSON list.")
        return 1

    failures: list[str] = []
    page_counts: dict[str, int] = {}
    for path in DATA_DIR.glob("*.pdf"):
        try:
            page_counts[path.name] = pdf_page_count(path)
        except Exception as exc:
            failures.append(f"Could not read {path.name}: {exc}")

    normalized_questions: dict[str, int] = {}
    seen_ids: set[str] = set()
    for index, item in enumerate(items, 1):
        label = item.get("id", f"item #{index}")
        missing = REQUIRED - set(item)
        if missing:
            failures.append(f"{label}: missing required fields {sorted(missing)}")
            continue
        if not str(item["question"]).strip():
            failures.append(f"{label}: question is empty")
        if not str(item["id"]).strip() or item["id"] in seen_ids:
            failures.append(f"{label}: ID is empty or duplicated")
        seen_ids.add(item["id"])
        if item["type"] not in VALID_TYPES:
            failures.append(f"{label}: invalid type {item['type']!r}")
        if not isinstance(item["answerable"], bool) or not isinstance(item["verified"], bool):
            failures.append(f"{label}: answerable and verified must be booleans")
        if item["type"] == "out_of_scope" and item["answerable"]:
            failures.append(f"{label}: out_of_scope must have answerable=false")
        if not item["answerable"] and item.get("reference_answer") not in (None, ""):
            failures.append(f"{label}: unanswerable item has a reference answer")
        if item["answerable"] and not str(item.get("reference_answer") or "").strip():
            failures.append(f"{label}: answerable item has no reference answer")
        source_file = item["source_file"]
        if source_file not in page_counts:
            failures.append(f"{label}: source file does not exist: {source_file}")
        elif not isinstance(item["source_pages"], list) or not item["source_pages"] or any(not isinstance(page, int) or page < 1 or page > page_counts[source_file] for page in item["source_pages"]):
            failures.append(f"{label}: source pages are outside 1..{page_counts[source_file]}")
        question_key = " ".join(str(item["question"]).lower().split())
        if question_key in normalized_questions:
            failures.append(f"{label}: duplicate question of item #{normalized_questions[question_key]}")
        normalized_questions[question_key] = index

    question_pairs = list(normalized_questions.items())
    for left_index, (left, left_number) in enumerate(question_pairs):
        for right, right_number in question_pairs[left_index + 1:]:
            if SequenceMatcher(None, left, right).ratio() >= 0.90:
                failures.append(f"Likely duplicate questions: item #{left_number} and item #{right_number}")

    type_counts = Counter(item.get("type", "invalid") for item in items)
    verified_count = sum(item.get("verified") is True for item in items)
    answerable_count = sum(item.get("answerable") is True for item in items)
    out_of_scope_count = type_counts["out_of_scope"]
    total = len(items)
    print(f"Total count: {total}")
    print("By type: " + ", ".join(f"{kind}={type_counts[kind]}" for kind in sorted(VALID_TYPES)))
    print(f"Answerable/unanswerable: {answerable_count}/{total - answerable_count}")
    print(f"Verified: {verified_count}")
    if verified_count < 150:
        failures.append(f"Target not met: need >=150 verified questions (have {verified_count})")
    if total == 0 or out_of_scope_count / total < 0.10:
        failures.append("Target not met: need >=10% out_of_scope questions")

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
