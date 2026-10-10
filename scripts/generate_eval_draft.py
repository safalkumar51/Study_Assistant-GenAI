"""Generate a review-only evaluation-set draft from the local PDFs."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag import (  # noqa: E402
    DATA_DIR,
    get_chunk_overlap,
    get_chunk_size,
    get_llm,
    list_pdf_files,
    load_pdf_documents,
    message_content_to_text,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

OUTPUT_PATH = ROOT / "eval" / "questions_draft.json"
TYPES = ("factual", "conceptual", "multi_chunk", "out_of_scope")


def chunk_documents() -> list:
    """Use the same PDF loader and splitting configuration as ingestion."""
    documents = load_pdf_documents(list_pdf_files())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=get_chunk_size(), chunk_overlap=get_chunk_overlap()
    )
    return splitter.split_documents(documents)


def evenly_sample(chunks: list, count: int, rng: random.Random) -> list:
    """Round-robin chunks by source/page so one document cannot dominate."""
    buckets: dict[tuple[str, int], list] = defaultdict(list)
    for chunk in chunks:
        buckets[(chunk.metadata.get("source", ""), chunk.metadata.get("page", -1))].append(chunk)
    groups = list(buckets.values())
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)
    sampled = []
    while groups and len(sampled) < count:
        next_groups = []
        for group in groups:
            if len(sampled) >= count:
                break
            if group:
                sampled.append(group.pop())
            if group:
                next_groups.append(group)
        groups = next_groups
    return sampled


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("model response was not a JSON object")
    return data


def ask_gemini(prompt: str, attempts: int = 3) -> dict[str, Any]:
    """Call Gemini with bounded retries for transient parsing/rate-limit failures."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = get_llm().invoke(prompt)
            return parse_json(message_content_to_text(response.content))
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Gemini draft generation failed after {attempts} attempts: {last_error}")


def build_prompt(question_type: str, chunk, second_chunk=None) -> str:
    source = chunk.metadata.get("source", "")
    page = int(chunk.metadata.get("page", 0)) + 1
    second = ""
    if second_chunk is not None:
        second = f"\nSECOND EXCERPT (page {int(second_chunk.metadata.get('page', 0)) + 1}):\n{second_chunk.page_content}"
    if question_type == "out_of_scope":
        task = """Write one plausible student study question related to the excerpt's topic but NOT answerable from the excerpt. Do not invent a reference answer. This will be hand-checked against every PDF before use."""
    else:
        task = f"""Write one concise, student-style {question_type} question and concise reference answer using ONLY the supplied excerpt(s). Do not add facts not present in them."""
    return f"""{task}
Return ONLY a JSON object with keys question, reference_answer, supporting_text.
source file: {source}; primary page: {page}
EXCERPT:\n{chunk.page_content}{second}"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-questions", type=int, default=200)
    args = parser.parse_args()
    if args.num_questions < 1:
        parser.error("--num-questions must be positive")
    if OUTPUT_PATH.exists():
        print(f"Refusing to overwrite existing {OUTPUT_PATH}. Rename or remove it first.", file=sys.stderr)
        return 2
    chunks = chunk_documents()
    if not chunks:
        print(f"No readable PDFs found in {DATA_DIR}", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    samples = evenly_sample(chunks, args.num_questions, rng)
    if len(samples) < args.num_questions:
        print(f"Only {len(samples)} chunks are available; generating that many drafts.")
    counts = {
        "factual": round(len(samples) * 0.45),
        "conceptual": round(len(samples) * 0.30),
        "multi_chunk": round(len(samples) * 0.15),
    }
    counts["out_of_scope"] = len(samples) - sum(counts.values())
    planned_types = [kind for kind, number in counts.items() for _ in range(number)]
    rng.shuffle(planned_types)
    draft = []
    for number, (chunk, question_type) in enumerate(zip(samples, planned_types), 1):
        second_chunk = None
        if question_type == "multi_chunk":
            same_source = [item for item in chunks if item.metadata.get("source") == chunk.metadata.get("source") and item.metadata.get("page") != chunk.metadata.get("page")]
            if same_source:
                second_chunk = min(same_source, key=lambda item: abs(item.metadata.get("page", 0) - chunk.metadata.get("page", 0)))
        try:
            generated = ask_gemini(build_prompt(question_type, chunk, second_chunk))
        except RuntimeError as exc:
            print(f"Skipping item {number}: {exc}", file=sys.stderr)
            continue
        pages = [int(chunk.metadata.get("page", 0)) + 1]
        if second_chunk is not None:
            pages.append(int(second_chunk.metadata.get("page", 0)) + 1)
        answerable = question_type != "out_of_scope"
        draft.append({
            "id": f"draft-{number:03d}",
            "question": str(generated.get("question", "")).strip(),
            "type": question_type,
            "answerable": answerable,
            "reference_answer": str(generated.get("reference_answer", "")).strip() if answerable else None,
            "source_file": chunk.metadata.get("source", ""),
            "source_pages": sorted(set(pages)),
            "supporting_text": str(generated.get("supporting_text", "")).strip() if answerable else "",
            "verified": False,
            "notes": "Must be hand-checked by searching all PDFs." if not answerable else "",
        })
        print(f"Generated {len(draft)}/{len(samples)}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(draft)} unverified draft items to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
