"""Create a blind, stratified human-audit sample from judge results."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval_utils import RESULTS_DIR, load_jsonl, selected_question_ids  # noqa: E402


def verdict_bucket(row: dict) -> str:
    judgment = row.get("judgment", {})
    if row.get("answerable"):
        return str(judgment.get("correctness", {}).get("verdict", "unknown"))
    return str(judgment.get("abstention", {}).get("verdict", "unknown"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-tag", required=True)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.size < 1:
        parser.error("--size must be positive")
    try:
        allowed_ids = selected_question_ids(args.split)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    generation = {str(row.get("id")): row for row in load_jsonl(RESULTS_DIR / f"{args.config_tag}_generation.jsonl")}
    judgments = {str(row.get("id")): row for row in load_jsonl(RESULTS_DIR / f"{args.config_tag}_judgments.jsonl")}
    rows = []
    for question_id in allowed_ids:
        if question_id in generation and question_id in judgments and not judgments[question_id].get("error"):
            row = dict(generation[question_id])
            row["judgment"] = judgments[question_id].get("judgment", {})
            rows.append(row)
    if not rows:
        print("No joined generation/judgment results available.", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("type")), verdict_bucket(row))].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict] = []
    # Guarantee a meaningful out-of-scope audit when that data exists.
    out_of_scope = [row for row in rows if not row.get("answerable")]
    rng.shuffle(out_of_scope)
    selected.extend(out_of_scope[:min(8, len(out_of_scope), args.size)])
    # Round-robin all type/verdict groups; verdict grouping naturally oversamples borderline cases.
    while len(selected) < min(args.size, len(rows)):
        progressed = False
        for bucket in buckets.values():
            while bucket and bucket[0] in selected:
                bucket.pop(0)
            if bucket and len(selected) < args.size:
                selected.append(bucket.pop(0))
                progressed = True
        if not progressed:
            break
    audit_dir = ROOT / "eval" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"{args.config_tag}_audit.csv"
    labels_path = audit_dir / f"{args.config_tag}_judge_labels.csv"
    audit_fields = ["id", "type", "answerable", "question", "answer", "retrieved_sources", "human_correctness", "human_faithful", "human_abstention", "human_notes"]
    with audit_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=audit_fields)
        writer.writeheader()
        for row in selected:
            sources = "; ".join(f"{chunk.get('source_file')} p.{chunk.get('page')}" for chunk in row.get("retrieved_chunks", []))
            writer.writerow({key: row.get(key, "") for key in audit_fields} | {"retrieved_sources": sources})
    with labels_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["id", "judge_correctness", "judge_faithful", "judge_abstention"])
        writer.writeheader()
        for row in selected:
            judgment = row["judgment"]
            writer.writerow({
                "id": row["id"],
                "judge_correctness": judgment.get("correctness", {}).get("verdict", ""),
                "judge_faithful": judgment.get("faithfulness", {}).get("faithful", ""),
                "judge_abstention": judgment.get("abstention", {}).get("verdict", ""),
            })
    print(f"Wrote {len(selected)} blind audit rows to {audit_path}")
    print(f"Judge labels were written separately to {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
