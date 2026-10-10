"""Summarize generation and LLM-judge evaluation results."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval_utils import RESULTS_DIR, load_json, load_jsonl, selected_question_ids  # noqa: E402


def percentile(values: list[float], point: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * point / 100
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_rate(values: list[float], seed: int = 42, resamples: int = 1000) -> dict[str, float]:
    """Return a seeded percentile bootstrap CI for a mean/rate."""
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    means = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(resamples)]
    return {"mean": sum(values) / len(values), "low": percentile(means, 2.5), "high": percentile(means, 97.5)}


def paired_bootstrap_difference(left: list[float], right: list[float], seed: int = 42, resamples: int = 1000) -> dict[str, float | bool]:
    """Return a paired bootstrap CI for left minus right."""
    if not left or len(left) != len(right):
        return {"difference": 0.0, "low": 0.0, "high": 0.0, "significant": False}
    rng = random.Random(seed)
    differences = [left[index] - right[index] for index in range(len(left))]
    samples = [sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(resamples)]
    low, high = percentile(samples, 2.5), percentile(samples, 97.5)
    return {"difference": sum(differences) / len(differences), "low": low, "high": high, "significant": not (low <= 0 <= high)}


def is_abstention(answer: str) -> bool:
    text = answer.lower()
    return "could not find" in text or "do not contain the answer" in text or "couldn't find" in text


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all requested aggregate measures for joined result records."""
    answerable = [row for row in records if row.get("answerable")]
    out_of_scope = [row for row in records if not row.get("answerable")]
    correctness = Counter(row.get("judgment", {}).get("correctness", {}).get("verdict") for row in answerable)
    faith_rows = [row for row in records if "faithfulness" in row.get("judgment", {})]
    support_numerator = sum(row["judgment"]["faithfulness"].get("number_supported", 0) for row in faith_rows)
    support_denominator = sum(row["judgment"]["faithfulness"].get("number_total", 0) for row in faith_rows)
    unsupported = [float(not row["judgment"]["faithfulness"].get("faithful", False)) for row in faith_rows]
    strict = [float(row.get("judgment", {}).get("correctness", {}).get("verdict") == "correct") for row in answerable]
    lenient = [float(row.get("judgment", {}).get("correctness", {}).get("verdict") in {"correct", "partially_correct"}) for row in answerable]
    fully_faithful = [float(row["judgment"]["faithfulness"].get("faithful", False)) for row in faith_rows]
    abstention = [float(row.get("judgment", {}).get("abstention", {}).get("verdict") == "properly_abstained") for row in out_of_scope if "abstention" in row.get("judgment", {})]
    false_abstention = [float(is_abstention(str(row.get("answer", "")))) for row in answerable]
    answer_relevance = [row["judgment"]["answer_relevance"]["score"] for row in answerable if "answer_relevance" in row.get("judgment", {})]
    context_relevance = [row["judgment"]["context_relevance"]["score"] for row in answerable if "context_relevance" in row.get("judgment", {})]

    result: dict[str, Any] = {
        "n": len(records), "n_answerable": len(answerable), "n_out_of_scope": len(out_of_scope),
        "correctness_distribution": dict(correctness),
        "strict_accuracy": bootstrap_rate(strict), "lenient_accuracy": bootstrap_rate(lenient),
        "fully_faithful": bootstrap_rate(fully_faithful),
        "claim_support_rate": support_numerator / support_denominator if support_denominator else 0.0,
        "hallucination_rate": bootstrap_rate(unsupported),
        "hallucination_rate_answerable": bootstrap_rate([float(not row["judgment"]["faithfulness"].get("faithful", False)) for row in answerable if "faithfulness" in row.get("judgment", {})]),
        "hallucination_rate_out_of_scope": bootstrap_rate([float(not row["judgment"]["faithfulness"].get("faithful", False)) for row in out_of_scope if "faithfulness" in row.get("judgment", {})]),
        "proper_abstention": bootstrap_rate(abstention), "false_abstention": bootstrap_rate(false_abstention),
        "mean_answer_relevance": statistics.mean(answer_relevance) if answer_relevance else 0.0,
        "mean_context_relevance": statistics.mean(context_relevance) if context_relevance else 0.0,
    }
    for timing in ("total_ms", "rewrite_ms", "retrieval_ms", "llm_ms"):
        values = [float(row.get(timing, 0)) for row in records]
        result[timing] = {"mean": statistics.mean(values) if values else 0.0, "p50": percentile(values, 50), "p95": percentile(values, 95)}
    for token in ("input_tokens", "output_tokens"):
        values = [float(row.get(token, 0)) for row in records]
        result[f"mean_{token}"] = statistics.mean(values) if values else 0.0
    costs = [float(row.get("estimated_cost_usd", 0)) for row in records]
    result["cost_usd"] = {"mean": statistics.mean(costs) if costs else 0.0, "total": sum(costs)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-tag", required=True, nargs="+")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    args = parser.parse_args()
    try:
        selected_ids = selected_question_ids(args.split)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    summaries: dict[str, dict[str, Any]] = {}
    strict_by_tag: dict[str, dict[str, float]] = {}
    for tag in args.config_tag:
        generation = {str(row.get("id")): row for row in load_jsonl(RESULTS_DIR / f"{tag}_generation.jsonl")}
        judgments = {str(row.get("id")): row for row in load_jsonl(RESULTS_DIR / f"{tag}_judgments.jsonl")}
        rows = []
        for question_id in sorted(selected_ids):
            if question_id not in generation or question_id not in judgments:
                continue
            if judgments[question_id].get("error"):
                continue
            row = dict(generation[question_id])
            row["judgment"] = judgments[question_id].get("judgment", {})
            rows.append(row)
        summary = summarize_records(rows)
        summary["config_tag"] = tag
        summary["split"] = args.split
        summary["by_type"] = {question_type: summarize_records([row for row in rows if row.get("type") == question_type]) for question_type in sorted({row.get("type") for row in rows})}
        retrieval = load_json(RESULTS_DIR / f"{tag}_retrieval.json", {})
        if retrieval:
            relevant = [row for row in retrieval.get("results", []) if str(row.get("id")) in selected_ids]
            if relevant:
                summary["retrieval"] = {"hit_rate_at_k": sum(row.get("hit", False) for row in relevant) / len(relevant), "mrr": sum((1 / row["rank"] if row.get("rank") else 0) for row in relevant) / len(relevant), "n": len(relevant)}
        (RESULTS_DIR / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries[tag] = summary
        strict_by_tag[tag] = {row["id"]: float(row.get("judgment", {}).get("correctness", {}).get("verdict") == "correct") for row in rows if row.get("answerable")}
        print(f"{tag}: n={summary['n']} strict={summary['strict_accuracy']['mean']:.3f} faithful={summary['fully_faithful']['mean']:.3f} hallucination={summary['hallucination_rate']['mean']:.3f}")

    if len(summaries) > 1:
        lines = ["# Evaluation comparison", "", "| Config | N | Strict accuracy | Faithful | Hallucination | Cost/query |", "|---|---:|---:|---:|---:|---:|"]
        baseline = summaries[args.config_tag[0]]
        for tag in args.config_tag:
            summary = summaries[tag]
            relative = ((summary["strict_accuracy"]["mean"] - baseline["strict_accuracy"]["mean"]) / baseline["strict_accuracy"]["mean"] if baseline["strict_accuracy"]["mean"] else 0.0)
            lines.append(f"| {tag} | {summary['n']} | {summary['strict_accuracy']['mean']:.3f} ({relative:+.1%}) | {summary['fully_faithful']['mean']:.3f} | {summary['hallucination_rate']['mean']:.3f} | ${summary['cost_usd']['mean']:.6f} |")
        for left, right in combinations(args.config_tag, 2):
            ids = sorted(set(strict_by_tag[left]) & set(strict_by_tag[right]))
            ci = paired_bootstrap_difference([strict_by_tag[left][item] for item in ids], [strict_by_tag[right][item] for item in ids])
            lines.append(f"\nStrict-accuracy difference {left} − {right}: {ci['difference']:.3f}, 95% CI [{ci['low']:.3f}, {ci['high']:.3f}], significant={ci['significant']}.")
        output = RESULTS_DIR / f"comparison_{'_'.join(args.config_tag)}.md"
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved comparison to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
