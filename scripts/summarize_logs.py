"""Summarize JSONL query logs produced by the RAG pipeline."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import LOG_PATH  # noqa: E402


def percentile(values: list[float], percent: float) -> float:
    """Return an interpolated percentile without adding a dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-tag", help="Only include this experiment tag.")
    parser.add_argument("--source", help="Only include this record source, such as app or eval.")
    parser.add_argument("--exclude-source", help="Exclude a record source, such as eval.")
    args = parser.parse_args()
    if not LOG_PATH.exists():
        print(f"No query log found at {LOG_PATH}")
        return 0

    records = []
    for line_number, line in enumerate(LOG_PATH.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(f"Skipping malformed JSON on line {line_number}.", file=sys.stderr)
            continue
        tag_matches = not args.config_tag or record.get("config_tag") == args.config_tag
        source_matches = not args.source or record.get("source") == args.source
        source_excluded = args.exclude_source and record.get("source") == args.exclude_source
        if tag_matches and source_matches and not source_excluded:
            records.append(record)

    if not records:
        print("No matching queries.")
        return 0

    def timings(name: str) -> list[float]:
        return [float(item.get("timings_ms", {}).get(name, 0)) for item in records]

    print(f"Queries: {len(records)}")
    print(f"Unique sessions: {len({item.get('session_id') for item in records})}")
    for name in ("total_ms", "rewrite_ms", "retrieval_ms", "llm_ms"):
        values = timings(name)
        print(f"{name}: avg={statistics.mean(values):.2f} ms  p50={percentile(values, 50):.2f} ms  p95={percentile(values, 95):.2f} ms")
    inputs = [int(item.get("input_tokens", 0)) for item in records]
    outputs = [int(item.get("output_tokens", 0)) for item in records]
    costs = [float(item.get("estimated_cost_usd", 0)) for item in records]
    print(f"Input tokens: avg={statistics.mean(inputs):.1f}")
    print(f"Output tokens: avg={statistics.mean(outputs):.1f}")
    print(f"Estimated cost: avg=${statistics.mean(costs):.8f}  total=${sum(costs):.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
