"""Create a deterministic type-stratified development/test question split."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval_utils import QUESTIONS_PATH, SPLIT_PATH, load_json  # noqa: E402


def stratified_split(items: list[dict], dev_fraction: float, seed: int) -> dict[str, list[str]]:
    """Split IDs by question type while preserving every item exactly once."""
    groups: dict[str, list[str]] = defaultdict(list)
    for item in items:
        groups[str(item["type"])].append(str(item["id"]))
    rng = random.Random(seed)
    dev, test = [], []
    for ids in groups.values():
        rng.shuffle(ids)
        dev_count = round(len(ids) * dev_fraction)
        # Keep both sides represented whenever a type has at least two examples.
        if len(ids) > 1:
            dev_count = min(len(ids) - 1, max(1, dev_count))
        dev.extend(ids[:dev_count])
        test.extend(ids[dev_count:])
    return {"dev": sorted(dev), "test": sorted(test)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.40)
    parser.add_argument("--force", action="store_true", help="Replace the generated split file.")
    args = parser.parse_args()
    if not 0 < args.dev_fraction < 1:
        parser.error("--dev-fraction must be between 0 and 1")
    if SPLIT_PATH.exists() and not args.force:
        print(f"Refusing to overwrite {SPLIT_PATH}; use --force to regenerate.", file=sys.stderr)
        return 2
    items = load_json(QUESTIONS_PATH, [])
    if not items:
        print(f"No questions found in {QUESTIONS_PATH}", file=sys.stderr)
        return 2
    split = stratified_split(items, args.dev_fraction, args.seed)
    split.update({"seed": args.seed, "dev_fraction": args.dev_fraction})
    SPLIT_PATH.write_text(__import__("json").dumps(split, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(split['dev'])} dev and {len(split['test'])} test IDs to {SPLIT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
