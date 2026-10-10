"""Compare blind human audit labels to held-out judge labels."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cohen_kappa(human: list[str], judge: list[str]) -> float:
    """Compute unweighted Cohen's kappa without a statistics dependency."""
    if not human or len(human) != len(judge):
        return 0.0
    observed = sum(left == right for left, right in zip(human, judge)) / len(human)
    left_counts, right_counts = Counter(human), Counter(judge)
    expected = sum((left_counts[label] / len(human)) * (right_counts[label] / len(judge)) for label in set(left_counts) | set(right_counts))
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def normalize_faithful(value: str) -> str:
    return "true" if value.strip().lower() in {"true", "yes", "y", "1", "faithful"} else "false"


def score_metric(name: str, human_key: str, judge_key: str, human_rows: dict[str, dict], judge_rows: dict[str, dict], transform=lambda value: value.strip().lower()) -> None:
    pairs = []
    for item_id, human in human_rows.items():
        value = human.get(human_key, "").strip()
        judge_value = judge_rows.get(item_id, {}).get(judge_key, "").strip()
        if value and judge_value:
            pairs.append((item_id, transform(value), transform(judge_value)))
    if not pairs:
        print(f"{name}: no completed human labels")
        return
    human, judge = [pair[1] for pair in pairs], [pair[2] for pair in pairs]
    agreement = sum(left == right for left, right in zip(human, judge)) / len(pairs)
    kappa = cohen_kappa(human, judge)
    print(f"{name}: n={len(pairs)}, agreement={agreement:.1%}, kappa={kappa:.3f}")
    if kappa < 0.4:
        print("WARNING: kappa < 0.4; do not trust the judge for this metric yet.")
    labels = sorted(set(human) | set(judge))
    print("  confusion matrix (human rows × judge columns):")
    print("  " + " | ".join(["human\\judge", *labels]))
    for label in labels:
        print("  " + " | ".join([label, *(str(sum(left == label and right == column for left, right in zip(human, judge))) for column in labels)]))
    disagreements = [item_id for item_id, left, right in pairs if left != right]
    if disagreements:
        print("  disagreements: " + ", ".join(disagreements))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-tag", required=True)
    args = parser.parse_args()
    audit_dir = ROOT / "eval" / "audit"
    audit_path = audit_dir / f"{args.config_tag}_audit.csv"
    labels_path = audit_dir / f"{args.config_tag}_judge_labels.csv"
    if not audit_path.exists() or not labels_path.exists():
        print("Audit CSV and judge-label CSV are both required.")
        return 2
    with audit_path.open(encoding="utf-8", newline="") as source:
        human_rows = {row["id"]: row for row in csv.DictReader(source)}
    with labels_path.open(encoding="utf-8", newline="") as source:
        judge_rows = {row["id"]: row for row in csv.DictReader(source)}
    score_metric("Correctness", "human_correctness", "judge_correctness", human_rows, judge_rows)
    score_metric("Faithfulness", "human_faithful", "judge_faithful", human_rows, judge_rows, normalize_faithful)
    score_metric("Abstention", "human_abstention", "judge_abstention", human_rows, judge_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
