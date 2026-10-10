"""Offline tests for deterministic evaluation statistics and splitting."""

from scripts.score_audit import cohen_kappa
from scripts.split_eval import stratified_split
from scripts.summarize_eval import bootstrap_rate, paired_bootstrap_difference


def test_bootstrap_rate_is_seeded_and_bounded() -> None:
    result = bootstrap_rate([0.0, 1.0, 1.0])
    assert result == bootstrap_rate([0.0, 1.0, 1.0])
    assert result["low"] <= result["mean"] <= result["high"]


def test_paired_bootstrap_detects_a_clear_difference() -> None:
    result = paired_bootstrap_difference([1.0] * 20, [0.0] * 20)
    assert result["significant"] is True
    assert result["low"] > 0


def test_kappa_perfect_agreement() -> None:
    assert cohen_kappa(["correct", "incorrect"], ["correct", "incorrect"]) == 1.0


def test_stratified_split_preserves_all_ids() -> None:
    items = [
        {"id": f"{question_type}-{number}", "type": question_type}
        for question_type in ("factual", "conceptual")
        for number in range(4)
    ]
    split = stratified_split(items, 0.4, 42)
    assert set(split["dev"]) | set(split["test"]) == {item["id"] for item in items}
    assert not set(split["dev"]) & set(split["test"])
