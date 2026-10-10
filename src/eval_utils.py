"""Shared, dependency-free helpers for evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = BASE_DIR / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
SPLIT_PATH = EVAL_DIR / "split.json"
RESULTS_DIR = EVAL_DIR / "results"


def load_json(path: Path, default: Any = None) -> Any:
    """Load JSON, returning the supplied default when the file is absent."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read valid JSON-object lines, ignoring blank lines."""
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append and flush one durable JSONL result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        output.flush()


def selected_question_ids(split: str) -> set[str]:
    """Return question IDs selected by a persisted dev/test split."""
    if split == "all":
        return {str(item["id"]) for item in load_json(QUESTIONS_PATH, [])}
    split_data = load_json(SPLIT_PATH)
    if not split_data or split not in split_data:
        raise ValueError("Split file missing. Run `python scripts/split_eval.py` first.")
    return {str(question_id) for question_id in split_data[split]}


def selected_questions(split: str) -> list[dict[str, Any]]:
    """Load evaluation questions matching a named split."""
    ids = selected_question_ids(split)
    return [item for item in load_json(QUESTIONS_PATH, []) if str(item.get("id")) in ids]


def confirm_api_run(description: str, calls: int, yes: bool) -> bool:
    """Require an explicit acknowledgement before a potentially paid API run."""
    estimate = f"about {calls} Gemini calls"
    print(f"This will make {estimate} ({description}).")
    print("Token use varies with retrieved context; this is an upper-bound call estimate.")
    if yes:
        return True
    try:
        return input("Continue? [y/N]: ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False
