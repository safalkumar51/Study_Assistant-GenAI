"""Interactively review draft evaluation questions in a terminal."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
DRAFT_PATH = EVAL_DIR / "questions_draft.json"
OUTPUT_PATH = EVAL_DIR / "questions.json"
PROGRESS_PATH = EVAL_DIR / "review_progress.json"


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_ids(items: list[dict]) -> None:
    for number, item in enumerate(items, 1):
        item["id"] = f"q{number:03d}"


def edit_item(item: dict) -> None:
    for field in ("question", "reference_answer", "supporting_text", "notes"):
        current = item.get(field) or ""
        replacement = input(f"{field} [{current}]: ").strip()
        if replacement:
            item[field] = replacement


def print_item(item: dict) -> None:
    print("\n" + "=" * 72)
    print(f"Draft ID: {item.get('id')}  |  {item.get('type')}  |  answerable={item.get('answerable')}")
    print(f"Question: {item.get('question')}")
    print(f"Reference answer: {item.get('reference_answer')}")
    print(f"Source: {item.get('source_file')} pages {item.get('source_pages')}")
    print(f"Supporting text:\n{item.get('supporting_text')}")
    if item.get("notes"):
        print(f"Notes: {item['notes']}")


def main() -> int:
    if not DRAFT_PATH.exists():
        print(f"Draft file not found: {DRAFT_PATH}")
        return 2
    draft = read_json(DRAFT_PATH, [])
    accepted = read_json(OUTPUT_PATH, [])
    progress = read_json(PROGRESS_PATH, {"accepted_draft_ids": [], "rejected_draft_ids": []})
    accepted_ids = set(progress.get("accepted_draft_ids", []))
    existing_ids = {item.get("id") for item in accepted}
    # Supports older/manual exports where the draft ID itself was retained.
    accepted_ids.update(item.get("id") for item in accepted if str(item.get("id", "")).startswith("draft-"))
    rejected_ids = set(progress.get("rejected_draft_ids", []))

    for item in draft:
        draft_id = item.get("id")
        if draft_id in accepted_ids or draft_id in existing_ids or draft_id in rejected_ids:
            continue
        print_item(item)
        while True:
            choice = input("[a]ccept, [e]dit then accept, [s]kip, [r]eject, [q]uit: ").strip().lower()
            if choice in {"a", "e", "s", "r", "q"}:
                break
        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "r":
            rejected_ids.add(draft_id)
        else:
            chosen = dict(item)
            if choice == "e":
                edit_item(chosen)
            chosen["verified"] = True
            accepted.append(chosen)
            accepted_ids.add(draft_id)
            normalize_ids(accepted)
            write_json(OUTPUT_PATH, accepted)
        write_json(PROGRESS_PATH, {
            "accepted_draft_ids": sorted(accepted_ids),
            "rejected_draft_ids": sorted(rejected_ids),
        })

    normalize_ids(accepted)
    write_json(OUTPUT_PATH, accepted)
    write_json(PROGRESS_PATH, {
        "accepted_draft_ids": sorted(accepted_ids),
        "rejected_draft_ids": sorted(rejected_ids),
    })
    print(f"Saved {len(accepted)} verified items to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
