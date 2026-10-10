"""Small, dependency-free helpers for recording RAG query metrics."""

from __future__ import annotations

import json
import os
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_PATH = BASE_DIR / "logs" / "queries.jsonl"

# Placeholder USD prices per one million tokens. Verify these values against the
# current Gemini pricing page before relying on the resulting cost estimates.
PRICING = {
    "input_per_million_usd": float(os.getenv("GEMINI_INPUT_PRICE_PER_1M", "0.10")),
    "output_per_million_usd": float(os.getenv("GEMINI_OUTPUT_PRICE_PER_1M", "0.40")),
}


class Timer(AbstractContextManager["Timer"]):
    """Measure elapsed wall-clock time with a high-resolution monotonic clock."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost using the configurable placeholder pricing."""
    return round(
        (input_tokens * PRICING["input_per_million_usd"]
        + output_tokens * PRICING["output_per_million_usd"])
        / 1_000_000,
        10,
    )


def log_query(record: dict[str, Any]) -> None:
    """Append one query record as JSONL, creating the log directory as needed."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
