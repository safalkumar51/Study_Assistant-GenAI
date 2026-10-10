"""Versioned, narrowly-scoped prompts for the evaluation judge."""

from __future__ import annotations

import json

JUDGE_PROMPT_VERSION = "judge-v1"


def _json_block(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def correctness_prompt(question: str, reference_answer: str, answer: str) -> str:
    return f"""You are a strict educational-answer evaluator. Compare the answer with the reference.
Question: {_json_block(question)}
Reference answer: {_json_block(reference_answer)}
System answer: {_json_block(answer)}
Return ONLY JSON: {{"verdict":"correct|partially_correct|incorrect|abstained","reason":"one sentence"}}.
Use abstained only when the system declined to answer. Do not reward verbosity."""


def faithfulness_prompt(answer: str, context: str) -> str:
    return f"""Evaluate grounding only. Extract factual claims from the answer, then assess each using ONLY context below.
Answer: {_json_block(answer)}
Context: {_json_block(context)}
Return ONLY JSON: {{"claims":[{{"claim":"...","supported":true,"evidence":"short quote or empty"}}],"number_supported":0,"number_total":0,"faithful":true}}.
Ignore pleasantries and citation formatting. faithful is true only if every factual claim is supported. Quote evidence exactly when supported."""


def answer_relevance_prompt(question: str, answer: str) -> str:
    return f"""Rate how directly the answer addresses the question, ignoring factual correctness.
Question: {_json_block(question)}
Answer: {_json_block(answer)}
Return ONLY JSON: {{"score":1,"reason":"one sentence"}} where score is an integer 1-5."""


def context_relevance_prompt(question: str, chunks: list[dict]) -> str:
    compact = [{"index": index, "text": chunk.get("text", "")} for index, chunk in enumerate(chunks)]
    return f"""Which retrieved chunks are relevant to answering this question?
Question: {_json_block(question)}
Chunks: {_json_block(compact)}
Return ONLY JSON: {{"relevant_indices":[0],"evidence":[{{"index":0,"quote":"short quote"}}]}}.
Use only valid chunk indices and quote evidence for every selected chunk."""


def abstention_prompt(question: str, answer: str) -> str:
    return f"""This question is intentionally out of scope for the uploaded documents. Judge whether the system safely abstained.
Question: {_json_block(question)}
System answer: {_json_block(answer)}
Return ONLY JSON: {{"verdict":"properly_abstained|hallucinated_answer|partial_hallucination","reason":"one sentence"}}."""
