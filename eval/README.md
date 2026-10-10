# Evaluation set

`questions.json` is a hand-verified JSON list. Each item has this schema:

```json
{
  "id": "q001",
  "question": "...",
  "type": "factual | conceptual | multi_chunk | out_of_scope",
  "answerable": true,
  "reference_answer": "...",
  "source_file": "name.pdf",
  "source_pages": [12, 13],
  "supporting_text": "short excerpt from the PDF that supports the answer",
  "verified": false,
  "notes": ""
}
```

`factual` questions have a direct, specific answer in one passage. `conceptual`
questions ask for an explanation grounded in the material. `multi_chunk` questions
need evidence from two related passages or pages. `out_of_scope` questions are
plausible study questions that the PDFs do not answer; set `answerable` to `false`
and leave `reference_answer` empty or `null`.

## Review guide

Generate a draft, then inspect every item in the terminal reviewer. For answerable
items, confirm the question and reference answer are supported by the cited pages
and edit the excerpt/pages if necessary. For out-of-scope items, search all PDFs to
confirm the information is actually absent; their source fields identify the related
material that inspired the question, not supporting evidence. Accept only items you
would be comfortable using as ground truth. The reviewer writes accepted items to
`questions.json`; do not treat `questions_draft.json` as ground truth.

Use `validate_eval.py` before an experiment. The targets are at least 150 verified
items and at least 10% out-of-scope questions.

## Development/test protocol

Create a deterministic, type-stratified split once:

```bash
python scripts/split_eval.py
```

Use only the **dev** split to tune prompts, thresholds, retrievers, chunking, and
other hyperparameters. Report the final, selected configuration only once on the
held-out **test** split. `questions.json` is never modified by splitting or runs.

## Generation and LLM judge

```bash
python scripts/run_eval_generation.py --config-tag baseline --split dev
python scripts/judge_eval.py --config-tag baseline --split dev
python scripts/summarize_eval.py --config-tag baseline --split dev
```

Generation and judging print an API-call estimate and require confirmation; pass
`--yes` only when you want to skip that prompt. The judge uses `JUDGE_MODEL`, or
the app's `GEMINI_MODEL` when it is unset. A different, stronger judge model than
the generator is preferable when available. Judge outputs are aids, not ground
truth: they can prefer their own style, reward verbosity, and be overly lenient.
Audit a sample blind before trusting scores:

```bash
python scripts/make_audit_sample.py --config-tag baseline
# Fill only the human_* columns in eval/audit/baseline_audit.csv.
python scripts/score_audit.py --config-tag baseline
```

Judge metrics: strict accuracy counts only `correct`; lenient accuracy includes
`partially_correct`; faithfulness is whether every extracted factual claim is
supported by retrieved context; hallucination is any unsupported claim; and
context relevance is the fraction of retrieved chunks the judge identifies as
relevant. Bootstrap confidence intervals use 1,000 seeded resamples.
