# Student Helper RAG

A Streamlit-based RAG app for asking questions over one or more uploaded PDF files.

## Features

- Upload one or more PDFs from the Streamlit sidebar.
- Uploaded PDFs are saved in the `data/` folder.
- Build or rebuild a local document index from every PDF in `data/`.
- Ask questions in a chat UI.
- The app first rewrites the user's question into a clearer retrieval query, then searches the indexed PDF chunks and sends the retrieved context to Gemini for the final answer.
- Source PDF names and page numbers are shown with answers.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with your Google API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

Optional environment variables:

```env
GEMINI_MODEL=gemini-3.1-flash-lite
RETRIEVER_PROVIDER=tfidf
```

`RETRIEVER_PROVIDER=tfidf` is the default and uses a fast local TF-IDF index. Set `RETRIEVER_PROVIDER=chroma` to use Chroma with embeddings.

## Run The App

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload PDFs, click **Save and index**, and start asking questions.

## Rebuild The Index From CLI

Put PDFs in `data/`, then run:

```bash
python src/ingest.py
```

## CLI Chat

After indexing documents, you can also use the terminal chat:

```bash
python src/main.py
```

Enter `0` to exit.

## Metrics and Evaluation

Every call to `answer_question` is appended to `logs/queries.jsonl`. Records include
timings for rewriting, retrieval, and answer generation; retrieved chunks; Gemini
usage metadata when available; a clearly flagged token estimate otherwise; and a
cost estimate based on placeholder prices. `logs/` is intentionally ignored by Git.
Set `PROMPT_VERSION` and `CONFIG_TAG` to label an experiment, and set
`GEMINI_INPUT_PRICE_PER_1M` / `GEMINI_OUTPUT_PRICE_PER_1M` only after checking the
current Gemini pricing for your model. The Streamlit chat has a collapsed **Debug
metrics** panel for each answer.

Summarize a log (optionally for one experiment) with:

```bash
python scripts/summarize_logs.py
python scripts/summarize_logs.py --config-tag baseline
python scripts/summarize_logs.py --exclude-source eval
```

The evaluation workflow is deliberately review-first, so no generated draft is
treated as ground truth:

```bash
# Requires GOOGLE_API_KEY and PDFs in data/; generates only an unverified draft.
python scripts/generate_eval_draft.py --seed 42 --num-questions 200
python scripts/review_eval.py
python scripts/validate_eval.py
python scripts/run_eval_retrieval.py --top-k 4 --config-tag baseline
```

See `eval/README.md` for the question schema and review criteria. The default
retrieval evaluation performs no LLM calls. Its optional `--use-rewrite` flag uses
the app's Gemini rewrite step, so it is useful for comparison but is not free.

For generation quality, first split the verified set, tune only on development,
and hold test results until your configuration is final:

```bash
python scripts/split_eval.py
python scripts/run_eval_generation.py --config-tag baseline --split dev
python scripts/judge_eval.py --config-tag baseline --split dev
python scripts/summarize_eval.py --config-tag baseline --split dev
```

Both API-backed scripts print an API-call estimate and ask for confirmation unless
you pass `--yes`. See `eval/README.md` for the blind human-audit workflow and the
limitations of LLM judges.

## Project Structure

```text
app.py              Streamlit UI
src/rag.py          Shared PDF loading, indexing, query rewrite, retrieval, and answering
src/ingest.py       CLI index rebuild script
src/main.py         CLI chat script
src/metrics.py      JSONL metrics helpers
scripts/            Metrics summary and evaluation tooling
eval/               Reviewable evaluation questions and results
data/               Uploaded/source PDFs
chroma_db/          Generated local index data
```
