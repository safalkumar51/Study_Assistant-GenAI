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

## Project Structure

```text
app.py              Streamlit UI
src/rag.py          Shared PDF loading, indexing, query rewrite, retrieval, and answering
src/ingest.py       CLI index rebuild script
src/main.py         CLI chat script
data/               Uploaded/source PDFs
chroma_db/          Generated local index data
```
