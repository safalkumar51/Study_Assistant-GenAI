import re
from pathlib import Path

import streamlit as st

from src.rag import DATA_DIR, answer_question, ingest_pdfs, list_pdf_files


def safe_pdf_name(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip()
    return name if name.lower().endswith(".pdf") else f"{name}.pdf"


def save_uploaded_pdfs(uploaded_files) -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for uploaded_file in uploaded_files:
        destination = DATA_DIR / safe_pdf_name(uploaded_file.name)
        destination.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(destination)

    return saved_paths


def debug_metrics(metrics: dict) -> dict:
    """Keep the chat debug view concise while the JSONL log retains full detail."""
    return {
        "timings_ms": metrics.get("timings_ms", {}),
        "input_tokens": metrics.get("input_tokens", 0),
        "output_tokens": metrics.get("output_tokens", 0),
        "tokens_estimated": metrics.get("tokens_estimated", False),
        "estimated_cost_usd": metrics.get("estimated_cost_usd", 0),
        "gemini_calls": metrics.get("gemini_calls", {}),
        "cache_hit": metrics.get("cache_hit", False),
        "error": metrics.get("error"),
    }


st.set_page_config(page_title="Student Helper RAG", layout="wide")
st.title("Student Helper RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    # Kept for the lifetime of this browser session and included in JSONL logs.
    import uuid

    st.session_state.session_id = str(uuid.uuid4())

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Save and index", type="primary", disabled=not uploaded_files):
        saved_paths = save_uploaded_pdfs(uploaded_files)
        with st.spinner("Indexing PDFs..."):
            result = ingest_pdfs()

        st.success(
            f"Saved {len(saved_paths)} file(s). Indexed "
            f"{result.pdf_count} PDF(s), {result.page_count} page(s), "
            f"and {result.chunk_count} chunk(s)."
        )

    if st.button("Rebuild index"):
        with st.spinner("Indexing PDFs..."):
            result = ingest_pdfs()

        if result.chunk_count:
            st.success(
                f"Indexed {result.pdf_count} PDF(s), "
                f"{result.page_count} page(s), and {result.chunk_count} chunk(s)."
            )
        else:
            st.warning("No PDFs found in the data folder.")

    st.divider()
    st.subheader("Data folder")
    pdf_files = list_pdf_files()

    if pdf_files:
        for pdf_file in pdf_files:
            st.caption(pdf_file.name)
    else:
        st.caption("No PDFs uploaded yet.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.caption(source)
        if message.get("metrics"):
            with st.expander("Debug metrics", expanded=False):
                st.json(debug_metrics(message["metrics"]))

query = st.chat_input("Ask a question")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = answer_question(query, session_id=st.session_state.session_id)
            answer, docs = result

        st.markdown(answer)

        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page")
            if page is not None:
                source = f"{source}, page {page + 1}"
            sources.append(source)

        if sources:
            unique_sources = list(dict.fromkeys(sources))
            with st.expander("Sources"):
                for source in unique_sources:
                    st.caption(source)
        else:
            unique_sources = []

        with st.expander("Debug metrics", expanded=False):
            st.json(debug_metrics(result.metrics))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": unique_sources,
            "metrics": result.metrics,
        }
    )
