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


st.set_page_config(page_title="Student Helper RAG", layout="wide")
st.title("Student Helper RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

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

query = st.chat_input("Ask a question")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer, docs = answer_question(query)

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

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": unique_sources,
        }
    )
