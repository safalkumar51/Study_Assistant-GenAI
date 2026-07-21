from dataclasses import dataclass
from functools import lru_cache
import os
import pickle
from pathlib import Path
import shutil
from typing import Iterable

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
TFIDF_INDEX_PATH = CHROMA_DIR / "tfidf_index.pkl"


@dataclass
class IngestResult:
    pdf_count: int
    page_count: int
    chunk_count: int


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.

            Use only the provided context to answer the question.

            If the answer is not present in the context, say:
            "I could not find the answer in the document."
            """,
        ),
        (
            "user",
            """
            Context: {context}

            Question: {question}
            """,
        ),
    ]
)

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Rewrite the user's question into a clear, standalone search query
            for retrieving relevant passages from study PDFs.

            Use the available PDF filenames as domain hints when helpful.
            For example, a filename can help expand acronyms in short questions.
            Keep important technical terms, names, formulas, and constraints.
            Do not answer the question.
            Return only the rewritten query.
            """,
        ),
        (
            "user",
            """
            Available PDF filenames: {pdf_sources}

            User question: {question}
            """,
        ),
    ]
)


def list_pdf_files() -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(DATA_DIR.glob("*.pdf"))


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    provider = os.getenv("EMBEDDING_PROVIDER", "google").lower()

    if provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=os.getenv(
                "EMBEDDING_MODEL_NAME",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
        )

    return GoogleGenerativeAIEmbeddings(
        model=os.getenv(
            "GOOGLE_EMBEDDING_MODEL",
            "models/gemini-embedding-001",
        )
    )


def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embedding_model(),
    )


def reset_vectorstore() -> None:
    resolved_chroma_dir = CHROMA_DIR.resolve()
    resolved_base_dir = BASE_DIR.resolve()

    if not resolved_chroma_dir.exists():
        return

    if resolved_chroma_dir.name != "chroma_db" or resolved_base_dir not in resolved_chroma_dir.parents:
        raise ValueError(f"Refusing to delete unexpected Chroma path: {resolved_chroma_dir}")

    shutil.rmtree(resolved_chroma_dir)


def get_retriever():
    return get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )


def get_retriever_provider() -> str:
    return os.getenv("RETRIEVER_PROVIDER", "tfidf").lower()


def load_pdf_documents(pdf_paths: Iterable[Path] | None = None):
    documents = []
    paths = list(pdf_paths) if pdf_paths is not None else list_pdf_files()

    for pdf_path in paths:
        documents.extend(load_single_pdf(pdf_path))

    return documents


def load_single_pdf(pdf_path: Path):
    try:
        import fitz
    except ImportError:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_path.name
            page.metadata["file_path"] = str(pdf_path)

        return pages

    pages = []
    pdf = fitz.open(str(pdf_path))
    try:
        for page_number in range(len(pdf)):
            page = pdf.load_page(page_number)
            text = page.get_text("text")
            if not isinstance(text, str) or not text.strip():
                continue

            pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "file_path": str(pdf_path),
                        "page": page_number,
                    },
                )
            )
    finally:
        pdf.close()

    return pages


def ingest_pdfs(pdf_paths: Iterable[Path] | None = None, reset: bool = True) -> IngestResult:
    documents = load_pdf_documents(pdf_paths)

    if not documents:
        if reset:
            reset_vectorstore()
        return IngestResult(pdf_count=0, page_count=0, chunk_count=0)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    if reset:
        reset_vectorstore()

    if get_retriever_provider() == "chroma":
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embedding_model(),
            persist_directory=str(CHROMA_DIR),
        )

        if hasattr(vectorstore, "persist"):
            vectorstore.persist()
    else:
        build_tfidf_index(chunks)

    pdf_count = len({doc.metadata.get("source") for doc in documents})
    return IngestResult(
        pdf_count=pdf_count,
        page_count=len(documents),
        chunk_count=len(chunks),
    )


def build_tfidf_index(chunks) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
    )
    matrix = vectorizer.fit_transform(texts)

    with TFIDF_INDEX_PATH.open("wb") as index_file:
        pickle.dump(
            {
                "texts": texts,
                "metadatas": metadatas,
                "vectorizer": vectorizer,
                "matrix": matrix,
            },
            index_file,
        )


def retrieve_tfidf_documents(question: str, k: int = 4) -> list[Document]:
    from sklearn.metrics.pairwise import cosine_similarity

    if not TFIDF_INDEX_PATH.exists():
        return []

    with TFIDF_INDEX_PATH.open("rb") as index_file:
        index = pickle.load(index_file)

    query_vector = index["vectorizer"].transform([question])
    scores = cosine_similarity(query_vector, index["matrix"]).ravel()
    ranked_indices = scores.argsort()[::-1]

    docs = []
    for doc_index in ranked_indices:
        if len(docs) >= k:
            break

        score = float(scores[doc_index])
        if score <= 0:
            continue

        metadata = dict(index["metadatas"][doc_index])
        metadata["score"] = score
        docs.append(
            Document(
                page_content=index["texts"][doc_index],
                metadata=metadata,
            )
        )

    return docs


def merge_documents(document_groups: Iterable[list[Document]], limit: int = 4) -> list[Document]:
    merged_docs = []
    seen = set()

    for documents in document_groups:
        for document in documents:
            key = (
                document.metadata.get("source"),
                document.metadata.get("page"),
                document.page_content[:120],
            )

            if key in seen:
                continue

            seen.add(key)
            merged_docs.append(document)

            if len(merged_docs) >= limit:
                return merged_docs

    return merged_docs


def retrieve_documents(question: str) -> list[Document]:
    if get_retriever_provider() == "chroma":
        return get_retriever().invoke(question)

    return retrieve_tfidf_documents(question)


def retrieve_with_rewrite(question: str, rewritten_question: str) -> list[Document]:
    if get_retriever_provider() == "chroma":
        rewritten_docs = get_retriever().invoke(rewritten_question)
        original_docs = get_retriever().invoke(question)
    else:
        combined_question = f"{question}\n{rewritten_question}"
        rewritten_docs = retrieve_tfidf_documents(rewritten_question)
        combined_docs = retrieve_tfidf_documents(combined_question)
        original_docs = retrieve_tfidf_documents(question)
        return merge_documents([rewritten_docs, combined_docs, original_docs])

    return merge_documents([rewritten_docs, original_docs])


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    )


def message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            else:
                text_parts.append(str(part))

        return "\n".join(part for part in text_parts if part)

    return str(content)


def rewrite_question(question: str) -> str:
    pdf_sources = ", ".join(pdf_path.name for pdf_path in list_pdf_files()) or "None"
    try:
        final_prompt = QUERY_REWRITE_PROMPT.invoke(
            {
                "question": question,
                "pdf_sources": pdf_sources,
            }
        )
        response = get_llm().invoke(final_prompt)
        rewritten_question = message_content_to_text(response.content).strip()
    except Exception:
        return question

    return rewritten_question or question


def answer_question(question: str) -> tuple[str, list]:
    rewritten_question = rewrite_question(question)
    docs = retrieve_with_rewrite(question, rewritten_question)

    if not docs:
        return "I could not find the answer in the document.", []

    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    response = get_llm().invoke(final_prompt)

    return message_content_to_text(response.content), docs
