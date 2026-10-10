from dataclasses import dataclass
from functools import lru_cache
import os
import pickle
from pathlib import Path
import shutil
from typing import Any, Iterable
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .metrics import Timer, estimate_cost_usd, log_query, utc_timestamp
except ImportError:  # Supports running modules directly from src/.
    from metrics import Timer, estimate_cost_usd, log_query, utc_timestamp

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
TFIDF_INDEX_PATH = CHROMA_DIR / "tfidf_index.pkl"
DEFAULT_SESSION_ID = str(uuid4())


@dataclass
class IngestResult:
    pdf_count: int
    page_count: int
    chunk_count: int


@dataclass
class AnswerResult:
    """Answer, retrieved documents, and metrics for one question.

    It remains iterable as ``(answer, docs)`` so existing callers that unpack
    the historical return value continue to work.
    """

    answer: str
    documents: list[Document]
    metrics: dict[str, Any]

    def __iter__(self):
        yield self.answer
        yield self.documents

    def __getitem__(self, index: int):
        return (self.answer, self.documents)[index]

    def __len__(self) -> int:
        return 2


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


def get_top_k() -> int:
    return max(1, int(os.getenv("RETRIEVER_TOP_K", "4")))


def get_chunk_size() -> int:
    return max(1, int(os.getenv("CHUNK_SIZE", "1000")))


def get_chunk_overlap() -> int:
    return max(0, int(os.getenv("CHUNK_OVERLAP", "200")))


def get_retriever(k: int | None = None):
    top_k = k or get_top_k()
    return get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": max(10, top_k * 2),
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
        chunk_size=get_chunk_size(),
        chunk_overlap=get_chunk_overlap(),
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


def retrieve_tfidf_documents(question: str, k: int | None = None) -> list[Document]:
    from sklearn.metrics.pairwise import cosine_similarity

    if not TFIDF_INDEX_PATH.exists():
        return []

    with TFIDF_INDEX_PATH.open("rb") as index_file:
        index = pickle.load(index_file)

    query_vector = index["vectorizer"].transform([question])
    scores = cosine_similarity(query_vector, index["matrix"]).ravel()
    ranked_indices = scores.argsort()[::-1]

    docs = []
    top_k = k or get_top_k()
    for doc_index in ranked_indices:
        if len(docs) >= top_k:
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


def merge_documents(document_groups: Iterable[list[Document]], limit: int | None = None) -> list[Document]:
    merged_docs = []
    seen = set()
    max_documents = limit or get_top_k()

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

            if len(merged_docs) >= max_documents:
                return merged_docs

    return merged_docs


def retrieve_documents(question: str, k: int | None = None) -> list[Document]:
    if get_retriever_provider() == "chroma":
        return get_retriever(k).invoke(question)

    return retrieve_tfidf_documents(question, k)


def retrieve_with_rewrite(
    question: str, rewritten_question: str, k: int | None = None
) -> list[Document]:
    top_k = k or get_top_k()
    if get_retriever_provider() == "chroma":
        rewritten_docs = get_retriever(top_k).invoke(rewritten_question)
        original_docs = get_retriever(top_k).invoke(question)
    else:
        combined_question = f"{question}\n{rewritten_question}"
        rewritten_docs = retrieve_tfidf_documents(rewritten_question, top_k)
        combined_docs = retrieve_tfidf_documents(combined_question, top_k)
        original_docs = retrieve_tfidf_documents(question, top_k)
        return merge_documents([rewritten_docs, combined_docs, original_docs], top_k)

    return merge_documents([rewritten_docs, original_docs], top_k)


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


def usage_from_response(response: Any, fallback_input_text: str = "", fallback_output_text: str = "") -> dict[str, Any]:
    """Extract Gemini token metadata, marking deterministic approximations clearly."""
    usage = getattr(response, "usage_metadata", None) or {}
    if not usage:
        response_metadata = getattr(response, "response_metadata", None) or {}
        if isinstance(response_metadata, dict):
            usage = response_metadata.get("usage_metadata", response_metadata.get("usage", {}))
    if not isinstance(usage, dict):
        usage = dict(usage) if hasattr(usage, "items") else {}

    input_tokens = usage.get("input_tokens", usage.get("prompt_token_count"))
    output_tokens = usage.get("output_tokens", usage.get("candidates_token_count"))
    estimated = input_tokens is None or output_tokens is None
    if estimated:
        # A transparent rough fallback: approximately four characters per token.
        input_tokens = input_tokens if input_tokens is not None else max(1, len(fallback_input_text) // 4)
        output_tokens = output_tokens if output_tokens is not None else max(1, len(fallback_output_text) // 4)

    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "tokens_estimated": estimated,
        "called": True,
    }


def rewrite_question_with_metrics(question: str) -> tuple[str, dict[str, Any], str | None]:
    """Rewrite a question and return query, call token data, and any error."""
    pdf_sources = ", ".join(pdf_path.name for pdf_path in list_pdf_files()) or "None"
    try:
        final_prompt = QUERY_REWRITE_PROMPT.invoke(
            {
                "question": question,
                "pdf_sources": pdf_sources,
            }
        )
        with Timer() as timer:
            response = get_llm().invoke(final_prompt)
        rewritten_question = message_content_to_text(response.content).strip()
        usage = usage_from_response(response, str(final_prompt), rewritten_question)
        usage["latency_ms"] = round(timer.elapsed_ms, 2)
        return rewritten_question or question, usage, None
    except Exception as exc:
        return question, {
            "input_tokens": 0,
            "output_tokens": 0,
            "tokens_estimated": False,
            "called": False,
            "latency_ms": 0.0,
        }, f"Rewrite failed: {exc}"


def rewrite_question(question: str) -> str:
    """Return a standalone query while retaining the original public helper."""
    rewritten_question, _, _ = rewrite_question_with_metrics(question)
    return rewritten_question


def _chunk_metrics(documents: list[Document]) -> list[dict[str, Any]]:
    metrics = []
    for doc in documents:
        page_index = doc.metadata.get("page")
        metrics.append(
            {
                "source_file": doc.metadata.get("source", "Unknown source"),
                "page": page_index + 1 if isinstance(page_index, int) else page_index,
                "page_index": page_index,
                "score": doc.metadata.get("score"),
                "preview": doc.page_content[:200],
            }
        )
    return metrics


def answer_question(
    question: str,
    session_id: str | None = None,
    source: str = "app",
    top_k: int | None = None,
) -> AnswerResult:
    """Answer a question and persist honest pipeline measurements as JSONL."""
    session = session_id or DEFAULT_SESSION_ID
    metrics: dict[str, Any] = {
        "timestamp": utc_timestamp(),
        "session_id": session,
        "source": source,
        "question": question,
        "rewritten_query": question,
        "retrieval_settings": {
            "retriever_provider": get_retriever_provider(),
            "top_k": top_k or get_top_k(),
            "chunk_size": get_chunk_size(),
            "chunk_overlap": get_chunk_overlap(),
        },
        "retrieved_chunks": [],
        "answer": "",
        "timings_ms": {"rewrite_ms": 0.0, "retrieval_ms": 0.0, "llm_ms": 0.0, "total_ms": 0.0},
        "gemini_calls": {
            "rewrite": {"input_tokens": 0, "output_tokens": 0, "tokens_estimated": False, "called": False},
            "answer": {"input_tokens": 0, "output_tokens": 0, "tokens_estimated": False, "called": False},
        },
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens_estimated": True,
        "estimated_cost_usd": 0.0,
        "cache_hit": False,
        "prompt_version": os.getenv("PROMPT_VERSION", "baseline"),
        "config_tag": os.getenv("CONFIG_TAG", "baseline"),
        "error": None,
    }
    docs: list[Document] = []
    answer = "I could not find the answer in the document."

    with Timer() as total_timer:
        rewritten_question, rewrite_usage, rewrite_error = rewrite_question_with_metrics(question)
        metrics["rewritten_query"] = rewritten_question
        metrics["gemini_calls"]["rewrite"] = rewrite_usage
        metrics["timings_ms"]["rewrite_ms"] = rewrite_usage["latency_ms"]
        if rewrite_error:
            metrics["error"] = rewrite_error

        try:
            with Timer() as retrieval_timer:
                docs = retrieve_with_rewrite(question, rewritten_question, top_k)
            metrics["timings_ms"]["retrieval_ms"] = round(retrieval_timer.elapsed_ms, 2)
        except Exception as exc:
            metrics["error"] = f"{metrics['error']}; Retrieval failed: {exc}" if metrics["error"] else f"Retrieval failed: {exc}"
            docs = []
        metrics["retrieved_chunks"] = _chunk_metrics(docs)

        if docs:
            context = "\n\n".join(doc.page_content for doc in docs)
            final_prompt = PROMPT.invoke({"context": context, "question": question})
            try:
                with Timer() as llm_timer:
                    response = get_llm().invoke(final_prompt)
                answer = message_content_to_text(response.content)
                answer_usage = usage_from_response(response, str(final_prompt), answer)
                answer_usage["latency_ms"] = round(llm_timer.elapsed_ms, 2)
                metrics["gemini_calls"]["answer"] = answer_usage
                metrics["timings_ms"]["llm_ms"] = answer_usage["latency_ms"]
            except Exception as exc:
                answer = "I could not generate an answer from the document."
                metrics["error"] = f"{metrics['error']}; Answer failed: {exc}" if metrics["error"] else f"Answer failed: {exc}"

    metrics["timings_ms"]["total_ms"] = round(total_timer.elapsed_ms, 2)
    metrics["answer"] = answer
    metrics["input_tokens"] = sum(call["input_tokens"] for call in metrics["gemini_calls"].values())
    metrics["output_tokens"] = sum(call["output_tokens"] for call in metrics["gemini_calls"].values())
    metrics["tokens_estimated"] = any(call["tokens_estimated"] for call in metrics["gemini_calls"].values())
    metrics["estimated_cost_usd"] = estimate_cost_usd(metrics["input_tokens"], metrics["output_tokens"])
    log_query(metrics)
    return AnswerResult(answer=answer, documents=docs, metrics=metrics)
