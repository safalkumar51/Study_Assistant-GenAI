from rag import ingest_pdfs


if __name__ == "__main__":
    result = ingest_pdfs()

    print(
        "Indexed "
        f"{result.pdf_count} PDF(s), "
        f"{result.page_count} page(s), "
        f"and {result.chunk_count} chunk(s)."
    )
