import argparse

from app.services.rag_pipeline import ingest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into the RAG vector store"
    )
    parser.add_argument("pdf_path", type=str)
    args = parser.parse_args()

    result = ingest(args.pdf_path)
    print(f"Document hash: {result.document_hash}")
    print(f"Filename: {result.filename}")
    print(f"Pages: {result.page_count}")
    print(f"Chunks: {result.chunk_count}")
    print(f"Images: {result.image_count}")
    print(f"Skipped (duplicate): {result.skipped_duplicate}")


if __name__ == "__main__":
    main()
