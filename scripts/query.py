import argparse

from app.services.rag_pipeline import query


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument("question", type=str)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    result = query(args.question, top_k=args.top_k)
    print("Answer:")
    print(result.answer)
    print("\nSources:")
    for source in result.sources:
        header = f"- [{source.filename} p.{source.page}] score={source.score:.3f}"
        print(f"{header} ({source.content_type})")
        print(f"  {source.text[:200]}")


if __name__ == "__main__":
    main()
