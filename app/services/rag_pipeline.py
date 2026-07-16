import logging
from dataclasses import dataclass
from pathlib import Path

from langsmith import traceable

from app.config.settings import get_settings
from app.services.chunker import chunk_text
from app.services.embeddings import embed_query, embed_texts
from app.services.hashing import sha256_file, sha256_text
from app.services.llm import generate_answer
from app.services.pdf_extractor import extract_pdf
from app.services.reranker import rerank
from app.services.vector_store import get_vector_store
from app.services.vision_captioner import caption_image
from app.utils.metadata_store import MetadataStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document_hash: str
    filename: str
    page_count: int
    chunk_count: int
    image_count: int
    skipped_duplicate: bool


@dataclass
class SourceChunk:
    text: str
    page: int
    filename: str
    content_type: str
    score: float
    image_path: str | None = None


@dataclass
class QueryResult:
    answer: str
    sources: list[SourceChunk]


def _get_metadata_store() -> MetadataStore:
    settings = get_settings()
    return MetadataStore(settings.metadata_db_path)


@traceable(name="ingest_pdf", run_type="chain")
def ingest(pdf_path: str | Path, filename: str | None = None) -> IngestResult:
    """Full ingestion pipeline: hash -> dedup check -> extract (text + OCR + images) ->
    caption images -> chunk -> dedup chunks -> embed -> upsert into the vector store."""
    settings = get_settings()
    pdf_path = Path(pdf_path)
    filename = filename or pdf_path.name

    doc_hash = sha256_file(str(pdf_path))
    metadata_store = _get_metadata_store()

    existing = metadata_store.get_document(doc_hash)
    if existing is not None:
        logger.info(
            "Document %s already ingested (hash=%s); skipping", filename, doc_hash
        )
        return IngestResult(
            document_hash=doc_hash,
            filename=existing.filename,
            page_count=existing.page_count,
            chunk_count=existing.chunk_count,
            image_count=existing.image_count,
            skipped_duplicate=True,
        )

    pages = extract_pdf(
        pdf_path,
        images_dir=settings.extracted_images_dir,
        doc_hash=doc_hash,
        min_text_chars_for_page_render=settings.min_page_text_chars_for_page_render,
        ocr_enabled=settings.ocr_enabled,
    )

    texts: list[str] = []
    chunk_meta: list[dict] = []
    image_count = 0

    for page in pages:
        for chunk in chunk_text(page.text):
            texts.append(chunk)
            chunk_meta.append(
                {
                    "text": chunk,
                    "content_type": "text",
                    "page": page.page_number,
                    "document_hash": doc_hash,
                    "filename": filename,
                }
            )

        if page.ocr_text:
            for chunk in chunk_text(page.ocr_text):
                texts.append(chunk)
                chunk_meta.append(
                    {
                        "text": chunk,
                        "content_type": "ocr_text",
                        "page": page.page_number,
                        "document_hash": doc_hash,
                        "filename": filename,
                    }
                )

        for image in page.images:
            image_count += 1
            try:
                caption = caption_image(image.path)
            except Exception as exc:
                logger.warning(
                    "Captioning failed for %s (%s); skipping", image.path, exc
                )
                continue
            texts.append(caption)
            chunk_meta.append(
                {
                    "text": caption,
                    "content_type": f"image_caption:{image.source}",
                    "page": image.page_number,
                    "document_hash": doc_hash,
                    "filename": filename,
                    "image_path": str(image.path),
                }
            )

    # Chunk-level SHA256 dedup: skip embedding text we've already indexed before.
    new_texts: list[str] = []
    new_meta: list[dict] = []
    new_hashes: list[str] = []
    for text, meta in zip(texts, chunk_meta):
        chunk_hash = sha256_text(text)
        if metadata_store.chunk_exists(chunk_hash):
            continue
        new_texts.append(text)
        new_meta.append(meta)
        new_hashes.append(chunk_hash)

    if new_texts:
        vectors = embed_texts(new_texts)
        vector_store = get_vector_store()
        vector_store.upsert(ids=new_hashes, vectors=vectors, metadatas=new_meta)

        for chunk_hash, text, meta in zip(new_hashes, new_texts, new_meta):
            metadata_store.upsert_chunk(
                chunk_hash=chunk_hash,
                document_hash=doc_hash,
                page=meta["page"],
                content_type=meta["content_type"],
                text=text,
                vector_store_id=chunk_hash,
            )

    metadata_store.upsert_document(
        doc_hash=doc_hash,
        filename=filename,
        page_count=len(pages),
        chunk_count=len(texts),
        image_count=image_count,
    )

    return IngestResult(
        document_hash=doc_hash,
        filename=filename,
        page_count=len(pages),
        chunk_count=len(texts),
        image_count=image_count,
        skipped_duplicate=False,
    )


@traceable(name="retrieve_context", run_type="retriever")
def retrieve(
    question: str, top_k: int | None = None, document_hash: str | None = None
) -> list[SourceChunk]:
    """Embed the question, retrieve cosine-similar candidates from the vector store,
    and rerank them with a cross-encoder for precision. Shared by both the
    non-streaming and streaming query paths. If document_hash is given, results are
    scoped to that single ingested document."""
    settings = get_settings()
    top_k = top_k or settings.top_k
    filter_metadata = {"document_hash": document_hash} if document_hash else None

    query_vector = embed_query(question)
    vector_store = get_vector_store()
    candidate_pool = (
        max(top_k, settings.reranker_candidate_pool)
        if settings.reranker_enabled
        else top_k
    )
    matches = vector_store.query(
        query_vector, top_k=candidate_pool, filter_metadata=filter_metadata
    )
    matches = rerank(question, matches, top_k=top_k)

    return [
        SourceChunk(
            text=metadata.get("text", ""),
            page=metadata.get("page", -1),
            filename=metadata.get("filename", ""),
            content_type=metadata.get("content_type", "text"),
            score=score,
            image_path=metadata.get("image_path"),
        )
        for _, score, metadata in matches
    ]


@traceable(name="rag_query", run_type="chain")
def query(
    question: str,
    top_k: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    document_hash: str | None = None,
) -> QueryResult:
    """Retrieve context and ask the chat model to synthesize a grounded answer."""
    sources = retrieve(question, top_k=top_k, document_hash=document_hash)
    answer = generate_answer(
        question,
        [source.text for source in sources],
        temperature=temperature,
        top_p=top_p,
    )
    return QueryResult(answer=answer, sources=sources)
