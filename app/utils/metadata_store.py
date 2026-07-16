import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    hash TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    image_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    hash TEXT PRIMARY KEY,
    document_hash TEXT NOT NULL REFERENCES documents(hash),
    page INTEGER NOT NULL,
    content_type TEXT NOT NULL,
    text TEXT NOT NULL,
    vector_store_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_hash ON chunks(document_hash);
"""


@dataclass
class DocumentRecord:
    hash: str
    filename: str
    ingested_at: str
    page_count: int
    chunk_count: int
    image_count: int


@dataclass
class ChunkRecord:
    hash: str
    document_hash: str
    page: int
    content_type: str
    text: str
    vector_store_id: str


class MetadataStore:
    """SQLite-backed registry of ingested documents/chunks, used for SHA256 dedup
    and to attach page/source metadata to vector search results."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_document(self, doc_hash: str) -> Optional[DocumentRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE hash = ?", (doc_hash,)
            ).fetchone()
            return DocumentRecord(**dict(row)) if row else None

    def upsert_document(
        self,
        doc_hash: str,
        filename: str,
        page_count: int,
        chunk_count: int,
        image_count: int,
    ) -> DocumentRecord:
        ingested_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO documents
                   (hash, filename, ingested_at, page_count, chunk_count, image_count)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(hash) DO UPDATE SET
                       filename=excluded.filename,
                       ingested_at=excluded.ingested_at,
                       page_count=excluded.page_count,
                       chunk_count=excluded.chunk_count,
                       image_count=excluded.image_count""",
                (doc_hash, filename, ingested_at, page_count, chunk_count, image_count),
            )
        return DocumentRecord(
            doc_hash, filename, ingested_at, page_count, chunk_count, image_count
        )

    def chunk_exists(self, chunk_hash: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE hash = ?", (chunk_hash,)
            ).fetchone()
            return row is not None

    def upsert_chunk(
        self,
        chunk_hash: str,
        document_hash: str,
        page: int,
        content_type: str,
        text: str,
        vector_store_id: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO chunks
                   (hash, document_hash, page, content_type, text, vector_store_id)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(hash) DO UPDATE SET vector_store_id=excluded.vector_store_id""",
                (chunk_hash, document_hash, page, content_type, text, vector_store_id),
            )

    def get_chunks_by_vector_ids(
        self, vector_ids: Iterable[str]
    ) -> dict[str, ChunkRecord]:
        ids = list(vector_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._connect() as conn:
            # placeholders is always internally generated "?" characters (not user input);
            # actual values are passed as parameterized args below.
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE vector_store_id IN ({placeholders})",  # nosec B608
                ids,
            ).fetchall()
            return {row["vector_store_id"]: ChunkRecord(**dict(row)) for row in rows}
