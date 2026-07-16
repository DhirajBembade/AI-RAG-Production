import logging

import numpy as np
from pinecone import Pinecone, ServerlessSpec

from app.config.settings import get_settings
from app.services.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


class PineconeStore(VectorStore):
    """Cosine similarity via a Pinecone serverless index (metric="cosine" is native)."""

    def __init__(self):
        settings = get_settings()
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")

        self._client = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index_name

        existing = {idx.name for idx in self._client.list_indexes()}
        if self._index_name not in existing:
            logger.info("Creating Pinecone index %s", self._index_name)
            self._client.create_index(
                name=self._index_name,
                dimension=settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud, region=settings.pinecone_region
                ),
            )

        self._index = self._client.Index(self._index_name)

    def upsert(
        self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]
    ) -> None:
        payload = [
            {"id": id_, "values": vector.tolist(), "metadata": metadata}
            for id_, vector, metadata in zip(ids, vectors, metadatas)
        ]
        self._index.upsert(vectors=payload)

    def query(
        self,
        vector: np.ndarray,
        top_k: int,
        filter_metadata: dict[str, str] | None = None,
    ) -> list[tuple[str, float, dict]]:
        query_filter = (
            {key: {"$eq": value} for key, value in filter_metadata.items()}
            if filter_metadata
            else None
        )
        response = self._index.query(
            vector=vector.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=query_filter,
        )
        return [
            (match.id, float(match.score), match.metadata or {})
            for match in response.matches
        ]
