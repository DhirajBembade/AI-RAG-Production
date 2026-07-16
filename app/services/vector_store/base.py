from abc import ABC, abstractmethod

import numpy as np


class VectorStore(ABC):
    """Cosine-similarity vector store abstraction. Implementations must pre/post-normalize
    as needed so that `query` returns cosine similarity scores, best match first."""

    @abstractmethod
    def upsert(
        self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]
    ) -> None: ...

    @abstractmethod
    def query(
        self,
        vector: np.ndarray,
        top_k: int,
        filter_metadata: dict[str, str] | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Return [(id, cosine_similarity_score, metadata), ...] ordered best match first.
        filter_metadata, if given, is an exact-match AND filter over stored metadata
        (e.g. {"document_hash": "..."} to scope results to a single document)."""
        ...
