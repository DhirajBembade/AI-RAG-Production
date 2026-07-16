import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

from app.services.vector_store.base import VectorStore

# Pinned to 1 thread: avoids an OpenMP runtime collision with torch (pulled in by the
# sentence-transformers reranker) that otherwise segfaults on macOS — see app/__init__.py.
faiss.omp_set_num_threads(1)


def _id_to_int64(id_str: str) -> int:
    """Deterministically map an arbitrary string id to a positive int64, since FAISS
    IndexIDMap only accepts integer ids (-1 is reserved for "not found")."""
    digest = hashlib.sha256(id_str.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, byteorder="big", signed=False) >> 1


class FaissStore(VectorStore):
    """Cosine similarity via IndexFlatIP over L2-normalized vectors, with a JSON sidecar
    mapping FAISS's internal int64 ids back to our string chunk ids + metadata."""

    def __init__(self, index_dir: Path, index_name: str, dimension: int):
        self._index_path = index_dir / f"{index_name}.index"
        self._meta_path = index_dir / f"{index_name}.meta.json"
        self._dimension = dimension
        self._id_to_metadata: dict[int, dict] = {}
        self._int_to_str_id: dict[int, str] = {}

        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            sidecar = json.loads(self._meta_path.read_text())
            self._id_to_metadata = {int(k): v["metadata"] for k, v in sidecar.items()}
            self._int_to_str_id = {int(k): v["id"] for k, v in sidecar.items()}
        else:
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

    def upsert(
        self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]
    ) -> None:
        int_ids = np.array([_id_to_int64(i) for i in ids], dtype="int64")
        self._index.add_with_ids(vectors.astype("float32"), int_ids)
        for int_id, str_id, meta in zip(int_ids.tolist(), ids, metadatas):
            self._id_to_metadata[int_id] = meta
            self._int_to_str_id[int_id] = str_id
        self._persist()

    def query(
        self,
        vector: np.ndarray,
        top_k: int,
        filter_metadata: dict[str, str] | None = None,
    ) -> list[tuple[str, float, dict]]:
        if self._index.ntotal == 0:
            return []

        # IndexFlatIP is already an exact brute-force search, so filtering just means:
        # score everything, keep results in score order, drop non-matches, take top_k.
        # Fine at FAISS's typical local/dev scale; avoids under-fetching a smaller
        # candidate window and missing a valid match.
        search_k = self._index.ntotal if filter_metadata else top_k
        scores, int_ids = self._index.search(
            vector.reshape(1, -1).astype("float32"), search_k
        )

        results = []
        for score, int_id in zip(scores[0], int_ids[0]):
            if int_id == -1:
                continue
            metadata = self._id_to_metadata[int_id]
            if filter_metadata and not all(
                metadata.get(k) == v for k, v in filter_metadata.items()
            ):
                continue
            results.append((self._int_to_str_id[int_id], float(score), metadata))
            if len(results) >= top_k:
                break
        return results

    def _persist(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        sidecar = {
            str(int_id): {
                "id": self._int_to_str_id[int_id],
                "metadata": self._id_to_metadata[int_id],
            }
            for int_id in self._id_to_metadata
        }
        self._meta_path.write_text(json.dumps(sidecar))
