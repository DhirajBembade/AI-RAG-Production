import numpy as np

from app.services.vector_store.faiss_store import FaissStore


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def test_faiss_store_upsert_and_query_orders_by_cosine_similarity(tmp_path):
    store = FaissStore(index_dir=tmp_path, index_name="test_index", dimension=4)

    vectors = _normalize(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
            ],
            dtype="float32",
        )
    )
    ids = ["chunk-a", "chunk-b", "chunk-c"]
    metadatas = [{"text": "a"}, {"text": "b"}, {"text": "c"}]

    store.upsert(ids=ids, vectors=vectors, metadatas=metadatas)

    query_vector = _normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"))
    results = store.query(query_vector, top_k=3)

    result_ids = [r[0] for r in results]
    assert result_ids[0] == "chunk-a"  # exact match, highest cosine similarity
    assert result_ids[1] == "chunk-c"  # close second, mostly aligned
    assert result_ids[2] == "chunk-b"  # orthogonal, lowest similarity

    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_faiss_store_persists_and_reloads(tmp_path):
    dimension = 4
    vectors = _normalize(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))

    store = FaissStore(
        index_dir=tmp_path, index_name="persist_test", dimension=dimension
    )
    store.upsert(ids=["only-chunk"], vectors=vectors, metadatas=[{"text": "hello"}])

    reloaded = FaissStore(
        index_dir=tmp_path, index_name="persist_test", dimension=dimension
    )
    results = reloaded.query(vectors[0], top_k=1)

    assert len(results) == 1
    assert results[0][0] == "only-chunk"
    assert results[0][2]["text"] == "hello"
