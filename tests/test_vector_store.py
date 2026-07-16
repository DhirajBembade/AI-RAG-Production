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


def test_faiss_store_query_filters_by_metadata(tmp_path):
    store = FaissStore(index_dir=tmp_path, index_name="filter_test", dimension=4)

    vectors = _normalize(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # doc1, exact match
                [0.9, 0.1, 0.0, 0.0],  # doc2, would rank #2 overall if unfiltered
                [0.5, 0.5, 0.0, 0.0],  # doc1, weaker match
                [0.0, 1.0, 0.0, 0.0],  # doc2, orthogonal
            ],
            dtype="float32",
        )
    )
    ids = ["a1", "b2", "c1", "d2"]
    metadatas = [
        {"text": "a1", "document_hash": "doc1"},
        {"text": "b2", "document_hash": "doc2"},
        {"text": "c1", "document_hash": "doc1"},
        {"text": "d2", "document_hash": "doc2"},
    ]
    store.upsert(ids=ids, vectors=vectors, metadatas=metadatas)

    query_vector = _normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"))
    results = store.query(
        query_vector, top_k=2, filter_metadata={"document_hash": "doc1"}
    )

    # doc2's "b2" would rank #2 overall (0.9 sim > 0.5 sim of "c1"), but the filter
    # must exclude it entirely rather than just deprioritize it.
    result_ids = [r[0] for r in results]
    assert result_ids == ["a1", "c1"]
    assert all(r[2]["document_hash"] == "doc1" for r in results)


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
