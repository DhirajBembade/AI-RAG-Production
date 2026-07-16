from app.services.reranker import rerank


def test_rerank_disabled_passes_through_truncated_to_top_k():
    # enabled=False takes the passthrough path — no cross-encoder model is loaded,
    # so this stays fast and offline (no download, no CI/network dependency).
    candidates = [
        ("a", 0.9, {"text": "a"}),
        ("b", 0.8, {"text": "b"}),
        ("c", 0.7, {"text": "c"}),
    ]

    result = rerank("question", candidates, top_k=2, enabled=False)

    assert [item[0] for item in result] == ["a", "b"]


def test_rerank_with_no_candidates_returns_empty_list():
    assert rerank("question", [], top_k=5) == []
