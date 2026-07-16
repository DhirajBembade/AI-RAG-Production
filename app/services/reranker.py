"""Second-stage reranking over the vector store's initial candidate set.

Vector similarity (cosine over independently-embedded query/chunk vectors) is fast but
approximate — it can't model interactions between the query and each candidate. A
cross-encoder reranker reads the query and each candidate *together* and scores their
relevance directly, which is much slower per-pair but far more precise. Standard RAG
pattern: over-fetch from the vector store (top ~20), then rerank down to the final top_k.

Default backend here is BAAI/bge-reranker-base, an open-source cross-encoder that runs
locally via sentence-transformers — no API key required. Alternative managed rerankers
(Cohere, Jina AI) are sketched at the bottom as reference code, not wired up.
"""

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

Candidate = tuple[str, float, dict]


@lru_cache
def _get_cross_encoder() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.reranker_model)


def rerank(
    question: str,
    candidates: list[Candidate],
    top_k: int,
    enabled: bool | None = None,
) -> list[Candidate]:
    """Re-score (id, vector_score, metadata) candidates with a cross-encoder and return
    the top_k, ordered best match first, with vector_score replaced by the cross-encoder
    relevance score."""
    if not candidates:
        return []

    settings = get_settings()
    if enabled is None:
        enabled = settings.reranker_enabled
    if not enabled:
        return candidates[:top_k]

    model = _get_cross_encoder()
    pairs = [(question, metadata.get("text", "")) for _, _, metadata in candidates]
    scores = model.predict(pairs)

    reranked = [
        (id_, float(score), metadata)
        for (id_, _, metadata), score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked[:top_k]


# ---------------------------------------------------------------------------------
# Alternative reranker backends — reference only, not wired into the pipeline.
# Swap the call in rag_pipeline.query() to use one of these instead if you'd rather
# use a managed API than run BGE locally.
# ---------------------------------------------------------------------------------

# --- Cohere Rerank (managed API, requires COHERE_API_KEY) ---
#
# import cohere
#
# def rerank_with_cohere(question: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
#     settings = get_settings()
#     co = cohere.Client(api_key=settings.cohere_api_key)
#     docs = [metadata.get("text", "") for _, _, metadata in candidates]
#     response = co.rerank(model="rerank-english-v3.0", query=question, documents=docs, top_n=top_k)
#     return [
#         (candidates[r.index][0], r.relevance_score, candidates[r.index][2])
#         for r in response.results
#     ]

# --- Jina AI Reranker (managed API, requires JINA_API_KEY) ---
#
# import requests
#
# def rerank_with_jina(question: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
#     settings = get_settings()
#     docs = [metadata.get("text", "") for _, _, metadata in candidates]
#     response = requests.post(
#         "https://api.jina.ai/v1/rerank",
#         headers={"Authorization": f"Bearer {settings.jina_api_key}"},
#         json={
#             "model": "jina-reranker-v2-base-multilingual",
#             "query": question,
#             "documents": docs,
#             "top_n": top_k,
#         },
#         timeout=10,
#     )
#     response.raise_for_status()
#     results = response.json()["results"]
#     return [
#         (candidates[r["index"]][0], r["relevance_score"], candidates[r["index"]][2])
#         for r in results
#     ]

# --- Smaller/faster open-source alternative to BGE (same CrossEncoder API) ---
#
# CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
