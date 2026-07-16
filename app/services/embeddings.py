import numpy as np
from langchain_openai import AzureOpenAIEmbeddings

from app.config.settings import get_settings


def get_embeddings_client() -> AzureOpenAIEmbeddings:
    settings = get_settings()
    return AzureOpenAIEmbeddings(
        azure_deployment=settings.azure_openai_embedding_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize so inner product == cosine similarity (the FAISS IndexFlatIP trick)."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms


def embed_texts(texts: list[str]) -> np.ndarray:
    client = get_embeddings_client()
    vectors = client.embed_documents(texts)
    return normalize(np.array(vectors, dtype="float32"))


def embed_query(text: str) -> np.ndarray:
    client = get_embeddings_client()
    vector = client.embed_query(text)
    return normalize(np.array([vector], dtype="float32"))[0]
