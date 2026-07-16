from app.config.settings import get_settings
from app.services.vector_store.base import VectorStore
from app.services.vector_store.faiss_store import FaissStore

__all__ = ["VectorStore", "FaissStore", "get_vector_store"]


def get_vector_store() -> VectorStore:
    settings = get_settings()
    if settings.vector_store_backend == "pinecone":
        from app.services.vector_store.pinecone_store import PineconeStore

        return PineconeStore()
    return FaissStore(
        index_dir=settings.faiss_index_dir,
        index_name=settings.faiss_index_name,
        dimension=settings.embedding_dimension,
    )
