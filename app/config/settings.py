from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Azure OpenAI (embeddings, chat, vision) ---
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    azure_openai_chat_deployment: str = "gpt-5-mini"
    azure_openai_vision_deployment: str = "gpt-5-mini"

    # --- OpenAI fallback (used only if the Azure vision/chat deployment can't handle it) ---
    openai_api_key: str = ""
    openai_vision_model: str = "gpt-4o-mini"
    openai_chat_model: str = "gpt-4o-mini"

    # --- Vector store ---
    vector_store_backend: str = "faiss"  # "faiss" | "pinecone"
    embedding_dimension: int = 1536  # text-embedding-3-small

    faiss_index_dir: Path = BASE_DIR / "embeddings"
    faiss_index_name: str = "faiss_index"

    pinecone_api_key: str = ""
    pinecone_index_name: str = "ai-rag-production"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- Ingestion / chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_page_text_chars_for_page_render: int = 40
    top_k: int = 4
    ocr_enabled: bool = (
        True  # requires the `tesseract` binary on PATH; degrades gracefully
    )

    # --- Answer generation ---
    llm_temperature: float = 0.7
    llm_top_p: float = 1.0

    # --- Reranking (second-stage precision boost over the initial vector search) ---
    reranker_enabled: bool = True
    reranker_model: str = (
        "BAAI/bge-reranker-base"  # open-source cross-encoder, runs locally
    )
    reranker_candidate_pool: int = (
        20  # how many vector-search hits to feed into the reranker
    )

    # Reference only (see app/services/reranker.py for commented alternative backends)
    cohere_api_key: str = ""
    jina_api_key: str = ""

    # --- Storage ---
    data_dir: Path = BASE_DIR / "data"
    extracted_images_dir: Path = BASE_DIR / "data" / "extracted_images"
    metadata_db_path: Path = BASE_DIR / "data" / "metadata.db"
    logs_dir: Path = BASE_DIR / "logs"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.faiss_index_dir.mkdir(parents=True, exist_ok=True)
    settings.extracted_images_dir.mkdir(parents=True, exist_ok=True)
    settings.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    return settings
