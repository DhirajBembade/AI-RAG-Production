from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import get_settings


def chunk_text(text: str) -> list[str]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]
