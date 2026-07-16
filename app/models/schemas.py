from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class IngestResponse(BaseModel):
    document_hash: str
    filename: str
    page_count: int
    chunk_count: int
    image_count: int
    skipped_duplicate: bool


class ChatRequest(BaseModel):
    question: str
    top_k: int | None = None
    temperature: float | None = None
    top_p: float | None = None


class SourceChunkResponse(BaseModel):
    text: str
    page: int
    filename: str
    content_type: str
    score: float
    image_path: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunkResponse]
