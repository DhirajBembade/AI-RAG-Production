import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config.settings import get_settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    SourceChunkResponse,
)
from app.services.guardrails import (
    PromptInjectionDetected,
    check_prompt_injection,
    redact_pii,
)
from app.services.llm import generate_answer_stream
from app.services.rag_pipeline import ingest, query, retrieve

logger = logging.getLogger(__name__)
router = APIRouter()


def _apply_input_guardrails(question: str) -> str:
    """Prompt-injection check + PII redaction on the incoming question. Raises
    HTTPException (400) if the question is blocked; otherwise returns the (possibly
    PII-redacted) question to use for retrieval and generation."""
    settings = get_settings()

    if settings.guardrails_prompt_injection_enabled:
        try:
            check_prompt_injection(question)
        except PromptInjectionDetected as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if settings.guardrails_pii_redaction_enabled:
        redacted, pii_types = redact_pii(question)
        if pii_types:
            logger.warning("Redacted PII types %s from incoming question", pii_types)
        return redacted

    return question


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="OK")


@router.post("/upload", response_model=IngestResponse)
async def upload(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        result = ingest(tmp_path, filename=file.filename)
    except Exception as exc:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(**asdict(result))


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    question = _apply_input_guardrails(request.question)

    try:
        result = query(
            question,
            top_k=request.top_k,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except Exception as exc:
        logger.exception("Query failed for question=%r", question)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    return ChatResponse(
        answer=result.answer,
        sources=[SourceChunkResponse(**asdict(source)) for source in result.sources],
    )


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Server-Sent Events variant of /chat: emits a `sources` event once retrieval is
    done, then a `token` event per streamed answer chunk, then a `done` event."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    question = _apply_input_guardrails(request.question)

    def event_stream():
        try:
            sources = retrieve(question, top_k=request.top_k)
            yield _sse(
                {"type": "sources", "sources": [asdict(source) for source in sources]}
            )

            for token in generate_answer_stream(
                question,
                [source.text for source in sources],
                temperature=request.temperature,
                top_p=request.top_p,
            ):
                yield _sse({"type": "token", "text": token})

            yield _sse({"type": "done"})
        except Exception as exc:
            logger.exception("Streaming query failed for question=%r", question)
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
