import logging
from collections.abc import Iterator

from langsmith import traceable
from openai import AzureOpenAI, OpenAI

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided context "
    "extracted from a document. If the context doesn't contain the answer, say so plainly "
    "instead of guessing. Cite the page number(s) you used when possible."
)


def _build_user_prompt(question: str, context_chunks: list[str]) -> str:
    context = (
        "\n\n---\n\n".join(context_chunks)
        if context_chunks
        else "(no matching context found)"
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def _messages(question: str, context_chunks: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(question, context_chunks)},
    ]


def _resolved_params(
    temperature: float | None, top_p: float | None
) -> tuple[float, float]:
    settings = get_settings()
    return (
        settings.llm_temperature if temperature is None else temperature,
        settings.llm_top_p if top_p is None else top_p,
    )


def _answer_with_azure(
    question: str, context_chunks: list[str], temperature: float, top_p: float
) -> str:
    settings = get_settings()
    client = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )
    response = client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=_messages(question, context_chunks),
        max_completion_tokens=800,
        temperature=temperature,
        top_p=top_p,
    )
    return (response.choices[0].message.content or "").strip()


def _answer_with_openai(
    question: str, context_chunks: list[str], temperature: float, top_p: float
) -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set — cannot fall back for answer generation"
        )
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=_messages(question, context_chunks),
        max_tokens=8000,
        temperature=temperature,
        top_p=top_p,
    )
    return (response.choices[0].message.content or "").strip()


@traceable(name="generate_answer", run_type="llm")
def generate_answer(
    question: str,
    context_chunks: list[str],
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Synthesize a final answer from retrieved chunks via the Azure chat deployment
    (default: gpt-5-mini), falling back to OpenAI directly on failure."""
    temperature, top_p = _resolved_params(temperature, top_p)
    try:
        return _answer_with_azure(question, context_chunks, temperature, top_p)
    except Exception as exc:
        logger.warning("Azure chat completion failed (%s); falling back to OpenAI", exc)
        return _answer_with_openai(question, context_chunks, temperature, top_p)


def _stream_delta_tokens(stream) -> Iterator[str]:
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _stream_with_azure(
    question: str, context_chunks: list[str], temperature: float, top_p: float
) -> Iterator[str]:
    settings = get_settings()
    client = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )
    stream = client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=_messages(question, context_chunks),
        max_completion_tokens=800,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )
    yield from _stream_delta_tokens(stream)


def _stream_with_openai(
    question: str, context_chunks: list[str], temperature: float, top_p: float
) -> Iterator[str]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set — cannot fall back for answer generation"
        )
    client = OpenAI(api_key=settings.openai_api_key)
    stream = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=_messages(question, context_chunks),
        max_tokens=800,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )
    yield from _stream_delta_tokens(stream)


@traceable(name="generate_answer_stream", run_type="llm")
def generate_answer_stream(
    question: str,
    context_chunks: list[str],
    temperature: float | None = None,
    top_p: float | None = None,
) -> Iterator[str]:
    """Streaming counterpart to generate_answer, yielding answer text token-by-token.

    Note on fallback: if the Azure stream fails after already yielding some tokens to
    the caller, the except block below still falls back and restarts from OpenAI —
    acceptable for this project (a rare mid-stream hiccup, not silent data loss), but
    worth knowing if you see a stream "restart" once in a while.
    """
    temperature, top_p = _resolved_params(temperature, top_p)
    try:
        yield from _stream_with_azure(question, context_chunks, temperature, top_p)
    except Exception as exc:
        logger.warning("Azure streaming chat failed (%s); falling back to OpenAI", exc)
        yield from _stream_with_openai(question, context_chunks, temperature, top_p)
