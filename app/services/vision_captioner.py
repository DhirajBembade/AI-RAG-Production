import base64
import logging
from pathlib import Path

from openai import AzureOpenAI, OpenAI

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "Describe this image from a technical document in 2-4 sentences. "
    "If it is a diagram, chart, or architecture figure, explain what it depicts, "
    "including labeled components and relationships. Be concise and factual."
)


def _image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lstrip(".") or "png"
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/{suffix};base64,{b64}"


def _vision_messages(image_path: Path) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_url(image_path)},
                },
            ],
        }
    ]


def _caption_with_azure(image_path: Path) -> str:
    settings = get_settings()
    client = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )
    response = client.chat.completions.create(
        model=settings.azure_openai_vision_deployment,
        messages=_vision_messages(image_path),
        max_completion_tokens=300,
    )
    return (response.choices[0].message.content or "").strip()


def _caption_with_openai(image_path: Path) -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set — cannot fall back for image captioning"
        )
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_vision_model,
        messages=_vision_messages(image_path),
        max_tokens=300,
    )
    return (response.choices[0].message.content or "").strip()


def caption_image(image_path: Path) -> str:
    """Caption an image via the Azure OpenAI vision deployment (default: gpt-5-mini).
    Falls back to OpenAI directly if the Azure deployment rejects the request, e.g.
    because that deployment doesn't accept image input."""
    try:
        return _caption_with_azure(image_path)
    except Exception as exc:
        logger.warning(
            "Azure vision captioning failed for %s (%s); falling back to OpenAI",
            image_path,
            exc,
        )
        return _caption_with_openai(image_path)
