"""Chat UI for the AI RAG Production API.

Deliberately a thin client: all retrieval/generation logic lives in the FastAPI
backend (app/). This talks to it over HTTP only (including real token-by-token
streaming via SSE), the same way any other frontend would in production.

Run with: uv run streamlit run streamlit_app.py
Requires the API to be running separately: uv run uvicorn app.main:app --reload
"""

import json
import os
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(page_title="AI RAG Production", page_icon="📚", layout="wide")

CONTENT_TYPE_LABELS = {
    "text": "📄 text",
    "ocr_text": "🔎 OCR",
}


def badge_for(content_type: str) -> str:
    if content_type.startswith("image_caption"):
        return "🖼️ image caption"
    return CONTENT_TYPE_LABELS.get(content_type, content_type)


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for source in sources:
            st.markdown(
                f"**{badge_for(source['content_type'])}** · "
                f"`{source['filename']}` p.{source['page']} · score={source['score']:.3f}"
            )
            st.caption(source["text"][:300] or "(empty)")
            image_path = source.get("image_path")
            if image_path and Path(image_path).exists():
                st.image(image_path, width=240)
            st.divider()


# --- Sidebar: connection, ingestion, generation controls ---

with st.sidebar:
    st.header("Settings")
    api_base_url = st.text_input(
        "API base URL", value=os.environ.get("API_BASE_URL", "http://localhost:8000")
    )

    st.divider()
    st.subheader("Ingest a PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF", type=["pdf"], label_visibility="collapsed"
    )
    if uploaded_file is not None and st.button("Ingest", use_container_width=True):
        with st.status("Ingesting...", expanded=True) as status:
            try:
                response = requests.post(
                    f"{api_base_url}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=600,
                )
                response.raise_for_status()
                result = response.json()
                label = (
                    "Already ingested (duplicate)"
                    if result["skipped_duplicate"]
                    else "Ingested"
                )
                status.update(label=label, state="complete")
                st.json(result)
            except Exception as exc:
                status.update(label="Ingestion failed", state="error")
                st.error(str(exc))

    st.divider()
    st.subheader("Generation")
    top_k = st.slider("top_k (chunks retrieved)", 1, 10, 4)
    temperature = st.slider("temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("top_p", 0.0, 1.0, 1.0, 0.05)

st.title("📚 AI RAG Production")
st.caption(
    "Ask a question about the PDFs you've ingested. Retrieval → cross-encoder rerank → "
    "streamed answer, exactly as the backend runs it."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer_text = ""
        sources: list[dict] = []
        try:
            response = requests.post(
                f"{api_base_url}/chat/stream",
                json={
                    "question": question,
                    "top_k": top_k,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                stream=True,
                timeout=300,
            )
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = json.loads(line[len("data: ") :])
                if payload["type"] == "sources":
                    sources = payload["sources"]
                elif payload["type"] == "token":
                    answer_text += payload["text"]
                    placeholder.markdown(answer_text + "▌")
                elif payload["type"] == "error":
                    answer_text = f"Error: {payload['detail']}"
                    break
            placeholder.markdown(answer_text or "(no answer)")
        except Exception as exc:
            answer_text = f"Request failed: {exc}"
            placeholder.markdown(answer_text)

        if sources:
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": sources}
    )
