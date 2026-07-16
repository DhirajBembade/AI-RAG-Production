"""RAGAS evaluation harness: runs the real retrieval+generation pipeline against a
small starter question set (data/eval_questions.json) and scores it on faithfulness,
answer relevancy, context precision, and context recall.

Run with: uv run python -m scripts.evaluate
Requires data/attention_is_all_you_need.pdf to already be ingested — run
`uv run python -m scripts.ingest data/attention_is_all_you_need.pdf` first.
"""

import json
import sys
import types
from pathlib import Path


def _patch_langchain_community_vertexai_shim() -> None:
    """ragas hard-imports langchain_community.chat_models.vertexai.ChatVertexAI at
    module load time, purely for an isinstance() check further down in that module.
    That submodule was removed from newer langchain-community releases (VertexAI
    support moved to a standalone package this project doesn't use). Inject a harmless
    stand-in so `import ragas` succeeds without pulling in unrelated Google Cloud deps.
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    shim = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - never instantiated, isinstance-check only
        pass

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim


_patch_langchain_community_vertexai_shim()

from datasets import Dataset  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.config.settings import get_settings  # noqa: E402
from app.services.embeddings import get_embeddings_client  # noqa: E402
from app.services.llm import generate_answer  # noqa: E402
from app.services.rag_pipeline import retrieve  # noqa: E402

EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_questions.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_results.json"


def build_eval_dataset() -> Dataset:
    eval_items = json.loads(EVAL_SET_PATH.read_text())

    rows = {"user_input": [], "retrieved_contexts": [], "response": [], "reference": []}
    for item in eval_items:
        question = item["question"]
        sources = retrieve(question)
        contexts = [source.text for source in sources] or [""]
        answer = generate_answer(question, contexts)

        rows["user_input"].append(question)
        rows["retrieved_contexts"].append(contexts)
        rows["response"].append(answer)
        rows["reference"].append(item["reference"])

    return Dataset.from_dict(rows)


def main() -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required to run evaluation: gpt-5-mini (Azure) only accepts "
            "its default temperature=1, but RAGAS's internal judge prompts need near-zero "
            "temperature for deterministic scoring. OpenAI's gpt-4o-mini is used as the "
            "judge model here — independent of whatever model serves live traffic."
        )
    judge_llm = ChatOpenAI(
        model=settings.openai_chat_model, api_key=settings.openai_api_key
    )
    judge_embeddings = get_embeddings_client()

    print(f"Building eval dataset from {EVAL_SET_PATH}")
    print("(runs real retrieval + generation for each question)...")
    dataset = build_eval_dataset()

    print(
        "Scoring with RAGAS (faithfulness, answer_relevancy, context_precision, recall)..."
    )
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print("\nAggregate scores:")
    print(result)

    RESULTS_PATH.write_text(json.dumps(result.scores, indent=2))
    print(f"\nPer-question scores saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
