# AI RAG Production

A production-style Retrieval-Augmented Generation service built to learn (and demonstrate)
the same end-to-end workflow used by enterprise AI teams: a real RAG pipeline, a real test
suite, real CI/CD, and a real git branching workflow — not just a notebook demo.

**What it does:** upload a PDF → extract its text, OCR any scanned/image-heavy pages, caption
every diagram/figure with a vision model → chunk, embed, and index everything → ask questions
and get answers grounded in retrieved, reranked context, streamed token-by-token.

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │                 Ingestion                    │
PDF ──▶ PyMuPDF (fitz) ──┼─▶ native text layer ─────────────────────────┼─┐
                         │─▶ embedded images ──▶ vision caption ────────┼─┤
                         │─▶ low-text pages ──▶ page render ──▶ OCR ────┼─┤
                         └─────────────────────────────────────────────┘ │
                                                                          ▼
                                            chunk ─▶ SHA256 dedup ─▶ embed (text-embedding-3-small)
                                                                          │
                                                                          ▼
                                          FAISS (local) or Pinecone (cloud) — cosine similarity
                                                     + SQLite metadata registry

                         ┌─────────────────────────────────────────────┐
                         │                   Query                      │
Question ──▶ embed ──▶ vector search (top ~20) ──▶ BGE cross-encoder rerank (top_k)
                                                                          │
                                                                          ▼
                                        gpt-5-mini (Azure) ──▶ streamed, grounded answer
                                        (falls back to OpenAI if Azure rejects the call)
```

**Clients:** the FastAPI backend (`app/`) is the single source of truth. `streamlit_app.py`
is a thin UI client that talks to it over HTTP (including real SSE streaming) — exactly how
any other frontend would in production.

### Content types indexed per chunk

| `content_type`              | Source                                              |
|------------------------------|------------------------------------------------------|
| `text`                       | Native PDF text layer                                 |
| `ocr_text`                   | Tesseract OCR on scanned / low-text pages             |
| `image_caption:embedded`     | Vision-model caption of an embedded raster image       |
| `image_caption:page_render`  | Vision-model caption of a rendered page (vector diagrams with no embedded image) |

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) (dependency manager)
- Docker (for containerized runs / matching CI)
- `tesseract` binary for OCR — optional, ingestion degrades gracefully without it
  - macOS: `brew install tesseract`
  - Debian/Ubuntu: `sudo apt-get install tesseract-ocr`
- An Azure OpenAI resource with `text-embedding-3-small` and a chat/vision deployment
  (this project defaults to `gpt-5-mini` for both)
- Optional: an OpenAI API key (fallback if the Azure deployment can't handle vision/chat)
- Optional: a Pinecone API key (only needed if you switch `VECTOR_STORE_BACKEND=pinecone`)

## Setup

```bash
git clone https://github.com/<your-username>/AI-RAG-Production.git
cd AI-RAG-Production

uv sync                        # installs everything, including dev tools, from uv.lock

cp .env.example .env           # then fill in your real keys
```

## Running locally

```bash
# Terminal 1 — API
uv run uvicorn app.main:app --reload

# Terminal 2 — chat UI
uv run streamlit run streamlit_app.py
```

Or via Docker Compose (build + run the containerized API):

```bash
docker compose up --build
```

### CLI, no server needed

```bash
uv run python -m scripts.ingest data/attention_is_all_you_need.pdf
uv run python -m scripts.query "What is self-attention?"
```

## API usage

```bash
# Health check
curl http://localhost:8000/health

# Ingest a PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@data/attention_is_all_you_need.pdf"

# Ask a question (non-streaming)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What optimizer did they use?", "top_k": 4}'

# Ask a question (streaming, Server-Sent Events)
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What optimizer did they use?", "temperature": 0.7, "top_p": 1.0}'
```

Interactive docs: `http://localhost:8000/docs`

## Running checks locally (mirrors CI exactly)

```bash
uv run flake8 app tests scripts conftest.py streamlit_app.py
uv run black --check app tests scripts conftest.py streamlit_app.py
uv run pytest -q
uv run bandit -r app -q
uv run pip-audit
docker build -t ai-rag-production .
```

---

## Git workflow

This is the part worth practicing hands-on — it's the same lifecycle used at HSBC, IBM,
Accenture, TCS, and most companies running trunk-based-ish development with protected
branches. Every command below is copy-pasteable.

### Branch model

| Branch          | Purpose                                            |
|------------------|-----------------------------------------------------|
| `main`           | Production-ready. Protected. Deploys via `cd.yml`.  |
| `develop`        | Integration branch — features merge here first.     |
| `feature/<name>` | One feature/fix, branched from `develop`.           |
| `release/<ver>`  | Stabilizing a `develop` snapshot before it hits `main`. |
| `hotfix/<name>`  | Urgent prod fix, branched from `main`.               |

### 1. Clone and set up

```bash
git clone https://github.com/<your-username>/AI-RAG-Production.git
cd AI-RAG-Production
git checkout -b develop          # first time only, if develop doesn't exist yet
git push -u origin develop
```

### 2. Start a feature branch

```bash
git checkout develop
git pull origin develop          # make sure you're starting from the latest
git checkout -b feature/add-reranker
```

### 3. Work, stage, commit

```bash
git status                       # see what changed
git add app/services/reranker.py tests/test_reranker.py
git commit -m "Add BGE cross-encoder reranking to the query pipeline"
```

Small, focused commits with a message that explains *why* > one giant commit at the end.

### 4. Push and open a Pull Request

```bash
git push -u origin feature/add-reranker
gh pr create --base develop --title "Add cross-encoder reranking" --body "Improves retrieval precision by reranking the top-20 vector hits down to top_k with BAAI/bge-reranker-base."
```

(No `gh` CLI? Push and open the PR from the GitHub web UI — same result.)

### 5. Let CI run, then get it reviewed

Opening the PR triggers `.github/workflows/ci.yml` automatically: lint → format check →
tests → security scan → dependency scan → Docker build. A red check blocks merging (once
you've enabled branch protection — see below). Ask a teammate to review, or self-review by
reading your own diff on the PR page.

### 6. Merge

Three ways, pick based on what you want the history to look like:

```bash
# Merge commit — preserves full branch history (good default for feature branches)
gh pr merge --merge

# Squash — collapses the branch into one clean commit on develop (good for messy WIP history)
gh pr merge --squash

# Rebase — replays your commits on top of develop, linear history, no merge commit
gh pr merge --rebase
```

### 7. Clean up and sync

```bash
git checkout develop
git pull origin develop
git branch -d feature/add-reranker           # delete local branch
git push origin --delete feature/add-reranker # delete remote branch
```

### 8. Handling a merge conflict

```bash
git checkout feature/add-reranker
git fetch origin
git merge origin/develop          # or: git rebase origin/develop
# fix conflicts in the flagged files
git add <fixed-files>
git commit                        # (merge) or `git rebase --continue` (rebase)
git push
```

### 9. Release: develop → main

```bash
git checkout develop
git pull origin develop
git checkout -b release/1.1.0
# final fixes/version bumps only, no new features
git push -u origin release/1.1.0
gh pr create --base main --title "Release 1.1.0"
gh pr merge --merge                # merging to main triggers cd.yml
git checkout develop
git merge main                     # bring the release commit back into develop
git push origin develop
```

### 10. Hotfix: urgent fix straight to main

```bash
git checkout main
git pull origin main
git checkout -b hotfix/fix-upload-crash
# fix, commit, push
gh pr create --base main --title "Hotfix: fix /upload crash on empty PDF"
gh pr merge --merge
git checkout develop
git merge main                     # don't let main and develop drift apart
git push origin develop
```

### Enabling branch protection (do this once, in the GitHub UI)

Repo → Settings → Branches → Add rule for `main` (and `develop`):
- Require a pull request before merging
- Require status checks to pass before merging → select the `lint-test-build` job from `ci.yml`
- Require branches to be up to date before merging

---

## CI/CD

### `ci.yml` — runs on every PR and every push to `develop`/`feature/**`/`release/**`/`hotfix/**`

Checkout → install `uv` → sync deps → **flake8** (lint) → **black --check** (format) →
**pytest** (unit tests) → **bandit** (security scan for hardcoded secrets, unsafe code) →
**pip-audit** (known-CVE dependency scan, informational) → **docker build** (build-only,
not pushed). Any hard failure blocks the PR once branch protection is enabled.

### `cd.yml` — runs on every push to `main` (i.e. after a PR merges)

Builds and tags the Docker image, pushes it to GHCR (`ghcr.io/<owner>/<repo>`, auth'd via
the workflow's built-in token — no extra secrets needed), smoke-tests the built container
locally in the runner (`docker run` + poll `/health`), then runs three sequential deploy
jobs (`development` → `staging` → `production`) gated by
[GitHub Environments](https://docs.github.com/en/actions/how-tos/manage-deployments/configure-and-manage-deployment-environments) —
add required reviewers on `staging`/`production` in repo settings to get real manual
approval gates. The deploy steps themselves are placeholders (`# TODO`) until you have
actual cloud infra (Azure Web App, ECS, Kubernetes, ...) to point them at.

### Rollback

```bash
# Kubernetes
kubectl rollout undo deployment/ai-rag-production

# Or just redeploy the previous image tag from GHCR — every push to main is tagged
# with its short commit SHA, so any prior version is always addressable.
```

---

## Configuration reference

All variables live in `.env` (copy from `.env.example`). Full list with defaults:

| Variable | Default | Notes |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` | — | Required |
| `AZURE_OPENAI_API_VERSION` | `2024-02-01` | Bump if your deployment needs a newer version for vision — see Troubleshooting |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | `text-embedding-3-small` | |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` / `AZURE_OPENAI_VISION_DEPLOYMENT` | `gpt-5-mini` | |
| `OPENAI_API_KEY` | — | Fallback only, used if Azure rejects a vision/chat call |
| `OPENAI_VISION_MODEL` / `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | |
| `VECTOR_STORE_BACKEND` | `faiss` | `faiss` or `pinecone` |
| `EMBEDDING_DIMENSION` | `1536` | Must match your embedding model |
| `FAISS_INDEX_DIR` / `FAISS_INDEX_NAME` | `embeddings` / `faiss_index` | |
| `PINECONE_API_KEY` / `PINECONE_INDEX_NAME` / `PINECONE_CLOUD` / `PINECONE_REGION` | — | Only used when backend is `pinecone` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | |
| `MIN_PAGE_TEXT_CHARS_FOR_PAGE_RENDER` | `40` | Below this, a page also gets rendered + OCR'd |
| `TOP_K` | `4` | Final number of chunks used to answer |
| `OCR_ENABLED` | `true` | Requires the `tesseract` binary |
| `LLM_TEMPERATURE` / `LLM_TOP_P` | `0.7` / `1.0` | Overridable per-request via the API |
| `RERANKER_ENABLED` | `true` | |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Local cross-encoder, no API key, downloads weights on first use |
| `RERANKER_CANDIDATE_POOL` | `20` | Vector hits fetched before reranking down to `TOP_K` |
| `COHERE_API_KEY` / `JINA_API_KEY` | — | Reference only — see `app/services/reranker.py` for commented alternative rerankers |
| `API_BASE_URL` | `http://localhost:8000` | Read only by `streamlit_app.py` |

## Switching vector store backend

```bash
# .env
VECTOR_STORE_BACKEND=faiss     # local, file-based, free — good for dev
VECTOR_STORE_BACKEND=pinecone  # cloud, needs PINECONE_API_KEY — good for prod
```

Both are hidden behind the same `VectorStore` interface (`app/services/vector_store/`), so
nothing else in the pipeline changes. FAISS uses `IndexFlatIP` over L2-normalized vectors
(cosine similarity via the inner-product trick); Pinecone's index is created natively with
`metric="cosine"`.

## Troubleshooting

**Vision/chat calls always fall back to OpenAI, never succeed on Azure.** Your
`gpt-5-mini` deployment may need a newer `AZURE_OPENAI_API_VERSION` to accept image input
or non-default `temperature`/`top_p` — try bumping it (e.g. `2024-12-01-preview`) or check
the Azure deployment's supported parameters in the portal.

**Segfault on macOS when running tests or the API.** FAISS and PyTorch (pulled in by the
reranker) bundle conflicting OpenMP runtimes on macOS. Already worked around in
`app/__init__.py` / `conftest.py` (`KMP_DUPLICATE_LIB_OK`, `OMP_NUM_THREADS=1`) and
`faiss_store.py` (`faiss.omp_set_num_threads(1)`) — if you still hit it, confirm those are
actually being imported before any other `faiss`/`torch` import in your entry point.

**First query is slow.** The reranker downloads `BAAI/bge-reranker-base` from Hugging Face
on first use (a few hundred MB) and caches it locally afterward. Set `RERANKER_ENABLED=false`
to skip reranking entirely.

**Rebuilding the FAISS index from scratch.** Delete `embeddings/faiss_index.index` and
`embeddings/faiss_index.meta.json`, then re-run ingestion. (The SQLite dedup registry at
`data/metadata.db` also needs clearing if you want chunks fully re-embedded, not just
re-indexed.)

**Creating the Pinecone index.** You don't need to — `PineconeStore` creates it
automatically (serverless, `metric="cosine"`) on first use if it doesn't already exist.

**OCR silently does nothing.** Confirm `tesseract` is on `PATH` (`which tesseract`).
Ingestion is designed to degrade gracefully (log a warning, continue) rather than fail if
it's missing.
