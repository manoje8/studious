# Medici - Agentic RAG Pipeline

> A production-grade Retrieval-Augmented Generation system orchestrated by a LangGraph state machine. Supports multi-turn memory, hybrid search, self-correcting retrieval loops, and multi-format document ingestion.

---

## What It Does

**Medici** routes every user query through a chain of specialised LLM agents:

1. **QueryRewriter** - resolves coreferences from Redis session history before routing.
2. **RouterAgent** - classifies the query into one of 8 categories: `factual`, `comparative`, `analytical`, `summarization`, `procedural`, `clarification`, `chitchat`, `meta`.
3. **PlannerAgent** - decomposes complex queries into 2–4 focused sub-questions.
4. **RetrievalAgent** - runs hybrid search (dense + BM25 + RRF + FlashRank re-ranking), then self-evaluates: `sufficient | refine_query | expand_search | exhausted`. Loops up to `MAX_RETRIEVAL_ROUND` times per sub-question; a global hard cap of 6 steps prevents runaway loops.
5. **GraderAgent** - post-filters retrieved chunks for relevance before synthesis; uses an `asyncio.Semaphore` to cap concurrent LLM calls.
6. **SynthesizerAgent** - selects from 7 category-specific prompt strategies, wraps context in `<retrieved_context>` XML tags to block prompt injection, and enforces source citations.

Simple queries (`chitchat`, `meta`) short-circuit directly to a response. Low-complexity `factual` queries go straight to synthesis without planning.

---

## Tech Stack

| Concern                                  | Technology                                                               |
|------------------------------------------|--------------------------------------------------------------------------|
| API                                      | FastAPI + Uvicorn                                                        |
| Graph Orchestration                      | LangGraph `StateGraph`                                                   |
| Graph Checkpointing                      | LangGraph PostgreSQL checkpointer (psycopg async pool)                   |
| LLM - routing / grading / retrieval eval | Groq `llama-3.3-70b-versatile`                                           |
| LLM - planning / rewriting / expansion   | Google Gemini `gemini-2.0-flash`                                         |
| LLM - additional provider                | Cerebras `llama3.1-70b`                                                  |
| Vector Store                             | Qdrant (async, cosine similarity, UUID5 idempotent IDs)                  |
| Sparse Search                            | BM25 (`rank-bm25`), rebuilt in-memory at startup                         |
| Fusion                                   | Reciprocal Rank Fusion (RRF)                                             |
| Re-ranking                               | FlashRank cross-encoder (CPU-friendly, lazy-initialised)                 |
| Document Parsing                         | Google Document AI, Docling (PDF, HTML, DOCX, PPTX, XLS/XLSX)            |
| Embeddings                               | Google Vertex AI `text-embedding-004` (batched, configurable dimensions) |
| Short-term Memory                        | Redis (2-hour TTL session store)                                         |
| Long-term Memory                         | PostgreSQL `episodic_memories` table (LLM-compressed summaries)          |
| Embedding Cache                          | PostgreSQL-backed, up to 50 k entries                                    |
| Semantic Query Cache                     | Redis vector cache - returns cached answers for near-duplicate queries   |
| Parse Cache                              | Filesystem gzip cache keyed by file path + mtime + parser config         |
| Retry / Resilience                       | tenacity - 3 attempts, exponential back-off + jitter on all Qdrant calls |
| Observability                            | Logfire (structured traces), LangSmith (optional)                        |
| Auth                                     | JWT (`SECRET_KEY` / `HS256`); guest mode when no accounts are configured |
| Dev tooling                              | Ruff, pre-commit                                                         |

---

## Ingestion Pipeline

```
Raw File (PDF / DOCX / PPTX / HTML / XLS)
    │
    ▼
Google Document AI  →  gzip filesystem cache
    │
    ▼
Chunker  (strategy: structure | fixed | splitter)
    │
    ▼
EmbeddingService  (Vertex AI text-embedding-004, batched)
    │
    ▼
QdrantStorageService.upsert_embedded_chunks()
    └── deterministic UUID5 point IDs  →  idempotent re-ingestion
    └── tenacity retry (3 attempts, exponential back-off + jitter)
```

### Chunking strategies

| Strategy | Behaviour |
|---|---|
| `structure` | Splits at heading boundaries; isolates tables as standalone chunks |
| `fixed` | Token-window with configurable size and overlap |
| `splitter` | Paragraph-level splitting by character delimiter |

---

## Installation

### As a library dependency

```bash
# Core library only
pip install "medici @ git+https://github.com/manoje8/medici.git"

# With optional extras
pip install "medici[api] @ git+https://github.com/manoje8/medici.git"          # + FastAPI server
pip install "medici[gcp,docling] @ git+https://github.com/manoje8/medici.git"  # + Google Cloud + Docling
pip install "medici[all] @ git+https://github.com/manoje8/medici.git"          # everything
```

Available extras: `api`, `ui`, `gcp`, `groq`, `cerebras`, `docling`, `documents`, `graph`, `observability`, `all`

### Library Usage (Quick Start)

```python
from medici.ingestion import Processor, EmbeddingService
from medici.common.llm import GeminiClient, GroqClient
from medici.common.services import QdrantStorageService, HybridSearch
from medici.agents import GraphPipeline, RetrievalAgent, RouterAgent
from medici.knowledge_graph import KGExtractor, KGStore
```

---

## Getting Started (Full Application)

### Prerequisites

- Python 3.12+
- Qdrant (Cloud or local Docker)
- Redis
- PostgreSQL
- Google Cloud project with **Document AI** and **Vertex AI** enabled
- Groq API key and/or Gemini API key

### Local development

```bash
git clone https://github.com/manoje8/medici.git
cd medici

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -e ".[all,dev,test]"
pre-commit install
```

### Docker (recommended)

```bash
cp .env.example .env   # fill in your secrets

make docker-build
make docker-up         # starts postgres, redis, app (port 8000), ui (port 8501)

make docker-logs       # tail logs
make docker-down       # stop
make docker-clean      # stop + remove volumes
```

---

## Configuration

Copy `.env.example` to `.env`. Key variables:

```env
# LLM providers
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile

GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash

CEREBRAS_API_KEY=          # optional

# Vector store
QDRANT_CLUSTER_ENDPOINT=https://your-cluster.qdrant.io
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=rag_docs

# Embeddings
EMBEDDING_MODEL_NAME=text-embedding-004
EMBEDDING_DIMENSIONS=768
EMBEDDING_BATCH_SIZE=100
EMBEDDING_CACHE_ENABLED=true

# Google Cloud (Document AI + Vertex AI)
PROJECT_ID=
LOCATION=us-central1
GCP_DOC_AI_LOCATION=us
GCP_DOC_AI_PROCESSOR_ID=

# Databases
POSTGRES_CONN_STRING=postgresql://postgres:postgres@postgres:5432/medici
REDIS_URL=redis://localhost:6379

# Pipeline limits
MAX_RETRIEVAL_ROUND=2
MAX_UPLOAD_BYTES=50          # MB
MAX_PAGE_PER_PARSE=20

# Auth (leave SECRET_KEY blank to disable auth and use guest mode)
SECRET_KEY=change-me-in-production

# Caches
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_THRESHOLD=0.88

# Observability (optional)
LOGFIRE_TOKEN=
LANGSMITH_API_KEY=
```

---

## Running

```bash
make server-run           # python -m medici.api.main  →  http://localhost:8000

make ui-run               # streamlit run web_ui/main.py  →  http://localhost:8501

# CLI - parse, ingest, or query without the API
medici parse  data/file.pdf --display-stats
medici ingest data/file.pdf --chunking-strategy structure
medici query  "What is multi-head attention?" --top-k 5
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Liveness ping |
| `GET` | `/health` | Qdrant ping → `200 ok` or `503 degraded` |
| `GET` | `/auth-status` | Returns auth mode; issues guest token if auth is disabled |
| `POST` | `/login` | Issues a JWT bearer token |
| `POST` | `/ingestion` | Upload and ingest a document (multipart form) |
| `POST` | `/bulk-ingestion` | Ingest a file already on disk by path (requires auth) |
| `POST` | `/query` | Submit a question to the agentic pipeline |

---

## Testing & Development

```bash
make test        # pytest tests/ -v --tb=short  (300+ unit tests across 11 modules)

make lint        # ruff check + format --check
make format      # ruff check --fix + ruff format
make clean       # remove __pycache__, *.pyc, egg-info
```

---

## Security

| Risk | Mitigation |
|---|---|
| Prompt injection via documents | Chunks wrapped in `<retrieved_context>` XML; all prompts include an immutable system rule to treat that content as untrusted plain text |
| Qdrant transient failures | All Qdrant calls retried up to 3×, exponential back-off + jitter |
| Secret exposure | All credentials loaded from `.env` via `python-dotenv`; `.env` is `.gitignore`d |
| Idempotent ingestion | Documents fingerprinted (SHA-256 of size + first 8 KB); Qdrant points use deterministic UUID5 IDs |
| File upload abuse | File size capped at `MAX_UPLOAD_BYTES`; extension allowlist enforced; path traversal blocked on bulk-ingestion |
