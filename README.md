# Studious - Agentic Retrieval-Augmented Generation

> A production-grade, agentic RAG pipeline built with FastAPI, LangGraph, Qdrant, Google Document AI, and Vertex AI embeddings. Designed for intelligent document ingestion, hybrid search, query expansion, multi-turn memory, and LLM-powered answer synthesis.

---

## Overview

**Studious** is an advanced Retrieval-Augmented Generation system that goes well beyond a naïve RAG setup. It is orchestrated by a **LangGraph state machine** that routes each query through a chain of specialised agents planning, retrieving, evaluating, grading, and synthesising with full multi-turn memory and self-correcting retrieval loops.

### What makes it "agentic"?

- The **Router** classifies questions into seven categories and selects the optimal execution path.
- The **Planner** decomposes complex questions into 2–4 focused sub-questions.
- The **RetrievalAgent** evaluates its own results and autonomously decides whether to refine the query, expand the search, or declare the context sufficient up to a configurable maximum number of rounds.
- The **Grader** post-filters retrieved chunks for relevance before synthesis.
- The **Synthesizer** selects from seven category-specific prompt strategies and enforces source citations.

---

## Architecture

### Ingestion Pipeline

```
Raw File  (PDF / DOCX / PPTX / HTML / XLS)
        │
        ▼
Google Document AI Parser
  └── filesystem gzip cache (avoids redundant API calls)
        │
        ▼
Chunker  ─── strategy: structure | fixed | splitter
  each chunk = { text, section_title, block_types, page_range, doc_id, … }
        │
        ▼
EmbeddingService  (Vertex AI text-embedding-004, batched)
        │
        ▼
QdrantStorageService.upsert_embedded_chunks()
  └── deterministic UUID5 point IDs  →  idempotent re-ingestion
  └── tenacity retry (3 attempts, exponential back-off + jitter)
```

### Agentic Query Graph (LangGraph)

```
user_message
    │
    ▼
rewrite_query  ── ShortTermMemory (Redis) resolves coreferences
    │
    ▼
route  ── RouterAgent classifies: factual | comparative | analytical |
    │      summarization | procedural | clarification | chitchat | meta
    │
    ├── chitchat / meta  ──────────────────────► handle_simple_response ─► END
    │
    ├── factual (low complexity)  ────────────► direct_synthesize ────────► END
    │
    └── all other categories
            │
            ▼
           plan  ── PlannerAgent decomposes into sub-questions
            │
            ▼
         retrieve  ◄──────────────────────────────────────────────┐
            │  RetrievalAgent:                                     │
            │   1. QueryExpander  →  3 LLM-derived query variants  │
            │   2. HybridSearch   →  dense (Qdrant) + BM25 + RRF   │
            │   3. Reranker       →  FlashRank cross-encoder        │
            │   4. Self-evaluate: sufficient | refine | expand      │
            │                                                       │
            ├── refine_query  ─────────────────────────────────────┘
            │
            ├── next_sub_question  (loops over sub-questions)
            │
            └── grade  ── GraderAgent filters accepted chunks
                    │
                    ▼
                synthesize  ── SynthesizerAgent (7 prompt strategies,
                    │           XML-delimited context, citation enforcement)
                    │
                   END
```

> **Global safety valve:** A `GLOBAL_MAX_RETRIEVAL_STEPS = 6` hard cap prevents runaway loops regardless of individual round limits.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI + Uvicorn |
| **Graph Orchestration** | LangGraph (`StateGraph`) |
| **Graph Checkpointing** | LangGraph Postgres checkpointer (psycopg async pool) |
| **LLM Providers** | Groq (`llama-3.3-70b-versatile`), Google Gemini (`gemini-2.5-flash-lite`), Cerebras |
| **Vector Store** | Qdrant (async client, cosine similarity, UUID5 idempotent IDs) |
| **Sparse Search** | BM25 (rank-bm25) in-memory, rebuilt at startup |
| **Fusion** | Reciprocal Rank Fusion (RRF) |
| **Re-ranking** | FlashRank (cross-encoder, CPU-friendly, lazy-initialised) |
| **Parsing** | Google Document AI (PDF, HTML, Office formats) |
| **Embeddings** | Google Vertex AI `text-embedding-004` (batched, configurable dimensions) |
| **Short-term Memory** | Redis (2-hour TTL session store) |
| **Long-term Memory** | PostgreSQL `episodic_memories` table (LLM-compressed summaries) |
| **Retry / Resilience** | tenacity (3 attempts, exponential back-off + jitter on all Qdrant calls) |
| **Parse Cache** | Filesystem gzip cache with JSON manifest (`DocumentCache`) |
| **Observability** | Logfire (structured traces), LangSmith (optional) |
| **Dev Tools** | Ruff, pre-commit |
| **Build** | setuptools via `pyproject.toml` |

---

## Key Features

### 1. LangGraph Agentic Orchestration
The entire query pipeline is a compiled `StateGraph` with typed `State`, conditional edges, and a PostgreSQL-backed checkpointer for durable conversation threads across restarts.

### 2. Hybrid Search with RRF
Dense vector search (Qdrant cosine similarity) and sparse keyword retrieval (BM25) run per query variant. Results are fused with Reciprocal Rank Fusion, then re-scored by a FlashRank cross-encoder.

### 3. Self-Correcting Retrieval Loop
`RetrievalAgent` evaluates its own retrieved chunks against the sub-question and returns one of four decisions: `sufficient`, `refine_query`, `expand_search`, or `exhausted`. The graph reacts accordingly, up to `MAX_RETRIEVAL_ROUND` per sub-question and a global 6-step budget.

### 4. Dual Memory System
- **Short-term** Redis-backed session store (2-hour TTL). `QueryRewriter` uses the last N turns to resolve coreferences in follow-up questions ("it", "that document", etc.).
- **Long-term (Episodic)** LLM compresses completed sessions into concise summaries with topic tags stored in PostgreSQL. Retrieved and injected into future sessions for the same user.

### 5. Prompt-Injection Firewall
All retrieved document chunks are wrapped in `<retrieved_context>` XML tags before being embedded in LLM prompts. Every retrieval-dependent prompt begins with a mandatory system rule instructing the model to treat that content as untrusted plain text and never follow instructions found within it.

### 6. Qdrant Resilience (tenacity)
Every Qdrant I/O call (`upsert`, `search`, `scroll`, `chunk_count`, `ping`) is wrapped in an async tenacity retry with 3 attempts, exponential back-off starting at 1 s (capped at 10 s), and per-attempt jitter. Retry attempts are logged as warnings via Logfire.

### 7. Multi-Format Document Parsing
Google Document AI provides layout-aware block extraction (heading, paragraph, table) for PDF, HTML, DOCX, PPTX, XLS/XLSX. Parse results are cached as gzip-compressed JSON keyed by file path + mtime + parser config.

### 8. Three Chunking Strategies
| Strategy | Description |
|---|---|
| `structure` | Splits at heading boundaries; isolates tables as standalone chunks; preserves section context |
| `fixed` | Token-window chunking with configurable size and overlap |
| `splitter` | Paragraph-level splitting by character delimiter |

### 9. Health-Check Endpoint
`GET /health` pings Qdrant and returns structured JSON with per-dependency status. Returns `HTTP 200` (`status: ok`) when all dependencies are reachable, `HTTP 503` (`status: degraded`) otherwise. Suitable for load-balancer or Kubernetes liveness probes.

---


## Getting Started

### Prerequisites

- Python 3.10+
- A running **Qdrant** instance (local Docker or Qdrant Cloud)
- A **Redis** instance (local Docker or managed)
- A **PostgreSQL** database (for LangGraph checkpointing + episodic memory)
- Google Cloud project with **Document AI** and **Vertex AI** APIs enabled
- A **Groq** API key and/or **Gemini** API key

### Installation

```bash
git clone https://github.com/manoje8/studious.git
cd advanced_rag

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

pre-commit install
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values (all settings are in `src/utils/config.py`):

```env
# Project
PROJECT_NAME=studious
HOST=localhost
PORT=8000
CORS_ORIGINS=http://localhost:3000

# Google Cloud
PROJECT_ID=<your-gcp-project>
LOCATION=<gcp-region>
GCP_DOC_AI_LOCATION=<doc-ai-region>
GCP_DOC_AI_PROCESSOR_ID=<processor-id>
GCP_RAW_BUCKET=<raw-bucket>
GCP_PROCESSED_BUCKET=<processed-bucket>

# Qdrant
QDRANT_CLUSTER_ENDPOINT=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=rag_docs

# Embedding
EMBEDDING_MODEL_NAME=text-embedding-004
EMBEDDING_DIMENSIONS=1356
EMBEDDING_BATCH_SIZE=100

# LLM Groq (used for routing, grading, retrieval evaluation)
GROQ_API_KEY=<your-groq-key>
GROQ_MODEL=llama-3.3-70b-versatile

# LLM Gemini (used for planning, query rewriting, query expansion)
GEMINI_API_KEY=<your-gemini-key>
GEMINI_MODEL=gemini-2.5-flash-lite

# Databases
POSTGRES_CONN_STRING=postgresql://postgres:pass@localhost:5432/studious
REDIS_URL=redis://localhost:6379

# Pipeline limits
MAX_RETRIEVAL_ROUND=3
MAX_CONTEXT_CHARS=100000
MAX_PAGE_PER_PARSE=20

# Observability (optional)
LOGFIRE_TOKEN=
LANGSMITH_TRACING=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
```

---

## Usage

### Start the API Server

```bash
make server-run
# (or)
python src/api/main.py
```

### Streamlit Chat UI

```bash
make ui-run
# or: streamlit run web_ui/main.py
```

### CLI: Parse, Ingest, or Query

```bash
# Parse a document (extract blocks, no embedding)
python -m src.ingestion.cli parse data/my_document.pdf --display-stats

# Ingest a document (parse → chunk → embed → store in Qdrant)
python -m src.ingestion.cli ingest data/my_document.pdf --chunking-strategy structure

# Direct vector search (no agentic pipeline)
python -m src.ingestion.cli query "What is multi-head attention?" --top-k 5
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Liveness ping returns `"running"` |
| `GET` | `/health` | Dependency health-check (Qdrant ping). Returns `200 ok` or `503 degraded` |
| `POST` | `/ingestion` | Ingest a document into the RAG pipeline |
| `POST` | `/query` | Submit a question to the agentic pipeline |

## Testing

The project ships with **300+ unit tests** across 11 modules:

```bash
make test

pytest tests/unit/test_services.py -v

```

---

## Development

```bash
# Lint (ruff)
make lint

# Auto-fix + format
make format

# Run pre-commit on all files
pre-commit run --all-files

```

### Observability

All pipeline stages emit structured traces via **Logfire**:

- **Ingestion:** parse cache hits/misses, chunk counts, batch upsert progress, retry warnings
- **Query:** retrieval decisions per round, query variants, grader accept/reject counts, synthesis category
- **Errors:** full structured error context on any exception

Set up Logfire with `logfire auth` and add your `LOGFIRE_TOKEN` to `.env`.

---

## Security Notes

| Concern | Mitigation |
|---|---|
| **Prompt injection via documents** | Retrieved chunks are wrapped in `<retrieved_context>` XML tags. All retrieval-dependent prompts include an immutable system rule instructing the LLM to treat that content as untrusted plain text. |
| **Qdrant transient failures** | All Qdrant calls are retried up to 3 times with exponential back-off and jitter before propagating the error. |
| **API key exposure** | All secrets are loaded from `.env` via `python-dotenv`; `.env` is `.gitignore`d. |
| **Idempotent ingestion** | Documents are fingerprinted (SHA-256 of size + first 8 KB); Qdrant points use deterministic UUID5 IDs safe to re-ingest. |
