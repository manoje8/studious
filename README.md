# Studios — Agentic Retrieval-Augmented Generation

> A production-grade, agentic RAG pipeline built with FastAPI, Qdrant, Google Document AI, and LangChain. Designed for intelligent document ingestion, hybrid search, query expansion, and LLM-powered answer synthesis.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Agentic Query Pipeline](#agentic-query-pipeline)
- [Observability](#observability)
- [Development](#development)

---

## Overview

**Advanced RAG** is a learning project that demonstrates how to build a sophisticated, agentic Retrieval-Augmented Generation system from the ground up. It goes well beyond a naive RAG setup by incorporating:

- **Agentic retrieval loops** — the system evaluates its own retrieval quality and decides whether to refine, expand, or accept results.
- **Hybrid search** — fuses dense (vector) and sparse (BM25) rankings via Reciprocal Rank Fusion (RRF).
- **Query expansion** — LLM-generated query variants improve recall.
- **Cross-encoder re-ranking** — FlashRank re-scores candidates for precision.
- **Multi-format document parsing** — PDFs, Office documents, and HTML via Google Document AI.
- **Structured and fixed-size chunking** — two strategies with rich per-chunk metadata.

---

## Architecture

### Ingestion Flow

```
Raw File (PDF / DOCX / HTML)
        ↓
    Google Document AI Parser
        ↓
  content_list  [block1, block2 … block_n]
        ↓
    Chunker  (structure | fixed)
        ↓
  chunks = [chunk1, chunk2 … chunk_m]
  each chunk = { text, metadata }
        ↓
  Embedding Model  (Vertex AI / OpenAI)
        ↓
  vectors = [vec1, vec2 … vec_m]
        ↓
  Qdrant Upsert  (batched, idempotent)
```

### Agentic Query Flow

```
User Question
      ↓
  Router Agent          ← classifies: factual | comparative | analytical | summarization
      ↓
  Query Expander        ← generates 3 alternative phrasings via LLM
      ↓
  Hybrid Search         ← dense (Qdrant) + sparse (BM25/Redis) → RRF merge
      ↓
  Cross-Encoder Reranker (FlashRank)
      ↓
  Retrieval Agent       ← evaluates quality: sufficient | refine | expand | exhausted
      ↑_________________| (self-loop until decision = sufficient or exhausted)
      ↓
  Synthesizer / LLM     ← generates final answer with citations
```

---

## Project Structure

```
advanced_rag/
├── src/
│   ├── main.py                  # FastAPI application entry point
│   ├── agents/
│   │   ├── agent_model.py       # Pydantic state models (AgentState, RetrievalRound)
│   │   ├── pipeline.py          # RAG pipeline orchestrator
│   │   ├── router.py            # RouterAgent — question classifier
│   │   ├── retrieval.py         # RetrievalAgent — agentic retrieval loop
│   │   ├── hybrid_search.py     # HybridSearch — dense + sparse + RRF
│   │   └── query_expander.py    # QueryExpander — LLM-based query augmentation
│   ├── ingestion/
│   │   ├── chunk.py             # Chunk dataclass + Chunking strategies
│   │   ├── embedding.py         # EmbeddingService (batched, async)
│   │   ├── processor.py         # Processor — full ingest orchestration
│   │   ├── cli.py               # CLI interface for ingestion
│   │   └── parser/
│   │       └── google_doc_ai.py # Google Document AI parser
│   ├── llm/
│   │   └── groq.py              # Groq LLM client wrapper
│   └── services/
│       ├── qdrant.py            # QdrantStorageService (async, batched upsert)
│       ├── reranker.py          # Reranker — FlashRank cross-encoder
│       └── sparse_index.py      # SparseSearchIndex — BM25 / Redis vector
├── utils/
│   ├── config.py                # Pydantic Settings config
│   └── constants.py             # Format constants (PDF, HTML, Office)
├── data/                        # Local test documents
├── Makefile                     # Dev shortcuts
├── pyproject.toml               # Build config (setuptools)
├── requirements.txt             # Python dependencies
├── .pre-commit-config.yaml      # Ruff + Black hooks
└── .env                         # Environment variables (not committed)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI + Uvicorn |
| **LLM** | Groq, LangChain (Vertex AI, OpenAI) |
| **Vector Store** | Qdrant (async client) |
| **Sparse Search** | BM25 (rank-bm25), Redis / RedisVL |
| **Re-ranking** | FlashRank (cross-encoder, CPU-friendly) |
| **Parsing** | Google Document AI, docling, unstructured, pypdf |
| **Embeddings** | Google Vertex AI / OpenAI (configurable) |
| **Observability** | Logfire (tracing + structured logs) |
| **Workflow** | LangGraph + LangChain |
| **Persistence** | PostgreSQL (psycopg3 + SQLAlchemy), Redis |
| **Dev Tools** | Ruff, Black, pre-commit, uv |

---

## Key Features

### 🔍 Hybrid Search with RRF
Combines dense vector similarity (Qdrant cosine) and sparse keyword retrieval (BM25) across all expanded query variants. Results are merged with Reciprocal Rank Fusion to produce a single ranked candidate list.

### 🤖 Agentic Retrieval Loop
The `RetrievalAgent` evaluates its own retrieved chunks against the original question and decides autonomously whether the context is `sufficient`, needs `refine_query`, `expand_search`, or is `exhausted`. This loop prevents under-retrieval without requiring manual tuning.

### 📄 Multi-Format Document Parsing
Supports PDF, HTML, DOCX, PPTX, and more. Google Document AI provides layout-aware parsing with block-level type metadata (heading, paragraph, table).

### ✂️ Two Chunking Strategies
- **Structure chunking** — splits at document headings and isolates tables as standalone chunks, preserving section context.
- **Fixed chunking** — token-window chunking with configurable overlap, suitable for unstructured or continuous text.

### Query Expansion
The LLM generates three semantically equivalent query variants before search, dramatically improving recall for ambiguous or technical questions.

### 📊 Observability with Logfire
All pipeline stages — parsing, chunking, embedding, search, retrieval decisions — are instrumented with Logfire structured traces for end-to-end visibility.

---

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A running **Qdrant** instance (local Docker or Qdrant Cloud)
- Google Cloud project with **Document AI** and **Vertex AI** APIs enabled
- (Optional) Redis for sparse index persistence

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd advanced_rag

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Pre-commit Hooks

```bash
pre-commit install
```

---

## Configuration

Copy `.env.example` to `.env` and populate all values:

```env
# Qdrant
QDRANT_CLUSTER_ENDPOINT=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=advanced_rag

# Embedding
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
EMBEDDING_BATCH_SIZE=64

# Google Cloud
GOOGLE_CLOUD_PROJECT=
GOOGLE_APPLICATION_CREDENTIALS=

# LLM / Groq
GROQ_API_KEY=

# Redis (optional)
REDIS_URL=redis://localhost:6379
```

---

## Usage

### Start the API server

```bash
uvicorn src.main:app --reload
```

### Ingest a document

```bash
# Via CLI
python -m src.ingestion.cli --file data/my_document.pdf --strategy structure

# Via API (POST /ingest)
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/my_document.pdf" \
  -F "chunking_strategy=structure"
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings in Q3?"}'
```

---

## Ingestion Pipeline

The `Processor` class orchestrates a 4-stage pipeline:

| Stage | Description |
|---|---|
| **1 – Parse** | Google Document AI extracts structured blocks (heading, paragraph, table) |
| **2 – Chunk** | `Chunking` applies structure or fixed strategy; returns `Chunk` objects with rich metadata |
| **3 – Embed** | `EmbeddingService` batch-encodes chunk texts into vectors |
| **4 – Store** | `QdrantStorageService` upserts vectors in batches of 100 with deterministic UUIDs |

Document IDs are SHA-256 fingerprints of file size + first 8 KB, enabling idempotent re-ingestion.

---

## Agentic Query Pipeline

| Agent | Role |
|---|---|
| `RouterAgent` | Classifies the question type and whether multi-step retrieval is needed |
| `QueryExpander` | Generates 3 LLM-derived query variants to improve recall |
| `HybridSearch` | Runs dense + sparse search per variant, merges with RRF |
| `Reranker` | Cross-encoder re-scores top candidates with FlashRank |
| `RetrievalAgent` | Self-evaluates retrieval quality and loops until sufficient or exhausted |

---

## Observability

All pipeline stages emit structured traces via **Logfire**:

- Service names: `RAG Pipeline`, `Hybrid search`
- Events: chunk counts, vector counts, retrieval decisions, query variants
- Error details on parse/upsert failures

Configure Logfire by running `logfire auth` and setting `LOGFIRE_TOKEN` in `.env`.

---

## Development

```bash
# Lint
ruff check .

# Format
black .

# Run pre-commit on all files
pre-commit run --all-files
```
