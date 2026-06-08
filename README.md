# Studious — Agentic Retrieval-Augmented Generation

> A production-grade, agentic RAG pipeline built with FastAPI, Qdrant, Google Document AI, and Vertex AI embeddings. Designed for intelligent document ingestion, hybrid search, query expansion, and LLM-powered answer synthesis.

---

## Overview

**Studious** is a learning project that demonstrates how to build a sophisticated, agentic Retrieval-Augmented Generation system from the ground up. It goes well beyond a naive RAG setup by incorporating:

- **Agentic retrieval loops** — a multi-agent system evaluates retrieval quality, decomposes complex questions, grades individual chunks, and self-corrects before synthesizing an answer.
- **Hybrid search** — fuses dense (Qdrant cosine) and sparse (BM25) rankings via Reciprocal Rank Fusion (RRF). *(BM25 path is currently disabled pending integration.)*
- **Query expansion** — LLM-generated query variants improve recall across multiple search passes.
- **Cross-encoder re-ranking** — FlashRank re-scores candidates for precision.
- **Multi-format document parsing** — PDFs, Office documents, and HTML via Google Document AI.
- **Three chunking strategies** — structure-aware, fixed-size with overlap, and a basic paragraph splitter, each with per-chunk metadata.
- **Document parse caching** — filesystem-backed gzip cache avoids redundant Document AI calls for unchanged files.

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
    ┌─ Cache Check ─┐
    │  HIT → return  │
    │  MISS → parse  │
    └────────────────┘
        ↓
    Chunker  (structure | fixed | splitter)
        ↓
  chunks = [chunk1, chunk2 … chunk_m]
  each chunk = { text, metadata }
        ↓
  Embedding Model  (Vertex AI text-embedding-004)
        ↓
  vectors = [vec1, vec2 … vec_m]
        ↓
  Qdrant Upsert  (batched, idempotent via UUID5)
```

### Agentic Graph Flow

```
                        ┌─────────────────────────────────────────┐
  user_message ─────►   │            ENTRY: rewrite_query         │
                        └──────────────────┬──────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────┐
                        │              route                      │
                        │  (RouterAgent.classify)                 │
                        └──┬───────────────┬───────────────┬──────┘
                           │               │               │
                     factual          analytical      summarization
                           │               │               │
                        ┌──▼───────────────▼───────────────▼──────┐
                        │              plan                       │
                        │  (PlannerAgent.decompose)               │
                        └──────────────────┬──────────────────────┘
                                           │  [sub_questions]
                        ┌──────────────────▼────────────────────── ┐
                        │          retrieve  (per sub-q)           │
                        │  HybridSearch → Reranker → Evaluate      │
                        └──┬───────────────────────────────────────┘
                           │
              ┌────────────┼──────────────────┐
         sufficient    refine_query      exhausted
              │            │                  │
              │     ┌──────▼──────┐           │
              │     │  refine     │───────────►│
              │     └─────────────┘           │
              └────────────┬──────────────────┘
                           │
                        ┌──▼──────────────────────────────────────┐
                        │              grade                      │
                        │  (GraderAgent.grade_chunks)             │
                        └──────────────────┬──────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────┐
                        │            synthesize                   │
                        │  (SynthesizerAgent.synthesize)          │
                        └──────────────────┬──────────────────────┘
                                           │
                                        END
```

\* *BM25 sparse search is implemented but commented out in `hybrid_search.py` pending runtime integration.*

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI + Uvicorn |
| **LLM Providers** | Groq (LangChain ChatGroq), Google Gemini (google-genai SDK) |
| **Vector Store** | Qdrant (async client, cosine similarity) |
| **Sparse Search** | BM25 (rank-bm25) — in-memory, built at ingestion |
| **Re-ranking** | FlashRank (cross-encoder, CPU-friendly, lazy-initialized) |
| **Parsing** | Google Document AI (PDF, HTML, Office formats) |
| **Embeddings** | Google Vertex AI `text-embedding-004` (batched, configurable dimensions) |
| **Caching** | Filesystem gzip cache with JSON manifest (`DocumentCache`) |
| **Observability** | Logfire (tracing + structured logs) |
| **Dev Tools** | Ruff, Black, pre-commit, uv |
| **Build** | setuptools via `pyproject.toml` |

---

## Key Features

### 🤖 Five-Agent Agentic Pipeline
The query pipeline orchestrates five specialized agents — **Router**, **Planner**, **Retriever**, **Grader**, and **Synthesizer** — that collaborate to classify questions, decompose complex queries, retrieve and evaluate evidence, filter noise, and generate cited answers.

### 🔍 Hybrid Search with RRF
Combines dense vector similarity (Qdrant cosine) and sparse keyword retrieval (BM25) across all expanded query variants. Results are merged with Reciprocal Rank Fusion to produce a single ranked candidate list.

### 🔄 Self-Correcting Retrieval Loop
The `RetrievalAgent` evaluates its own retrieved chunks against the original question and decides autonomously whether the context is `sufficient`, needs `refine_query`, `expand_search`, or is `exhausted`. When exhausted or at max rounds, it falls back to best-effort chunks rather than returning nothing.

### 📄 Multi-Format Document Parsing
Supports PDF, HTML, DOCX, PPTX, XLS/XLSX, and more. Google Document AI provides layout-aware parsing with block-level type metadata (heading, paragraph, table).

### ✂️ Three Chunking Strategies
- **Structure** — splits at document headings and isolates tables as standalone chunks, preserving section context and block-type metadata.
- **Fixed** — token-window chunking with configurable size and overlap, suitable for unstructured or continuous text.
- **Splitter** — basic paragraph-level splitting by character delimiter.

### 🔀 Query Expansion
The LLM generates three semantically equivalent query variants before search, dramatically improving recall for ambiguous or technical questions.

### 💾 Document Parse Caching
Parsed results are cached as gzip-compressed JSON with a manifest file. Cache keys incorporate file path, modification time, and parser config — unchanged files are never re-parsed.

### 📊 Observability with Logfire
All pipeline stages — parsing, chunking, embedding, search, retrieval decisions — are instrumented with Logfire structured traces for end-to-end visibility.

---

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A running **Qdrant** instance (local Docker or Qdrant Cloud)
- Google Cloud project with **Document AI** and **Vertex AI** APIs enabled
- A **Groq** API key (or Gemini API key) for LLM inference

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

Create a `.env` file in the project root (see all available settings in `src/utils/config.py`):

```env
# Project
PROJECT_NAME=studious
HOST=localhost
PORT=8000

# Google Cloud
PROJECT_ID=<your-gcp-project>
LOCATION=<gcp-region>
GCP_DOC_AI_LOCATION=<doc-ai-region>
GCP_DOC_AI_PROCESSOR_ID=<processor-id>

# Qdrant
QDRANT_CLUSTER_ENDPOINT=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=rag_docs

# Embedding
EMBEDDING_MODEL_NAME=text-embedding-004
EMBEDDING_DIMENSIONS=1536
EMBEDDING_BATCH_SIZE=100

# LLM — Groq
GROQ_API_KEY=<your-groq-key>
GROQ_MODEL=llama-3.3-70b-versatile

# LLM — Gemini (alternative)
GEMINI_API_KEY=<your-gemini-key>
GEMINI_MODEL=gemini-2.5-flash-lite

# Observability
LOGFIRE_TOKEN=
LANGSMITH_TRACING=
LANGSMITH_API_KEY=

# Pipeline
MAX_RETRIEVAL_ROUND=1
MAX_PAGE_PER_PARSE=20
```

---

## Usage

### Start the API Server

```bash
# Via Makefile
make run-server

# Or directly
python src/api/main.py
```

### CLI — Parse, Ingest, or Query

```bash
# Parse a document (extract content blocks, no embedding)
python -m src.ingestion.cli parse data/my_document.pdf --display-stats

# Ingest a document (parse → chunk → embed → store in Qdrant)
python -m src.ingestion.cli ingest data/my_document.pdf --chunking-strategy structure

# Query Qdrant directly (dense search, no agentic pipeline)
python -m src.ingestion.cli query "What is multi-head attention?" --top-k 5
```

### API Endpoints

```bash
# Ingest a document
curl -X POST "http://localhost:8000/ingestion?file_path=data/my_document.pdf&chunking_strategy=structure"

# Query via agentic pipeline
curl -X POST "http://localhost:8000/query?question=What+are+the+key+findings"
```

### Pipeline Script (Direct)

```bash
# Run the agentic pipeline directly
python -m src.agents.pipeline
```

---

## Ingestion Pipeline

The `Processor` class orchestrates a 4-stage pipeline:

| Stage | Description |
|---|---|
| **1 — Parse** | Google Document AI extracts structured blocks (heading, paragraph, table); results are cached to `.cache/doc_parser/` as gzip JSON |
| **2 — Chunk** | `Chunking` applies structure, fixed, or splitter strategy; returns `Chunk` objects with rich metadata (section title, block types, page numbers) |
| **3 — Embed** | `EmbeddingService` batch-encodes chunk texts into vectors via Vertex AI `text-embedding-004` with retry logic |
| **4 — Store** | `QdrantStorageService` upserts vectors in batches of 100 with deterministic UUID5 identifiers |

Document IDs are SHA-256 fingerprints of file size + first 8 KB, enabling idempotent re-ingestion.

---

## Agentic Query Pipeline

| Agent | Role |
|---|---|
| `RouterAgent` | Classifies the question type (factual, comparative, analytical, summarization) and whether multi-step retrieval is needed |
| `PlannerAgent` | Decomposes non-factual questions into 2–4 focused sub-questions |
| `QueryExpander` | Generates 3 LLM-derived query variants per sub-question to improve recall |
| `HybridSearch` | Runs dense search per variant, merges with RRF |
| `Reranker` | Cross-encoder re-scores top candidates with FlashRank |
| `RetrievalAgent` | Self-evaluates retrieval quality and loops until sufficient or exhausted |
| `GraderAgent` | Filters accepted chunks for relevance to the original question |
| `SynthesizerAgent` | Generates a final answer with section citations from graded context |

---

## Observability

All pipeline stages emit structured traces via **Logfire**:

- Service names: `Studious`, `PROCESS CLI`, `Hybrid search`
- Events: chunk counts, vector counts, retrieval decisions, query variants, cache hits/misses
- Error details on parse/upsert failures

Configure Logfire by running `logfire auth` and setting `LOGFIRE_TOKEN` in `.env`.

---

## Development

```bash
# Lint
make lint
# or: ruff check . && black --check .

# Format
make format
# or: ruff check --fix . && black .

# Run pre-commit on all files
pre-commit run --all-files

# Start the server
make run-server
```

---

_Last updated: 2026-06-02_
