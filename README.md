---
title: Dokumentassistent
emoji: 📄
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
noindex: true
---

# Doc Assistant

A RAG-based document assistant for Danish-language PDFs, featuring hybrid search, cross-encoder reranking, and structured evaluation.

## Architecture

The system follows a three-stage RAG pipeline with an optional Agent Flows mode:

**Ingestion:** PDF documents are parsed with PyMuPDF, cleaned, and split into chunks using one of three strategies (fixed-size, recursive, or semantic). Each chunk is embedded via a multilingual sentence-transformer and stored in a Qdrant vector collection. A parallel BM25 index is built from the same chunks for sparse keyword matching.

**Retrieval:** User queries run through both dense (Qdrant cosine similarity) and sparse (BM25) search paths. Results are merged via reciprocal rank fusion, then a cross-encoder reranker scores each candidate for final ordering.

**Generation:** Top-ranked chunks are assembled into a prompt context and passed to the LLM. The response is returned via a FastAPI endpoint with full SSE streaming and displayed in a Streamlit UI. Retrieval quality can be measured offline using RAGAS metrics.

**Routing — two modes (switchable via `AGENT_MODE`):**

- **Pipeline mode** (default, `AGENT_MODE=pipeline`): Fixed LangGraph DAG — language detection → optional translation → hybrid retrieval → cross-encoder reranking → intent-specific generation. Works with lightweight models such as `gemma4`.

- **ReAct Agent mode** (`AGENT_MODE=react`): Replaces the fixed DAG with a multi-step reasoning loop. The LLM decides which tools to call and how many times, then produces a grounded answer citing source documents. Supports multi-hop questions, comparisons across documents, and procedural queries that benefit from iterative retrieval. Requires an LLM with tool-calling support (OpenAI, Anthropic, Google GenAI, or compatible Ollama models such as `llama3.1` / `qwen3`).

  Available tools in ReAct mode:

  | Tool | When the LLM uses it |
  |------|----------------------|
  | `hybrid_search(query, top_k)` | Find relevant passages — called once or multiple times with refined queries |
  | `list_documents()` | Discover which documents are in the knowledge base |
  | `fetch_document(document_id)` | Read the full text of a named document (e.g. for summaries) |

## Tech Stack

| Category | Technology |
|---|---|
| Framework | FastAPI, uvicorn |
| Orchestration | LangChain, LangGraph |
| Vector Store | Qdrant (local mode, no server required) |
| Embedding | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` (384 dim) |
| LLM | `gemma4` (default, runs locally via Ollama) |
| Sparse Search | rank_bm25 |
| Reranking | sentence-transformers `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| PDF Parsing | PyMuPDF (fitz) |
| Evaluation | RAGAS |
| UI | Streamlit |
| Config | python-dotenv |
| Agent Flows | LangGraph `create_react_agent` + LangChain `@tool` |

## Provider Support

Both LLM and embedding backends are swappable through environment variables — no code changes required. Supported providers:

- **Ollama** — default, fully local, no API keys needed
- **OpenAI**
- **Azure OpenAI**
- **Anthropic**
- **Google GenAI**
- **Groq**

Switch providers by editing `LLM_PROVIDER` and `EMBEDDING_PROVIDER` in your `.env` file. See `.env.example` for per-provider configuration details.

## Agent Mode

The system supports two routing modes, controlled by `AGENT_MODE` in `.env`:

| Mode | Value | Description |
|------|-------|-------------|
| Pipeline (default) | `AGENT_MODE=pipeline` | Fixed LangGraph DAG. Works with lightweight models such as `gemma4` via Ollama — no cloud API required. |
| ReAct Agent | `AGENT_MODE=react` | Multi-step reasoning loop. The LLM calls tools as many times as needed — `hybrid_search` for targeted passages, `list_documents` to navigate the knowledge base, `fetch_document` for full document reads — then cites sources in the final answer. |

**LLM compatibility for ReAct mode:**

`AGENT_MODE=react` requires a model with native tool-calling support. Use `AGENT_MODE=pipeline` (the default) if your model does not support it.

| Provider | Tool-calling support |
|----------|---------------------|
| OpenAI (`gpt-4o-mini`, `gpt-4o`) | Yes |
| Anthropic (`claude-*`) | Yes |
| Google GenAI (`gemini-*`) | Yes |
| Azure OpenAI | Yes |
| Groq (`qwen/qwen3-32b`, `llama-3.3-70b-versatile`) | Yes |
| Ollama — `llama3.1`, `qwen2.5`, `mistral-nemo` | Yes (model-dependent) |
| Ollama — `gemma4` (default) | No → use `pipeline` mode |

Example `.env` for ReAct mode with OpenAI:

```dotenv
AGENT_MODE=react
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Example `.env` for pipeline mode with local Ollama (default, no API key needed):

```dotenv
AGENT_MODE=pipeline
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma4
```

## Quick Start

Prerequisites: Python 3.11+ and [Ollama](https://ollama.com/) installed.

```bash
# Clone and install
git clone <repo-url>
cd Dokumentassistent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure (defaults work out of the box with Ollama)
cp .env.example .env

# Pull the default LLM
ollama pull gemma4:e4b

# Ingest documents (place PDFs in docs/ first)
python -m scripts.ingest

# Start the API server
uvicorn src.api.main:app --reload
# → http://localhost:8000  (docs at /docs)

# Start the Streamlit UI
streamlit run src/ui/app.py
# → http://localhost:8501
```

## Docker

Docker Compose starts Qdrant, the API server, and the Streamlit UI. On first launch, the API container waits for Qdrant and runs ingestion automatically if the collection is empty.

### Local mode (Ollama + HuggingFace, no API keys)

```bash
cp .env.example .env
docker compose --profile local up --build
```

Starts Qdrant + Ollama + API + UI. The `ollama-init` sidecar pulls `gemma4` on first run.

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| Qdrant dashboard | http://localhost:6333/dashboard |

### Cloud mode (OpenAI / Azure / Anthropic / Google / Groq)

```bash
cp .env.example .env
# Edit .env: set your provider and API key
docker compose up --build
```

Starts Qdrant + API + UI (no Ollama). Example `.env` for OpenAI:

```dotenv
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### Hugging Face Spaces

The project includes a `Dockerfile` and supervisor config for one-click deployment on HF Spaces. The Space runs Qdrant, the API, and the Streamlit UI behind an nginx reverse proxy on port 7860.

## Project Structure

```
src/
  config.py                # Centralized configuration from environment variables
  provider.py              # Factory functions: create_llm() / create_embeddings()
  models.py                # Shared dataclasses (DocumentChunk, QueryResult, etc.)
  ingestion/
    pdf_parser.py          # PDF text extraction via PyMuPDF
    text_cleaner.py        # Danish/English text normalization
    chunker.py             # Fixed-size, recursive, and semantic chunking
    pipeline.py            # End-to-end ingestion orchestration
  retrieval/
    embedder.py            # Document and query embedding
    vector_store.py        # Qdrant collection management and search
    bm25_search.py         # BM25 sparse keyword search
    hybrid.py              # Reciprocal rank fusion of dense + sparse results
    reranker.py            # Cross-encoder reranking
  api/
    main.py                # FastAPI application setup
    routes.py              # REST endpoints (query, ingest, health)
  agent/
    intent_classifier.py   # Query intent detection
    router.py              # Fixed-DAG pipeline router (AGENT_MODE=pipeline)
    tools.py               # @tool-decorated hybrid_search + ToolResultStore
    react_router.py        # ReAct agent router with tool-calling loop (AGENT_MODE=react)
  evaluation/
    evaluator.py           # RAGAS-based retrieval quality metrics
  ui/
    app.py                 # Streamlit frontend
scripts/
  ingest.py                # CLI document ingestion
  e2e_test.py              # End-to-end integration test
tests/                     # pytest test suite
docs/                      # Example PDFs/Texts (KU AI public documents)
```
