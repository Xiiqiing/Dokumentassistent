---
title: Dokumentassistent
emoji: 📄
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
---

# Doc Assistant

A RAG-based document assistant. Handles Danish-language PDFs with hybrid search, reranking, and structured evaluation.

## Features

- PDF ingestion with multiple chunking strategies (fixed-size, recursive, semantic)
- Hybrid retrieval: dense (Qdrant) + sparse (BM25) with reciprocal rank fusion
- Cross-encoder reranking for improved relevance
- Intent classification and query routing
- RAGAS-based retrieval quality evaluation
- Provider-agnostic: Ollama, OpenAI, Azure OpenAI, Anthropic, Google GenAI
- Streamlit frontend with Danish/English UI
- FastAPI REST interface

## Tech Stack

| Component | Default | Alternatives |
|---|---|---|
| LLM | Ollama (`gemma3:4b`) | OpenAI, Azure OpenAI, Anthropic, Google GenAI |
| Embeddings | HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dim) | OpenAI, Azure OpenAI, Google GenAI |
| Vector Store | Qdrant (local mode, no server) | |
| Sparse Search | rank_bm25 | |
| Reranking | sentence-transformers (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) | |
| PDF Parsing | PyMuPDF (fitz) | |
| Evaluation | RAGAS | |
| Frontend | Streamlit | |
| Orchestration | LangChain | |

## Quick Start (Local, No API Keys)

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running

### 1. Clone and set up

```bash
git clone <repo-url>
cd DocAssistant

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Default settings use Ollama + local embeddings, no edits needed
```

### 3. Pull the default LLM model

```bash
ollama pull gemma3:4b
```

### 4. Ingest documents

Place PDF files in the `docs/` directory, then run:

```bash
python -m scripts.ingest
```

### 5. Start the API server

```bash
uvicorn src.api.main:app --reload
```

The API is available at http://localhost:8000. Interactive docs at http://localhost:8000/docs.

### 6. Start the Streamlit UI (optional)

```bash
streamlit run src/ui/app.py
```

The UI is available at http://localhost:8501.

## Docker

Docker Compose starts Qdrant, the API server, and the Streamlit UI.
On first launch the API container waits for Qdrant, checks whether the
collection already contains data, and runs ingestion automatically if needed.

### Local mode (Ollama + HuggingFace, no API keys)

```bash
cp .env.example .env          # defaults are already set for local mode
docker compose --profile local up --build
```

This starts **Qdrant + Ollama + API + UI**. The `ollama-init` sidecar pulls
`gemma3:4b` on first run. Embeddings use the bundled HuggingFace model
(no external service required during ingestion).

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| Qdrant dashboard | http://localhost:6333/dashboard |

### Cloud mode (OpenAI / Azure / Anthropic / Google)

```bash
cp .env.example .env
# Edit .env: uncomment the cloud provider block you need,
# comment out the local-mode block, and fill in your API key.
docker compose up --build
```

This starts **Qdrant + API + UI** (no Ollama). Example for OpenAI:

```dotenv
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

See `.env.example` for Azure OpenAI, Anthropic, and Google GenAI examples.

> **Note:** Ingestion needs the embedding provider to be reachable. For cloud
> embedding providers the relevant API key must be set in `.env` before first
> `docker compose up`.

## Testing

```bash
pytest tests/
```

End-to-end test (requires Ollama running):

```bash
python -m scripts.e2e_test
```

## Project Structure

```
src/
  config.py          # Centralized config from env vars
  provider.py        # Factory: create_llm() / create_embeddings()
  models.py          # Shared dataclasses (DocumentChunk, QueryResult, etc.)
  ingestion/         # PDF parsing, text cleaning, chunking strategies
  retrieval/         # Embedder, vector store, BM25, hybrid fusion, reranker
  api/               # FastAPI endpoints
  agent/             # Intent classification and query routing
  evaluation/        # RAGAS retrieval quality evaluation
  ui/                # Streamlit frontend
scripts/
  ingest.py          # CLI document ingestion
  e2e_test.py        # End-to-end integration test
tests/               # pytest test suite
docs/                # Danish test documents (KU public policy PDFs)
```
