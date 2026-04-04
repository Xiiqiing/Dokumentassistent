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

The system follows a three-stage RAG pipeline:

**Ingestion:** PDF documents are parsed with PyMuPDF, cleaned, and split into chunks using one of three strategies (fixed-size, recursive, or semantic). Each chunk is embedded via a multilingual sentence-transformer and stored in a Qdrant vector collection. A parallel BM25 index is built from the same chunks for sparse keyword matching.

**Retrieval:** User queries run through both dense (Qdrant cosine similarity) and sparse (BM25) search paths. Results are merged via reciprocal rank fusion, then a cross-encoder reranker scores each candidate for final ordering. An intent classifier routes queries to the appropriate retrieval strategy.

**Generation:** Top-ranked chunks are assembled into a prompt context and passed to the LLM through LangChain. The response is returned via a FastAPI endpoint and displayed in a Streamlit UI. Retrieval quality can be measured offline using RAGAS metrics.

## Tech Stack

| Category | Technology |
|---|---|
| Framework | FastAPI, uvicorn |
| Orchestration | LangChain |
| Vector Store | Qdrant (local mode, no server required) |
| Embedding | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` (384 dim) |
| LLM | `gemma3:4b` (default, runs locally via Ollama) |
| Sparse Search | rank_bm25 |
| Reranking | sentence-transformers `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| PDF Parsing | PyMuPDF (fitz) |
| Evaluation | RAGAS |
| UI | Streamlit |
| Config | python-dotenv |

## Provider Support

Both LLM and embedding backends are swappable through environment variables — no code changes required. Supported providers:

- **Ollama** — default, fully local, no API keys needed
- **OpenAI**
- **Azure OpenAI**
- **Anthropic**
- **Google GenAI**
- **Groq**

Switch providers by editing `LLM_PROVIDER` and `EMBEDDING_PROVIDER` in your `.env` file. See `.env.example` for per-provider configuration details.

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
ollama pull gemma3:4b

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

Starts Qdrant + Ollama + API + UI. The `ollama-init` sidecar pulls `gemma3:4b` on first run.

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| Qdrant dashboard | http://localhost:6333/dashboard |

### Cloud mode (OpenAI / Azure / Anthropic / Google)

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
    router.py              # Strategy routing based on intent
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
