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

**Live Demo:** [xq-dokumentassistent.hf.space](https://xq-dokumentassistent.hf.space) — hosted on Hugging Face Spaces

A document assistant for Danish-language PDFs. Queries run through hybrid dense+BM25 search, cross-encoder reranking, and an LLM that cites the source passages in its answer.

## How it works

PDFs are parsed with PyMuPDF, split into chunks (fixed-size, recursive, or semantic), embedded with a multilingual sentence-transformer, and stored in Qdrant. A BM25 index is built from the same chunks for keyword search.

At query time both indexes are searched and their results merged with reciprocal rank fusion. A cross-encoder then rescores the candidates before the top chunks are passed to the LLM. The API streams the response over SSE and the Streamlit UI displays it with source attribution.

**Two routing modes, switchable via `AGENT_MODE`:**

- **Pipeline** (default): a fixed LangGraph DAG — language detection → optional translation → hybrid retrieval → reranking → generation. Works with lightweight local models like `gemma4`.

- **ReAct Agent** (`AGENT_MODE=react`): replaces the DAG with a reasoning loop where the LLM calls tools as many times as it needs before answering. Useful for multi-hop questions or comparisons across documents. Requires a model with tool-calling support.

  | Tool | Purpose |
  |------|---------|
  | `hybrid_search(query, top_k)` | Retrieve relevant passages |
  | `list_documents()` | See what's in the knowledge base |
  | `fetch_document(document_id)` | Read a full document |

## Tech Stack

| Category | Technology |
|---|---|
| Framework | FastAPI, uvicorn |
| Orchestration | LangChain, LangGraph |
| Vector Store | Qdrant (local mode) |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (384 dim) |
| LLM | `gemma4` via Ollama (default) |
| Sparse Search | rank_bm25 |
| Reranking | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| PDF Parsing | PyMuPDF |
| Evaluation | RAGAS |
| UI | Streamlit |

## Provider Support

LLM and embedding backends are configured through environment variables. Supported providers: Ollama, OpenAI, Azure OpenAI, Anthropic, Google GenAI, Groq. The default (Ollama + HuggingFace) runs locally without any API keys.

See `.env.example` for per-provider configuration.

## Agent Mode

| Mode | `AGENT_MODE` | Notes |
|------|-------------|-------|
| Pipeline | `pipeline` (default) | Fixed DAG, works with `gemma4` |
| ReAct | `react` | Tool-calling loop, needs a model that supports tool use |

Tool-calling is supported by OpenAI, Anthropic, Google GenAI, Azure OpenAI, Groq, and some Ollama models (`llama3.1`, `qwen2.5`, `mistral-nemo`). The default `gemma4` does not support it — use `pipeline` mode with Ollama.

ReAct with OpenAI:

```dotenv
AGENT_MODE=react
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Pipeline with local Ollama:

```dotenv
AGENT_MODE=pipeline
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma4
```

## Quick Start

Requires Python 3.11+ and [Ollama](https://ollama.com/).

```bash
git clone https://github.com/Xiiqiing/Dokumentassistent.git
cd Dokumentassistent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

ollama pull gemma4:e4b
python -m scripts.ingest          # place PDFs in docs/ first

uvicorn src.api.main:app --reload  # → http://localhost:8000
streamlit run src/ui/app.py        # → http://localhost:8501
```

## Docker

Docker Compose handles Qdrant, the API, and the Streamlit UI together. The API container waits for Qdrant on startup and runs ingestion automatically if the collection is empty.

**Local (Ollama + HuggingFace):**

```bash
cp .env.example .env
docker compose --profile local up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| Qdrant dashboard | http://localhost:6333/dashboard |

**Cloud (OpenAI / Anthropic / etc.):**

```bash
cp .env.example .env
# set LLM_PROVIDER, EMBEDDING_PROVIDER, and your API key
docker compose up --build
```

**Hugging Face Spaces:** a `Dockerfile` and supervisor config are included. The Space runs Qdrant, the API, and the UI behind nginx on port 7860.

## Project Structure

```
src/
  config.py                # env-based configuration
  provider.py              # create_llm() / create_embeddings() factory
  models.py                # shared dataclasses
  ingestion/
    pdf_parser.py          # PyMuPDF extraction
    text_cleaner.py        # Danish/English normalization
    chunker.py             # fixed-size, recursive, semantic chunking
    pipeline.py            # ingestion orchestration
  retrieval/
    embedder.py
    vector_store.py        # Qdrant
    bm25_search.py
    hybrid.py              # reciprocal rank fusion
    reranker.py            # cross-encoder
  api/
    main.py
    routes.py              # /query, /ingest, /health
  agent/
    intent_classifier.py
    router.py              # pipeline mode (AGENT_MODE=pipeline)
    tools.py               # hybrid_search + ToolResultStore
    react_router.py        # ReAct mode (AGENT_MODE=react)
  evaluation/
    evaluator.py           # RAGAS metrics
  ui/
    app.py                 # Streamlit frontend
scripts/
  ingest.py
  e2e_test.py
tests/
docs/                      # example PDFs (KU AI public documents)
```
