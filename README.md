---
title: Dokumentintelligens-system
emoji: 📄
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
noindex: true
---

# Doc Assistant

**Live Demo:** [xq-dokumentassistent.hf.space](https://xq-dokumentassistent.hf.space) — hosted on Hugging Face Spaces

A document intelligence system built on a RAG architecture, covering PDF ingestion, semantic chunking, hybrid retrieval with reranking, and LLM-generated answers with source citations. The LLM layer is provider-agnostic. Two modes: a pipeline for lightweight models, and a Plan-and-Execute agent flow with conversation memory for complex multi-step queries. Retrieval quality is evaluated with RAGAS.

## How it works

PDFs are parsed with PyMuPDF, split into chunks (fixed-size, recursive, or semantic), embedded with a multilingual sentence-transformer, and stored in Qdrant. A BM25 index is built from the same chunks for keyword search.

At query time both indexes are searched and their results merged with reciprocal rank fusion. A cross-encoder then rescores the candidates before the top chunks are passed to the LLM. The API streams the response over SSE and the Streamlit UI displays it with source attribution.

**Two routing modes, switchable via `AGENT_MODE`:**

- **Pipeline**: a predefined LangGraph graph — language detection → optional translation → hybrid retrieval → reranking → generation, with a confidence-based retry loop. Works with lightweight local models.

- **Plan-and-Execute Agent** (default, `AGENT_MODE=react`): a structured multi-step pipeline where a planner decomposes the query into steps, an executor runs each step via a ReAct sub-agent with tool access, and a synthesizer produces the final cited answer. Includes conversation memory for multi-turn follow-ups. Requires a model with tool-calling support.

  | Tool | Purpose |
  |------|---------|
  | `hybrid_search(query, top_k)` | Retrieve relevant passages via hybrid search + reranking |
  | `multi_query_search(question, top_k)` | Decompose complex questions into sub-queries, search each, merge results |
  | `search_within_document(document_id, query, top_k)` | Find specific sections inside a known document |
  | `summarize_document(document_id)` | Generate a structured summary of a document |
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
| Pipeline | `pipeline` | Predefined graph, works with lightweight models |
| Plan-and-Execute (default) | `react` | Structured multi-step agent with conversation memory |

Tool-calling is supported by OpenAI, Anthropic, Google GenAI, Azure OpenAI, Groq, and some Ollama models (`llama3.1`, `qwen2.5`, `mistral-nemo`).

Plan-and-Execute with OpenAI:

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
OLLAMA_MODEL=gemma3
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
    tools.py               # 6 retrieval tools + ToolResultStore
    plan_and_execute.py    # Plan-and-Execute agent (AGENT_MODE=react)
    memory.py              # conversation memory for multi-turn
  evaluation/
    evaluator.py           # RAGAS metrics
  ui/
    app.py                 # Streamlit frontend
scripts/
  ingest.py
  e2e_test.py
tests/
docs/                      # example PDFs or texts (KU AI public documents)
```
