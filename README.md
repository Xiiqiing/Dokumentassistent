---
title: Dokumentintelligens-system
emoji: 📄
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
noindex: true
---

# Dokumentassistent

## Live demo
Hosted on Hugging Face Spaces: [xq-dokumentassistent.hf.space](https://xq-dokumentassistent.hf.space)

[Skip to English ↓](#english)

## Dansk

En RAG-applikation, der lader brugeren stille spørgsmål til dokumenter på et hvilket som helst sprog og få svar med kildehenvisninger. Systemet er bygget på open source-komponenter (LangChain, LangGraph, Qdrant, Ollama) og kan køre lokalt uden API-nøgler. Det bruger hybrid søgning med reranking, en Plan-and-Execute-agent med samtalehukommelse, og RAGAS-baseret evaluering af svarkvaliteten.

### Funktioner

| Område | Implementering |
|---|---|
| Ustruktureret data | PyMuPDF-parser, dansk og engelsk tekstrensning, tre opdelingsstrategier (fast størrelse, rekursiv, semantisk) |
| Hybrid søgning | Qdrant til vektorsøgning kombineret med BM25, flettet med reciprocal rank fusion |
| Reranking | Cross-encoder `mmarco-mMiniLMv2-L12-H384-v1` |
| Agent-flows | Plan-and-Execute med seks værktøjer, ReAct-subagent og samtalehukommelse |
| Evaluering | RAGAS-metrikker (faithfulness, answer relevancy, context precision) |
| Sporbarhed | Hvert svar har kildehenvisninger med chunk-ID og sidenummer, samt struktureret logning |
| Provider-abstraktion | Factory-mønster, der gør det muligt at skifte mellem Ollama, OpenAI, Azure OpenAI, Anthropic og Google GenAI uden at ændre forretningskoden |
| Deployment | Docker Compose til lokal kørsel, Hugging Face Spaces til den offentlige demo |

### Sådan fungerer det

PDF-filer bliver læst med PyMuPDF, renset, opdelt i tekststykker (fast størrelse, rekursivt eller semantisk), indlejret med en flersproget sentence-transformer og gemt i Qdrant. Et BM25-indeks bygges på de samme tekststykker til nøgleordssøgning.

Når en bruger stiller et spørgsmål, kører systemet både den semantiske og den leksikale søgning, fletter resultaterne sammen med reciprocal rank fusion, og lader en cross-encoder rescore kandidaterne. De øverste tekststykker bliver sendt til en LLM, og svaret streames tilbage via SSE og vises i Streamlit-grænsefladen sammen med kilderne.

### To agent-tilstande

Systemet kan køre i to forskellige tilstande, der vælges via miljøvariablen `AGENT_MODE`.

**Pipeline** (`AGENT_MODE=pipeline`) er en fast LangGraph-DAG, der kører sprogdetektion, valgfri oversættelse, hybrid søgning, reranking, generering, plus en confidence-baseret retry-loop. Den fungerer fint med små lokale modeller, der ikke understøtter tool calling.

**Plan-and-Execute-agent** (`AGENT_MODE=react`, standard) er flertrinet: en planner nedbryder først spørgsmålet i delopgaver, en executor kører hver delopgave gennem en ReAct-subagent med adgang til værktøjerne nedenfor, og en synthesizer samler resultaterne til ét svar med kildehenvisninger. Den bruger samtalehukommelse til opfølgende spørgsmål og kræver en model, der understøtter tool calling.

| Værktøj | Formål |
|---|---|
| `hybrid_search(query, top_k)` | Henter relevante tekststykker via hybrid søgning og reranking |
| `multi_query_search(question, top_k)` | Nedbryder komplekse spørgsmål i delspørgsmål, søger på hver og fletter resultaterne |
| `search_within_document(document_id, query, top_k)` | Finder bestemte afsnit i et kendt dokument |
| `summarize_document(document_id)` | Laver et struktureret resumé af et dokument |
| `list_documents()` | Viser hvilke dokumenter, der ligger i vidensbasen |
| `fetch_document(document_id)` | Henter et helt dokument |

### Produktionshensyn

Hvert svar henviser tilbage til de tekststykker, det bygger på, med dokument-ID, sidenummer og selve teksten, så svarene kan kontrolleres bagefter. RAGAS-evalueringen i `src/evaluation/` måler faithfulness og context precision, så man kan opdage forringelser, før en ændring går i drift.

Konfigurationen ligger i miljøvariabler via `src/config.py`; der er ingen hardkodede stier, modelnavne eller API-nøgler. Koden importerer aldrig en provider-SDK direkte — LLM- og embedding-backends hentes gennem `create_llm()` og `create_embeddings()`, så man kan skifte mellem Ollama, OpenAI og andre uden at røre den øvrige kode. Standardopsætningen kører lokalt uden eksterne API-kald.

### Teknologivalg

| Kategori | Teknologi |
|---|---|
| Framework | FastAPI, uvicorn |
| Orkestrering | LangChain, LangGraph |
| Vektorlager | Qdrant (lokal tilstand) |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (384 dim) |
| LLM | `gemma4:e4b` via Ollama som standard |
| Sparse-søgning | rank_bm25 |
| Reranking | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| PDF-parsing | PyMuPDF |
| Evaluering | RAGAS |
| Grænseflade | Streamlit |

### Provider-understøttelse

LLM- og embedding-backends konfigureres via miljøvariabler. De understøttede providers er Ollama, OpenAI, Azure OpenAI, Anthropic, Google GenAI og Groq. Standardopsætningen (Ollama og HuggingFace) kører helt lokalt uden API-nøgler.

Se `.env.example` for konfiguration pr. provider.

### Prøv den live

Demoen ligger på [xq-dokumentassistent.hf.space](https://xq-dokumentassistent.hf.space).

Prøv disse spørgsmål — eller dine egne — på et hvilket som helst sprog.

- "Hvad er KU's politik for brug af AI-værktøjer?"
- "Hvilke regler gælder for brug af generativ AI i eksamen?"
- "Sammenlign reglerne for AI-brug i forskning og undervisning."

Det tredje spørgsmål udløser Plan-and-Execute-agenten, så man kan se den nedbryde spørgsmålet i delopgaver i realtid.

### Kom i gang

Kræver Python 3.11+ og [Ollama](https://ollama.com/).

```bash
git clone https://github.com/Xiiqiing/Dokumentassistent.git
cd Dokumentassistent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

ollama pull gemma4:e4b
python -m scripts.ingest          # læg PDF-filer i docs/ først

uvicorn src.api.main:app --reload  # http://localhost:8000
streamlit run src/ui/app.py        # http://localhost:8501
```

### Docker

Docker Compose håndterer Qdrant, API'et og Streamlit-grænsefladen samlet. API-containeren venter på, at Qdrant er oppe, og kører ingestion automatisk, hvis samlingen er tom.

#### Lokalt setup med Ollama og HuggingFace

```bash
cp .env.example .env
docker compose --profile local up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API-dokumentation | http://localhost:8000/docs |
| Streamlit-grænseflade | http://localhost:8501 |
| Qdrant-dashboard | http://localhost:6333/dashboard |

#### Cloud-setup med OpenAI, Anthropic eller andre

```bash
cp .env.example .env
# sæt LLM_PROVIDER, EMBEDDING_PROVIDER og din API-nøgle
docker compose up --build
```

#### Hugging Face Spaces

Et `Dockerfile` og en supervisor-konfiguration er inkluderet. Spacet kører Qdrant, API'et og grænsefladen bag nginx på port 7860.

### Projektstruktur

```
src/
  config.py                # konfiguration via miljøvariabler
  provider.py              # create_llm() og create_embeddings() factory
  models.py                # delte dataklasser
  ingestion/
    pdf_parser.py          # PyMuPDF-udtræk
    text_cleaner.py        # dansk og engelsk normalisering
    chunker.py             # fast størrelse, rekursiv og semantisk opdeling
    pipeline.py            # ingestion-orkestrering
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
    router.py              # pipeline-tilstand (AGENT_MODE=pipeline)
    tools.py               # seks retrieval-værktøjer og ToolResultStore
    plan_and_execute.py    # Plan-and-Execute-agent (AGENT_MODE=react)
    memory.py              # samtalehukommelse til flere spørgsmål
  evaluation/
    evaluator.py           # RAGAS-metrikker
  ui/
    app.py                 # Streamlit-frontend
scripts/
  ingest.py
  e2e_test.py
tests/
docs/                      # eksempel-PDF'er eller tekster (KU AI-dokumenter)
```

---

## English

A RAG application that lets users ask questions about documents in any language and get answers with source citations. The system is built on open source components (LangChain, LangGraph, Qdrant, Ollama) and can run locally without API keys. It uses hybrid search with reranking, a Plan-and-Execute agent with conversation memory, and RAGAS-based evaluation of answer quality.

### Capabilities

| Area | Implementation |
|---|---|
| Unstructured data | PyMuPDF parser, Danish and English text cleaning, three chunking strategies (fixed-size, recursive, semantic) |
| Hybrid retrieval | Qdrant dense vectors combined with BM25, fused via reciprocal rank fusion |
| Reranking | Cross-encoder `mmarco-mMiniLMv2-L12-H384-v1` |
| Agent flows | Plan-and-Execute with six tools, ReAct sub-agent and conversation memory |
| Evaluation | RAGAS metrics (faithfulness, answer relevancy, context precision) |
| Traceability | Each answer includes source references with chunk ID and page number, plus structured logging |
| Provider abstraction | Factory pattern that allows swapping between Ollama, OpenAI, Azure OpenAI, Anthropic and Google GenAI without touching business code |
| Deployment | Docker Compose for local setup, Hugging Face Spaces for the public demo |

### How it works

PDFs are parsed with PyMuPDF, cleaned, split into chunks (fixed-size, recursive, or semantic), embedded with a multilingual sentence-transformer, and stored in Qdrant. A BM25 index is built from the same chunks for keyword search.

At query time, both indexes are searched and the results merged with reciprocal rank fusion. A cross-encoder then rescores the candidates before the top chunks are passed to the LLM. The API streams the response over SSE and the Streamlit UI displays it together with the sources.

### Two agent modes

The system can run in two different modes, switchable via the `AGENT_MODE` environment variable.

**Pipeline** (`AGENT_MODE=pipeline`) is a fixed LangGraph DAG that runs language detection, optional translation, hybrid retrieval, reranking, generation, and a confidence-based retry loop. It works well with small local models that don't support tool calling.

**Plan-and-Execute agent** (`AGENT_MODE=react`, default) is multi-step: a planner first decomposes the query into sub-tasks, an executor runs each sub-task through a ReAct sub-agent with access to the tools listed below, and a synthesizer combines the results into a single cited answer. It uses conversation memory for follow-up questions and requires a model that supports tool calling.

| Tool | Purpose |
|---|---|
| `hybrid_search(query, top_k)` | Retrieves relevant passages via hybrid search and reranking |
| `multi_query_search(question, top_k)` | Decomposes complex questions into sub-queries, searches each, and merges the results |
| `search_within_document(document_id, query, top_k)` | Finds specific sections inside a known document |
| `summarize_document(document_id)` | Generates a structured summary of a document |
| `list_documents()` | Shows what is in the knowledge base |
| `fetch_document(document_id)` | Reads a full document |

### Production considerations

Every answer points back to the chunks it was built on, with document ID, page number and the chunk text itself, so answers can be checked after the fact. The RAGAS evaluation in `src/evaluation/` measures faithfulness and context precision, which lets you catch regressions before a change goes live.

Configuration lives in environment variables via `src/config.py`; there are no hardcoded paths, model names or API keys. The application code never imports a provider SDK directly — LLM and embedding backends are loaded through `create_llm()` and `create_embeddings()`, so you can switch between Ollama, OpenAI and others without touching the rest of the code. The default setup runs locally without any external API calls.

### Tech stack

| Category | Technology |
|---|---|
| Framework | FastAPI, uvicorn |
| Orchestration | LangChain, LangGraph |
| Vector store | Qdrant (local mode) |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (384 dim) |
| LLM | `gemma4:e4b` via Ollama (default) |
| Sparse search | rank_bm25 |
| Reranking | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| PDF parsing | PyMuPDF |
| Evaluation | RAGAS |
| UI | Streamlit |

### Provider support

LLM and embedding backends are configured through environment variables. Supported providers are Ollama, OpenAI, Azure OpenAI, Anthropic, Google GenAI and Groq. The default setup (Ollama and HuggingFace) runs entirely locally without any API keys.

See `.env.example` for per-provider configuration.

### Try it live

The demo lives at [xq-dokumentassistent.hf.space](https://xq-dokumentassistent.hf.space).

Try these questions, or ask one of your own in any language.

- "Hvad er KU's politik for brug af AI-værktøjer?"
- "Hvilke regler gælder for brug af generativ AI i eksamen?"
- "Sammenlign reglerne for AI-brug i forskning og undervisning."

The third question triggers the Plan-and-Execute agent, so you can watch it decompose the query into sub-tasks in real time.

### Quick start

Requires Python 3.11+ and [Ollama](https://ollama.com/).

```bash
git clone https://github.com/Xiiqiing/Dokumentassistent.git
cd Dokumentassistent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

ollama pull gemma4:e4b
python -m scripts.ingest          # place PDFs in docs/ first

uvicorn src.api.main:app --reload  # http://localhost:8000
streamlit run src/ui/app.py        # http://localhost:8501
```

### Docker

Docker Compose handles Qdrant, the API and the Streamlit UI together. The API container waits for Qdrant on startup and runs ingestion automatically if the collection is empty.

#### Local setup with Ollama and HuggingFace

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

#### Cloud setup with OpenAI, Anthropic or others

```bash
cp .env.example .env
# set LLM_PROVIDER, EMBEDDING_PROVIDER and your API key
docker compose up --build
```

#### Hugging Face Spaces

A `Dockerfile` and supervisor configuration are included. The Space runs Qdrant, the API and the UI behind nginx on port 7860.

### Project structure

```
src/
  config.py                # env-based configuration
  provider.py              # create_llm() and create_embeddings() factory
  models.py                # shared dataclasses
  ingestion/
    pdf_parser.py          # PyMuPDF extraction
    text_cleaner.py        # Danish and English normalization
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
    tools.py               # six retrieval tools and ToolResultStore
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

