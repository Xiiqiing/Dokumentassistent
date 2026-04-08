# Dokumentassistent

## Live demo 
Hosted on Hugging Face Spaces: [xq-dokumentassistent.hf.space](https://xq-dokumentassistent.hf.space)

[Skip to English ↓](#english)

## Dansk

En produktionsklar RAG-applikation, der gør det muligt at stille spørgsmål til dokumenter på dansk og få svar med kildehenvisninger. Systemet er bygget på open source-komponenter (LangChain, LangGraph, Qdrant, Ollama) og kan køre helt lokalt uden eksterne API-kald. Det implementerer hybrid søgning med reranking, en Plan-and-Execute agent med samtalehukommelse, og RAGAS-baseret evaluering af svarkvaliteten.

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

**Pipeline** (`AGENT_MODE=pipeline`) bygger på en fast LangGraph-graf med sprogdetektion, valgfri oversættelse, hybrid søgning, reranking og generering. Tilstanden har en confidence-baseret retry-loop og fungerer fint med lette lokale modeller.

**Plan-and-Execute Agent** (`AGENT_MODE=react`, standard) er en flertrinsagent, hvor en planner først nedbryder spørgsmålet i delopgaver, en executor kører hver delopgave gennem en ReAct-subagent med adgang til et sæt værktøjer, og en synthesizer producerer det endelige svar med kildehenvisninger. Tilstanden indeholder samtalehukommelse til opfølgende spørgsmål og kræver en model, der understøtter tool calling.

| Værktøj | Formål |
|---|---|
| `hybrid_search(query, top_k)` | Henter relevante tekststykker via hybrid søgning og reranking |
| `multi_query_search(question, top_k)` | Nedbryder komplekse spørgsmål i delspørgsmål, søger på hver og fletter resultaterne |
| `search_within_document(document_id, query, top_k)` | Finder bestemte afsnit i et kendt dokument |
| `summarize_document(document_id)` | Laver et struktureret resumé af et dokument |
| `list_documents()` | Viser hvilke dokumenter, der ligger i vidensbasen |
| `fetch_document(document_id)` | Henter et helt dokument |

### Produktionshensyn

- **Sporbarhed.** Hvert genereret svar har kildehenvisninger på chunk-niveau med dokument-ID, sidenummer og tekststykke, så det kan revideres bagudrettet.
- **Governance.** RAGAS-evalueringspipelinen i `src/evaluation/` gør det muligt at måle faithfulness og context precision, før ændringer slippes løs i produktion.
- **Konfigurerbarhed.** Ingen hardkodede stier, modelnavne eller API-nøgler. Alt styres via miljøvariabler gennem `src/config.py`.
- **Provider-neutralitet.** Forretningskoden importerer aldrig en provider-SDK direkte. LLM- og embedding-backends skiftes via factory-funktionerne `create_llm()` og `create_embeddings()`, hvilket undgår vendor lock-in.
- **Lokal som standard.** Standardkonfigurationen kører helt uden eksterne API-kald og passer til miljøer med strenge krav til datahjemsted.
- **Pakket i containere.** Docker Compose til lokal kørsel og Hugging Face Spaces til den offentlige demo.

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

Prøv for eksempel disse spørgsmål på dansk.

- "Hvad er KU's politik for brug af AI-værktøjer?"
- "Hvilke regler gælder for brug af generativ AI i eksamen?"
- "Sammenlign reglerne for AI-brug i forskning og undervisning."

Det sidste spørgsmål udløser Plan-and-Execute-agenten, så man kan se den nedbryde spørgsmålet i delopgaver i realtid.

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

A production-ready RAG application that lets users ask questions about documents in Danish and receive answers with source citations. The system is built on open source components (LangChain, LangGraph, Qdrant, Ollama) and can run fully local without any external API calls. It implements hybrid search with reranking, a Plan-and-Execute agent with conversation memory, and RAGAS-based evaluation of answer quality.

### Capabilities

| Area | Implementation |
|---|---|
| Unstructured data | PyMuPDF parser, Danish and English text cleaning, three chunking strategies (fixed-size, recursive, semantic) |
| Hybrid retrieval | Qdrant dense vectors combined with BM25, fused via reciprocal rank fusion |
| Reranking | Cross-encoder `mmarco-mMiniLMv2-L12-H384-v1` |
| Agent flows | Plan-and-Execute with six tools, ReAct sub-agent and conversation memory |
| Evaluation | RAGAS metrics (faithfulness, answer relevancy, context precision) |
| Traceability | Each answer carries source references with chunk ID and page number, plus structured logging |
| Provider abstraction | Factory pattern that allows swapping between Ollama, OpenAI, Azure OpenAI, Anthropic and Google GenAI without touching business code |
| Deployment | Docker Compose for local setup, Hugging Face Spaces for the public demo |

### How it works

PDFs are parsed with PyMuPDF, cleaned, split into chunks (fixed-size, recursive, or semantic), embedded with a multilingual sentence-transformer, and stored in Qdrant. A BM25 index is built from the same chunks for keyword search.

At query time, both indexes are searched and the results merged with reciprocal rank fusion. A cross-encoder then rescores the candidates before the top chunks are passed to the LLM. The API streams the response over SSE and the Streamlit UI displays it together with the sources.

### Two agent modes

The system can run in two different modes, switchable via the `AGENT_MODE` environment variable.

**Pipeline** (`AGENT_MODE=pipeline`) is built on a fixed LangGraph graph with language detection, optional translation, hybrid retrieval, reranking, and generation. The mode has a confidence-based retry loop and works well with lightweight local models.

**Plan-and-Execute Agent** (`AGENT_MODE=react`, default) is a multi-step agent where a planner first decomposes the query into sub-tasks, an executor runs each sub-task through a ReAct sub-agent with access to a set of tools, and a synthesizer produces the final answer with citations. The mode includes conversation memory for follow-up questions and requires a model that supports tool calling.

| Tool | Purpose |
|---|---|
| `hybrid_search(query, top_k)` | Retrieves relevant passages via hybrid search and reranking |
| `multi_query_search(question, top_k)` | Decomposes complex questions into sub-queries, searches each, and merges the results |
| `search_within_document(document_id, query, top_k)` | Finds specific sections inside a known document |
| `summarize_document(document_id)` | Generates a structured summary of a document |
| `list_documents()` | Shows what is in the knowledge base |
| `fetch_document(document_id)` | Reads a full document |

### Production considerations

- **Traceability.** Every generated answer carries chunk-level source references with document ID, page number and span, so it can be audited and reviewed afterwards.
- **Governance.** The RAGAS evaluation pipeline in `src/evaluation/` lets you measure faithfulness and context precision before promoting changes to production.
- **Configurability.** No hardcoded paths, model names or API keys. Everything is controlled via environment variables through `src/config.py`.
- **Provider neutrality.** Business code never imports a provider SDK directly. LLM and embedding backends swap via the `create_llm()` and `create_embeddings()` factory functions, which avoids vendor lock-in.
- **Local-first.** The default configuration runs entirely without external API calls and fits environments with strict data residency requirements.
- **Containerized.** Docker Compose for local runs and Hugging Face Spaces for the public demo.

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

Try asking these questions in Danish.

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
