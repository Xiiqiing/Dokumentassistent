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

A RAG application that lets users ask questions about documents in any language and get answers with source citations. Built with LangChain, LangGraph, and Qdrant, it supports multiple LLM backends (Azure OpenAI, AWS Bedrock, Ollama) switchable via environment variables, and can run fully locally without API keys. Features include hybrid search with cross-encoder reranking, two LangGraph agent modes, RAGAS-based evaluation, and CI/CD pipelines for Azure Container Apps and AWS ECS Fargate.

### Capabilities

| Area | Implementation |
|---|---|
| Unstructured data | PyMuPDF parser, Danish and English text cleaning, three chunking strategies (fixed-size, recursive, semantic) |
| Hybrid retrieval | Qdrant dense vectors combined with BM25, fused via reciprocal rank fusion |
| Reranking | Cross-encoder `mmarco-mMiniLMv2-L12-H384-v1` |
| Agent flows | Plan-and-Execute with six tools, ReAct sub-agent and conversation memory |
| Evaluation | RAGAS metrics (faithfulness, answer relevancy, context precision) |
| Traceability | Each answer includes source references with chunk ID and page number, plus structured logging |
| Provider abstraction | Factory pattern that allows swapping between Ollama, OpenAI, Azure OpenAI, AWS Bedrock, Anthropic and Google GenAI without touching business code |
| Deployment | Docker Compose (local), Azure Container Apps, AWS ECS Fargate, Hugging Face Spaces (demo) |

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

LLM and embedding backends are configured through environment variables. Supported providers are Ollama, OpenAI, Azure OpenAI, AWS Bedrock, Anthropic, Google GenAI and Groq. The default setup (Ollama and HuggingFace) runs entirely locally without any API keys.

See `.env.example` for per-provider configuration.

### Cloud deployment

The application is cloud-agnostic by design. Business code depends only on LangChain abstract interfaces; the concrete provider is selected at deploy time via environment variables.

| Layer | Azure | AWS | Local |
|---|---|---|---|
| LLM / Embeddings | Azure OpenAI | Bedrock (Claude, Titan) | Ollama + HuggingFace |
| Container registry | ACR | ECR | - |
| Runtime | Container Apps | ECS Fargate | docker-compose |
| CI/CD | GitHub Actions | GitHub Actions | - |

GitHub Actions workflows are included for both clouds:

- `ci.yml` runs lint, type check, and tests on every push and PR
- `deploy-azure.yml` builds, pushes to ACR, and deploys to Azure Container Apps
- `deploy-aws.yml` builds, pushes to ECR, and deploys to ECS Fargate

Health probes (`/health/live` for liveness, `/health/ready` for readiness) are used by both Kubernetes and container orchestrators to manage rolling deployments.

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

#### Azure Container Apps

```bash
# Login and set variables
az login
ACR_NAME=<your-acr-name>
RG=<your-resource-group>
APP_NAME=doc-assistant

# Build and push
az acr login --name $ACR_NAME
docker build -f Dockerfile.compose -t $ACR_NAME.azurecr.io/doc-assistant:latest .
docker push $ACR_NAME.azurecr.io/doc-assistant:latest

# Deploy
az containerapp create \
  --name $APP_NAME \
  --resource-group $RG \
  --image $ACR_NAME.azurecr.io/doc-assistant:latest \
  --target-port 8000 \
  --env-vars LLM_PROVIDER=azure_openai \
    AZURE_OPENAI_API_KEY=secretref:aoai-key \
    AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/ \
    AZURE_OPENAI_DEPLOYMENT=<deployment> \
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment>
```

CI/CD is automated via `.github/workflows/deploy-azure.yml`.

#### AWS ECS Fargate

```bash
# Login and set variables
AWS_REGION=eu-west-1
ECR_REPO=<account-id>.dkr.ecr.$AWS_REGION.amazonaws.com/doc-assistant

# Build and push
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO
docker build -f Dockerfile.compose -t $ECR_REPO:latest .
docker push $ECR_REPO:latest

# Deploy (update existing ECS service)
aws ecs update-service \
  --cluster doc-assistant \
  --service doc-assistant \
  --force-new-deployment
```

CI/CD is automated via `.github/workflows/deploy-aws.yml`. The task definition should set `LLM_PROVIDER=bedrock` and grant the task role Bedrock access.

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
    routes.py              # /query, /ingest, /health/live, /health/ready
  agent/
    intent_classifier.py
    router.py              # pipeline mode (AGENT_MODE=pipeline)
    tools.py               # six retrieval tools and ToolResultStore
    plan_and_execute.py    # Plan-and-Execute agent (AGENT_MODE=react)
    memory.py              # conversation memory for multi-turn
    session_store.py       # SQLite-backed per-session memory persistence
  evaluation/
    evaluator.py           # RAGAS metrics
  ui/
    app.py                 # Streamlit frontend
scripts/
  ingest.py
  evaluate.py              # RAGAS evaluation CLI
  e2e_test.py
tests/
docs/                      # example PDFs or texts (KU AI public documents)
infra/
  bicep/
    main.bicep             # Azure Container Apps (Bicep)
    main.bicepparam        # parameter file
  terraform/
    main.tf                # Azure Container Apps (Terraform)
    variables.tf           # input variables
    outputs.tf             # deployment outputs
.github/
  workflows/
    ci.yml                 # lint + test on push/PR
    deploy-azure.yml       # build, push ACR, deploy Container Apps
    deploy-aws.yml         # build, push ECR, deploy ECS Fargate
```

