"""API route definitions for the document assistant."""

import logging
import os
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

if TYPE_CHECKING:
    from src.agent.router import QueryRouter
    from src.config import Settings
    from src.ingestion.pipeline import IngestionPipeline
    from src.retrieval.bm25_search import BM25Search
    from src.retrieval.embedder import Embedder
    from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

_query_router: "QueryRouter | None" = None
_ingestion_pipeline: "IngestionPipeline | None" = None
_embedder: "Embedder | None" = None
_vector_store: "VectorStore | None" = None
_bm25_search: "BM25Search | None" = None
_settings: "Settings | None" = None


def set_dependencies(
    query_router: "QueryRouter",
    ingestion_pipeline: "IngestionPipeline",
    embedder: "Embedder",
    vector_store: "VectorStore",
    bm25_search: "BM25Search",
    settings: "Settings",
) -> None:
    """Inject dependencies from the application factory.

    Args:
        query_router: Configured QueryRouter instance.
        ingestion_pipeline: Configured IngestionPipeline instance.
        embedder: Embedder instance for generating embeddings.
        vector_store: VectorStore instance for dense indexing.
        bm25_search: BM25Search instance for sparse indexing.
        settings: Application settings.
    """
    global _query_router, _ingestion_pipeline, _embedder, _vector_store, _bm25_search, _settings
    _query_router = query_router
    _ingestion_pipeline = ingestion_pipeline
    _embedder = embedder
    _vector_store = vector_store
    _bm25_search = bm25_search
    _settings = settings


class QueryRequest(BaseModel):
    """Request body for the query endpoint."""

    question: str
    top_k: int = 5
    strategy: str = "recursive"


class PipelineResultItem(BaseModel):
    """A single result item in pipeline details."""

    document_id: str
    chunk_id: str
    score: float
    source: str


class PipelineDetailsResponse(BaseModel):
    """Intermediate pipeline data for the query response."""

    original_query: str = ""
    retrieval_query: str = ""
    detected_language: str = ""
    translated: bool = False
    dense_results: list[PipelineResultItem] = []
    sparse_results: list[PipelineResultItem] = []
    fused_results: list[PipelineResultItem] = []
    reranked_results: list[PipelineResultItem] = []


class QueryResponse(BaseModel):
    """Response body for the query endpoint."""

    answer: str
    sources: list[dict[str, str | float]]
    intent: str
    confidence: float
    pipeline_details: PipelineDetailsResponse = PipelineDetailsResponse()


class IngestRequest(BaseModel):
    """Request body for the document ingestion endpoint."""

    file_path: str
    strategy: str = "recursive"


class IngestResponse(BaseModel):
    """Response body for the document ingestion endpoint."""

    document_id: str
    chunks_created: int


class HealthResponse(BaseModel):
    """Response body for the health check endpoint."""

    status: str
    version: str
    llm_provider: str = ""
    llm_model: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse with service status and version.
    """
    llm_provider = ""
    llm_model = ""
    embedding_provider = ""
    embedding_model = ""
    if _settings is not None:
        llm_provider = _settings.llm_provider
        embedding_provider = _settings.embedding_provider
        embedding_model = _settings.embedding_model
        model_map = {
            "ollama": _settings.ollama_model,
            "openai": _settings.openai_model,
            "azure_openai": _settings.azure_openai_deployment,
            "anthropic": _settings.anthropic_model,
            "google_genai": _settings.google_model,
        }
        llm_model = model_map.get(llm_provider, _settings.generation_model)
    return HealthResponse(
        status="ok",
        version="0.1.0",
        llm_provider=llm_provider,
        llm_model=llm_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Query the document knowledge base.

    Args:
        request: Query parameters including question and retrieval settings.

    Returns:
        QueryResponse with generated answer and source documents.
    """
    logger.info("Received query: %s", request.question)

    response = _query_router.route(query=request.question, top_k=request.top_k)

    sources = [
        {
            "chunk_id": result.chunk.chunk_id,
            "document_id": result.chunk.document_id,
            "text": result.chunk.text,
            "score": result.score,
            "source": result.source,
        }
        for result in response.sources
    ]

    def _to_pipeline_items(results: list) -> list[PipelineResultItem]:
        return [
            PipelineResultItem(
                document_id=r.chunk.document_id,
                chunk_id=r.chunk.chunk_id,
                score=r.score,
                source=r.source,
            )
            for r in results
        ]

    pd = response.pipeline_details
    pipeline_details = PipelineDetailsResponse(
        original_query=pd.original_query,
        retrieval_query=pd.retrieval_query,
        detected_language=pd.detected_language,
        translated=pd.translated,
        dense_results=_to_pipeline_items(pd.dense_results),
        sparse_results=_to_pipeline_items(pd.sparse_results),
        fused_results=_to_pipeline_items(pd.fused_results),
        reranked_results=_to_pipeline_items(pd.reranked_results),
    )

    return QueryResponse(
        answer=response.answer,
        sources=sources,
        intent=response.intent.value,
        confidence=response.confidence,
        pipeline_details=pipeline_details,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest) -> IngestResponse:
    """Ingest a new document into the knowledge base.

    Args:
        request: Ingestion parameters including file path and strategy.

    Returns:
        IngestResponse with document ID and number of chunks created.
    """
    if not os.path.isfile(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    logger.info("Ingesting document: %s", request.file_path)

    chunks = _ingestion_pipeline.ingest_pdf(request.file_path)

    if chunks:
        embeddings = _embedder.embed_batch([chunk.text for chunk in chunks])
        _vector_store.add_chunks(chunks, embeddings)
        _bm25_search.index(chunks)

    document_id = os.path.basename(request.file_path)
    logger.info("Ingested %d chunks for document %s", len(chunks), document_id)

    return IngestResponse(
        document_id=document_id,
        chunks_created=len(chunks),
    )
