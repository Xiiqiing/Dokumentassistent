"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser

from src.config import load_settings
from src.provider import create_llm, create_embeddings, create_reranker
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.agent.intent_classifier import IntentClassifier
from src.agent.router import QueryRouter
from src.agent.react_router import ReActRouter
from src.ingestion.pipeline import IngestionPipeline
from src.api.routes import router, set_dependencies

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = load_settings()

    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))

    llm = create_llm(settings)
    embeddings = create_embeddings(settings)

    embedder = Embedder(embeddings=embeddings)
    vector_store = VectorStore(
        path=settings.qdrant_path,
        collection_name=settings.collection_name,
        dimension=settings.embedding_dimension,
        url=settings.qdrant_url,
    )
    bm25_search = BM25Search()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Load stored chunks from Qdrant and rebuild the BM25 index on startup."""
        chunks = vector_store.get_all_chunks()
        if chunks:
            bm25_search.index(chunks)
            logger.info("Rebuilt BM25 index with %d chunks from Qdrant", len(chunks))
        else:
            logger.info("No existing chunks in Qdrant; BM25 index is empty")
        yield

    application = FastAPI(
        title="KU Doc Assistant",
        description="RAG-based document assistant for University of Copenhagen.",
        version="0.1.0",
        lifespan=lifespan,
    )
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_search=bm25_search,
        embedder=embedder,
        dense_weight=settings.dense_weight,
        bm25_weight=settings.bm25_weight,
    )
    reranker = Reranker(model=create_reranker(settings.reranker_model))

    if settings.agent_mode == "react":
        logger.info("Agent mode: ReAct (tool-calling loop)")
        query_router: QueryRouter | ReActRouter = ReActRouter(
            llm=llm,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            vector_store=vector_store,
            default_top_k=settings.top_k,
        )
    else:
        logger.info("Agent mode: pipeline (fixed DAG)")
        intent_classifier = IntentClassifier(llm=llm, model_name=settings.generation_model)
        generator = llm | StrOutputParser()
        query_router = QueryRouter(
            intent_classifier=intent_classifier,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            generator=generator,
            translate_query=settings.translate_query,
        )

    set_dependencies(
        query_router=query_router,
        ingestion_pipeline=IngestionPipeline(
            strategy=_parse_strategy(settings),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            embeddings=embeddings,
        ),
        embedder=embedder,
        vector_store=vector_store,
        bm25_search=bm25_search,
        settings=settings,
    )

    application.include_router(router)

    logger.info("KU Doc Assistant application created successfully")
    return application


def _parse_strategy(settings: "Settings") -> "ChunkStrategy":  # noqa: F821
    """Return the default chunking strategy from config."""
    from src.models import ChunkStrategy
    return ChunkStrategy.SEMANTIC


app: FastAPI = create_app()
