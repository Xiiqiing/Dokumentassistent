"""End-to-end test: PDF ingestion → chunking → embedding → hybrid search → rerank → answer.

Runs the full RAG pipeline directly against src/ modules without FastAPI.
Uses local providers (Ollama for LLM, HuggingFace for embeddings).
"""

import logging
import os
import shutil
import sys
import tempfile

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.config import load_settings
from src.models import ChunkStrategy
from src.provider import create_embeddings, create_llm, create_reranker
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.agent.intent_classifier import IntentClassifier
from src.agent.router import QueryRouter

from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
TEST_QUERY = "Hvad er reglerne for brug af AI på KU?"


def main() -> None:
    """Run full end-to-end RAG pipeline test."""
    # --- Config ---
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=== E2E Test Start ===")
    logger.info("LLM provider: %s | Embedding provider: %s", settings.llm_provider, settings.embedding_provider)

    # Use a temporary Qdrant path so we don't pollute the main store
    qdrant_tmp = tempfile.mkdtemp(prefix="e2e_qdrant_")
    logger.info("Qdrant temp path: %s", qdrant_tmp)

    try:
        # --- 1) Create providers ---
        logger.info("Creating LLM and embeddings...")
        llm = create_llm(settings)
        embeddings = create_embeddings(settings)

        # --- 2) Ingest all PDFs from docs/ ---
        logger.info("Ingesting PDFs from %s ...", DOCS_DIR)
        pipeline = IngestionPipeline(
            strategy=ChunkStrategy.RECURSIVE,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = pipeline.ingest_directory(DOCS_DIR)
        logger.info("Total chunks created: %d", len(chunks))

        if not chunks:
            logger.error("No chunks produced. Check that docs/ contains valid PDFs.")
            sys.exit(1)

        # --- 3) Embed and index ---
        logger.info("Embedding %d chunks...", len(chunks))
        embedder = Embedder(embeddings)
        vectors = embedder.embed_batch([c.text for c in chunks])
        logger.info("Embedding complete (dim=%d)", len(vectors[0]))

        logger.info("Indexing into Qdrant...")
        vector_store = VectorStore(
            path=qdrant_tmp,
            collection_name="e2e_test",
            dimension=settings.embedding_dimension,
        )
        vector_store.add_chunks(chunks, vectors)

        logger.info("Building BM25 index...")
        bm25 = BM25Search()
        bm25.index(chunks)

        # --- 4) Build retrieval + generation pipeline ---
        hybrid = HybridRetriever(
            vector_store=vector_store,
            bm25_search=bm25,
            embedder=embedder,
            dense_weight=settings.dense_weight,
            bm25_weight=settings.bm25_weight,
        )
        reranker = Reranker(model=create_reranker(settings.reranker_model))
        classifier = IntentClassifier(llm=llm)
        generator = llm | StrOutputParser()
        router = QueryRouter(
            intent_classifier=classifier,
            hybrid_retriever=hybrid,
            reranker=reranker,
            generator=generator,
        )

        # --- 5) Run query ---
        logger.info("Query: %s", TEST_QUERY)
        response = router.route(query=TEST_QUERY, top_k=settings.top_k)

        # --- Print results ---
        print("\n" + "=" * 70)
        print("QUERY:", TEST_QUERY)
        print("=" * 70)
        print(f"\nINTENT: {response.intent.value}")
        print(f"CONFIDENCE: {response.confidence:.3f}")
        print(f"\nANSWER:\n{response.answer}")
        print("\nSOURCES:")
        for i, result in enumerate(response.sources, 1):
            src = result.chunk.metadata.get("source", "unknown")
            page = result.chunk.metadata.get("page_number", "?")
            print(f"  [{i}] {os.path.basename(src)} (p.{page}) — score: {result.score:.4f}")
            print(f"      {result.chunk.text[:120]}...")
        print("=" * 70)

        logger.info("=== E2E Test Complete ===")

    finally:
        # Clean up temp Qdrant data
        shutil.rmtree(qdrant_tmp, ignore_errors=True)
        logger.info("Cleaned up temp Qdrant at %s", qdrant_tmp)


if __name__ == "__main__":
    main()
