"""Standalone ingestion script: parse PDFs, chunk, embed, and persist to Qdrant.

Usage:
    python -m scripts.ingest [--docs-dir DOCS_DIR] [--strategy recursive]
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports resolve.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import load_settings
from src.models import ChunkStrategy
from src.provider import create_embeddings
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the Qdrant vector store.",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=None,
        help="Path to the directory containing PDFs (default: <project_root>/docs).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[s.value for s in ChunkStrategy],
        default=None,
        help="Chunking strategy (default: from CHUNK_STRATEGY env or 'recursive').",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full ingestion pipeline and persist results to Qdrant."""
    args = parse_args()
    settings = load_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    docs_dir = args.docs_dir or str(Path(_PROJECT_ROOT) / "docs")
    strategy_value = args.strategy or "recursive"
    strategy = ChunkStrategy(strategy_value)

    logger.info("=== KU Doc Assistant — Ingestion ===")
    logger.info("Docs directory : %s", docs_dir)
    logger.info("Chunk strategy : %s", strategy.value)
    logger.info("Chunk size     : %d", settings.chunk_size)
    logger.info("Chunk overlap  : %d", settings.chunk_overlap)
    logger.info("Qdrant path    : %s", settings.qdrant_path)

    # 1. Parse and chunk PDFs
    pipeline = IngestionPipeline(
        strategy=strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = pipeline.ingest_directory(docs_dir)
    if not chunks:
        logger.warning("No chunks produced — nothing to ingest.")
        return

    # 2. Embed chunks
    embeddings_model = create_embeddings(settings)
    embedder = Embedder(embeddings_model)
    texts = [chunk.text for chunk in chunks]
    logger.info("Embedding %d chunks ...", len(texts))
    vectors = embedder.embed_batch(texts)

    # 3. Store in Qdrant (persistent path from config)
    store = VectorStore(
        path=settings.qdrant_path,
        collection_name=settings.collection_name,
        dimension=settings.embedding_dimension,
        url=settings.qdrant_url,
    )
    store.add_chunks(chunks, vectors)

    logger.info("=== Ingestion complete: %d chunks indexed ===", len(chunks))


if __name__ == "__main__":
    main()
