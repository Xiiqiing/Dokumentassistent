"""RAGAS evaluation script: runs a fixed QA test set through the RAG pipeline
and reports retrieval + generation quality metrics.

Usage:
    python -m scripts.evaluate [--top-k 5] [--retrieval-only]

Output:
    A table of RAGAS scores printed to stdout.
    - Full mode:       faithfulness, answer_relevancy, context_precision, context_recall
    - Retrieval-only:  context_precision, context_recall
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from langchain_core.output_parsers import StrOutputParser

from src.config import load_settings
from src.evaluation.evaluator import RAGEvaluator
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

logger = logging.getLogger(__name__)

DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

# ---------------------------------------------------------------------------
# Test set: (question, ground_truth)
# Questions are in Danish to match the document language.
# Ground truths are short reference answers used by RAGAS context_recall.
# ---------------------------------------------------------------------------
TEST_SET: list[tuple[str, str]] = [
    (
        "Hvad er reglerne for brug af AI på KU?",
        "KU har retningslinjer for ansvarlig brug af AI-værktøjer, "
        "herunder krav om gennemsigtighed og akademisk integritet.",
    ),
    (
        "Hvilke regler gælder for behandling af personoplysninger?",
        "Behandling af personoplysninger på KU skal ske i overensstemmelse "
        "med GDPR og universitetets databeskyttelsespolitik.",
    ),
    (
        "Hvad er KUs politik for informationssikkerhed?",
        "KU kræver, at medarbejdere følger informationssikkerhedspolitikken, "
        "herunder adgangskontrol og beskyttelse af følsomme data.",
    ),
    (
        "Hvordan håndteres brud på datasikkerheden?",
        "Sikkerhedsbrud skal indberettes til IT-sikkerhedsteamet inden for "
        "72 timer i overensstemmelse med GDPR-kravene.",
    ),
    (
        "Hvad er reglerne for eksamen og snyd?",
        "KU har regler for eksamenssnyd, herunder konsekvenser som bortvisning "
        "fra eksamen og i alvorlige tilfælde bortvisning fra universitetet.",
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation over a fixed QA test set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks per query (default: 5).",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only measure context_precision and context_recall (no generation).",
    )
    return parser.parse_args()


def main() -> None:
    """Build the RAG pipeline, run the test set, and print RAGAS scores."""
    args = parse_args()
    settings = load_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=== RAGAS Evaluation Start ===")
    logger.info("Test set size: %d | top_k: %d | retrieval_only: %s",
                len(TEST_SET), args.top_k, args.retrieval_only)

    qdrant_tmp = tempfile.mkdtemp(prefix="eval_qdrant_")
    logger.info("Qdrant temp path: %s", qdrant_tmp)

    try:
        # --- 1) Create providers ---
        llm = create_llm(settings)
        embeddings = create_embeddings(settings)

        # --- 2) Ingest docs ---
        logger.info("Ingesting PDFs from %s ...", DOCS_DIR)
        pipeline = IngestionPipeline(
            strategy=ChunkStrategy.RECURSIVE,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = pipeline.ingest_directory(DOCS_DIR)
        logger.info("Total chunks: %d", len(chunks))

        if not chunks:
            logger.error("No chunks produced. Place PDFs in docs/ and retry.")
            sys.exit(1)

        # --- 3) Embed and index ---
        embedder = Embedder(embeddings)
        vectors = embedder.embed_batch([c.text for c in chunks])

        vector_store = VectorStore(
            path=qdrant_tmp,
            collection_name="eval",
            dimension=settings.embedding_dimension,
        )
        vector_store.add_chunks(chunks, vectors)

        bm25 = BM25Search()
        bm25.index(chunks)

        # --- 4) Build router ---
        hybrid = HybridRetriever(
            vector_store=vector_store,
            bm25_search=bm25,
            embedder=embedder,
            dense_weight=settings.dense_weight,
            bm25_weight=settings.bm25_weight,
        )
        reranker = Reranker(model=create_reranker(settings.reranker_model))
        classifier = IntentClassifier(llm=llm, model_name=settings.generation_model)
        generator = llm | StrOutputParser()
        router = QueryRouter(
            intent_classifier=classifier,
            hybrid_retriever=hybrid,
            reranker=reranker,
            generator=generator,
        )

        # --- 5) Run test set ---
        questions: list[str] = []
        answers: list[str] = []
        contexts: list[list[str]] = []
        ground_truths: list[str] = []

        for question, ground_truth in TEST_SET:
            logger.info("Running query: %s", question)
            response = router.route(query=question, top_k=args.top_k)
            questions.append(question)
            answers.append(response.answer)
            contexts.append([r.chunk.text for r in response.sources])
            ground_truths.append(ground_truth)

        # --- 6) Evaluate ---
        evaluator = RAGEvaluator(llm=llm)

        if args.retrieval_only:
            scores = evaluator.evaluate_retrieval(
                questions=questions,
                contexts=contexts,
                ground_truths=ground_truths,
            )
        else:
            scores = evaluator.evaluate(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths,
            )

        # --- 7) Print results ---
        print("\n" + "=" * 50)
        print("RAGAS EVALUATION RESULTS")
        print("=" * 50)
        for metric, score in scores.items():
            print(f"  {metric:<30} {score:.4f}")
        print("=" * 50)

        logger.info("=== RAGAS Evaluation Complete ===")

    finally:
        shutil.rmtree(qdrant_tmp, ignore_errors=True)
        logger.info("Cleaned up temp Qdrant at %s", qdrant_tmp)


if __name__ == "__main__":
    main()
