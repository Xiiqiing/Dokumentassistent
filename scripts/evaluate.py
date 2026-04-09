"""RAGAS matrix evaluation: run a curated test set through multiple
configurations of the RAG pipeline and persist results.

Two pre-defined experiments:

    chunking — varies CHUNK_STRATEGY (FIXED_SIZE / RECURSIVE / SEMANTIC),
               fixes router=react, top_k=5
    router   — varies router (react / pipeline), fixes
               chunking=RECURSIVE, top_k=5
    all      — runs both experiments
    quick    — single cell (chunking=RECURSIVE, router=react), for smoke testing

Each cell:
  1. Builds an in-memory Qdrant + BM25 index for its chunking strategy
     (indices are reused across cells with the same chunking).
  2. Runs each test question through the chosen router.
  3. Sends the (question, answer, contexts, reference) tuples to RAGAS,
     using the judge LLM from EVALUATOR_LLM_PROVIDER (or generation LLM if unset).

Output:
    eval/runs/<timestamp>_<config>.json   — full result (config + per-sample)
    eval/runs/<timestamp>_<config>.md     — human-readable aggregate table

Usage:
    python -m scripts.evaluate --experiment quick
    python -m scripts.evaluate --experiment chunking
    python -m scripts.evaluate --experiment all --top-k 5

Env vars:
    LLM_PROVIDER=groq                       (generation LLM)
    GROQ_API_KEY=gsk_...                    (required for groq)
    EVALUATOR_LLM_PROVIDER=groq             (judge LLM; empty = reuse generation)
"""

import argparse
import datetime as dt
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain_core.output_parsers import StrOutputParser

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.agent.intent_classifier import IntentClassifier  # noqa: E402
from src.agent.plan_and_execute import PlanAndExecuteRouter  # noqa: E402
from src.agent.router import QueryRouter  # noqa: E402
from src.config import Settings, load_settings  # noqa: E402
from src.evaluation.evaluator import RAGEvaluator  # noqa: E402
from src.ingestion.pipeline import IngestionPipeline  # noqa: E402
from src.models import ChunkStrategy, GenerationResponse  # noqa: E402
from src.provider import (  # noqa: E402
    create_embeddings,
    create_evaluator_llm,
    create_llm,
    create_reranker,
)
from src.retrieval.bm25_search import BM25Search  # noqa: E402
from src.retrieval.embedder import Embedder  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402
from src.retrieval.reranker import Reranker  # noqa: E402
from src.retrieval.vector_store import VectorStore  # noqa: E402

logger = logging.getLogger(__name__)

DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
QA_SET_PATH = os.path.join(PROJECT_ROOT, "eval", "qa_set.yaml")
RUNS_DIR = os.path.join(PROJECT_ROOT, "eval", "runs")


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """Single experiment cell configuration."""

    name: str
    chunking: ChunkStrategy
    router: str  # "react" or "pipeline"
    top_k: int


def _build_run_configs(experiment: str, top_k: int) -> list[RunConfig]:
    """Build the list of RunConfig cells for the requested experiment.

    Args:
        experiment: One of "quick", "chunking", "router", "all".
        top_k: Top-K passed to every cell.

    Returns:
        Ordered list of RunConfig cells to execute.

    Raises:
        ValueError: If the experiment name is unknown.
    """
    if experiment == "quick":
        return [
            RunConfig("recursive_react", ChunkStrategy.RECURSIVE, "react", top_k),
        ]
    if experiment == "chunking":
        return [
            RunConfig("fixed_react", ChunkStrategy.FIXED_SIZE, "react", top_k),
            RunConfig("recursive_react", ChunkStrategy.RECURSIVE, "react", top_k),
            RunConfig("semantic_react", ChunkStrategy.SEMANTIC, "react", top_k),
        ]
    if experiment == "router":
        return [
            RunConfig("recursive_pipeline", ChunkStrategy.RECURSIVE, "pipeline", top_k),
            RunConfig("recursive_react", ChunkStrategy.RECURSIVE, "react", top_k),
        ]
    if experiment == "all":
        return [
            RunConfig("fixed_react", ChunkStrategy.FIXED_SIZE, "react", top_k),
            RunConfig("recursive_react", ChunkStrategy.RECURSIVE, "react", top_k),
            RunConfig("semantic_react", ChunkStrategy.SEMANTIC, "react", top_k),
            RunConfig("recursive_pipeline", ChunkStrategy.RECURSIVE, "pipeline", top_k),
        ]
    raise ValueError(
        f"Unknown experiment '{experiment}'. Expected one of: "
        f"quick, chunking, router, all."
    )


# ---------------------------------------------------------------------------
# QA set loading
# ---------------------------------------------------------------------------


def _load_qa_set(path: str) -> list[dict[str, Any]]:
    """Load and validate the curated QA set, keeping only reviewed entries.

    Args:
        path: Path to eval/qa_set.yaml.

    Returns:
        List of reviewed question dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is malformed or contains no reviewed entries.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"QA set not found at {path}. Run scripts/generate_qa_set.py first, "
            f"then curate the draft into eval/qa_set.yaml."
        )
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if isinstance(data, dict):
        questions = data.get("questions", [])
    else:
        questions = data
    if not isinstance(questions, list):
        raise ValueError(f"Invalid QA set format in {path}: expected list of questions.")
    reviewed = [q for q in questions if isinstance(q, dict) and q.get("reviewed")]
    if not reviewed:
        raise ValueError(
            f"No reviewed questions in {path}. "
            f"Set 'reviewed: true' on entries you want to keep."
        )
    logger.info("Loaded %d reviewed questions from %s", len(reviewed), path)
    return reviewed


# ---------------------------------------------------------------------------
# Index building (with caching across cells)
# ---------------------------------------------------------------------------


@dataclass
class BuiltIndex:
    """Bundle of artifacts produced by ingesting docs/ with one chunking strategy."""

    vector_store: VectorStore
    bm25: BM25Search
    embedder: Embedder
    qdrant_path: str  # tmp dir owned by this index, cleaned up at the end


def _build_index(
    chunking: ChunkStrategy,
    settings: Settings,
    embeddings: Any,
) -> BuiltIndex:
    """Ingest docs/ with the given chunking strategy and build dense + sparse indices.

    Args:
        chunking: ChunkStrategy enum value.
        settings: Application settings.
        embeddings: LangChain Embeddings instance.

    Returns:
        BuiltIndex bundle ready to feed into a router.
    """
    logger.info("Building index for chunking=%s ...", chunking.value)
    pipeline = IngestionPipeline(
        strategy=chunking,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embeddings=embeddings if chunking == ChunkStrategy.SEMANTIC else None,
    )
    chunks = pipeline.ingest_directory(DOCS_DIR)
    if not chunks:
        raise RuntimeError(f"No chunks produced for chunking={chunking.value}")
    logger.info("  -> %d chunks", len(chunks))

    embedder = Embedder(embeddings)
    vectors = embedder.embed_batch([c.text for c in chunks])

    qdrant_path = tempfile.mkdtemp(prefix=f"eval_qdrant_{chunking.value}_")
    vector_store = VectorStore(
        path=qdrant_path,
        collection_name=f"eval_{chunking.value}",
        dimension=settings.embedding_dimension,
    )
    vector_store.add_chunks(chunks, vectors)

    bm25 = BM25Search()
    bm25.index(chunks)

    return BuiltIndex(
        vector_store=vector_store,
        bm25=bm25,
        embedder=embedder,
        qdrant_path=qdrant_path,
    )


# ---------------------------------------------------------------------------
# Router construction
# ---------------------------------------------------------------------------


def _build_router(
    router_kind: str,
    index: BuiltIndex,
    reranker: Reranker,
    llm: Any,
    settings: Settings,
) -> Any:
    """Build the requested router (PlanAndExecuteRouter or QueryRouter).

    Args:
        router_kind: "react" for PlanAndExecuteRouter, "pipeline" for QueryRouter.
        index: BuiltIndex bundle.
        reranker: Shared Reranker instance.
        llm: Generation LLM instance.
        settings: Application settings.

    Returns:
        A router instance exposing ``route(query, top_k)``.
    """
    hybrid = HybridRetriever(
        vector_store=index.vector_store,
        bm25_search=index.bm25,
        embedder=index.embedder,
        dense_weight=settings.dense_weight,
        bm25_weight=settings.bm25_weight,
    )

    if router_kind == "react":
        return PlanAndExecuteRouter(
            llm=llm,
            hybrid_retriever=hybrid,
            reranker=reranker,
            vector_store=index.vector_store,
            default_top_k=settings.top_k,
        )
    if router_kind == "pipeline":
        classifier = IntentClassifier(llm=llm, model_name=settings.generation_model)
        llm_chain = llm | StrOutputParser()
        return QueryRouter(
            intent_classifier=classifier,
            hybrid_retriever=hybrid,
            reranker=reranker,
            llm_chain=llm_chain,
            translate_query=settings.translate_query,
        )
    raise ValueError(f"Unknown router kind: {router_kind!r}")


# ---------------------------------------------------------------------------
# Per-cell run
# ---------------------------------------------------------------------------


def _generate_records(
    config: RunConfig,
    qa_set: list[dict[str, Any]],
    index: BuiltIndex,
    reranker: Reranker,
    llm: Any,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Run the router over the QA set and collect raw records.

    Args:
        config: Cell configuration.
        qa_set: Reviewed test questions.
        index: Pre-built dense + sparse indices for this chunking.
        reranker: Shared Reranker.
        llm: Generation LLM.
        settings: Application settings.

    Returns:
        List of raw record dicts (one per question), each containing the
        generated answer and retrieved contexts.
    """
    router = _build_router(config.router, index, reranker, llm, settings)
    records: list[dict[str, Any]] = []

    for entry in qa_set:
        question = entry["question"]
        logger.info("  Q: %s", question)
        try:
            response: GenerationResponse = router.route(query=question, top_k=config.top_k)
        except Exception as exc:
            logger.error("Router failed for question %r: %s", question, exc)
            response = GenerationResponse(
                answer="",
                sources=[],
                intent=None,  # type: ignore[arg-type]
                confidence=0.0,
            )
        ctx_texts = [r.chunk.text for r in response.sources]
        records.append(
            {
                "question": question,
                "answer": response.answer,
                "retrieved_contexts": ctx_texts,
                "reference_en": entry.get("reference_en", ""),
                "source_quote_da": entry.get("source_quote_da", ""),
                "source_doc": entry.get("source_doc", ""),
                "category": entry.get("category", ""),
            }
        )
    return records


def _judge_records(
    raw_records: list[dict[str, Any]],
    judge: RAGEvaluator,
) -> dict[str, Any]:
    """Run the RAGAS judge over a list of pre-computed raw records.

    Args:
        raw_records: List of records as produced by ``_generate_records``.
        judge: RAGEvaluator wrapping the judge LLM.

    Returns:
        Dict with ``aggregate`` and ``per_sample`` keys from the judge.
    """
    questions = [r["question"] for r in raw_records]
    answers = [r["answer"] for r in raw_records]
    contexts = [r["retrieved_contexts"] for r in raw_records]
    ground_truths: list[str | dict[str, Any]] = [
        {
            "reference_en": r.get("reference_en", ""),
            "source_quote_da": r.get("source_quote_da", ""),
        }
        for r in raw_records
    ]
    return judge.evaluate(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
    )


def _run_cell(
    config: RunConfig,
    qa_set: list[dict[str, Any]],
    index: BuiltIndex,
    reranker: Reranker,
    llm: Any,
    judge: RAGEvaluator,
    settings: Settings,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Run one experiment cell end-to-end.

    The raw records (questions / answers / contexts) are checkpointed to
    ``checkpoint_path`` BEFORE the RAGAS judge is invoked, so a judge crash
    does not waste the generation work. Use ``--rejudge <checkpoint>`` to
    re-run the judge step on a saved checkpoint.

    Args:
        config: The cell configuration.
        qa_set: List of reviewed test questions.
        index: Pre-built dense + sparse indices for this chunking.
        reranker: Shared Reranker.
        llm: Generation LLM.
        judge: RAGEvaluator wrapping the judge LLM.
        settings: Application settings.
        checkpoint_path: Where to write the raw-records checkpoint.

    Returns:
        Dict with config metadata, aggregate scores, per-sample scores, and
        the raw answers/contexts collected from the router.
    """
    logger.info("=== Running cell %s ===", config.name)
    raw_records = _generate_records(config, qa_set, index, reranker, llm, settings)

    # Checkpoint BEFORE the judge so a RAGAS crash does not waste generation.
    checkpoint_payload = {
        "config": {
            "name": config.name,
            "chunking": config.chunking.value,
            "router": config.router,
            "top_k": config.top_k,
        },
        "raw_records": raw_records,
        "n_samples": len(raw_records),
    }
    _write_json(checkpoint_path, checkpoint_payload)
    logger.info("Wrote raw-records checkpoint: %s", checkpoint_path)

    logger.info("  -> calling RAGAS judge ...")
    eval_result = _judge_records(raw_records, judge)

    return {
        "config": {
            "name": config.name,
            "chunking": config.chunking.value,
            "router": config.router,
            "top_k": config.top_k,
        },
        "aggregate": eval_result["aggregate"],
        "per_sample": eval_result["per_sample"],
        "raw_records": raw_records,
        "n_samples": len(qa_set),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _safe_json(obj: Any) -> Any:
    """Convert non-JSON-serialisable values (numpy / pandas) to plain Python."""
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            return str(obj)
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON result file with safe serialisation."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_safe_json(payload), fh, indent=2, ensure_ascii=False)


def _format_markdown(
    timestamp: str,
    settings: Settings,
    cell_results: list[dict[str, Any]],
) -> str:
    """Format the matrix results as a Markdown report.

    Args:
        timestamp: ISO timestamp for the report header.
        settings: Application settings (used for env metadata).
        cell_results: One dict per cell, as produced by _run_cell.

    Returns:
        Markdown string with a metadata block and one aggregate table.
    """
    judge_provider = settings.evaluator_llm_provider or settings.llm_provider
    judge_model_label = (
        settings.evaluator_llm_model
        or {
            "groq": settings.groq_model,
            "openai": settings.openai_model,
            "anthropic": settings.anthropic_model,
            "google_genai": settings.google_model,
            "ollama": settings.ollama_model,
        }.get(judge_provider, "(provider default)")
    )

    lines: list[str] = []
    lines.append(f"# RAGAS Evaluation — {timestamp}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Generation LLM: `{settings.llm_provider}` / "
                 f"`{settings.groq_model if settings.llm_provider == 'groq' else settings.generation_model}`")
    lines.append(f"- Judge LLM:      `{judge_provider}` / `{judge_model_label}`")
    lines.append(f"- Embeddings:     `{settings.embedding_provider}` / `{settings.local_embedding_model}`")
    lines.append(f"- Reranker:       `{settings.reranker_model}`")
    lines.append(f"- Samples:        {cell_results[0]['n_samples'] if cell_results else 0}")
    lines.append("")

    if not cell_results:
        return "\n".join(lines)

    metric_keys: list[str] = []
    for cell in cell_results:
        for key in cell["aggregate"].keys():
            if key not in metric_keys:
                metric_keys.append(key)

    header = ["Config", "Chunking", "Router", "top_k", *metric_keys]
    lines.append("## Aggregate Scores")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for cell in cell_results:
        cfg = cell["config"]
        row = [
            cfg["name"],
            cfg["chunking"],
            cfg["router"],
            str(cfg["top_k"]),
        ]
        for key in metric_keys:
            value = cell["aggregate"].get(key)
            row.append(f"{value:.4f}" if isinstance(value, (int, float)) else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a RAGAS evaluation matrix over the curated test set.",
    )
    parser.add_argument(
        "--experiment",
        choices=["quick", "chunking", "router", "all"],
        default="quick",
        help="Pre-defined experiment to run (default: quick).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K per query (default 5).")
    parser.add_argument(
        "--qa-set",
        type=str,
        default=QA_SET_PATH,
        help=f"Path to curated QA set YAML (default: {QA_SET_PATH}).",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=RUNS_DIR,
        help=f"Directory for output JSON+MD files (default: {RUNS_DIR}).",
    )
    parser.add_argument(
        "--rejudge",
        type=str,
        default="",
        metavar="CHECKPOINT_PATH",
        help=(
            "Re-run only the RAGAS judge on a previously saved checkpoint "
            "(eval/runs/<ts>_<cell>.checkpoint.json). Skips index building "
            "and router invocation entirely."
        ),
    )
    return parser.parse_args()


def _rejudge_from_checkpoint(
    checkpoint_path: str,
    settings: Settings,
    runs_dir: Path,
    timestamp: str,
) -> None:
    """Re-run only the RAGAS judge against a saved raw-records checkpoint.

    Skips index building, router invocation, and the temp Qdrant lifecycle
    entirely. The new judged result is written next to the runs dir under a
    fresh timestamp.

    Args:
        checkpoint_path: Path to a checkpoint.json file from a prior run.
        settings: Application settings (used to build the judge LLM).
        runs_dir: Directory for output files.
        timestamp: Timestamp string used in output filenames.
    """
    logger.info("Rejudge mode: loading checkpoint %s", checkpoint_path)
    with open(checkpoint_path, "r", encoding="utf-8") as fh:
        checkpoint = json.load(fh)

    config_dict = checkpoint.get("config", {})
    raw_records = checkpoint.get("raw_records", [])
    if not raw_records:
        raise ValueError(f"Checkpoint {checkpoint_path} contains no raw_records.")
    logger.info(
        "Checkpoint config: %s | %d records",
        config_dict.get("name", "(unknown)"),
        len(raw_records),
    )

    judge_llm = create_evaluator_llm(settings)
    embeddings = create_embeddings(settings)
    judge = RAGEvaluator(llm=judge_llm, embeddings=embeddings)

    logger.info("Calling RAGAS judge on cached records ...")
    eval_result = _judge_records(raw_records, judge)

    cell_result = {
        "config": config_dict,
        "aggregate": eval_result["aggregate"],
        "per_sample": eval_result["per_sample"],
        "raw_records": raw_records,
        "n_samples": len(raw_records),
    }

    cell_name = config_dict.get("name", "rejudged")
    out_json = runs_dir / f"{timestamp}_{cell_name}.rejudged.json"
    out_md = runs_dir / f"{timestamp}_{cell_name}.rejudged.md"
    _write_json(out_json, cell_result)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(_format_markdown(timestamp, settings, [cell_result]))
    logger.info("Wrote rejudged JSON: %s", out_json)
    logger.info("Wrote rejudged Markdown: %s", out_md)
    print(f"\nRejudge done. Markdown report: {out_md}")


def main() -> None:
    """Run the requested experiment matrix and persist results."""
    args = parse_args()
    settings = load_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if args.rejudge:
        logger.info("=== RAGAS Rejudge Start (%s) ===", timestamp)
        _rejudge_from_checkpoint(args.rejudge, settings, runs_dir, timestamp)
        return

    logger.info("=== RAGAS Matrix Evaluation Start (%s) ===", timestamp)

    qa_set = _load_qa_set(args.qa_set)
    run_configs = _build_run_configs(args.experiment, args.top_k)
    logger.info("Experiment: %s | %d cells | %d questions",
                args.experiment, len(run_configs), len(qa_set))

    # --- Build shared resources -------------------------------------------
    llm = create_llm(settings)
    judge_llm = create_evaluator_llm(settings)
    embeddings = create_embeddings(settings)
    judge = RAGEvaluator(llm=judge_llm, embeddings=embeddings)
    reranker = Reranker(model=create_reranker(settings.reranker_model))

    # --- Build indices once per chunking strategy -------------------------
    indices: dict[ChunkStrategy, BuiltIndex] = {}
    needed_chunkings = {cfg.chunking for cfg in run_configs}
    for chunking in needed_chunkings:
        indices[chunking] = _build_index(chunking, settings, embeddings)

    # --- Run cells ---------------------------------------------------------
    cell_results: list[dict[str, Any]] = []

    try:
        for config in run_configs:
            checkpoint_path = runs_dir / f"{timestamp}_{config.name}.checkpoint.json"
            cell_result = _run_cell(
                config=config,
                qa_set=qa_set,
                index=indices[config.chunking],
                reranker=reranker,
                llm=llm,
                judge=judge,
                settings=settings,
                checkpoint_path=checkpoint_path,
            )
            cell_results.append(cell_result)

            # Per-cell judged-result JSON
            cell_json_path = runs_dir / f"{timestamp}_{config.name}.json"
            _write_json(cell_json_path, cell_result)
            logger.info("Wrote cell result: %s", cell_json_path)
    finally:
        # --- Combined JSON + Markdown report -------------------------------
        combined_path = runs_dir / f"{timestamp}_{args.experiment}.json"
        md_path = runs_dir / f"{timestamp}_{args.experiment}.md"
        _write_json(
            combined_path,
            {
                "timestamp": timestamp,
                "experiment": args.experiment,
                "qa_set_path": args.qa_set,
                "n_samples": len(qa_set),
                "cells": cell_results,
            },
        )
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(_format_markdown(timestamp, settings, cell_results))
        logger.info("Wrote combined JSON: %s", combined_path)
        logger.info("Wrote Markdown report: %s", md_path)

        # --- Cleanup tmp Qdrant dirs ---------------------------------------
        for index in indices.values():
            shutil.rmtree(index.qdrant_path, ignore_errors=True)
        logger.info("Cleaned up %d tmp Qdrant dirs", len(indices))

    print(f"\nDone. Markdown report: {md_path}")


if __name__ == "__main__":
    main()
