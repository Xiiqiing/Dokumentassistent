"""Cross-encoder reranking."""

import logging
import math

from src.models import QueryResult

logger = logging.getLogger(__name__)


def _sigmoid(score: float) -> float:
    """Normalize a raw cross-encoder score to 0-1 via sigmoid."""
    score = max(-500.0, min(500.0, score))
    return 1.0 / (1.0 + math.exp(-score))


class Reranker:
    """Reranks retrieval results using a cross-encoder model."""

    def __init__(self, model: object) -> None:
        """Initialize the reranker with a cross-encoder model.

        Args:
            model: A cross-encoder model instance (e.g. from provider.create_reranker).
        """
        self._model = model
        logger.info("Loaded cross-encoder reranker")

    def rerank(self, query: str, results: list[QueryResult], top_k: int) -> list[QueryResult]:
        """Rerank retrieval results using the cross-encoder.

        Args:
            query: The original search query.
            results: Candidate results to rerank.
            top_k: Number of top results to keep after reranking.

        Returns:
            Reranked list of QueryResult objects.
        """
        if not results:
            return []

        pairs = [[query, result.chunk.text] for result in results]
        scores = self._model.predict(pairs)

        reranked = [
            QueryResult(chunk=result.chunk, score=_sigmoid(float(score)), source="reranked")
            for result, score in zip(results, scores)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)

        logger.debug("Reranked %d results, keeping top %d", len(reranked), top_k)
        return reranked[:top_k]
