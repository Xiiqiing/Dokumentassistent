"""Cross-encoder reranking using sentence-transformers."""

import logging

from sentence_transformers import CrossEncoder

from src.models import QueryResult

logger = logging.getLogger(__name__)


class Reranker:
    """Reranks retrieval results using a cross-encoder model."""

    def __init__(self, model_name: str) -> None:
        """Initialize the reranker with a cross-encoder model.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
        """
        self._model = CrossEncoder(model_name)
        logger.info("Loaded cross-encoder model '%s'", model_name)

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
            QueryResult(chunk=result.chunk, score=float(score), source="reranked")
            for result, score in zip(results, scores)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)

        logger.debug("Reranked %d results, keeping top %d", len(reranked), top_k)
        return reranked[:top_k]
