"""BM25 sparse retrieval using rank_bm25."""

import logging

from rank_bm25 import BM25Okapi

from src.models import DocumentChunk, QueryResult

logger = logging.getLogger(__name__)


class BM25Search:
    """Sparse keyword-based retrieval using BM25 scoring."""

    def __init__(self) -> None:
        """Initialize the BM25 search index."""
        self._chunks: list[DocumentChunk] = []
        self._index: BM25Okapi | None = None

    def index(self, chunks: list[DocumentChunk]) -> None:
        """Build the BM25 index from document chunks.

        Args:
            chunks: List of document chunks to index.
        """
        self._chunks = chunks
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in chunks]
        self._index = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index with %d chunks", len(chunks))

    def search(self, query: str, top_k: int) -> list[QueryResult]:
        """Search the BM25 index for relevant chunks.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of QueryResult objects sorted by BM25 score.
        """
        if self._index is None or not self._chunks:
            logger.warning("BM25 search called before indexing")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        positive_indices = [i for i in range(len(scores)) if scores[i] > 0.0]
        ranked_indices = sorted(positive_indices, key=lambda i: scores[i], reverse=True)[:top_k]

        results = [
            QueryResult(
                chunk=self._chunks[i],
                score=float(scores[i]),
                source="bm25",
            )
            for i in ranked_indices
        ]
        logger.debug("BM25 search returned %d results for query: %s", len(results), query)
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text by lowercasing and splitting on whitespace.

        Args:
            text: The text to tokenize.

        Returns:
            List of lowercase tokens.
        """
        return text.lower().split()
