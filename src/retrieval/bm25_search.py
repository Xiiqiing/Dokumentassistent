"""BM25 sparse retrieval using rank_bm25."""

import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
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

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = [
            QueryResult(
                chunk=self._chunks[i],
                score=float(scores[i]),
                source="bm25",
            )
            for i in ranked_indices
            if scores[i] > 0.0
        ]
        logger.debug("BM25 search returned %d results for query: %s", len(results), query)
        return results

    def as_retriever(self, top_k: int) -> BaseRetriever:
        """Return a LangChain BaseRetriever wrapping this BM25 index.

        Args:
            top_k: Number of results to return per query.

        Returns:
            A BaseRetriever that calls search() and returns Documents.
        """
        return _BM25RetrieverAdapter(bm25_search=self, top_k=top_k)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text by lowercasing and splitting on whitespace.

        Args:
            text: The text to tokenize.

        Returns:
            List of lowercase tokens.
        """
        return text.lower().split()


class _BM25RetrieverAdapter(BaseRetriever):
    """LangChain BaseRetriever adapter over BM25Search."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bm25_search: Any
    top_k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = self.bm25_search.search(query, self.top_k)
        return [
            Document(
                page_content=r.chunk.text,
                metadata={
                    "chunk_id": r.chunk.chunk_id,
                    "document_id": r.chunk.document_id,
                    "chunk_metadata": r.chunk.metadata,
                    "strategy": r.chunk.strategy.value,
                    "score": r.score,
                },
            )
            for r in results
        ]
