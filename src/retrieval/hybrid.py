"""Hybrid search combining dense and sparse retrieval with reciprocal rank fusion."""

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from src.models import ChunkStrategy, DocumentChunk, QueryResult
from src.retrieval.bm25_search import BM25Search
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Container for hybrid search results including intermediate stages.

    Attributes:
        dense_results: Results from dense (vector) retrieval.
        sparse_results: Results from sparse (BM25) retrieval.
        fused_results: Results after reciprocal rank fusion.
    """

    dense_results: list[QueryResult]
    sparse_results: list[QueryResult]
    fused_results: list[QueryResult]


class HybridRetriever:
    """Combines dense (vector) and sparse (BM25) retrieval using rank fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_search: BM25Search,
        embedder: Embedder,
        dense_weight: float,
        bm25_weight: float,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            vector_store: VectorStore instance for dense retrieval.
            bm25_search: BM25Search instance for sparse retrieval.
            embedder: Embedder instance for query embedding.
            dense_weight: Weight for dense retrieval scores in fusion.
            bm25_weight: Weight for BM25 scores in fusion.
        """
        self._vector_store = vector_store
        self._bm25_search = bm25_search
        self._embedder = embedder
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight

    def search(self, query: str, top_k: int) -> list[QueryResult]:
        """Execute hybrid search combining dense and sparse results.

        Args:
            query: The search query string.
            top_k: Number of top results to return after fusion.

        Returns:
            List of QueryResult objects sorted by fused score.
        """
        result = self.search_detailed(query, top_k)
        return result.fused_results

    def search_detailed(self, query: str, top_k: int) -> HybridSearchResult:
        """Execute hybrid search and return all intermediate results.

        Uses LangChain BaseRetriever.invoke() for both dense and sparse retrieval.

        Args:
            query: The search query string.
            top_k: Number of top results to return after fusion.

        Returns:
            HybridSearchResult containing dense, sparse, and fused results.
        """
        dense_retriever = self._vector_store.as_retriever(self._embedder, top_k)
        sparse_retriever = self._bm25_search.as_retriever(top_k)

        dense_docs: list[Document] = dense_retriever.invoke(query)
        sparse_docs: list[Document] = sparse_retriever.invoke(query)

        dense_results = [self._doc_to_query_result(doc, "dense") for doc in dense_docs]
        sparse_results = [self._doc_to_query_result(doc, "bm25") for doc in sparse_docs]

        logger.debug(
            "Hybrid search: %d dense, %d sparse results",
            len(dense_results),
            len(sparse_results),
        )

        fused = self.reciprocal_rank_fusion(dense_results, sparse_results, k=60)
        return HybridSearchResult(
            dense_results=dense_results,
            sparse_results=sparse_results,
            fused_results=fused[:top_k],
        )

    @staticmethod
    def _doc_to_query_result(doc: Document, source: str) -> QueryResult:
        """Convert a LangChain Document to a QueryResult.

        Args:
            doc: Document returned by a BaseRetriever.
            source: Retrieval source label (e.g. 'dense' or 'bm25').

        Returns:
            QueryResult with chunk and score populated from document metadata.
        """
        meta = doc.metadata
        chunk = DocumentChunk(
            chunk_id=meta.get("chunk_id", ""),
            document_id=meta.get("document_id", ""),
            text=doc.page_content,
            metadata=meta.get("chunk_metadata", {}),
            strategy=ChunkStrategy(meta.get("strategy", ChunkStrategy.RECURSIVE.value)),
        )
        return QueryResult(chunk=chunk, score=float(meta.get("score", 0.0)), source=source)

    def reciprocal_rank_fusion(
        self,
        dense_results: list[QueryResult],
        sparse_results: list[QueryResult],
        k: int,
    ) -> list[QueryResult]:
        """Merge two ranked lists using reciprocal rank fusion.

        Args:
            dense_results: Results from dense retrieval.
            sparse_results: Results from sparse retrieval.
            k: Smoothing constant for RRF (typically 60).

        Returns:
            Merged and re-ranked list of QueryResult objects.
        """
        # Map chunk_id -> (rrf_score, best QueryResult)
        scores: dict[str, float] = {}
        best_chunk: dict[str, QueryResult] = {}

        for rank, result in enumerate(dense_results):
            cid = result.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + self._dense_weight / (k + rank + 1)
            best_chunk[cid] = result.chunk

        for rank, result in enumerate(sparse_results):
            cid = result.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + self._bm25_weight / (k + rank + 1)
            if cid not in best_chunk:
                best_chunk[cid] = result.chunk

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

        fused_results = [
            QueryResult(chunk=best_chunk[cid], score=scores[cid], source="hybrid")
            for cid in sorted_ids
        ]

        logger.debug("RRF produced %d fused results", len(fused_results))
        return fused_results
