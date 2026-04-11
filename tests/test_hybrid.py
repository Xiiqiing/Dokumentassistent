"""Tests for hybrid search with reciprocal rank fusion."""

from unittest.mock import MagicMock

import pytest

from src.models import DocumentChunk, QueryResult
from src.retrieval.hybrid import HybridRetriever


def _make_result(chunk_id: str, score: float = 0.0, source: str = "test") -> QueryResult:
    """Create a QueryResult with a minimal DocumentChunk."""
    chunk = DocumentChunk(chunk_id=chunk_id, document_id="doc1", text=f"text-{chunk_id}")
    return QueryResult(chunk=chunk, score=score, source=source)


def _build_retriever(
    dense_results: list[QueryResult],
    sparse_results: list[QueryResult],
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> HybridRetriever:
    """Build a HybridRetriever with mocked dependencies.

    Mocks vector_store.search() and bm25_search.search() since
    HybridRetriever calls them directly.
    """
    vector_store = MagicMock()
    vector_store.search.return_value = dense_results

    bm25_search = MagicMock()
    bm25_search.search.return_value = sparse_results

    embedder = MagicMock()
    embedder.embed_text.return_value = [0.0] * 384

    return HybridRetriever(
        vector_store=vector_store,
        bm25_search=bm25_search,
        embedder=embedder,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )


class TestReciprocalRankFusion:
    """Tests for RRF score computation correctness."""

    def test_rrf_scores_with_equal_weights(self) -> None:
        """Both sources return same doc; RRF score equals sum of reciprocal ranks."""
        dense = [_make_result("a")]
        sparse = [_make_result("a")]

        retriever = _build_retriever(dense, sparse, dense_weight=1.0, bm25_weight=1.0)
        results = retriever.reciprocal_rank_fusion(dense, sparse, k=60)

        # rank 0 in both lists: score = 1/(60+0+1) + 1/(60+0+1) = 2/61
        expected = 2.0 / 61.0
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "a"
        assert results[0].score == pytest.approx(expected)
        assert results[0].source == "hybrid"

    def test_rrf_scores_with_different_weights(self) -> None:
        """Weighted RRF applies dense_weight and bm25_weight correctly."""
        dense = [_make_result("a")]
        sparse = [_make_result("a")]

        retriever = _build_retriever(dense, sparse, dense_weight=0.7, bm25_weight=0.3)
        results = retriever.reciprocal_rank_fusion(dense, sparse, k=60)

        expected = 0.7 / 61.0 + 0.3 / 61.0
        assert results[0].score == pytest.approx(expected)

    def test_rrf_ranking_order(self) -> None:
        """Doc appearing in both lists outranks doc in only one list."""
        dense = [_make_result("a"), _make_result("b")]
        sparse = [_make_result("a"), _make_result("c")]

        retriever = _build_retriever(dense, sparse, dense_weight=1.0, bm25_weight=1.0)
        results = retriever.reciprocal_rank_fusion(dense, sparse, k=60)

        # "a" appears in both -> highest score
        assert results[0].chunk.chunk_id == "a"
        # "b" and "c" each appear once at different ranks
        remaining_ids = {r.chunk.chunk_id for r in results[1:]}
        assert remaining_ids == {"b", "c"}

    def test_rrf_rank_positions_matter(self) -> None:
        """Higher rank (lower index) in a list yields a higher RRF contribution."""
        dense = [_make_result("a"), _make_result("b")]
        sparse: list[QueryResult] = []

        retriever = _build_retriever(dense, sparse, dense_weight=1.0, bm25_weight=1.0)
        results = retriever.reciprocal_rank_fusion(dense, sparse, k=60)

        # "a" at rank 0: 1/61, "b" at rank 1: 1/62
        assert results[0].chunk.chunk_id == "a"
        assert results[0].score == pytest.approx(1.0 / 61.0)
        assert results[1].chunk.chunk_id == "b"
        assert results[1].score == pytest.approx(1.0 / 62.0)


class TestDegradation:
    """Tests for graceful degradation when one source returns no results."""

    def test_only_dense_results(self) -> None:
        """When BM25 returns nothing, results come solely from dense retrieval."""
        dense = [_make_result("a"), _make_result("b")]
        sparse: list[QueryResult] = []

        retriever = _build_retriever(dense, sparse)
        results = retriever.search("query", top_k=5)

        assert len(results) == 2
        ids = [r.chunk.chunk_id for r in results]
        assert ids == ["a", "b"]

    def test_only_sparse_results(self) -> None:
        """When dense returns nothing, results come solely from BM25."""
        dense: list[QueryResult] = []
        sparse = [_make_result("x"), _make_result("y")]

        retriever = _build_retriever(dense, sparse)
        results = retriever.search("query", top_k=5)

        assert len(results) == 2
        ids = [r.chunk.chunk_id for r in results]
        assert ids == ["x", "y"]

    def test_both_sources_empty(self) -> None:
        """When both sources return nothing, result is an empty list."""
        retriever = _build_retriever([], [])
        results = retriever.search("query", top_k=5)

        assert results == []


class TestTopKTruncation:
    """Tests for top-k truncation after fusion."""

    def test_truncation_to_top_k(self) -> None:
        """Fused results are truncated to the requested top_k."""
        dense = [_make_result(f"d{i}") for i in range(5)]
        sparse = [_make_result(f"s{i}") for i in range(5)]

        retriever = _build_retriever(dense, sparse)
        results = retriever.search("query", top_k=3)

        assert len(results) == 3

    def test_top_k_larger_than_results(self) -> None:
        """When fewer results exist than top_k, return all available."""
        dense = [_make_result("a")]
        sparse = [_make_result("b")]

        retriever = _build_retriever(dense, sparse)
        results = retriever.search("query", top_k=10)

        assert len(results) == 2

    def test_top_k_one(self) -> None:
        """top_k=1 returns exactly the highest-scored result."""
        dense = [_make_result("a"), _make_result("b")]
        sparse = [_make_result("a")]

        retriever = _build_retriever(dense, sparse)
        results = retriever.search("query", top_k=1)

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "a"
