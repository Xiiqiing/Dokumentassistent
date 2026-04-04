"""Tests for BM25 sparse retrieval."""

import pytest

from src.models import DocumentChunk, QueryResult
from src.retrieval.bm25_search import BM25Search


def _make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    """Create a DocumentChunk helper."""
    return DocumentChunk(chunk_id=chunk_id, document_id="doc1", text=text)


class TestBM25Index:
    """Tests for index construction."""

    def test_index_stores_chunks(self) -> None:
        bm25 = BM25Search()
        chunks = [_make_chunk("1", "hello world"), _make_chunk("2", "foo bar")]
        bm25.index(chunks)
        assert bm25._chunks == chunks
        assert bm25._index is not None

    def test_index_replaces_previous(self) -> None:
        bm25 = BM25Search()
        bm25.index([_make_chunk("1", "old text")])
        bm25.index([_make_chunk("2", "new text")])
        assert len(bm25._chunks) == 1
        assert bm25._chunks[0].chunk_id == "2"

    def test_index_empty_list_raises(self) -> None:
        bm25 = BM25Search()
        with pytest.raises(ZeroDivisionError):
            bm25.index([])


class TestBM25Search:
    """Tests for query and ranking correctness."""

    def test_search_returns_relevant_results(self) -> None:
        bm25 = BM25Search()
        bm25.index([
            _make_chunk("1", "university policy on student enrollment"),
            _make_chunk("2", "library opening hours and access"),
            _make_chunk("3", "student enrollment deadline and requirements"),
        ])
        results = bm25.search("student enrollment", top_k=3)
        assert len(results) >= 2
        # The two chunks mentioning "student enrollment" should rank highest
        top_ids = [r.chunk.chunk_id for r in results[:2]]
        assert "1" in top_ids
        assert "3" in top_ids

    def test_search_respects_top_k(self) -> None:
        bm25 = BM25Search()
        bm25.index([
            _make_chunk("1", "alpha beta gamma"),
            _make_chunk("2", "beta gamma delta"),
            _make_chunk("3", "gamma delta epsilon"),
            _make_chunk("4", "delta epsilon zeta"),
        ])
        # "alpha" only in chunk 1, "beta" in 1&2 — at most 2 have nonzero scores
        results = bm25.search("alpha beta", top_k=2)
        assert len(results) <= 2

    def test_search_scores_descending(self) -> None:
        bm25 = BM25Search()
        bm25.index([
            _make_chunk("1", "data"),
            _make_chunk("2", "data data data"),
            _make_chunk("3", "data data"),
        ])
        results = bm25.search("data", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_result_fields(self) -> None:
        bm25 = BM25Search()
        bm25.index([
            _make_chunk("1", "test document content"),
            _make_chunk("2", "unrelated other stuff"),
            _make_chunk("3", "more filler material here"),
        ])
        results = bm25.search("test", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, QueryResult)
        assert r.source == "bm25"
        assert r.score > 0.0
        assert r.chunk.chunk_id == "1"

    def test_search_no_match_returns_empty(self) -> None:
        bm25 = BM25Search()
        bm25.index([_make_chunk("1", "hello world")])
        results = bm25.search("zzzznotfound", top_k=5)
        assert results == []

    def test_search_filters_zero_scores(self) -> None:
        bm25 = BM25Search()
        bm25.index([
            _make_chunk("1", "relevant keyword"),
            _make_chunk("2", "completely unrelated text"),
        ])
        results = bm25.search("keyword", top_k=10)
        for r in results:
            assert r.score > 0.0


class TestBM25Danish:
    """Tests for Danish text with æ, ø, å characters."""

    def test_danish_characters_indexed_and_searchable(self) -> None:
        bm25 = BM25Search()
        bm25.index([
            _make_chunk("1", "københavns universitet uddannelsespolitik"),
            _make_chunk("2", "studerende skal følge reglerne"),
            _make_chunk("3", "årsrapport for forskningsområdet"),
        ])
        results = bm25.search("københavns", top_k=3)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "1"

    def test_danish_oe_character(self) -> None:
        bm25 = BM25Search()
        bm25.index([
            _make_chunk("1", "følgende bestemmelser gælder"),
            _make_chunk("2", "other english text here"),
            _make_chunk("3", "mere dansk tekst uden søgeord"),
        ])
        results = bm25.search("følgende", top_k=3)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "1"

    def test_danish_aa_character(self) -> None:
        bm25 = BM25Search()
        bm25.index([
            _make_chunk("1", "åben adgang til dokumenter"),
            _make_chunk("2", "lukket periode for eksamen"),
            _make_chunk("3", "generel information om kurser"),
        ])
        results = bm25.search("åben", top_k=3)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "1"

    def test_danish_case_insensitive(self) -> None:
        bm25 = BM25Search()
        bm25.index([
            _make_chunk("1", "Ændringer i studieordningen"),
            _make_chunk("2", "andet dokument uden relevans"),
            _make_chunk("3", "tredje dokument om noget helt andet"),
        ])
        results = bm25.search("ændringer", top_k=3)
        assert len(results) == 1


class TestBM25EmptyIndex:
    """Tests for querying before or on an empty index."""

    def test_search_before_indexing(self) -> None:
        bm25 = BM25Search()
        results = bm25.search("anything", top_k=5)
        assert results == []

    def test_search_on_empty_index_not_possible(self) -> None:
        """BM25Okapi raises ZeroDivisionError on empty corpus,
        so searching an empty index is only possible if index() was never called."""
        bm25 = BM25Search()
        with pytest.raises(ZeroDivisionError):
            bm25.index([])
