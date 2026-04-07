"""Tests for agent tools (hybrid_search, list_documents, fetch_document,
search_within_document, multi_query_search, summarize_document)."""

from unittest.mock import MagicMock

import pytest

from src.agent.tools import ToolResultStore, make_retrieval_tools, _merge_results, _format_results
from src.models import DocumentChunk, QueryResult
from src.retrieval.hybrid import HybridSearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str = "c1", document_id: str = "doc.pdf", text: str = "text",
           page_number: int = 1, chunk_index: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        metadata={"page_number": page_number, "chunk_index": chunk_index},
    )


def _qr(chunk_id: str = "c1", document_id: str = "doc.pdf", text: str = "text",
         score: float = 0.8, source: str = "hybrid", page_number: int = 1) -> QueryResult:
    return QueryResult(
        chunk=_chunk(chunk_id=chunk_id, document_id=document_id, text=text, page_number=page_number),
        score=score,
        source=source,
    )


def _hybrid_result(results: list[QueryResult]) -> HybridSearchResult:
    return HybridSearchResult(
        dense_results=results,
        sparse_results=results,
        fused_results=results,
    )


@pytest.fixture
def components():
    """Create mock retriever, reranker, vector_store, and store."""
    retriever = MagicMock()
    reranker = MagicMock()
    vector_store = MagicMock()
    store = ToolResultStore()
    return retriever, reranker, vector_store, store


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestMergeResults:
    def test_merge_empty(self) -> None:
        assert _merge_results([], []) == []

    def test_merge_keeps_higher_score(self) -> None:
        old = [_qr(chunk_id="c1", score=0.5)]
        new = [_qr(chunk_id="c1", score=0.9)]
        merged = _merge_results(old, new)
        assert len(merged) == 1
        assert merged[0].score == 0.9

    def test_merge_keeps_old_if_higher(self) -> None:
        old = [_qr(chunk_id="c1", score=0.9)]
        new = [_qr(chunk_id="c1", score=0.5)]
        merged = _merge_results(old, new)
        assert merged[0].score == 0.9

    def test_merge_combines_different_ids(self) -> None:
        old = [_qr(chunk_id="c1", score=0.5)]
        new = [_qr(chunk_id="c2", score=0.9)]
        merged = _merge_results(old, new)
        assert len(merged) == 2
        assert merged[0].chunk.chunk_id == "c2"  # higher score first

    def test_merge_sorted_descending(self) -> None:
        results = [_qr(chunk_id=f"c{i}", score=s) for i, s in enumerate([0.3, 0.9, 0.6])]
        merged = _merge_results([], results)
        scores = [r.score for r in merged]
        assert scores == sorted(scores, reverse=True)


class TestFormatResults:
    def test_empty_returns_no_results_message(self) -> None:
        result = _format_results([])
        assert "Ingen relevante" in result

    def test_includes_document_id_and_score(self) -> None:
        results = [_qr(document_id="policy.pdf", score=0.85)]
        text = _format_results(results)
        assert "policy.pdf" in text
        assert "0.850" in text

    def test_includes_page_number(self) -> None:
        results = [_qr(page_number=5)]
        text = _format_results(results)
        assert "side 5" in text


# ---------------------------------------------------------------------------
# hybrid_search
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def test_returns_formatted_results(self, components) -> None:
        retriever, reranker, vector_store, store = components
        results = [_qr(document_id="a.pdf", score=0.9, text="answer")]
        retriever.search_detailed.return_value = _hybrid_result(results)
        reranker.rerank.return_value = results

        tools = make_retrieval_tools(retriever, reranker, vector_store, store)
        hybrid_search = tools[0]
        output = hybrid_search.invoke({"query": "test", "top_k": 5})

        assert "a.pdf" in output
        assert "answer" in output
        retriever.search_detailed.assert_called_once_with("test", top_k=5)

    def test_accumulates_in_store(self, components) -> None:
        retriever, reranker, vector_store, store = components
        results = [_qr(chunk_id="c1", score=0.8)]
        retriever.search_detailed.return_value = _hybrid_result(results)
        reranker.rerank.return_value = results

        tools = make_retrieval_tools(retriever, reranker, vector_store, store)
        tools[0].invoke({"query": "q1"})

        assert len(store.retrieved) == 1
        assert store.retrieved[0].chunk.chunk_id == "c1"
        assert len(store.tool_calls) == 1
        assert store.tool_calls[0] == ("hybrid_search", "q1")

    def test_no_results(self, components) -> None:
        retriever, reranker, vector_store, store = components
        retriever.search_detailed.return_value = _hybrid_result([])
        reranker.rerank.return_value = []

        tools = make_retrieval_tools(retriever, reranker, vector_store, store)
        output = tools[0].invoke({"query": "nothing"})
        assert "Ingen relevante" in output


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------

class TestListDocuments:
    def test_returns_document_list(self, components) -> None:
        retriever, reranker, vector_store, store = components
        vector_store.list_document_ids.return_value = ["a.pdf", "b.pdf"]

        tools = make_retrieval_tools(retriever, reranker, vector_store, store)
        list_docs = tools[1]
        output = list_docs.invoke({})

        assert "a.pdf" in output
        assert "b.pdf" in output
        assert "2 i alt" in output

    def test_empty_knowledge_base(self, components) -> None:
        retriever, reranker, vector_store, store = components
        vector_store.list_document_ids.return_value = []

        tools = make_retrieval_tools(retriever, reranker, vector_store, store)
        output = tools[1].invoke({})
        assert "empty" in output.lower() or "Ingen" in output


# ---------------------------------------------------------------------------
# fetch_document
# ---------------------------------------------------------------------------

class TestFetchDocument:
    def test_returns_full_text(self, components) -> None:
        retriever, reranker, vector_store, store = components
        chunks = [_chunk(chunk_id="c1", text="page1"), _chunk(chunk_id="c2", text="page2")]
        vector_store.get_chunks_by_document_id.return_value = chunks

        tools = make_retrieval_tools(retriever, reranker, vector_store, store)
        fetch = tools[2]
        output = fetch.invoke({"document_id": "doc.pdf"})

        assert "page1" in output
        assert "page2" in output
        assert len(store.retrieved) == 2

    def test_document_not_found(self, components) -> None:
        retriever, reranker, vector_store, store = components
        vector_store.get_chunks_by_document_id.return_value = []

        tools = make_retrieval_tools(retriever, reranker, vector_store, store)
        output = tools[2].invoke({"document_id": "missing.pdf"})
        assert "ikke fundet" in output


# ---------------------------------------------------------------------------
# search_within_document
# ---------------------------------------------------------------------------

class TestSearchWithinDocument:
    def test_reranks_document_chunks(self, components) -> None:
        retriever, reranker, vector_store, store = components
        chunks = [
            _chunk(chunk_id="c1", text="irrelevant"),
            _chunk(chunk_id="c2", text="relevant answer"),
        ]
        vector_store.get_chunks_by_document_id.return_value = chunks
        reranker.rerank.return_value = [_qr(chunk_id="c2", text="relevant answer", score=0.95)]

        tools = make_retrieval_tools(retriever, reranker, vector_store, store)
        search_within = tools[3]
        output = search_within.invoke({"document_id": "doc.pdf", "query": "answer"})

        assert "relevant answer" in output
        assert "0.950" in output
        reranker.rerank.assert_called_once()
        # Verify it passed all chunks to reranker
        candidates = reranker.rerank.call_args[0][1]
        assert len(candidates) == 2

    def test_document_not_found(self, components) -> None:
        retriever, reranker, vector_store, store = components
        vector_store.get_chunks_by_document_id.return_value = []

        tools = make_retrieval_tools(retriever, reranker, vector_store, store)
        output = tools[3].invoke({"document_id": "missing.pdf", "query": "test"})
        assert "ikke fundet" in output

    def test_accumulates_in_store(self, components) -> None:
        retriever, reranker, vector_store, store = components
        chunks = [_chunk(chunk_id="c1")]
        vector_store.get_chunks_by_document_id.return_value = chunks
        reranker.rerank.return_value = [_qr(chunk_id="c1", score=0.7)]

        tools = make_retrieval_tools(retriever, reranker, vector_store, store)
        tools[3].invoke({"document_id": "doc.pdf", "query": "q"})

        assert len(store.retrieved) == 1
        assert store.tool_calls[-1][0] == "search_within_document"


# ---------------------------------------------------------------------------
# multi_query_search (requires llm_chain)
# ---------------------------------------------------------------------------

class TestMultiQuerySearch:
    def test_decomposes_and_searches(self, components) -> None:
        retriever, reranker, vector_store, store = components
        llm_chain = MagicMock()

        # LLM returns 2 sub-queries
        llm_chain.invoke.return_value = "eksamenregler bachelor\neksamensregler kandidat"

        results_a = [_qr(chunk_id="c1", score=0.9, text="bachelor exam")]
        results_b = [_qr(chunk_id="c2", score=0.85, text="master exam")]
        retriever.search_detailed.side_effect = [
            _hybrid_result(results_a),
            _hybrid_result(results_b),
        ]
        reranker.rerank.side_effect = [results_a, results_b]

        tools = make_retrieval_tools(retriever, reranker, vector_store, store, llm_chain=llm_chain)
        multi_search = tools[4]
        output = multi_search.invoke({"question": "Compare exam rules"})

        assert "delforespørgsler" in output
        assert retriever.search_detailed.call_count == 2
        assert reranker.rerank.call_count == 2
        assert len(store.retrieved) == 2

    def test_fallback_when_decompose_fails(self, components) -> None:
        retriever, reranker, vector_store, store = components
        llm_chain = MagicMock()

        # LLM returns empty/garbage
        llm_chain.invoke.return_value = ""

        results = [_qr(chunk_id="c1", score=0.8)]
        retriever.search_detailed.return_value = _hybrid_result(results)
        reranker.rerank.return_value = results

        tools = make_retrieval_tools(retriever, reranker, vector_store, store, llm_chain=llm_chain)
        output = tools[4].invoke({"question": "original question"})

        # Should fall back to the original question as single query
        assert retriever.search_detailed.call_count == 1
        assert "0.800" in output

    def test_not_available_without_llm(self, components) -> None:
        retriever, reranker, vector_store, store = components
        tools = make_retrieval_tools(retriever, reranker, vector_store, store, llm_chain=None)
        tool_names = [t.name for t in tools]
        assert "multi_query_search" not in tool_names
        assert "summarize_document" not in tool_names

    def test_deduplicates_across_sub_queries(self, components) -> None:
        retriever, reranker, vector_store, store = components
        llm_chain = MagicMock()
        llm_chain.invoke.return_value = "query1\nquery2"

        # Both sub-queries return the same chunk
        same_result = [_qr(chunk_id="c1", score=0.8)]
        retriever.search_detailed.return_value = _hybrid_result(same_result)
        reranker.rerank.return_value = same_result

        tools = make_retrieval_tools(retriever, reranker, vector_store, store, llm_chain=llm_chain)
        tools[4].invoke({"question": "test"})

        # Should be deduplicated to 1
        assert len(store.retrieved) == 1


# ---------------------------------------------------------------------------
# summarize_document (requires llm_chain)
# ---------------------------------------------------------------------------

class TestSummarizeDocument:
    def test_generates_summary(self, components) -> None:
        retriever, reranker, vector_store, store = components
        llm_chain = MagicMock()
        llm_chain.invoke.return_value = "This document covers exam policies."

        chunks = [_chunk(chunk_id="c1", text="Exam rules...")]
        vector_store.get_chunks_by_document_id.return_value = chunks

        tools = make_retrieval_tools(retriever, reranker, vector_store, store, llm_chain=llm_chain)
        summarize = tools[5]
        output = summarize.invoke({"document_id": "exam.pdf"})

        assert "Resumé af exam.pdf" in output
        assert "exam policies" in output
        llm_chain.invoke.assert_called_once()
        # Verify the prompt includes the document text
        prompt = llm_chain.invoke.call_args[0][0]
        assert "Exam rules" in prompt

    def test_document_not_found(self, components) -> None:
        retriever, reranker, vector_store, store = components
        llm_chain = MagicMock()
        vector_store.get_chunks_by_document_id.return_value = []

        tools = make_retrieval_tools(retriever, reranker, vector_store, store, llm_chain=llm_chain)
        output = tools[5].invoke({"document_id": "missing.pdf"})
        assert "ikke fundet" in output
        llm_chain.invoke.assert_not_called()

    def test_truncates_long_documents(self, components) -> None:
        retriever, reranker, vector_store, store = components
        llm_chain = MagicMock()
        llm_chain.invoke.return_value = "summary"

        # Create a document longer than 8000 chars
        long_text = "x" * 10000
        chunks = [_chunk(chunk_id="c1", text=long_text)]
        vector_store.get_chunks_by_document_id.return_value = chunks

        tools = make_retrieval_tools(retriever, reranker, vector_store, store, llm_chain=llm_chain)
        tools[5].invoke({"document_id": "long.pdf"})

        prompt = llm_chain.invoke.call_args[0][0]
        assert "forkortet" in prompt

    def test_registers_chunks_as_sources(self, components) -> None:
        retriever, reranker, vector_store, store = components
        llm_chain = MagicMock()
        llm_chain.invoke.return_value = "summary"

        chunks = [_chunk(chunk_id="c1"), _chunk(chunk_id="c2")]
        vector_store.get_chunks_by_document_id.return_value = chunks

        tools = make_retrieval_tools(retriever, reranker, vector_store, store, llm_chain=llm_chain)
        tools[5].invoke({"document_id": "doc.pdf"})

        assert len(store.retrieved) == 2
        assert store.tool_calls[-1] == ("summarize_document", "doc.pdf")
