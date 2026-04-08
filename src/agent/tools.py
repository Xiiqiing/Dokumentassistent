"""LangChain tools for the ReAct agent."""

import logging
import re
from dataclasses import dataclass, field

from langchain_core.runnables import Runnable
from langchain_core.tools import tool

from src.models import QueryResult
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove ``<think>`` blocks — both closed and unclosed."""
    text = _THINK_CLOSED_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


def _extract_content(result: object) -> str:
    """Extract plain text from an LLM invoke result.

    Handles AIMessage (content: str or list), plain strings, etc.
    """
    if hasattr(result, "content"):
        content = result.content
    else:
        content = result

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        text = "\n".join(parts)
    else:
        text = str(content)

    return _strip_think(text)


@dataclass
class ToolResultStore:
    """Captures structured retrieval results produced during tool invocations.

    Attributes:
        retrieved: Accumulated QueryResult list across all tool calls,
            merged by chunk_id and sorted by descending score.
        tool_calls: Log of (tool_name, query_or_arg) tuples in invocation order.
        dense_results: Accumulated dense retrieval results across hybrid_search calls.
        sparse_results: Accumulated sparse (BM25) retrieval results across hybrid_search calls.
        fused_results: Accumulated RRF-fused results across hybrid_search calls.
    """

    retrieved: list[QueryResult] = field(default_factory=list)
    tool_calls: list[tuple[str, str]] = field(default_factory=list)
    dense_results: list[QueryResult] = field(default_factory=list)
    sparse_results: list[QueryResult] = field(default_factory=list)
    fused_results: list[QueryResult] = field(default_factory=list)


def _merge_results(existing: list[QueryResult], new: list[QueryResult]) -> list[QueryResult]:
    """Merge two QueryResult lists by chunk_id, keeping the highest score.

    Args:
        existing: Previously accumulated results.
        new: New results to merge in.

    Returns:
        Merged list sorted by descending score.
    """
    by_id = {r.chunk.chunk_id: r for r in existing}
    for r in new:
        cid = r.chunk.chunk_id
        if cid not in by_id or r.score > by_id[cid].score:
            by_id[cid] = r
    return sorted(by_id.values(), key=lambda r: r.score, reverse=True)


def _format_results(results: list[QueryResult]) -> str:
    """Format a list of QueryResult into a readable string.

    Args:
        results: Ranked results to format.

    Returns:
        Formatted string with numbered entries, or a no-results message.
    """
    if not results:
        return "Ingen relevante dokumenter fundet. (No relevant documents found.)"
    parts: list[str] = []
    for i, r in enumerate(results, 1):
        page_info = ""
        page = r.chunk.metadata.get("page_number")
        if page is not None:
            page_info = f"  side {page}"
        parts.append(
            f"[{i}] {r.chunk.document_id}{page_info}  (relevance: {r.score:.3f})\n{r.chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def make_retrieval_tools(
    hybrid_retriever: HybridRetriever,
    reranker: Reranker,
    vector_store: VectorStore,
    store: ToolResultStore,
    default_top_k: int = 5,
    llm_chain: Runnable | None = None,
) -> list:
    """Create retrieval tools bound to the given components and result store.

    The returned tools write structured QueryResult objects into *store* on each
    invocation so the calling router can surface them as sources without having
    to re-parse the tool's text output.

    Args:
        hybrid_retriever: HybridRetriever instance.
        reranker: Reranker instance.
        vector_store: VectorStore instance for document-level access.
        store: Shared ToolResultStore that captures structured results.
        default_top_k: Default number of results to return per call.
        llm_chain: Optional LLM chain for tools that need generation
            (summarize_document, multi_query_search). When None, those
            tools are excluded from the returned list.

    Returns:
        List of LangChain tool callables ready for bind_tools / ToolNode.
    """

    # ------------------------------------------------------------------
    # Core search tool
    # ------------------------------------------------------------------

    @tool
    def hybrid_search(query: str, top_k: int = default_top_k) -> str:
        """Search the KU document knowledge base using hybrid retrieval.

        Combines dense semantic search (Qdrant) and sparse keyword search (BM25),
        then re-ranks results with a cross-encoder. Use this tool to find relevant
        passages from ingested KU policy documents about rules, regulations, exam
        procedures, employment conditions, and administrative guidelines.

        Call this tool before answering any question that requires factual
        information from KU documents. You may call it multiple times with
        different queries if the first result is insufficient.

        Args:
            query: Search query. Danish gives the best recall against KU documents.
            top_k: Number of top results to return (1–20). Default is 5.

        Returns:
            Formatted string of ranked document passages with source references
            and relevance scores.
        """
        logger.info("Tool hybrid_search: query=%r top_k=%d", query, top_k)
        store.tool_calls.append(("hybrid_search", query))

        hybrid_result = hybrid_retriever.search_detailed(query, top_k=top_k)
        results = reranker.rerank(query, hybrid_result.fused_results, top_k=top_k)

        store.dense_results = _merge_results(store.dense_results, hybrid_result.dense_results)
        store.sparse_results = _merge_results(store.sparse_results, hybrid_result.sparse_results)
        store.fused_results = _merge_results(store.fused_results, hybrid_result.fused_results)
        store.retrieved = _merge_results(store.retrieved, results)

        return _format_results(results)

    # ------------------------------------------------------------------
    # Document-level tools
    # ------------------------------------------------------------------

    @tool
    def list_documents() -> str:
        """List all documents currently available in the KU knowledge base.

        Use this tool when the user asks which documents are available, wants to
        know what topics are covered, or before fetching a specific document by ID.

        Returns:
            Newline-separated list of document IDs, or a message if the
            knowledge base is empty.
        """
        logger.info("Tool list_documents called")
        store.tool_calls.append(("list_documents", ""))

        ids = vector_store.list_document_ids()
        if not ids:
            return "Ingen dokumenter i vidensbasen. (Knowledge base is empty.)"
        lines = "\n".join(f"- {doc_id}" for doc_id in ids)
        return f"Dokumenter i vidensbasen ({len(ids)} i alt):\n{lines}"

    @tool
    def fetch_document(document_id: str) -> str:
        """Fetch the full text of a specific document from the knowledge base.

        Use this tool when the user asks for a summary or overview of a named
        document, or when hybrid_search results reference a document that
        warrants deeper reading. Prefer hybrid_search for targeted questions.

        Args:
            document_id: The exact document ID as returned by list_documents or
                seen in hybrid_search results (e.g. 'ku_ai_policy.pdf').

        Returns:
            The concatenated text of all chunks belonging to the document, or
            an error message if the document ID is not found.
        """
        logger.info("Tool fetch_document: document_id=%r", document_id)
        store.tool_calls.append(("fetch_document", document_id))

        chunks = vector_store.get_chunks_by_document_id(document_id)
        if not chunks:
            return (
                f"Dokumentet '{document_id}' blev ikke fundet i vidensbasen. "
                f"(Document not found. Use list_documents to see available IDs.)"
            )

        chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0))

        existing = {r.chunk.chunk_id: r for r in store.retrieved}
        for chunk in chunks:
            if chunk.chunk_id not in existing:
                existing[chunk.chunk_id] = QueryResult(chunk=chunk, score=1.0, source="fetch_document")
        store.retrieved = sorted(existing.values(), key=lambda r: r.score, reverse=True)

        full_text = "\n\n".join(c.text for c in chunks)
        return (
            f"Dokument: {document_id}  ({len(chunks)} afsnit)\n\n"
            f"{full_text}"
        )

    # ------------------------------------------------------------------
    # Targeted within-document search
    # ------------------------------------------------------------------

    @tool
    def search_within_document(document_id: str, query: str, top_k: int = 3) -> str:
        """Search for specific information within a single document.

        Retrieves all chunks belonging to the document and uses the cross-encoder
        reranker to find the most relevant passages for the query. Use this when
        you already know which document to look in and need to pinpoint the exact
        section (e.g. a specific clause, page, or paragraph).

        Args:
            document_id: The exact document ID to search within.
            query: What to look for inside the document.
            top_k: Number of top passages to return (1–10). Default is 3.

        Returns:
            The most relevant passages within the document, ranked by relevance.
        """
        logger.info(
            "Tool search_within_document: doc=%r query=%r top_k=%d",
            document_id, query, top_k,
        )
        store.tool_calls.append(("search_within_document", f"{document_id}: {query}"))

        chunks = vector_store.get_chunks_by_document_id(document_id)
        if not chunks:
            return (
                f"Dokumentet '{document_id}' blev ikke fundet i vidensbasen. "
                f"(Document not found. Use list_documents to see available IDs.)"
            )

        # Wrap chunks as QueryResult so the reranker can score them
        candidates = [
            QueryResult(chunk=c, score=0.0, source="search_within_document")
            for c in chunks
        ]
        results = reranker.rerank(query, candidates, top_k=top_k)

        store.retrieved = _merge_results(store.retrieved, results)

        return _format_results(results)

    # ------------------------------------------------------------------
    # LLM-powered tools (only available when llm_chain is provided)
    # ------------------------------------------------------------------

    tools: list = [hybrid_search, list_documents, fetch_document, search_within_document]

    if llm_chain is not None:

        @tool
        def multi_query_search(question: str, top_k: int = default_top_k) -> str:
            """Decompose a complex question into sub-queries and search each independently.

            Use this tool instead of hybrid_search when the question involves
            multiple aspects, comparisons, or requires information from different
            topics. For example: "How do exam rules differ between bachelor and
            master programmes?" would be split into separate searches for each
            programme's exam rules, then merged.

            Args:
                question: The complex user question to decompose and search.
                top_k: Number of results to return per sub-query (1–10). Default is 5.

            Returns:
                Combined results from all sub-queries, deduplicated and ranked.
            """
            logger.info("Tool multi_query_search: question=%r", question)
            store.tool_calls.append(("multi_query_search", question))

            # Step 1: Ask LLM to decompose the question
            decompose_prompt = (
                "You are a search query planner. Given a complex question, "
                "decompose it into 2-4 simple, independent search queries that "
                "together cover all aspects of the question. The queries should "
                "be in Danish (since the document base is Danish).\n\n"
                "Reply with ONLY the queries, one per line, nothing else.\n\n"
                f"Question: {question}"
            )
            raw = _extract_content(llm_chain.invoke(decompose_prompt))
            sub_queries = [q.strip().lstrip("0123456789.-) ") for q in raw.splitlines() if q.strip()]
            if not sub_queries:
                sub_queries = [question]

            logger.info("Decomposed into %d sub-queries: %s", len(sub_queries), sub_queries)

            # Step 2: Search each sub-query independently
            all_results: list[QueryResult] = []
            for sq in sub_queries:
                hybrid_result = hybrid_retriever.search_detailed(sq, top_k=top_k)
                reranked = reranker.rerank(sq, hybrid_result.fused_results, top_k=top_k)
                all_results = _merge_results(all_results, reranked)

                store.dense_results = _merge_results(store.dense_results, hybrid_result.dense_results)
                store.sparse_results = _merge_results(store.sparse_results, hybrid_result.sparse_results)
                store.fused_results = _merge_results(store.fused_results, hybrid_result.fused_results)

            # Step 3: Keep top results across all sub-queries
            final = all_results[:top_k]
            store.retrieved = _merge_results(store.retrieved, final)

            header = f"Søgning opdelt i {len(sub_queries)} delforespørgsler:\n"
            header += "\n".join(f"  • {sq}" for sq in sub_queries)
            header += "\n\n"
            return header + _format_results(final)

        @tool
        def summarize_document(document_id: str) -> str:
            """Generate a structured summary of a document in the knowledge base.

            Fetches the full document and uses the LLM to produce a concise summary
            covering the main topics, key rules, and important details. Use this
            when the user asks "what is this document about?" or wants an overview
            before diving into specifics.

            Args:
                document_id: The exact document ID to summarize.

            Returns:
                A structured summary of the document, or an error if not found.
            """
            logger.info("Tool summarize_document: document_id=%r", document_id)
            store.tool_calls.append(("summarize_document", document_id))

            chunks = vector_store.get_chunks_by_document_id(document_id)
            if not chunks:
                return (
                    f"Dokumentet '{document_id}' blev ikke fundet i vidensbasen. "
                    f"(Document not found. Use list_documents to see available IDs.)"
                )

            chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0))
            full_text = "\n\n".join(c.text for c in chunks)

            # Register chunks as sources
            existing = {r.chunk.chunk_id: r for r in store.retrieved}
            for chunk in chunks:
                if chunk.chunk_id not in existing:
                    existing[chunk.chunk_id] = QueryResult(
                        chunk=chunk, score=1.0, source="summarize_document",
                    )
            store.retrieved = sorted(existing.values(), key=lambda r: r.score, reverse=True)

            # Truncate to avoid exceeding context limits
            max_chars = 8000
            if len(full_text) > max_chars:
                full_text = full_text[:max_chars] + "\n\n[... teksten er forkortet ... (text truncated)]"

            summary_prompt = (
                "Produce a structured summary of the following document. "
                "Include:\n"
                "1. Document title/topic\n"
                "2. Key points (3-7 bullet points)\n"
                "3. Important rules, deadlines, or requirements mentioned\n"
                "4. Who the document applies to\n\n"
                "Write the summary in the same language as the document.\n\n"
                f"Document ID: {document_id}\n\n"
                f"Document text:\n{full_text}"
            )
            summary = _extract_content(llm_chain.invoke(summary_prompt))
            return f"Resumé af {document_id}:\n\n{summary}"

        tools.extend([multi_query_search, summarize_document])

    return tools
