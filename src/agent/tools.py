"""LangChain tools for the ReAct agent."""

import logging
from dataclasses import dataclass, field

from langchain_core.tools import tool

from src.models import QueryResult
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class ToolResultStore:
    """Captures structured retrieval results produced during tool invocations.

    Attributes:
        retrieved: Accumulated QueryResult list across all hybrid_search calls,
            merged by chunk_id and sorted by descending score.
        tool_calls: Log of (tool_name, query_or_arg) tuples in invocation order.
    """

    retrieved: list[QueryResult] = field(default_factory=list)
    tool_calls: list[tuple[str, str]] = field(default_factory=list)


def make_retrieval_tools(
    hybrid_retriever: HybridRetriever,
    reranker: Reranker,
    vector_store: VectorStore,
    store: ToolResultStore,
    default_top_k: int = 5,
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

    Returns:
        List of LangChain tool callables ready for bind_tools / ToolNode.
    """

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

        # Accumulate results across multiple calls (union by chunk_id, keep highest score)
        existing = {r.chunk.chunk_id: r for r in store.retrieved}
        for r in results:
            cid = r.chunk.chunk_id
            if cid not in existing or r.score > existing[cid].score:
                existing[cid] = r
        store.retrieved = sorted(existing.values(), key=lambda r: r.score, reverse=True)

        if not results:
            return "Ingen relevante dokumenter fundet. (No relevant documents found.)"

        parts: list[str] = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] {r.chunk.document_id}  (relevance: {r.score:.3f})\n{r.chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

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
            return "Ingen dokumenter i videnbasen. (Knowledge base is empty.)"
        lines = "\n".join(f"- {doc_id}" for doc_id in ids)
        return f"Dokumenter i videnbasen ({len(ids)} i alt):\n{lines}"

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
                f"Dokumentet '{document_id}' blev ikke fundet i videnbasen. "
                f"(Document not found. Use list_documents to see available IDs.)"
            )

        # Sort chunks by chunk_id to preserve document order
        chunks.sort(key=lambda c: c.chunk_id)
        full_text = "\n\n".join(c.text for c in chunks)
        return (
            f"Dokument: {document_id}  ({len(chunks)} afsnit)\n\n"
            f"{full_text}"
        )

    return [hybrid_search, list_documents, fetch_document]
