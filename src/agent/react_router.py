"""ReAct agent router using a LangGraph tool-calling loop.

Replaces the fixed detect→translate→retrieve→rerank→generate DAG with a
multi-step reasoning loop where the LLM decides which tools to call and
when it has gathered enough information to produce a final answer.

Requires an LLM that supports bind_tools (OpenAI, Anthropic, Google GenAI,
and compatible Ollama models such as llama3.1 / qwen2.5). Set
AGENT_MODE=react in .env to activate; falls back to QueryRouter otherwise.
"""

import logging
from collections.abc import Generator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.prebuilt import create_react_agent

from src.models import GenerationResponse, IntentType, PipelineDetails, QueryResult
from src.agent.tools import ToolResultStore, make_retrieval_tools
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant for administrative staff at the University of Copenhagen (KU).\n\n"
    "You have access to a hybrid_search tool that searches KU policy documents stored in the "
    "knowledge base.\n\n"
    "Guidelines:\n"
    "- Always call hybrid_search before answering questions about KU rules, policies, exams, "
    "employment conditions, or administrative procedures.\n"
    "- If the first search does not return sufficient information, call hybrid_search again "
    "with a refined or more specific query.\n"
    "- For comparison questions, search for each item separately.\n"
    "- Cite the document sources ([1], [2], …) in your answer.\n"
    "- Answer in the same language as the user's question."
)


def _ser_sources(sources: list[QueryResult]) -> list[dict]:
    """Serialise QueryResult list to a JSON-safe list of dicts."""
    return [
        {
            "chunk_id": r.chunk.chunk_id,
            "document_id": r.chunk.document_id,
            "text": r.chunk.text,
            "score": r.score,
            "source": r.source,
        }
        for r in sources
    ]


class ReActRouter:
    """Routes queries through a multi-step ReAct agent with tool-calling LLM.

    The agent runs in a loop: the LLM reasons about the query, calls
    hybrid_search as many times as needed, observes results, and finally
    produces a grounded answer.  Results from every tool call are merged into
    a single ranked source list that is returned alongside the answer.
    """

    def __init__(
        self,
        llm: Runnable,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        vector_store: VectorStore,
        default_top_k: int = 5,
    ) -> None:
        """Initialise the ReAct router.

        Args:
            llm: LLM with tool-calling support (must implement bind_tools).
            hybrid_retriever: HybridRetriever instance.
            reranker: Reranker instance.
            vector_store: VectorStore instance for document-level tool access.
            default_top_k: Default number of results returned per tool call.
        """
        self._llm = llm
        self._hybrid_retriever = hybrid_retriever
        self._reranker = reranker
        self._vector_store = vector_store
        self._default_top_k = default_top_k

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_graph(self, store: ToolResultStore) -> object:
        """Build a fresh ReAct graph bound to *store* for one request."""
        tools = make_retrieval_tools(
            self._hybrid_retriever,
            self._reranker,
            self._vector_store,
            store,
            self._default_top_k,
        )
        return create_react_agent(self._llm, tools)

    @staticmethod
    def _extract_answer(messages: list) -> str:
        """Return the last non-tool-call AIMessage content as the final answer."""
        for msg in reversed(messages):
            if (
                isinstance(msg, AIMessage)
                and msg.content
                and not getattr(msg, "tool_calls", None)
            ):
                return str(msg.content)
        return ""

    # ------------------------------------------------------------------
    # Public interface (mirrors QueryRouter)
    # ------------------------------------------------------------------

    def route(self, query: str, top_k: int) -> GenerationResponse:
        """Route a query through the ReAct agent pipeline.

        Args:
            query: The user's natural language query.
            top_k: Number of top documents to retrieve per tool call.

        Returns:
            GenerationResponse with answer, sources, intent, and confidence.
        """
        logger.info("ReAct routing query: %s", query)
        store = ToolResultStore()
        graph = self._make_graph(store)

        result = graph.invoke(
            {
                "messages": [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=query),
                ]
            }
        )

        messages = result.get("messages", [])
        answer = self._extract_answer(messages)

        sources = store.retrieved[:top_k]
        confidence = max((r.score for r in sources), default=0.0)

        logger.info(
            "ReAct answer ready (confidence=%.4f, sources=%d, tool_calls=%d)",
            confidence,
            len(sources),
            len(store.tool_calls),
        )

        return GenerationResponse(
            answer=answer,
            sources=sources,
            intent=IntentType.RAG if sources else IntentType.FACTUAL,
            confidence=confidence,
            pipeline_details=PipelineDetails(
                original_query=query,
                retrieval_query=", ".join(q for _, q in store.tool_calls) or query,
                reranked_results=sources,
            ),
        )

    def route_stream(self, query: str, top_k: int) -> Generator[dict, None, None]:
        """Stream ReAct agent events step by step.

        Yields event dicts with the following step types (in addition to the
        existing pipeline steps understood by the UI):

        - ``tool_call``   — LLM decided to call a tool; carries ``tool`` and ``query``.
        - ``tool_result`` — Tool returned; carries ``tool``, ``result_count``.
        - ``generate``    — LLM is writing the final answer.
        - ``done``        — Final event with the full result payload.

        Args:
            query: User query.
            top_k: Number of results to retrieve per tool call.

        Yields:
            Step event dicts.
        """
        store = ToolResultStore()
        graph = self._make_graph(store)

        all_messages: list = []

        for chunk in graph.stream(
            {
                "messages": [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=query),
                ]
            },
            stream_mode="updates",
        ):
            for _node_name, update in chunk.items():
                if update is None:
                    continue
                node_messages = update.get("messages", [])
                all_messages.extend(node_messages)

                for msg in node_messages:
                    if isinstance(msg, AIMessage):
                        for tc in getattr(msg, "tool_calls", []):
                            yield {
                                "step": "tool_call",
                                "tool": tc.get("name", ""),
                                "query": tc.get("args", {}).get("query", ""),
                            }
                        if msg.content and not getattr(msg, "tool_calls", None):
                            yield {"step": "generate"}

                    elif isinstance(msg, ToolMessage):
                        yield {
                            "step": "tool_result",
                            "tool": getattr(msg, "name", ""),
                            "result_count": len(store.retrieved),
                        }

        answer = self._extract_answer(all_messages)
        sources = store.retrieved[:top_k]
        confidence = max((r.score for r in sources), default=0.0)

        yield {
            "step": "done",
            "result": {
                "answer": answer,
                "sources": _ser_sources(sources),
                "intent": (IntentType.RAG if sources else IntentType.FACTUAL).value,
                "confidence": confidence,
                "pipeline_details": {
                    "original_query": query,
                    "retrieval_query": ", ".join(q for _, q in store.tool_calls) or query,
                    "detected_language": "unknown",
                    "translated": False,
                    "dense_results": [],
                    "sparse_results": [],
                    "fused_results": [],
                    "reranked_results": [
                        {
                            "document_id": r.chunk.document_id,
                            "chunk_id": r.chunk.chunk_id,
                            "score": r.score,
                            "source": r.source,
                        }
                        for r in sources
                    ],
                },
            },
        }
