"""Query router that selects retrieval strategy based on intent.
--------------------------------------------------------------------
This is to support lightweight local models (e.g. gemma3) that lack
tool/function-calling capability. LangGraph moves all routing decisions
(intent branching, confidence-based retry) into graph edges so the
pipeline works identically regardless of the underlying model.

This pipeline has a conditional retry loop (low confidence → broaden query → re-retrieve).
LangGraph makes that cycle, the conditional skip, and per-node streaming
explicit and testable without hand-rolled flags or callback plumbing.
"""

import logging
import re
from collections.abc import Generator
from typing import TypedDict

from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph

from src.models import IntentType, GenerationResponse, PipelineDetails, QueryResult
from src.agent.intent_classifier import IntentClassifier
from src.agent.prompts import render_prompt
from src.agent.token_budget import measure as _measure_tokens
from src.agent.tools import detect_document_languages
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

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


# Reranker confidence below this triggers a query-broadening retry.
# Cross-encoder sigmoid scores below 0.3 generally indicate poor relevance.
_LOW_CONFIDENCE_THRESHOLD = 0.3
_MAX_RETRIES = 1


class RouterState(TypedDict):
    """LangGraph state passed between routing nodes.

    Attributes:
        query: The user's original query.
        top_k: Number of results to retrieve.
        user_language: Detected language of the query.
        intent: Classified intent type.
        retrieval_query: Query used for retrieval (may be translated).
        translated: Whether the query was translated.
        dense_results: Results from vector retrieval.
        sparse_results: Results from BM25 retrieval.
        fused_results: Results after RRF fusion.
        reranked: Results after cross-encoder reranking.
        confidence: Max reranker score (0.0-1.0).
        retry_count: Number of query-broadening retries performed so far.
        answer: Final generated answer.
    """

    query: str
    top_k: int
    user_language: str
    intent: IntentType
    retrieval_query: str
    translated: bool
    dense_results: list[QueryResult]
    sparse_results: list[QueryResult]
    fused_results: list[QueryResult]
    reranked: list[QueryResult]
    confidence: float
    retry_count: int
    answer: str


def _make_initial_state(query: str, top_k: int) -> RouterState:
    """Create a fresh RouterState with sensible defaults.

    Args:
        query: The user's original query.
        top_k: Number of results to retrieve.

    Returns:
        RouterState ready to be passed into the graph.
    """
    return RouterState(
        query=query,
        top_k=top_k,
        user_language="Danish",
        intent=IntentType.UNKNOWN,
        retrieval_query=query,
        translated=False,
        dense_results=[],
        sparse_results=[],
        fused_results=[],
        reranked=[],
        confidence=0.0,
        retry_count=0,
        answer="",
    )


class QueryRouter:
    """Routes queries to appropriate retrieval and generation pipelines."""

    def __init__(
        self,
        intent_classifier: IntentClassifier,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        llm_chain: Runnable,
        *,
        translate_query: bool = True,
        document_languages: list[str] | None = None,
        token_budget_enabled: bool = False,
    ) -> None:
        """Initialize the query router.

        Args:
            intent_classifier: IntentClassifier instance.
            hybrid_retriever: HybridRetriever instance.
            reranker: Reranker instance.
            llm_chain: LLM chain (llm | StrOutputParser) for generation,
                translation, and language detection.
            translate_query: Whether to translate the user query into a
                corpus language before BM25 retrieval when the query
                language does not already match one of the corpus languages.
                When False, no translation is performed.
            document_languages: Optional pre-detected list of corpus
                languages. When omitted, the router lazily detects them
                from the vector store on first translation/generation via
                the LLM.
        """
        self._intent_classifier = intent_classifier
        self._hybrid_retriever = hybrid_retriever
        self._reranker = reranker
        self._llm_chain = llm_chain
        self._translate_query_enabled = translate_query
        self._document_languages: list[str] | None = (
            list(document_languages) if document_languages else None
        )
        self._token_budget_enabled = token_budget_enabled
        self._graph = self._build_graph()

    def _ensure_document_languages(self) -> list[str]:
        """Lazily detect and cache the document corpus languages via the LLM.

        Returns:
            List of detected language names (e.g. ``["Danish"]`` or
            ``["Danish", "English"]``). Empty list when the corpus is empty
            or no readable text could be sampled.
        """
        if self._document_languages is not None:
            return self._document_languages
        self._document_languages = detect_document_languages(
            self._hybrid_retriever.vector_store, self._llm_chain
        )
        if self._document_languages:
            logger.info("Detected document corpus languages: %s", self._document_languages)
        return self._document_languages

    def _detect_language_and_intent(self, query: str) -> tuple[str, IntentType]:
        """Detect the query language and classify intent in a single LLM call.

        Args:
            query: The user's original query.

        Returns:
            Tuple of (detected_language, intent).
        """
        valid_intents = "factual, summary, comparison, procedural, unknown"
        prompt = render_prompt(
            "detect_language_and_intent",
            valid_intents=valid_intents,
            query=query,
        )
        raw = _extract_content(self._llm_chain.invoke(prompt))
        logger.debug("Combined detection raw response: %s", raw)

        # Parse response
        detected = "Danish"
        intent = IntentType.UNKNOWN
        for line in raw.splitlines():
            line = line.strip().lower()
            if line.startswith("language:"):
                detected = line.split(":", 1)[1].strip().strip(".")
            elif line.startswith("intent:"):
                raw_intent = line.split(":", 1)[1].strip().strip(".")
                if raw_intent in {i.value for i in IntentType}:
                    intent = IntentType(raw_intent)
                else:
                    logger.warning("Unrecognized intent '%s' from combined call, falling back to UNKNOWN", raw_intent)

        # Capitalize language name for display
        detected = detected.capitalize()
        logger.info("Detected query language: %s", detected)
        logger.info("Classified intent: %s", intent.value)
        return detected, intent

    def _translate_query(self, query: str, detected_language: str) -> str:
        """Translate the query into a corpus language when needed.

        BM25 needs token-level matches against the corpus, so when the user's
        query language is not present in the corpus we translate it to the
        primary corpus language. When the corpus contains the user's
        language already (single- or multi-language corpus), no translation
        is performed — the original query is used as-is.

        Args:
            query: The user's original query.
            detected_language: Detected language of the query.

        Returns:
            The retrieval query, translated when necessary.
        """
        doc_langs = self._ensure_document_languages()

        # Without a known corpus language we cannot pick a translation target.
        if not doc_langs:
            return query

        user_lang = detected_language.lower().strip()
        doc_lang_set = {lang.lower() for lang in doc_langs}
        # Accept the Danish autonym so legacy "dansk" detection still matches.
        if user_lang == "dansk":
            user_lang = "danish"

        # Query already in one of the corpus languages → BM25 will work as-is.
        if user_lang in doc_lang_set:
            return query

        if not self._translate_query_enabled:
            logger.info("Query translation disabled; using original query for retrieval")
            return query

        target = doc_langs[0]
        translate_prompt = render_prompt(
            "translate_query", target=target, query=query
        )
        translated = _extract_content(self._llm_chain.invoke(translate_prompt))
        logger.info("Translated query to %s: %s", target, translated)
        return translated

    # ------------------------------------------------------------------
    # LangGraph node functions
    # ------------------------------------------------------------------

    def _detect_node(self, state: RouterState) -> dict:
        """Detect language and classify intent."""
        user_language, intent = self._detect_language_and_intent(state["query"])
        return {"user_language": user_language, "intent": intent}

    def _translate_node(self, state: RouterState) -> dict:
        """Translate query to Danish if needed."""
        retrieval_query = self._translate_query(state["query"], state["user_language"])
        return {
            "retrieval_query": retrieval_query,
            "translated": retrieval_query != state["query"],
        }

    def _retrieve_node(self, state: RouterState) -> dict:
        """Run hybrid search."""
        hybrid_result = self._hybrid_retriever.search_detailed(
            state["retrieval_query"], top_k=state["top_k"]
        )
        logger.info("Retrieved %d results from hybrid search", len(hybrid_result.fused_results))
        return {
            "dense_results": hybrid_result.dense_results,
            "sparse_results": hybrid_result.sparse_results,
            "fused_results": hybrid_result.fused_results,
        }

    def _rerank_node(self, state: RouterState) -> dict:
        """Rerank fused results with cross-encoder."""
        results = state.get("fused_results", [])
        reranked = (
            self._reranker.rerank(state["retrieval_query"], results, top_k=state["top_k"])
            if results
            else []
        )
        confidence = max(r.score for r in reranked) if reranked else 0.0
        logger.info("Reranked to %d results", len(reranked))
        if reranked:
            logger.info("Confidence: %.4f (sigmoid-normalized by reranker)", confidence)
        return {"reranked": reranked, "confidence": confidence}

    def _broaden_query_node(self, state: RouterState) -> dict:
        """Rewrite the retrieval query when reranker confidence is low.

        Uses the LLM to generate alternative search terms while preserving
        the original meaning, then increments the retry counter.
        """
        prompt = render_prompt(
            "broaden_query",
            query=state["query"],
            retrieval_query=state["retrieval_query"],
        )
        broadened = _extract_content(self._llm_chain.invoke(prompt))
        logger.info(
            "Broadened query for retry %d: %s",
            state["retry_count"] + 1,
            broadened,
        )
        return {
            "retrieval_query": broadened,
            "retry_count": state["retry_count"] + 1,
        }

    @staticmethod
    def _check_confidence(state: RouterState) -> str:
        """Decide whether to retry retrieval or proceed to generation.

        Triggers a retry when results exist but confidence is below
        the threshold and retries remain.  Empty results (no documents
        matched at all) are not retried — broadening cannot help when
        the knowledge base simply lacks coverage.
        """
        if (
            state.get("reranked")
            and state["confidence"] < _LOW_CONFIDENCE_THRESHOLD
            and state["retry_count"] < _MAX_RETRIES
        ):
            logger.info(
                "Low confidence (%.4f < %.2f), retrying with broadened query",
                state["confidence"],
                _LOW_CONFIDENCE_THRESHOLD,
            )
            return "retry"
        return "accept"

    @staticmethod
    def _update_intent_node(state: RouterState) -> dict:
        """Promote FACTUAL to RAG when sources are found."""
        if state.get("reranked") and state["intent"] == IntentType.FACTUAL:
            logger.info("Overriding intent to RAG (sources retrieved)")
            return {"intent": IntentType.RAG}
        return {}

    def _generate_node(self, state: RouterState) -> dict:
        """Build prompt and call LLM."""
        reranked = state.get("reranked", [])
        context = "\n\n".join(r.chunk.text for r in reranked)
        prompt = self._build_prompt(
            state["query"], state["intent"], context, state["user_language"]
        )
        _measure_tokens("generate_answer", prompt, enabled=self._token_budget_enabled)
        answer = _extract_content(self._llm_chain.invoke(prompt))
        logger.info("Generated answer for intent=%s", state["intent"].value)
        return {"answer": answer}

    @staticmethod
    def _should_retrieve(state: RouterState) -> str:
        """Skip retrieval entirely when intent is UNKNOWN."""
        return "retrieve" if state["intent"] != IntentType.UNKNOWN else "generate"

    def _build_graph(self) -> object:
        """Build the LangGraph routing graph.

        Graph topology::

            detect → translate ─┬─ (UNKNOWN) ──────────────→ generate
                                └─ (other)  → retrieve → rerank
                                                 ↑          │
                                                 │      check_confidence
                                                 │        │       │
                                              broaden ←─ retry  accept
                                              _query        → update_intent
                                                                  │
                                                               generate

        Key LangGraph features demonstrated:
            - Conditional edges: intent-based skip, confidence-based routing
            - Cycle: low-confidence retry loop (broaden_query → retrieve)
            - Shared state: retry_count controls loop termination

        Returns:
            Compiled LangGraph graph.
        """
        graph: StateGraph = StateGraph(RouterState)
        graph.add_node("detect", self._detect_node)
        graph.add_node("translate", self._translate_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("rerank", self._rerank_node)
        graph.add_node("broaden_query", self._broaden_query_node)
        graph.add_node("update_intent", self._update_intent_node)
        graph.add_node("generate", self._generate_node)

        graph.set_entry_point("detect")
        graph.add_edge("detect", "translate")

        # Branch: skip retrieval entirely for off-topic queries
        graph.add_conditional_edges(
            "translate",
            self._should_retrieve,
            {"retrieve": "retrieve", "generate": "generate"},
        )

        graph.add_edge("retrieve", "rerank")

        # Branch + cycle: retry with broadened query on low confidence
        graph.add_conditional_edges(
            "rerank",
            self._check_confidence,
            {"retry": "broaden_query", "accept": "update_intent"},
        )
        graph.add_edge("broaden_query", "retrieve")  # ← the loop

        graph.add_edge("update_intent", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    def route(self, query: str, top_k: int) -> GenerationResponse:
        """Route a query through the full RAG pipeline via LangGraph.

        Args:
            query: The user's natural language query.
            top_k: Number of top documents to retrieve.

        Returns:
            GenerationResponse with answer, sources, and metadata.
        """
        logger.info("Routing query: %s", query)

        final_state: RouterState = self._graph.invoke(_make_initial_state(query, top_k))

        pipeline = PipelineDetails(
            original_query=query,
            retrieval_query=final_state["retrieval_query"],
            detected_language=final_state["user_language"],
            translated=final_state["translated"],
            dense_results=final_state.get("dense_results", []),
            sparse_results=final_state.get("sparse_results", []),
            fused_results=final_state.get("fused_results", []),
            reranked_results=final_state.get("reranked", []),
        )

        return GenerationResponse(
            answer=final_state["answer"],
            sources=final_state.get("reranked", []),
            intent=final_state["intent"],
            confidence=final_state["confidence"],
            pipeline_details=pipeline,
        )

    def route_stream(self, query: str, top_k: int) -> Generator[dict, None, None]:
        """Stream pipeline events as each LangGraph node completes.

        Each yielded dict contains a ``step`` key (the node name) plus
        node-specific fields.  A final synthetic event with ``step='done'``
        carries the fully serialised response under ``result``.

        Args:
            query: User query.
            top_k: Number of results to retrieve.

        Yields:
            Step event dicts, then a final ``done`` event with the result.
        """
        accumulated: dict = dict(_make_initial_state(query, top_k))

        for chunk in self._graph.stream(_make_initial_state(query, top_k), stream_mode="updates"):
            for node_name, update in chunk.items():
                if update is None:
                    continue
                accumulated.update(update)

                event: dict = {"step": node_name}
                if node_name == "detect":
                    event["intent"] = update.get("intent", IntentType.UNKNOWN).value
                    event["language"] = update.get("user_language", "")
                elif node_name == "translate":
                    event["translated"] = update.get("translated", False)
                    event["retrieval_query"] = update.get("retrieval_query", query)
                elif node_name == "retrieve":
                    event["dense_count"] = len(update.get("dense_results", []))
                    event["sparse_count"] = len(update.get("sparse_results", []))
                elif node_name == "rerank":
                    event["reranked_count"] = len(update.get("reranked", []))
                    event["confidence"] = round(update.get("confidence", 0.0), 4)
                elif node_name == "broaden_query":
                    event["retrieval_query"] = update.get("retrieval_query", "")
                    event["retry_count"] = update.get("retry_count", 0)

                yield event

        # Build the final response from accumulated state and emit as "done"
        reranked: list = accumulated.get("reranked", [])

        pd_acc = PipelineDetails(
            original_query=query,
            retrieval_query=accumulated.get("retrieval_query", query),
            detected_language=accumulated.get("user_language", "Danish"),
            translated=accumulated.get("translated", False),
            dense_results=accumulated.get("dense_results", []),
            sparse_results=accumulated.get("sparse_results", []),
            fused_results=accumulated.get("fused_results", []),
            reranked_results=reranked,
        )

        yield {
            "step": "done",
            "result": {
                "answer": accumulated.get("answer", ""),
                "sources": [r.to_dict() for r in reranked],
                "intent": accumulated.get("intent", IntentType.UNKNOWN).value,
                "confidence": accumulated.get("confidence", 0.0),
                "pipeline_details": {
                    "original_query": pd_acc.original_query,
                    "retrieval_query": pd_acc.retrieval_query,
                    "detected_language": pd_acc.detected_language,
                    "translated": pd_acc.translated,
                    "dense_results": [r.to_dict(include_text=False) for r in pd_acc.dense_results],
                    "sparse_results": [r.to_dict(include_text=False) for r in pd_acc.sparse_results],
                    "fused_results": [r.to_dict(include_text=False) for r in pd_acc.fused_results],
                    "reranked_results": [r.to_dict(include_text=False) for r in pd_acc.reranked_results],
                },
            },
        }

    def _build_prompt(
        self, query: str, intent: IntentType, context: str, user_language: str
    ) -> str:
        """Build a generation prompt tailored to the query intent.

        Args:
            query: The user's query.
            intent: Classified intent type.
            context: Retrieved context text.
            user_language: Detected language of the user's query.

        Returns:
            Formatted prompt string for the LLM.
        """
        intent_instructions = {
            IntentType.FACTUAL: (
                "Answer the question directly and concisely. "
                "No relevant source documents were found."
            ),
            IntentType.RAG: (
                "Answer the question directly and concisely based on the provided context. "
                "Cite specific details from the source documents."
            ),
            IntentType.SUMMARY: (
                "Provide a clear and comprehensive summary of the relevant information "
                "from the provided context."
            ),
            IntentType.COMPARISON: (
                "Compare and contrast the relevant items mentioned in the query "
                "using the provided context. Highlight key differences and similarities."
            ),
            IntentType.PROCEDURAL: (
                "Provide step-by-step instructions based on the provided context. "
                "Be clear and actionable."
            ),
            IntentType.UNKNOWN: (
                "This question is outside the KU document knowledge base. "
                "Begin your answer with a brief note that you are a document assistant for the "
                "University of Copenhagen and this topic is not covered in the available documents. "
                "Then answer the question as helpfully as possible from general knowledge."
            ),
        }

        instruction = intent_instructions[intent]

        doc_langs = self._ensure_document_languages()
        if doc_langs:
            corpus_clause = (
                f"The context documents may be in {' or '.join(doc_langs)} — "
                f"use them as reference but always reply in {user_language}."
            )
        else:
            corpus_clause = (
                f"The context documents may be in a different language — "
                f"use them as reference but always reply in {user_language}."
            )
        language_rule = (
            f"IMPORTANT: You MUST answer in {user_language}. "
            f"The user asked in {user_language}, so your entire response must be in {user_language}. "
            f"{corpus_clause}"
        )

        return (
            f"You are a helpful assistant for administrative staff at the University of Copenhagen (KU).\n\n"
            f"{language_rule}\n\n"
            f"Instruction: {instruction}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"REMINDER: {language_rule}\n\n"
            f"Answer in {user_language}:"
        )
