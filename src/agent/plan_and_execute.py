"""Plan-and-Execute agent router using LangGraph.

Replaces the flat ReAct loop with a structured three-phase pipeline:

1. **Planner** — analyses the user query and produces an ordered list of
   steps (e.g. "search for exam rules", "search for grading policy",
   "compare both").
2. **Executor** — runs each step via a short ReAct sub-graph that has
   access to all retrieval tools.
3. **Synthesizer** — collects the results from all executed steps and
   produces a final, cited answer.

The separation gives the pipeline *predictable structure* while still
allowing the executor to reason freely within each step.
"""

import json
import logging
import re
from collections.abc import Generator
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from src.agent.memory import ConversationMemory
from src.agent.tools import ToolResultStore, make_retrieval_tools
from src.models import GenerationResponse, IntentType, PipelineDetails, QueryResult
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_MAX_STEPS = 6

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

_PLANNER_PROMPT = (
    "You are a planning assistant for the University of Copenhagen (KU) document system.\n\n"
    "Given a user question, produce a JSON list of 1–4 steps needed to answer it.\n"
    "Each step is an object with:\n"
    '  - "action": one of "search", "search_within", "multi_search", '
    '"summarize", "list_docs", "fetch_doc"\n'
    '  - "detail": a short description of what to do (e.g. the search query, document ID)\n\n'
    "Rules:\n"
    "- IMPORTANT: Most questions probably only need 1 step. Only use 2+ steps when the question explicitly asks about multiple distinct topics.\n"
    "- For simple factual questions: 1 search step is enough.\n"
    "- For comparison questions: use multi_search or separate search steps.\n"
    "- For document overview requests: use summarize.\n"
    "- For questions with multiple aspects: use 2–4 separate steps.\n"
    "- Always end with the steps needed; do NOT include a final 'answer' step.\n\n"
    "Reply with ONLY the JSON array, nothing else. No explanation, no thinking.\n\n"
    "Examples:\n"
    'Question: "What is the exam policy?"\n'
    '[{"action": "search", "detail": "KU eksamensregler"}]\n\n'
    'Question: "Compare vacation rules for academic vs administrative staff"\n'
    '[{"action": "search", "detail": "ferieregler videnskabeligt personale"}, '
    '{"action": "search", "detail": "ferieregler administrativt personale"}]\n\n'
    'Question: "Summarize the AI policy document"\n'
    '[{"action": "summarize", "detail": "ku_ai_policy.pdf"}]\n\n'
    'Question: "Which documents are about AI? Summarize and find the rules for written exams"\n'
    '[{"action": "list_docs", "detail": "list all available documents"}, '
    '{"action": "search", "detail": "AI dokumenter KU"}, '
    '{"action": "search", "detail": "regler skriftlige opgaver eksamen GAI"}]\n\n'
    "Now plan for this question:\n"
)

_EXECUTOR_SYSTEM = (
    "/no_think\n"
    "You are executing ONE step of a plan to answer a user's question about "
    "University of Copenhagen (KU) documents.\n\n"
    "You have retrieval tools available. Execute the step described below, "
    "then summarise what you found in 2-3 sentences. If you find nothing "
    "relevant, say so clearly.\n\n"
    "Do NOT produce a final answer — just report what you found for this step."
)

_SYNTHESIZER_PROMPT = (
    "You are a helpful assistant for administrative staff at the University "
    "of Copenhagen (KU).\n\n"
    "Below are the results gathered from multiple research steps. "
    "Synthesize them into a single coherent answer to the user's original question.\n\n"
    "Guidelines:\n"
    "- Cite document sources using [1], [2], etc.\n"
    "- Answer in the same language as the user's question.\n"
    "- Be concise but thorough.\n"
    "- If some steps found no results, acknowledge gaps honestly.\n\n"
)


# ------------------------------------------------------------------
# Graph state
# ------------------------------------------------------------------

class PlanStep(TypedDict):
    """A single step in the execution plan."""

    action: str
    detail: str


class PlanExecState(TypedDict):
    """State for the Plan-and-Execute graph.

    Attributes:
        query: The user's original question.
        top_k: Number of results per retrieval call.
        plan: Ordered list of steps produced by the planner.
        step_index: Index of the next step to execute.
        step_results: List of (step_description, result_text) pairs.
        answer: Final synthesised answer.
    """

    query: str
    top_k: int
    plan: list[PlanStep]
    step_index: int
    step_results: list[tuple[str, str]]
    answer: str


# ------------------------------------------------------------------
# Router class
# ------------------------------------------------------------------

class PlanAndExecuteRouter:
    """Routes queries through a Plan-and-Execute pipeline.

    Graph topology::

        plan → should_execute? ─┬─ yes → execute_step → should_execute?
                                └─ no  → synthesize → END
    """

    def __init__(
        self,
        llm: Runnable,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        vector_store: VectorStore,
        default_top_k: int = 5,
        memory: ConversationMemory | None = None,
    ) -> None:
        """Initialise the Plan-and-Execute router.

        Args:
            llm: LLM with tool-calling support.
            hybrid_retriever: HybridRetriever instance.
            reranker: Reranker instance.
            vector_store: VectorStore instance.
            default_top_k: Default number of results per retrieval call.
            memory: Optional ConversationMemory for multi-turn context.
                When provided, prior conversation history is injected into
                planner and synthesizer prompts, and each completed turn
                is automatically recorded.
        """
        self._llm = llm
        self._hybrid_retriever = hybrid_retriever
        self._reranker = reranker
        self._vector_store = vector_store
        self._default_top_k = default_top_k
        self._memory = memory or ConversationMemory()

    # ------------------------------------------------------------------
    # Node functions
    # ------------------------------------------------------------------

    def _plan_node(self, state: PlanExecState) -> dict:
        """Generate an execution plan from the user query."""
        history = self._memory.format_history()
        history_section = ""
        if history:
            history_section = (
                f"Conversation history (for context on follow-up questions):\n"
                f"{history}\n\n"
            )
        prompt = _PLANNER_PROMPT + history_section + f'Question: "{state["query"]}"'
        raw = _extract_content(self._llm.invoke(prompt))
        logger.info("Planner raw output: %s", raw)

        plan = _parse_plan(raw)
        logger.info("Plan: %d steps — %s", len(plan), plan)
        return {"plan": plan, "step_index": 0, "step_results": []}

    @staticmethod
    def _should_execute(state: PlanExecState) -> str:
        """Decide whether to execute the next step or synthesize."""
        if state["step_index"] < len(state["plan"]) and state["step_index"] < _MAX_STEPS:
            return "execute"
        return "synthesize"

    def _make_execute_step_node(self, store: ToolResultStore):
        """Create an execute_step node closure bound to a request-scoped store.

        Args:
            store: ToolResultStore for this specific request.

        Returns:
            Node function for LangGraph.
        """

        def _execute_step_node(state: PlanExecState) -> dict:
            idx = state["step_index"]
            step = state["plan"][idx]
            step_desc = f'{step["action"]}: {step["detail"]}'
            logger.info("Executing step %d/%d: %s", idx + 1, len(state["plan"]), step_desc)

            tools = make_retrieval_tools(
                self._hybrid_retriever,
                self._reranker,
                self._vector_store,
                store,
                self._default_top_k,
                llm_chain=self._llm,
            )
            sub_agent = create_react_agent(self._llm, tools)

            step_prompt = (
                f'Step to execute: {step_desc}\n\n'
                f'Original user question (for context): {state["query"]}'
            )

            result = sub_agent.invoke({
                "messages": [
                    SystemMessage(content=_EXECUTOR_SYSTEM),
                    HumanMessage(content=step_prompt),
                ]
            })

            answer = _extract_last_ai_text(result.get("messages", []))
            logger.info("Step %d result: %s", idx + 1, answer[:200])

            new_results = list(state["step_results"]) + [(step_desc, answer)]
            return {"step_index": idx + 1, "step_results": new_results}

        return _execute_step_node

    def _synthesize_node(self, state: PlanExecState) -> dict:
        """Synthesize a final answer from all step results."""
        step_texts = []
        for i, (desc, result) in enumerate(state["step_results"], 1):
            step_texts.append(f"### Step {i}: {desc}\n{result}")
        gathered = "\n\n".join(step_texts)

        history = self._memory.format_history()
        history_section = ""
        if history:
            history_section = (
                f"Prior conversation:\n{history}\n\n"
            )

        prompt = (
            f"{_SYNTHESIZER_PROMPT}"
            f"{history_section}"
            f"Original question: {state['query']}\n\n"
            f"Research results:\n{gathered}\n\n"
            f"Answer:"
        )
        answer = _extract_content(self._llm.invoke(prompt))
        logger.info("Synthesized final answer (%d chars)", len(answer))
        return {"answer": answer}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self, store: ToolResultStore) -> object:
        """Build the Plan-and-Execute LangGraph.

        Args:
            store: Request-scoped ToolResultStore for this invocation.

        Returns:
            Compiled LangGraph.
        """
        graph: StateGraph = StateGraph(PlanExecState)

        graph.add_node("plan", self._plan_node)
        graph.add_node("execute_step", self._make_execute_step_node(store))
        graph.add_node("synthesize", self._synthesize_node)

        graph.set_entry_point("plan")
        graph.add_conditional_edges(
            "plan",
            self._should_execute,
            {"execute": "execute_step", "synthesize": "synthesize"},
        )
        graph.add_conditional_edges(
            "execute_step",
            self._should_execute,
            {"execute": "execute_step", "synthesize": "synthesize"},
        )
        graph.add_edge("synthesize", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Public interface (mirrors QueryRouter)
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        top_k: int,
        memory: ConversationMemory | None = None,
    ) -> GenerationResponse:
        """Route a query through the Plan-and-Execute pipeline.

        Args:
            query: The user's natural language query.
            top_k: Number of top documents to retrieve per tool call.
            memory: Optional per-session memory override.  When provided
                this memory is used instead of the router's default memory.

        Returns:
            GenerationResponse with answer, sources, intent, and confidence.
        """
        original_memory = self._memory
        if memory is not None:
            self._memory = memory
        try:
            logger.info("PlanExec routing query: %s", query)
            store = ToolResultStore()

            initial_state = PlanExecState(
                query=query,
                top_k=top_k,
                plan=[],
                step_index=0,
                step_results=[],
                answer="",
            )

            graph = self._build_graph(store)
            final_state: PlanExecState = graph.invoke(initial_state)

            sources = store.retrieved[:top_k]
            confidence = max((r.score for r in sources), default=0.0)

            plan_step_strs = [
                f'{s["action"]}: {s["detail"]}' for s in final_state.get("plan", [])
            ]
            tool_call_strs = [f"{name}: {arg}" for name, arg in store.tool_calls]

            response = GenerationResponse(
                answer=final_state["answer"],
                sources=sources,
                intent=IntentType.RAG if sources else IntentType.FACTUAL,
                confidence=confidence,
                pipeline_details=PipelineDetails(
                    original_query=query,
                    retrieval_query=", ".join(
                        q for name, q in store.tool_calls if name == "hybrid_search"
                    ) or query,
                    dense_results=store.dense_results,
                    sparse_results=store.sparse_results,
                    fused_results=store.fused_results,
                    reranked_results=sources,
                    plan_steps=plan_step_strs,
                    tool_calls=tool_call_strs,
                ),
            )

            self._memory.add_turn(query, response.answer, sources)
            return response
        finally:
            self._memory = original_memory

    def route_stream(
        self,
        query: str,
        top_k: int,
        memory: ConversationMemory | None = None,
    ) -> Generator[dict, None, None]:
        """Stream Plan-and-Execute events step by step.

        Yields event dicts with step types:
        - ``plan`` — plan was generated; carries ``steps``.
        - ``execute_step`` — a step was executed; carries ``step_index``,
          ``step_desc``, ``result_preview``.
        - ``synthesize`` — final answer generated.
        - ``done`` — final event with full result payload.

        Args:
            query: User query.
            top_k: Number of results to retrieve per tool call.
            memory: Optional per-session memory override.

        Yields:
            Step event dicts.
        """
        original_memory = self._memory
        if memory is not None:
            self._memory = memory
        try:
            yield from self._route_stream_inner(query, top_k)
        finally:
            self._memory = original_memory

    def _route_stream_inner(self, query: str, top_k: int) -> Generator[dict, None, None]:
        """Internal streaming implementation."""
        store = ToolResultStore()

        initial_state = PlanExecState(
            query=query,
            top_k=top_k,
            plan=[],
            step_index=0,
            step_results=[],
            answer="",
        )

        graph = self._build_graph(store)
        accumulated: dict = dict(initial_state)

        for chunk in graph.stream(initial_state, stream_mode="updates"):
            for node_name, update in chunk.items():
                if update is None:
                    continue
                accumulated.update(update)

                if node_name == "plan":
                    yield {
                        "step": "plan",
                        "steps": [
                            f'{s["action"]}: {s["detail"]}'
                            for s in update.get("plan", [])
                        ],
                    }
                elif node_name == "execute_step":
                    results = update.get("step_results", [])
                    if results:
                        last_desc, last_result = results[-1]
                        yield {
                            "step": "execute_step",
                            "step_index": update.get("step_index", 0),
                            "step_desc": last_desc,
                            "result_preview": last_result[:300],
                        }
                elif node_name == "synthesize":
                    yield {"step": "synthesize"}

        sources = store.retrieved[:top_k]
        confidence = max((r.score for r in sources), default=0.0)
        answer = accumulated.get("answer", "")

        self._memory.add_turn(query, answer, sources)

        yield {
            "step": "done",
            "result": {
                "answer": answer,
                "sources": [r.to_dict() for r in sources],
                "intent": (IntentType.RAG if sources else IntentType.FACTUAL).value,
                "confidence": confidence,
                "pipeline_details": {
                    "original_query": query,
                    "retrieval_query": ", ".join(
                        q for name, q in store.tool_calls if name == "hybrid_search"
                    ) or query,
                    "detected_language": "",
                    "translated": False,
                    "dense_results": [r.to_dict(include_text=False) for r in store.dense_results],
                    "sparse_results": [r.to_dict(include_text=False) for r in store.sparse_results],
                    "fused_results": [r.to_dict(include_text=False) for r in store.fused_results],
                    "reranked_results": [r.to_dict(include_text=False) for r in sources],
                    "plan_steps": [
                        f'{s["action"]}: {s["detail"]}'
                        for s in accumulated.get("plan", [])
                    ],
                    "tool_calls": [f"{n}: {a}" for n, a in store.tool_calls],
                },
            },
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove ``<think>`` blocks — both closed and unclosed.

    Some models (Qwen3) always emit ``<think>...</think>``; others may
    leave the tag unclosed.  This handles both cases.
    """
    text = _THINK_CLOSED_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.strip()


def _extract_content(result: object) -> str:
    """Extract plain text from an LLM invoke result.

    Handles:
    - AIMessage with ``content: str``
    - AIMessage with ``content: list[str | dict]`` (some providers)
    - Plain strings (e.g. from StrOutputParser or test mocks)

    Args:
        result: Return value of ``llm.invoke()`` or ``chain.invoke()``.

    Returns:
        Cleaned text with ``<think>`` blocks removed.
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


def _parse_plan(raw: str) -> list[PlanStep]:
    """Parse the planner's JSON output into a list of PlanStep dicts.

    Robust against markdown fences, trailing text, and minor formatting issues.

    Args:
        raw: Raw LLM output expected to contain a JSON array.

    Returns:
        List of PlanStep dicts. Falls back to a single search step on failure.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove opening and closing fences
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract a JSON array from the text
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                logger.warning("Failed to parse plan, falling back to single search")
                return [PlanStep(action="search", detail=cleaned[:200])]
        else:
            logger.warning("No JSON array found in plan output, falling back")
            return [PlanStep(action="search", detail=cleaned[:200])]

    if not isinstance(parsed, list):
        logger.warning("Plan is not a list, wrapping")
        parsed = [parsed]

    steps: list[PlanStep] = []
    for item in parsed:
        if isinstance(item, dict) and "action" in item and "detail" in item:
            steps.append(PlanStep(action=str(item["action"]), detail=str(item["detail"])))
        else:
            logger.warning("Skipping malformed plan step: %s", item)

    if not steps:
        return [PlanStep(action="search", detail="general search")]

    return steps


def _extract_last_ai_text(messages: list) -> str:
    """Return the text content of the last non-tool-call AI message.

    Args:
        messages: List of LangChain message objects.

    Returns:
        The extracted text, or empty string if none found.
    """
    for msg in reversed(messages):
        if (
            isinstance(msg, AIMessage)
            and msg.content
            and not getattr(msg, "tool_calls", None)
        ):
            return _extract_content(msg)
    return ""
