"""Tests for the Plan-and-Execute agent router."""

from unittest.mock import MagicMock, patch
import json

from src.agent.plan_and_execute import (
    PlanAndExecuteRouter,
    PlanExecState,
    PlanStep,
    _extract_last_ai_text,
    _parse_plan,
)
from src.models import (
    DocumentChunk,
    GenerationResponse,
    IntentType,
    QueryResult,
)
from src.retrieval.hybrid import HybridSearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str = "c1", text: str = "text") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id, document_id="doc.pdf", text=text,
        metadata={"page_number": 1, "chunk_index": 0},
    )


def _qr(chunk_id: str = "c1", score: float = 0.8, text: str = "text") -> QueryResult:
    return QueryResult(chunk=_chunk(chunk_id=chunk_id, text=text), score=score, source="hybrid")


def _hybrid_result(results: list[QueryResult]) -> HybridSearchResult:
    return HybridSearchResult(
        dense_results=results, sparse_results=results, fused_results=results,
    )


# ---------------------------------------------------------------------------
# _parse_plan
# ---------------------------------------------------------------------------

class TestParsePlan:
    def test_valid_json(self) -> None:
        raw = '[{"action": "search", "detail": "exam rules"}]'
        steps = _parse_plan(raw)
        assert len(steps) == 1
        assert steps[0]["action"] == "search"
        assert steps[0]["detail"] == "exam rules"

    def test_multiple_steps(self) -> None:
        raw = json.dumps([
            {"action": "search", "detail": "policy A"},
            {"action": "search", "detail": "policy B"},
            {"action": "summarize", "detail": "doc.pdf"},
        ])
        steps = _parse_plan(raw)
        assert len(steps) == 3

    def test_markdown_fenced(self) -> None:
        raw = '```json\n[{"action": "search", "detail": "test"}]\n```'
        steps = _parse_plan(raw)
        assert len(steps) == 1
        assert steps[0]["action"] == "search"

    def test_json_with_surrounding_text(self) -> None:
        raw = 'Here is the plan:\n[{"action": "search", "detail": "x"}]\nDone.'
        steps = _parse_plan(raw)
        assert len(steps) == 1

    def test_invalid_json_falls_back(self) -> None:
        raw = "this is not json at all"
        steps = _parse_plan(raw)
        assert len(steps) == 1
        assert steps[0]["action"] == "search"

    def test_empty_array_falls_back(self) -> None:
        raw = "[]"
        steps = _parse_plan(raw)
        assert len(steps) == 1  # fallback to single search

    def test_malformed_items_skipped(self) -> None:
        raw = json.dumps([
            {"action": "search", "detail": "good"},
            {"bad": "step"},
            {"action": "search", "detail": "also good"},
        ])
        steps = _parse_plan(raw)
        assert len(steps) == 2

    def test_non_list_wrapped(self) -> None:
        raw = '{"action": "search", "detail": "test"}'
        steps = _parse_plan(raw)
        assert len(steps) == 1


# ---------------------------------------------------------------------------
# _extract_last_ai_text
# ---------------------------------------------------------------------------

class TestExtractLastAIText:
    def test_returns_last_ai_message(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage
        messages = [
            HumanMessage(content="question"),
            AIMessage(content="first"),
            AIMessage(content="second"),
        ]
        assert _extract_last_ai_text(messages) == "second"

    def test_skips_tool_calls(self) -> None:
        from langchain_core.messages import AIMessage
        msg_with_tools = AIMessage(content="calling tool", tool_calls=[{"name": "t", "args": {}, "id": "1"}])
        msg_final = AIMessage(content="the answer")
        assert _extract_last_ai_text([msg_with_tools, msg_final]) == "the answer"

    def test_empty_messages(self) -> None:
        assert _extract_last_ai_text([]) == ""


# ---------------------------------------------------------------------------
# PlanAndExecuteRouter — plan node
# ---------------------------------------------------------------------------

class TestPlanNode:
    def test_plan_node_generates_steps(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = '[{"action": "search", "detail": "KU regler"}]'

        router = PlanAndExecuteRouter(
            llm=llm,
            hybrid_retriever=MagicMock(),
            reranker=MagicMock(),
            vector_store=MagicMock(),
        )

        state = PlanExecState(
            query="What are the rules?",
            top_k=5, plan=[], step_index=0, step_results=[], answer="",
        )
        result = router._plan_node(state)
        assert len(result["plan"]) == 1
        assert result["plan"][0]["action"] == "search"
        assert result["step_index"] == 0

    def test_plan_node_handles_bad_llm_output(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = "I cannot produce JSON"

        router = PlanAndExecuteRouter(
            llm=llm,
            hybrid_retriever=MagicMock(),
            reranker=MagicMock(),
            vector_store=MagicMock(),
        )

        state = PlanExecState(
            query="test", top_k=5, plan=[], step_index=0, step_results=[], answer="",
        )
        result = router._plan_node(state)
        assert len(result["plan"]) >= 1  # fallback plan


# ---------------------------------------------------------------------------
# PlanAndExecuteRouter — should_execute
# ---------------------------------------------------------------------------

class TestShouldExecute:
    def test_more_steps_returns_execute(self) -> None:
        state = PlanExecState(
            query="q", top_k=5,
            plan=[PlanStep(action="search", detail="x")],
            step_index=0, step_results=[], answer="",
        )
        assert PlanAndExecuteRouter._should_execute(state) == "execute"

    def test_all_steps_done_returns_synthesize(self) -> None:
        state = PlanExecState(
            query="q", top_k=5,
            plan=[PlanStep(action="search", detail="x")],
            step_index=1, step_results=[], answer="",
        )
        assert PlanAndExecuteRouter._should_execute(state) == "synthesize"

    def test_empty_plan_returns_synthesize(self) -> None:
        state = PlanExecState(
            query="q", top_k=5, plan=[], step_index=0, step_results=[], answer="",
        )
        assert PlanAndExecuteRouter._should_execute(state) == "synthesize"

    def test_max_steps_cap(self) -> None:
        """Step index at _MAX_STEPS should stop execution."""
        state = PlanExecState(
            query="q", top_k=5,
            plan=[PlanStep(action="search", detail=f"q{i}") for i in range(10)],
            step_index=6,  # == _MAX_STEPS
            step_results=[], answer="",
        )
        assert PlanAndExecuteRouter._should_execute(state) == "synthesize"


# ---------------------------------------------------------------------------
# PlanAndExecuteRouter — synthesize node
# ---------------------------------------------------------------------------

class TestSynthesizeNode:
    def test_synthesize_combines_results(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = "Combined answer about exams."

        router = PlanAndExecuteRouter(
            llm=llm,
            hybrid_retriever=MagicMock(),
            reranker=MagicMock(),
            vector_store=MagicMock(),
        )

        state = PlanExecState(
            query="exam rules",
            top_k=5, plan=[],
            step_index=2,
            step_results=[
                ("search: exam bachelor", "Found bachelor exam rules..."),
                ("search: exam master", "Found master exam rules..."),
            ],
            answer="",
        )
        result = router._synthesize_node(state)
        assert result["answer"] == "Combined answer about exams."

        # Verify prompt includes both step results
        prompt = llm.invoke.call_args[0][0]
        assert "bachelor exam rules" in prompt
        assert "master exam rules" in prompt


# ---------------------------------------------------------------------------
# PlanAndExecuteRouter — full route (integration with mocks)
# ---------------------------------------------------------------------------

class TestFullRoute:
    def test_route_produces_response(self) -> None:
        """Full route with mocked LLM and retrieval components."""
        llm = MagicMock()
        retriever = MagicMock()
        reranker = MagicMock()
        vector_store = MagicMock()

        # Plan: single search step
        plan_json = '[{"action": "search", "detail": "test query"}]'
        # Sub-agent answer after executing step
        from langchain_core.messages import AIMessage
        sub_agent_result = {"messages": [AIMessage(content="Found relevant info about test.")]}
        # Final synthesis
        final_answer = "The test policy states..."

        # LLM calls: plan, executor system/tools, synthesis
        # We mock the LLM and also mock the sub-agent creation
        llm.invoke.side_effect = [plan_json, final_answer]

        results = [_qr(chunk_id="c1", score=0.9, text="test policy")]
        retriever.search_detailed.return_value = _hybrid_result(results)
        reranker.rerank.return_value = results
        vector_store.list_document_ids.return_value = ["doc.pdf"]

        router = PlanAndExecuteRouter(llm, retriever, reranker, vector_store)

        # Patch create_react_agent to return a mock that returns our sub_agent_result
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = sub_agent_result

        with patch("src.agent.plan_and_execute.create_react_agent", return_value=mock_agent):
            response = router.route("test question", top_k=5)

        assert isinstance(response, GenerationResponse)
        assert response.answer == "The test policy states..."

    def test_route_with_no_results(self) -> None:
        """Route when retrieval finds nothing."""
        llm = MagicMock()
        retriever = MagicMock()
        reranker = MagicMock()
        vector_store = MagicMock()

        plan_json = '[{"action": "search", "detail": "nonexistent"}]'
        from langchain_core.messages import AIMessage
        sub_agent_result = {"messages": [AIMessage(content="No relevant documents found.")]}

        llm.invoke.side_effect = [plan_json, "I could not find information."]

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = sub_agent_result

        router = PlanAndExecuteRouter(llm, retriever, reranker, vector_store)

        with patch("src.agent.plan_and_execute.create_react_agent", return_value=mock_agent):
            response = router.route("nonexistent topic", top_k=5)

        assert response.intent == IntentType.FACTUAL
        assert response.confidence == 0.0

    def test_route_multi_step(self) -> None:
        """Route with a multi-step plan."""
        llm = MagicMock()
        retriever = MagicMock()
        reranker = MagicMock()
        vector_store = MagicMock()

        plan_json = json.dumps([
            {"action": "search", "detail": "policy A"},
            {"action": "search", "detail": "policy B"},
        ])
        from langchain_core.messages import AIMessage
        sub_result_1 = {"messages": [AIMessage(content="Policy A info")]}
        sub_result_2 = {"messages": [AIMessage(content="Policy B info")]}

        llm.invoke.side_effect = [plan_json, "Comparison of A and B."]

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [sub_result_1, sub_result_2]

        router = PlanAndExecuteRouter(llm, retriever, reranker, vector_store)

        with patch("src.agent.plan_and_execute.create_react_agent", return_value=mock_agent):
            response = router.route("Compare A and B", top_k=5)

        assert response.answer == "Comparison of A and B."
        # Sub-agent should have been called twice
        assert mock_agent.invoke.call_count == 2


# ---------------------------------------------------------------------------
# PlanAndExecuteRouter — route_stream
# ---------------------------------------------------------------------------

class TestRouteStream:
    def test_stream_yields_plan_execute_synthesize_done(self) -> None:
        """Streaming should yield events in order: plan, execute, synthesize, done."""
        llm = MagicMock()
        retriever = MagicMock()
        reranker = MagicMock()
        vector_store = MagicMock()

        plan_json = '[{"action": "search", "detail": "test"}]'
        from langchain_core.messages import AIMessage
        sub_agent_result = {"messages": [AIMessage(content="Found info.")]}

        llm.invoke.side_effect = [plan_json, "Final answer."]

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = sub_agent_result

        router = PlanAndExecuteRouter(llm, retriever, reranker, vector_store)

        with patch("src.agent.plan_and_execute.create_react_agent", return_value=mock_agent):
            events = list(router.route_stream("test", top_k=5))

        step_names = [e["step"] for e in events]
        assert "plan" in step_names
        assert "done" in step_names
        # done event has the result
        done_event = [e for e in events if e["step"] == "done"][0]
        assert "result" in done_event
        assert done_event["result"]["answer"] == "Final answer."
