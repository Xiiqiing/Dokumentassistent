"""Tests for conversation memory."""

from src.agent.memory import ConversationMemory, Turn
from src.models import DocumentChunk, QueryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qr(chunk_id: str = "c1", doc_id: str = "doc.pdf", score: float = 0.8) -> QueryResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id, document_id=doc_id, text="text",
        metadata={"page_number": 1},
    )
    return QueryResult(chunk=chunk, score=score, source="test")


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------

class TestConversationMemory:
    def test_initially_empty(self) -> None:
        mem = ConversationMemory()
        assert mem.is_empty
        assert mem.turns == []
        assert mem.last_query() == ""
        assert mem.last_sources() == []

    def test_add_turn(self) -> None:
        mem = ConversationMemory()
        mem.add_turn("What is X?", "X is Y.", [_qr()])
        assert not mem.is_empty
        assert len(mem.turns) == 1
        assert mem.last_query() == "What is X?"

    def test_multiple_turns(self) -> None:
        mem = ConversationMemory()
        mem.add_turn("Q1", "A1")
        mem.add_turn("Q2", "A2")
        assert len(mem.turns) == 2
        assert mem.last_query() == "Q2"

    def test_clear(self) -> None:
        mem = ConversationMemory()
        mem.add_turn("Q1", "A1")
        mem.clear()
        assert mem.is_empty

    def test_turns_returns_copy(self) -> None:
        mem = ConversationMemory()
        mem.add_turn("Q1", "A1")
        turns = mem.turns
        turns.append(Turn(query="fake", answer="fake"))
        assert len(mem.turns) == 1  # original unaffected


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

class TestEviction:
    def test_max_turns_eviction(self) -> None:
        mem = ConversationMemory(max_turns=3)
        for i in range(5):
            mem.add_turn(f"Q{i}", f"A{i}")
        assert len(mem.turns) == 3
        # Oldest should be Q2 (Q0 and Q1 evicted)
        assert mem.turns[0].query == "Q2"

    def test_max_turns_one(self) -> None:
        mem = ConversationMemory(max_turns=1)
        mem.add_turn("Q1", "A1")
        mem.add_turn("Q2", "A2")
        assert len(mem.turns) == 1
        assert mem.turns[0].query == "Q2"


# ---------------------------------------------------------------------------
# format_history
# ---------------------------------------------------------------------------

class TestFormatHistory:
    def test_empty_history(self) -> None:
        mem = ConversationMemory()
        assert mem.format_history() == ""

    def test_includes_query_and_answer(self) -> None:
        mem = ConversationMemory()
        mem.add_turn("What is X?", "X is a policy.")
        text = mem.format_history()
        assert "What is X?" in text
        assert "X is a policy." in text

    def test_includes_source_doc_ids(self) -> None:
        mem = ConversationMemory()
        sources = [_qr(doc_id="policy.pdf"), _qr(chunk_id="c2", doc_id="rules.pdf")]
        mem.add_turn("Q", "A", sources)
        text = mem.format_history()
        assert "policy.pdf" in text
        assert "rules.pdf" in text

    def test_max_recent_limits_output(self) -> None:
        mem = ConversationMemory()
        for i in range(10):
            mem.add_turn(f"Q{i}", f"A{i}")
        text = mem.format_history(max_recent=2)
        assert "Q8" in text
        assert "Q9" in text
        assert "Q0" not in text

    def test_long_answer_truncated(self) -> None:
        mem = ConversationMemory()
        mem.add_turn("Q", "x" * 1000)
        text = mem.format_history()
        # Answer should be truncated to 500 chars
        assert len(text) < 1000


# ---------------------------------------------------------------------------
# get_prior_sources
# ---------------------------------------------------------------------------

class TestGetPriorSources:
    def test_empty_returns_empty(self) -> None:
        mem = ConversationMemory()
        assert mem.get_prior_sources() == []

    def test_collects_across_turns(self) -> None:
        mem = ConversationMemory()
        mem.add_turn("Q1", "A1", [_qr(chunk_id="c1", score=0.8)])
        mem.add_turn("Q2", "A2", [_qr(chunk_id="c2", score=0.9)])
        sources = mem.get_prior_sources()
        assert len(sources) == 2
        # Sorted by score descending
        assert sources[0].score == 0.9

    def test_deduplicates_by_chunk_id(self) -> None:
        mem = ConversationMemory()
        mem.add_turn("Q1", "A1", [_qr(chunk_id="c1", score=0.5)])
        mem.add_turn("Q2", "A2", [_qr(chunk_id="c1", score=0.9)])
        sources = mem.get_prior_sources()
        assert len(sources) == 1
        assert sources[0].score == 0.9  # keeps higher score

    def test_no_sources_turns(self) -> None:
        mem = ConversationMemory()
        mem.add_turn("Q1", "A1")  # no sources
        assert mem.get_prior_sources() == []


# ---------------------------------------------------------------------------
# Integration: memory in PlanAndExecuteRouter
# ---------------------------------------------------------------------------

class TestMemoryIntegration:
    def test_route_records_turn(self) -> None:
        """After route(), the conversation turn should be recorded in memory."""
        from unittest.mock import MagicMock, patch
        from langchain_core.messages import AIMessage
        from src.agent.plan_and_execute import PlanAndExecuteRouter

        llm = MagicMock()
        retriever = MagicMock()
        reranker = MagicMock()
        vector_store = MagicMock()
        memory = ConversationMemory()

        plan_json = '[{"action": "search", "detail": "test"}]'
        llm.invoke.side_effect = [plan_json, "The answer."]

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [AIMessage(content="Found info.")]}

        router = PlanAndExecuteRouter(
            llm, retriever, reranker, vector_store, memory=memory,
        )

        with patch("src.agent.plan_and_execute.create_react_agent", return_value=mock_agent):
            router.route("test question", top_k=5)

        assert not memory.is_empty
        assert memory.last_query() == "test question"
        assert memory.turns[0].answer == "The answer."

    def test_history_injected_into_planner(self) -> None:
        """On a follow-up query, conversation history should appear in the planner prompt."""
        from unittest.mock import MagicMock, patch
        from langchain_core.messages import AIMessage
        from src.agent.plan_and_execute import PlanAndExecuteRouter

        llm = MagicMock()
        memory = ConversationMemory()
        memory.add_turn("What is the exam policy?", "The exam policy says...")

        plan_json = '[{"action": "search", "detail": "follow-up"}]'
        llm.invoke.side_effect = [plan_json, "Follow-up answer."]

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [AIMessage(content="More info.")]}

        router = PlanAndExecuteRouter(
            llm, MagicMock(), MagicMock(), MagicMock(), memory=memory,
        )

        with patch("src.agent.plan_and_execute.create_react_agent", return_value=mock_agent):
            router.route("What about the grading?", top_k=5)

        # The first LLM call is the planner — check it includes history
        planner_prompt = llm.invoke.call_args_list[0][0][0]
        assert "exam policy" in planner_prompt
        assert "Conversation history" in planner_prompt

    def test_multi_turn_accumulates(self) -> None:
        """Multiple route() calls should accumulate turns in memory."""
        from unittest.mock import MagicMock, patch
        from langchain_core.messages import AIMessage
        from src.agent.plan_and_execute import PlanAndExecuteRouter

        llm = MagicMock()
        memory = ConversationMemory()

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [AIMessage(content="info")]}

        router = PlanAndExecuteRouter(
            llm, MagicMock(), MagicMock(), MagicMock(), memory=memory,
        )

        for i in range(3):
            plan_json = f'[{{"action": "search", "detail": "q{i}"}}]'
            llm.invoke.side_effect = [plan_json, f"Answer {i}"]
            with patch("src.agent.plan_and_execute.create_react_agent", return_value=mock_agent):
                router.route(f"Question {i}", top_k=5)

        assert len(memory.turns) == 3
