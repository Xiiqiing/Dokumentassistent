"""Tests for query router with mock LLM and components."""

from unittest.mock import MagicMock

import pytest

from src.agent.router import QueryRouter
from src.models import (
    DocumentChunk,
    GenerationResponse,
    IntentType,
    QueryResult,
)


def _make_query_result(text: str, score: float) -> QueryResult:
    """Create a QueryResult for testing."""
    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="d1",
        text=text,
        metadata={"page": 1},
    )
    return QueryResult(chunk=chunk, score=score, source="hybrid")


def _make_hybrid_result(results: list[QueryResult]) -> MagicMock:
    """Create a mock HybridSearchResult."""
    hybrid = MagicMock()
    hybrid.dense_results = results
    hybrid.sparse_results = results
    hybrid.fused_results = results
    return hybrid


@pytest.fixture
def mock_components():
    """Create mock intent classifier, retriever, reranker, and generator."""
    classifier = MagicMock()
    retriever = MagicMock()
    reranker = MagicMock()
    generator = MagicMock()
    return classifier, retriever, reranker, generator


def _setup_generator_danish(generator: MagicMock, final_answer: str) -> None:
    """Configure generator mock for Danish queries (no translation needed)."""
    generator.invoke.side_effect = ["Danish", final_answer]


def _setup_generator_english(generator: MagicMock, translated_query: str, final_answer: str) -> None:
    """Configure generator mock for English queries (detection + translation + answer)."""
    generator.invoke.side_effect = ["English", translated_query, final_answer]


class TestQueryRouterRAG:
    """Tests for queries routed as RAG (factual/summary/comparison/procedural)."""

    @pytest.mark.parametrize("intent,expected_intent", [
        (IntentType.FACTUAL, IntentType.RAG),  # FACTUAL overridden to RAG when sources exist
        (IntentType.SUMMARY, IntentType.SUMMARY),
        (IntentType.COMPARISON, IntentType.COMPARISON),
        (IntentType.PROCEDURAL, IntentType.PROCEDURAL),
    ])
    def test_rag_intent_returns_answer_with_sources(
        self, mock_components, intent: IntentType, expected_intent: IntentType
    ) -> None:
        """RAG intents should retrieve, rerank, and generate an answer."""
        classifier, retriever, reranker, generator = mock_components
        classifier.classify.return_value = intent

        results = [_make_query_result("policy text", 0.85)]
        retriever.search_detailed.return_value = _make_hybrid_result(results)
        reranker.rerank.return_value = results
        _setup_generator_danish(generator, "Generated answer")

        router = QueryRouter(classifier, retriever, reranker, generator)
        response = router.route("Hvad er KU's feriepolitik?", top_k=3)

        assert isinstance(response, GenerationResponse)
        assert response.answer == "Generated answer"
        assert response.intent == expected_intent
        assert response.confidence == pytest.approx(0.85, abs=1e-6)
        assert len(response.sources) == 1

        retriever.search_detailed.assert_called_once_with(
            "Hvad er KU's feriepolitik?", top_k=3
        )
        reranker.rerank.assert_called_once_with(
            "Hvad er KU's feriepolitik?", results, top_k=3
        )

    def test_prompt_contains_context_and_query(self, mock_components) -> None:
        """The prompt sent to the generator should include context and query."""
        classifier, retriever, reranker, generator = mock_components
        classifier.classify.return_value = IntentType.FACTUAL

        results = [_make_query_result("Relevant context text", 0.9)]
        retriever.search_detailed.return_value = _make_hybrid_result(results)
        reranker.rerank.return_value = results
        _setup_generator_danish(generator, "answer")

        router = QueryRouter(classifier, retriever, reranker, generator)
        router.route("test query", top_k=3)

        # The final invoke call is the generation call
        prompt = generator.invoke.call_args_list[-1][0][0]
        assert "Relevant context text" in prompt
        assert "test query" in prompt

    def test_prompt_contains_language_rule(self, mock_components) -> None:
        """The prompt should contain a language instruction matching user language."""
        classifier, retriever, reranker, generator = mock_components
        classifier.classify.return_value = IntentType.RAG

        results = [_make_query_result("ctx", 0.5)]
        retriever.search_detailed.return_value = _make_hybrid_result(results)
        reranker.rerank.return_value = results
        _setup_generator_english(generator, "oversæt forespørgsel", "answer")

        router = QueryRouter(classifier, retriever, reranker, generator)
        router.route("What is KU's vacation policy?", top_k=3)

        prompt = generator.invoke.call_args_list[-1][0][0]
        assert "MUST answer in English" in prompt


class TestQueryRouterDirect:
    """Tests for queries that get a direct answer (UNKNOWN intent, no retrieval hits)."""

    def test_unknown_intent_still_generates_answer(self, mock_components) -> None:
        """UNKNOWN intent skips retrieval and returns zero confidence."""
        classifier, retriever, reranker, generator = mock_components
        classifier.classify.return_value = IntentType.UNKNOWN

        reranker.rerank.return_value = []
        _setup_generator_danish(generator, "Fallback answer")

        router = QueryRouter(classifier, retriever, reranker, generator)
        response = router.route("Hej, hvad kan du hjælpe med?", top_k=3)

        assert response.answer == "Fallback answer"
        assert response.intent == IntentType.UNKNOWN
        assert response.confidence == 0.0
        retriever.search_detailed.assert_not_called()

    def test_unknown_intent_prompt_uses_generic_instruction(
        self, mock_components
    ) -> None:
        """UNKNOWN intent should use the generic helpful instruction."""
        classifier, retriever, reranker, generator = mock_components
        classifier.classify.return_value = IntentType.UNKNOWN

        reranker.rerank.return_value = []
        _setup_generator_danish(generator, "answer")

        router = QueryRouter(classifier, retriever, reranker, generator)
        router.route("random input", top_k=3)

        prompt = generator.invoke.call_args_list[-1][0][0]
        assert "as helpfully as possible" in prompt


class TestQueryRouterFallback:
    """Tests for ambiguous input and fallback/degradation behaviour."""

    def test_empty_reranked_results_gives_zero_confidence(
        self, mock_components
    ) -> None:
        """When reranker returns no results, confidence should be 0.0."""
        classifier, retriever, reranker, generator = mock_components
        classifier.classify.return_value = IntentType.FACTUAL

        retriever.search_detailed.return_value = _make_hybrid_result([])
        reranker.rerank.return_value = []
        _setup_generator_danish(generator, "No information found")

        router = QueryRouter(classifier, retriever, reranker, generator)
        response = router.route("asdfghjkl", top_k=3)

        assert response.confidence == 0.0
        assert response.sources == []
        assert response.answer == "No information found"

    def test_empty_context_passed_to_generator(self, mock_components) -> None:
        """When no chunks are retrieved, the prompt context should be empty."""
        classifier, retriever, reranker, generator = mock_components
        classifier.classify.return_value = IntentType.FACTUAL

        retriever.search_detailed.return_value = _make_hybrid_result([])
        reranker.rerank.return_value = []
        _setup_generator_danish(generator, "answer")

        router = QueryRouter(classifier, retriever, reranker, generator)
        router.route("gibberish", top_k=3)

        prompt = generator.invoke.call_args_list[-1][0][0]
        assert "Context:\n\n" in prompt

    def test_multiple_results_confidence_uses_max_score(
        self, mock_components
    ) -> None:
        """Confidence should be the maximum score among reranked results."""
        classifier, retriever, reranker, generator = mock_components
        classifier.classify.return_value = IntentType.SUMMARY

        results = [
            _make_query_result("low", 0.3),
            _make_query_result("high", 0.95),
            _make_query_result("mid", 0.6),
        ]
        retriever.search_detailed.return_value = _make_hybrid_result(results)
        reranker.rerank.return_value = results
        _setup_generator_danish(generator, "summary")

        router = QueryRouter(classifier, retriever, reranker, generator)
        response = router.route("opsummer politikken", top_k=5)

        assert response.confidence == pytest.approx(0.95, abs=1e-6)


class TestQueryTranslation:
    """Tests for query language detection and translation."""

    def test_danish_query_not_translated(self, mock_components) -> None:
        """Danish queries should be passed directly to retrieval without translation."""
        classifier, retriever, reranker, generator = mock_components
        classifier.classify.return_value = IntentType.RAG

        results = [_make_query_result("ctx", 0.5)]
        retriever.search_detailed.return_value = _make_hybrid_result(results)
        reranker.rerank.return_value = results
        _setup_generator_danish(generator, "svar")

        router = QueryRouter(classifier, retriever, reranker, generator)
        router.route("Hvad er reglerne?", top_k=3)

        # Only 2 invoke calls: language detection + generation (no translation)
        assert generator.invoke.call_count == 2
        retriever.search_detailed.assert_called_once_with("Hvad er reglerne?", top_k=3)

    def test_english_query_translated_for_retrieval(self, mock_components) -> None:
        """English queries should be translated to Danish for retrieval."""
        classifier, retriever, reranker, generator = mock_components
        classifier.classify.return_value = IntentType.RAG

        results = [_make_query_result("ctx", 0.5)]
        retriever.search_detailed.return_value = _make_hybrid_result(results)
        reranker.rerank.return_value = results
        _setup_generator_english(generator, "Hvad er reglerne?", "The rules are...")

        router = QueryRouter(classifier, retriever, reranker, generator, translate_query=True)
        response = router.route("What are the rules?", top_k=3)

        # 3 invoke calls: detection + translation + generation
        assert generator.invoke.call_count == 3
        retriever.search_detailed.assert_called_once_with("Hvad er reglerne?", top_k=3)
        reranker.rerank.assert_called_once_with("Hvad er reglerne?", results, top_k=3)
        assert response.answer == "The rules are..."

    def test_translation_disabled_skips_translate(self, mock_components) -> None:
        """When translate_query=False, English queries go straight to retrieval untranslated."""
        classifier, retriever, reranker, generator = mock_components
        classifier.classify.return_value = IntentType.RAG

        results = [_make_query_result("ctx", 0.5)]
        retriever.search_detailed.return_value = _make_hybrid_result(results)
        reranker.rerank.return_value = results
        # Only 2 calls: detection + generation (no translation)
        generator.invoke.side_effect = ["English", "The answer"]

        router = QueryRouter(classifier, retriever, reranker, generator, translate_query=False)
        response = router.route("What are the rules?", top_k=3)

        assert generator.invoke.call_count == 2
        retriever.search_detailed.assert_called_once_with("What are the rules?", top_k=3)
        assert response.answer == "The answer"


class TestSigmoidInReranker:
    """Tests that sigmoid normalization is in the reranker, not the router."""

    def test_confidence_equals_max_reranked_score(self, mock_components) -> None:
        """Confidence should equal the max reranked score (already sigmoid-normalized)."""
        classifier, retriever, reranker, generator = mock_components
        classifier.classify.return_value = IntentType.RAG

        results = [
            _make_query_result("a", 0.7),
            _make_query_result("b", 0.9),
        ]
        retriever.search_detailed.return_value = _make_hybrid_result(results)
        reranker.rerank.return_value = results
        generator.invoke.side_effect = ["Danish", "answer"]

        router = QueryRouter(classifier, retriever, reranker, generator)
        response = router.route("test", top_k=3)

        assert response.confidence == pytest.approx(0.9, abs=1e-6)
