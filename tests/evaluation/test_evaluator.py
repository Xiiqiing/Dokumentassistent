"""Tests for src.evaluation.evaluator."""

from unittest.mock import MagicMock, patch

from src.evaluation.evaluator import RAGEvaluator

EVAL_MODULE = "src.evaluation.evaluator"


def _make_evaluator() -> RAGEvaluator:
    """Create a RAGEvaluator with a mocked LLM."""
    mock_llm = MagicMock()
    with patch(f"{EVAL_MODULE}.LangchainLLMWrapper"):
        return RAGEvaluator(llm=mock_llm)


class TestRAGEvaluator:
    """Tests for the RAGEvaluator class."""

    @patch(f"{EVAL_MODULE}.LangchainLLMWrapper")
    def test_init_stores_llm(self, mock_wrapper: MagicMock) -> None:
        """Test that __init__ creates the LLM wrapper."""
        mock_llm = MagicMock()
        evaluator = RAGEvaluator(llm=mock_llm)
        mock_wrapper.assert_called_once_with(mock_llm)
        assert evaluator._llm is not None

    @patch(f"{EVAL_MODULE}.evaluate")
    @patch(f"{EVAL_MODULE}.ContextRecall")
    @patch(f"{EVAL_MODULE}.ContextPrecision")
    @patch(f"{EVAL_MODULE}.AnswerRelevancy")
    @patch(f"{EVAL_MODULE}.Faithfulness")
    def test_evaluate_returns_metrics(
        self,
        mock_faith: MagicMock,
        mock_relevancy: MagicMock,
        mock_precision: MagicMock,
        mock_recall: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        """Test that evaluate returns a dict of metric scores."""
        mock_result = MagicMock()
        mock_result._repr_dict = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.90,
            "context_precision": 0.75,
            "context_recall": 0.80,
        }
        mock_evaluate.return_value = mock_result

        evaluator = _make_evaluator()
        scores = evaluator.evaluate(
            questions=["What is KU?"],
            answers=["KU is a university."],
            contexts=[["KU is the University of Copenhagen."]],
            ground_truths=["KU is the University of Copenhagen."],
        )

        assert isinstance(scores, dict)
        assert scores["faithfulness"] == 0.85
        assert scores["answer_relevancy"] == 0.90
        assert scores["context_precision"] == 0.75
        assert scores["context_recall"] == 0.80
        mock_evaluate.assert_called_once()

    @patch(f"{EVAL_MODULE}.evaluate")
    @patch(f"{EVAL_MODULE}.ContextRecall")
    @patch(f"{EVAL_MODULE}.ContextPrecision")
    def test_evaluate_retrieval_returns_metrics(
        self,
        mock_precision: MagicMock,
        mock_recall: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        """Test that evaluate_retrieval returns retrieval-specific metrics."""
        mock_result = MagicMock()
        mock_result._repr_dict = {
            "context_precision": 0.75,
            "context_recall": 0.80,
        }
        mock_evaluate.return_value = mock_result

        evaluator = _make_evaluator()
        scores = evaluator.evaluate_retrieval(
            questions=["What is KU?"],
            contexts=[["KU is the University of Copenhagen."]],
            ground_truths=["KU is the University of Copenhagen."],
        )

        assert isinstance(scores, dict)
        assert "context_precision" in scores
        assert "context_recall" in scores
        assert "faithfulness" not in scores
        mock_evaluate.assert_called_once()
