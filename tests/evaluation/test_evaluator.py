"""Tests for src.evaluation.evaluator."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.evaluation.evaluator import (
    _CORRECTNESS_REF_CHAIN,
    _GROUNDING_REF_CHAIN,
    RAGEvaluator,
)

EVAL_MODULE = "src.evaluation.evaluator"


def _make_evaluator() -> RAGEvaluator:
    """Create a RAGEvaluator with a mocked LLM and embeddings."""
    mock_llm = MagicMock()
    mock_embeddings = MagicMock()
    with patch(f"{EVAL_MODULE}.LangchainLLMWrapper"), patch(
        f"{EVAL_MODULE}.LangchainEmbeddingsWrapper"
    ):
        return RAGEvaluator(llm=mock_llm, embeddings=mock_embeddings)


def _make_grounding_df() -> pd.DataFrame:
    """Fake RAGAS dataframe for the grounding pass (Danish reference)."""
    return pd.DataFrame(
        [
            {
                "user_input": "What is KU?",
                "retrieved_contexts": ["KU er Københavns Universitet."],
                "reference": "KU er Københavns Universitet.",
                "response": "KU is a university.",
                "faithfulness": 0.85,
                "answer_relevancy": 0.90,
                "llm_context_precision_with_reference": 0.75,
                "context_recall": 0.80,
            }
        ]
    )


def _make_correctness_df() -> pd.DataFrame:
    """Fake RAGAS dataframe for the correctness pass (English reference)."""
    return pd.DataFrame(
        [
            {
                "user_input": "What is KU?",
                "retrieved_contexts": ["KU er Københavns Universitet."],
                "reference": "KU is the University of Copenhagen.",
                "response": "KU is a university.",
                "answer_correctness": 0.72,
                "factual_correctness(mode=f1)": 0.65,
            }
        ]
    )


def _make_retrieval_result_df() -> pd.DataFrame:
    """Build a fake RAGAS result dataframe with only retrieval metrics."""
    return pd.DataFrame(
        [
            {
                "user_input": "What is KU?",
                "retrieved_contexts": ["KU er Københavns Universitet."],
                "reference": "KU er Københavns Universitet.",
                "llm_context_precision_with_reference": 0.75,
                "context_recall": 0.80,
            }
        ]
    )


class TestRAGEvaluator:
    """Tests for the RAGEvaluator class."""

    @patch(f"{EVAL_MODULE}.LangchainEmbeddingsWrapper")
    @patch(f"{EVAL_MODULE}.LangchainLLMWrapper")
    def test_init_stores_llm_and_embeddings(
        self,
        mock_llm_wrapper: MagicMock,
        mock_emb_wrapper: MagicMock,
    ) -> None:
        """Test that __init__ wraps both the LLM and the embeddings."""
        mock_llm = MagicMock()
        mock_embeddings = MagicMock()
        evaluator = RAGEvaluator(llm=mock_llm, embeddings=mock_embeddings)
        mock_llm_wrapper.assert_called_once_with(mock_llm)
        mock_emb_wrapper.assert_called_once_with(mock_embeddings)
        assert evaluator._llm is not None
        assert evaluator._embeddings is not None

    def test_resolve_reference_with_string(self) -> None:
        """Plain-string ground truths pass through unchanged."""
        assert (
            RAGEvaluator._resolve_reference("hello", _GROUNDING_REF_CHAIN) == "hello"
        )

    def test_resolve_reference_grounding_prefers_danish_quote(self) -> None:
        """Grounding chain prefers source_quote_da over reference_en."""
        gt = {
            "source_quote_da": "Danish text",
            "reference_en": "English text",
        }
        assert (
            RAGEvaluator._resolve_reference(gt, _GROUNDING_REF_CHAIN) == "Danish text"
        )

    def test_resolve_reference_correctness_prefers_english(self) -> None:
        """Correctness chain prefers reference_en over source_quote_da."""
        gt = {
            "source_quote_da": "Danish text",
            "reference_en": "English text",
        }
        assert (
            RAGEvaluator._resolve_reference(gt, _CORRECTNESS_REF_CHAIN)
            == "English text"
        )

    def test_resolve_reference_falls_back_to_other_key(self) -> None:
        """When the preferred key is missing, falls back to the next chain entry."""
        gt = {"reference_en": "English text"}
        # Grounding prefers Danish but only English is available — should fall back.
        assert (
            RAGEvaluator._resolve_reference(gt, _GROUNDING_REF_CHAIN) == "English text"
        )

    @patch(f"{EVAL_MODULE}.evaluate")
    @patch(f"{EVAL_MODULE}.FactualCorrectness")
    @patch(f"{EVAL_MODULE}.AnswerCorrectness")
    @patch(f"{EVAL_MODULE}.LLMContextRecall")
    @patch(f"{EVAL_MODULE}.LLMContextPrecisionWithReference")
    @patch(f"{EVAL_MODULE}.AnswerRelevancy")
    @patch(f"{EVAL_MODULE}.Faithfulness")
    def test_evaluate_runs_two_passes_and_merges_results(
        self,
        mock_faith: MagicMock,
        mock_relevancy: MagicMock,
        mock_precision: MagicMock,
        mock_recall: MagicMock,
        mock_answer_correctness: MagicMock,
        mock_factual_correctness: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        """evaluate() runs grounding + correctness passes and merges per-sample rows."""
        # Two distinct results returned by the two evaluate() calls.
        result_grounding = MagicMock()
        result_grounding.to_pandas.return_value = _make_grounding_df()
        result_correctness = MagicMock()
        result_correctness.to_pandas.return_value = _make_correctness_df()
        mock_evaluate.side_effect = [result_grounding, result_correctness]

        evaluator = _make_evaluator()
        result = evaluator.evaluate(
            questions=["What is KU?"],
            answers=["KU is a university."],
            contexts=[["KU er Københavns Universitet."]],
            ground_truths=[
                {
                    "source_quote_da": "KU er Københavns Universitet.",
                    "reference_en": "KU is the University of Copenhagen.",
                }
            ],
        )

        # evaluate() should be called exactly twice (one per metric family).
        assert mock_evaluate.call_count == 2

        agg = result["aggregate"]
        # Grounding metrics
        assert agg["faithfulness"] == 0.85
        assert agg["answer_relevancy"] == 0.90
        assert agg["llm_context_precision_with_reference"] == 0.75
        assert agg["context_recall"] == 0.80
        # Correctness metrics
        assert agg["answer_correctness"] == 0.72
        assert agg["factual_correctness(mode=f1)"] == 0.65

        # Per-sample rows are merged, including reference_en for transparency.
        per_sample = result["per_sample"]
        assert len(per_sample) == 1
        row = per_sample[0]
        assert row["user_input"] == "What is KU?"
        # The Danish reference (from pass 1) is the canonical reference.
        assert row["reference"] == "KU er Københavns Universitet."
        # The English reference (from pass 2) lives in reference_en.
        assert row["reference_en"] == "KU is the University of Copenhagen."
        # Both metric families are present in the merged row.
        assert row["faithfulness"] == 0.85
        assert row["answer_correctness"] == 0.72

    @patch(f"{EVAL_MODULE}.evaluate")
    @patch(f"{EVAL_MODULE}.LLMContextRecall")
    @patch(f"{EVAL_MODULE}.LLMContextPrecisionWithReference")
    def test_evaluate_retrieval_returns_only_retrieval_metrics(
        self,
        mock_precision: MagicMock,
        mock_recall: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        """evaluate_retrieval() returns only context precision/recall."""
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = _make_retrieval_result_df()
        mock_evaluate.return_value = mock_result

        evaluator = _make_evaluator()
        result = evaluator.evaluate_retrieval(
            questions=["What is KU?"],
            contexts=[["KU er Københavns Universitet."]],
            ground_truths=["KU er Københavns Universitet."],
        )

        agg = result["aggregate"]
        assert "llm_context_precision_with_reference" in agg
        assert "context_recall" in agg
        assert "faithfulness" not in agg
        assert "answer_relevancy" not in agg
        assert "answer_correctness" not in agg
        assert len(result["per_sample"]) == 1
        mock_evaluate.assert_called_once()
