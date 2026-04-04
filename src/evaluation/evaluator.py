"""RAGAS-based evaluation for retrieval and generation quality."""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG pipeline quality using RAGAS metrics."""

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the evaluator.

        Args:
            llm: A LangChain BaseChatModel instance from provider.py.
        """
        self._llm = LangchainLLMWrapper(llm)
        logger.info("RAGEvaluator initialized")

    def _build_dataset(
        self,
        questions: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
        answers: list[str] | None = None,
    ) -> EvaluationDataset:
        """Build a RAGAS EvaluationDataset from raw inputs.

        Args:
            questions: List of input questions.
            contexts: List of retrieved context lists per question.
            ground_truths: List of expected correct answers.
            answers: Optional list of generated answers.

        Returns:
            EvaluationDataset ready for evaluation.
        """
        samples = []
        for i, question in enumerate(questions):
            sample_kwargs: dict = {
                "user_input": question,
                "retrieved_contexts": contexts[i],
                "reference": ground_truths[i],
            }
            if answers is not None:
                sample_kwargs["response"] = answers[i]
            samples.append(SingleTurnSample(**sample_kwargs))
        return EvaluationDataset(samples=samples)

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
    ) -> dict[str, float]:
        """Run RAGAS evaluation on a set of QA pairs.

        Args:
            questions: List of input questions.
            answers: List of generated answers.
            contexts: List of retrieved context lists per question.
            ground_truths: List of expected correct answers.

        Returns:
            Dictionary mapping metric names to scores.
        """
        logger.info("Running full evaluation on %d samples", len(questions))
        dataset = self._build_dataset(questions, contexts, ground_truths, answers)
        metrics = [
            Faithfulness(llm=self._llm),
            AnswerRelevancy(llm=self._llm),
            ContextPrecision(llm=self._llm),
            ContextRecall(llm=self._llm),
        ]
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
        )
        scores: dict[str, float] = dict(result._repr_dict)
        logger.info("Evaluation scores: %s", scores)
        return scores

    def evaluate_retrieval(
        self,
        questions: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
    ) -> dict[str, float]:
        """Evaluate retrieval quality only (context precision/recall).

        Args:
            questions: List of input questions.
            contexts: List of retrieved context lists per question.
            ground_truths: List of expected correct answers.

        Returns:
            Dictionary mapping retrieval metric names to scores.
        """
        logger.info("Running retrieval evaluation on %d samples", len(questions))
        dataset = self._build_dataset(questions, contexts, ground_truths)
        metrics = [
            ContextPrecision(llm=self._llm),
            ContextRecall(llm=self._llm),
        ]
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
        )
        scores: dict[str, float] = dict(result._repr_dict)
        logger.info("Retrieval evaluation scores: %s", scores)
        return scores
