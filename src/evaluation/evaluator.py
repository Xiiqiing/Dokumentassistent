"""RAGAS-based evaluation for retrieval and generation quality.

Uses the legacy ``ragas.metrics`` classes (``LLMContextPrecisionWithReference``,
``LLMContextRecall``, ``Faithfulness``, ``AnswerRelevancy``,
``AnswerCorrectness``, ``FactualCorrectness``) rather than the newer
``ragas.metrics.collections`` API. The collections API only accepts
``InstructorLLM`` instances and would force us to import provider-specific
clients (openai / anthropic / ...) directly, which violates the project's
``provider.py``-only rule.

The legacy classes are wired with a ``LangchainLLMWrapper`` and a
``LangchainEmbeddingsWrapper`` at ``evaluate()`` time, so we keep using the
LangChain abstractions returned by ``src/provider.py`` everywhere.

Two metric families, two evaluation passes
------------------------------------------

For a multilingual test set (English questions querying Danish documents,
English-language answers, English+Danish reference fields), different metrics
want different reference languages:

- **Grounding metrics** (``ContextPrecision``, ``ContextRecall``) compare the
  reference against retrieved Danish chunks. They work best with a Danish
  reference (mono-lingual matching), so we feed them ``source_quote_da``.
  ``Faithfulness`` and ``AnswerRelevancy`` ignore the reference field but ride
  along in this pass to share the dataset.

- **Correctness metrics** (``AnswerCorrectness``, ``FactualCorrectness``)
  compare the generated English answer against the reference directly. They
  work best with an English reference, so we feed them ``reference_en``.

``evaluate()`` builds two ``EvaluationDataset`` instances (same questions,
contexts and answers, different ``reference`` field per pass) and runs RAGAS
twice. The aggregate scores are unioned and per-sample rows are merged on
``user_input``.

Why both families
-----------------

``Faithfulness`` measures whether each claim in the answer is supported by the
retrieved chunks. It does **not** check whether those chunks (and therefore
the answer) are actually correct. An answer that confidently quotes the wrong
chunks scores 1.0; an answer that hedges with prior-knowledge inference but
matches the ground truth scores low. Adding ``AnswerCorrectness`` /
``FactualCorrectness`` (which compare directly to ``reference_en``) closes
this gap by measuring **whether the answer is right**, not just whether it
is grounded in whatever was retrieved.

Note: ``AnswerRelevancy`` is constructed with ``strictness=1``. The default of
3 issues an OpenAI-style ``n=3`` request to generate three hypothetical
questions in one API call, which Groq's API rejects with HTTP 400
``'n' : number must be at most 1``. With ``strictness=1`` the metric makes a
single call per sample, which all providers support.
"""

import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics._faithfulness import Faithfulness

logger = logging.getLogger(__name__)

# Columns produced by RAGAS dataframes that are NOT metric scores. Used to
# separate metric columns from sample fields when computing aggregates.
_NON_METRIC_COLS: frozenset[str] = frozenset(
    {
        "user_input",
        "retrieved_contexts",
        "reference",
        "response",
        "ground_truth",
        "question",
        "answer",
        "contexts",
    }
)

# Reference-key fallback chains used by ``_resolve_reference``. Each chain
# starts with the preferred key for a particular metric family, then falls
# back to whichever other key is available so legacy plain-string ground
# truths still work.
_GROUNDING_REF_CHAIN: tuple[str, ...] = ("source_quote_da", "reference_en", "reference")
_CORRECTNESS_REF_CHAIN: tuple[str, ...] = ("reference_en", "source_quote_da", "reference")


class RAGEvaluator:
    """Evaluates RAG pipeline quality using two complementary RAGAS metric families.

    The judge LLM is independent from the generation LLM. This is critical
    when generation runs on a small local model: a stronger judge gives
    substantially less noisy scores.

    Each ground-truth entry may be either a plain string or a dict. The dict
    form is used by the multilingual test set::

        {
            "reference_en":    "English reference answer (informational)",
            "source_quote_da": "Verbatim Danish quote from the source document"
        }

    See the module docstring for why two metric families and two evaluation
    passes are used.
    """

    def __init__(self, llm: BaseChatModel, embeddings: Embeddings) -> None:
        """Initialize the evaluator.

        Args:
            llm: A LangChain BaseChatModel instance to use as the RAGAS judge.
                Should be a strong model (>= ~30B params) for reliable scoring.
            embeddings: A LangChain Embeddings instance. Required because
                ``AnswerRelevancy`` and ``AnswerCorrectness`` compute cosine
                similarity between text pairs.
        """
        self._llm = LangchainLLMWrapper(llm)
        self._embeddings = LangchainEmbeddingsWrapper(embeddings)
        logger.info("RAGEvaluator initialized")

    @staticmethod
    def _resolve_reference(
        ground_truth: str | dict[str, Any],
        ref_chain: tuple[str, ...],
    ) -> str:
        """Pick the best reference string for a given metric family.

        Args:
            ground_truth: Either a plain reference string or a dict with
                ``source_quote_da`` / ``reference_en`` / ``reference`` keys.
            ref_chain: Ordered tuple of dict keys to try, most preferred first.

        Returns:
            The reference string to feed into RAGAS.
        """
        if isinstance(ground_truth, str):
            return ground_truth
        if isinstance(ground_truth, dict):
            for key in ref_chain:
                value = ground_truth.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return str(ground_truth)

    def _build_dataset(
        self,
        questions: list[str],
        contexts: list[list[str]],
        ground_truths: list[str | dict[str, Any]],
        answers: list[str] | None = None,
        *,
        ref_chain: tuple[str, ...] = _GROUNDING_REF_CHAIN,
    ) -> EvaluationDataset:
        """Build a RAGAS EvaluationDataset from raw inputs.

        Args:
            questions: List of input questions.
            contexts: Retrieved context lists per question.
            ground_truths: Reference answers (str or dict).
            answers: Optional list of generated answers.
            ref_chain: Reference-key fallback chain to use when resolving
                each ground-truth entry.

        Returns:
            EvaluationDataset ready for evaluation.
        """
        samples: list[SingleTurnSample] = []
        for i, question in enumerate(questions):
            sample_kwargs: dict[str, Any] = {
                "user_input": question,
                "retrieved_contexts": contexts[i],
                "reference": self._resolve_reference(ground_truths[i], ref_chain),
            }
            if answers is not None:
                sample_kwargs["response"] = answers[i]
            samples.append(SingleTurnSample(**sample_kwargs))
        return EvaluationDataset(samples=samples)

    @staticmethod
    def _result_to_dicts(
        result: Any,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Convert a RAGAS EvaluationResult into aggregate scores + per-sample rows.

        Uses the public ``to_pandas()`` API instead of the private
        ``_repr_dict`` so the code does not break across RAGAS minor versions.

        Args:
            result: RAGAS EvaluationResult instance.

        Returns:
            Tuple of (aggregate metric → mean score, list of per-sample row dicts).
        """
        df = result.to_pandas()
        metric_cols = [c for c in df.columns if c not in _NON_METRIC_COLS]
        aggregate: dict[str, float] = {}
        for col in metric_cols:
            if df[col].dtype.kind in "fi":
                aggregate[col] = float(df[col].mean())
        per_sample: list[dict[str, Any]] = df.to_dict(orient="records")
        return aggregate, per_sample

    @staticmethod
    def _merge_per_sample(
        rows_a: list[dict[str, Any]],
        rows_b: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge two per-sample row lists by ``user_input``.

        Both lists are produced by ``_result_to_dicts`` and contain the same
        questions but different metric columns. The reference field will
        differ between the two passes (Danish vs English), so we keep the
        Danish one (from rows_a) as the canonical ``reference`` and add an
        ``reference_en`` column from rows_b for transparency.

        Args:
            rows_a: Rows from the grounding pass (reference = Danish quote).
            rows_b: Rows from the correctness pass (reference = English answer).

        Returns:
            Merged list of row dicts with all metric columns from both passes.
        """
        rows_b_by_q: dict[str, dict[str, Any]] = {row["user_input"]: row for row in rows_b}
        merged: list[dict[str, Any]] = []
        for row_a in rows_a:
            row_b = rows_b_by_q.get(row_a["user_input"], {})
            merged_row: dict[str, Any] = dict(row_a)
            # Preserve the English reference for transparency.
            if "reference" in row_b:
                merged_row["reference_en"] = row_b["reference"]
            # Add metric columns from pass B that are not already present.
            for key, value in row_b.items():
                if key in _NON_METRIC_COLS:
                    continue
                if key not in merged_row:
                    merged_row[key] = value
            merged.append(merged_row)
        return merged

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str | dict[str, Any]],
    ) -> dict[str, Any]:
        """Run full RAGAS evaluation across grounding + correctness metric families.

        Performs two passes against the same questions / answers / contexts
        but with different reference languages (see module docstring).

        Args:
            questions: Input questions.
            answers: Generated answers from the RAG pipeline.
            contexts: Retrieved context lists per question.
            ground_truths: Reference answers (str or dict per
                ``_resolve_reference``).

        Returns:
            Dict with two keys:
                ``aggregate``:  metric name → mean score across all samples
                ``per_sample``: list of per-sample dicts (one row per question)
        """
        n = len(questions)
        logger.info("Running full evaluation on %d samples (two passes)", n)

        # ---- Pass 1: grounding metrics with Danish reference ----
        dataset_da = self._build_dataset(
            questions,
            contexts,
            ground_truths,
            answers,
            ref_chain=_GROUNDING_REF_CHAIN,
        )
        grounding_metrics = [
            Faithfulness(),
            AnswerRelevancy(strictness=1),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ]
        logger.info("Pass 1/2: grounding metrics (reference = Danish quote)")
        result_a = evaluate(
            dataset=dataset_da,
            metrics=grounding_metrics,
            llm=self._llm,
            embeddings=self._embeddings,
            show_progress=False,
        )
        agg_a, samples_a = self._result_to_dicts(result_a)
        logger.info("Pass 1/2 aggregate: %s", agg_a)

        # ---- Pass 2: correctness metrics with English reference ----
        dataset_en = self._build_dataset(
            questions,
            contexts,
            ground_truths,
            answers,
            ref_chain=_CORRECTNESS_REF_CHAIN,
        )
        correctness_metrics = [
            AnswerCorrectness(),
            FactualCorrectness(),
        ]
        logger.info("Pass 2/2: correctness metrics (reference = English answer)")
        result_b = evaluate(
            dataset=dataset_en,
            metrics=correctness_metrics,
            llm=self._llm,
            embeddings=self._embeddings,
            show_progress=False,
        )
        agg_b, samples_b = self._result_to_dicts(result_b)
        logger.info("Pass 2/2 aggregate: %s", agg_b)

        # ---- Merge ----
        aggregate: dict[str, float] = {**agg_a, **agg_b}
        per_sample = self._merge_per_sample(samples_a, samples_b)
        logger.info("Combined aggregate: %s", aggregate)
        return {"aggregate": aggregate, "per_sample": per_sample}

    def evaluate_retrieval(
        self,
        questions: list[str],
        contexts: list[list[str]],
        ground_truths: list[str | dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate retrieval quality only (ContextPrecision + ContextRecall).

        Single-pass against the Danish reference, no correctness metrics.

        Args:
            questions: Input questions.
            contexts: Retrieved context lists per question.
            ground_truths: Reference answers (str or dict).

        Returns:
            Dict with ``aggregate`` and ``per_sample`` keys, same shape as
            ``evaluate()`` but only retrieval metrics.
        """
        logger.info("Running retrieval evaluation on %d samples", len(questions))
        dataset = self._build_dataset(
            questions,
            contexts,
            ground_truths,
            ref_chain=_GROUNDING_REF_CHAIN,
        )
        metrics = [
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ]
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._llm,
            embeddings=self._embeddings,
            show_progress=False,
        )
        aggregate, per_sample = self._result_to_dicts(result)
        logger.info("Retrieval evaluation aggregate: %s", aggregate)
        return {"aggregate": aggregate, "per_sample": per_sample}
