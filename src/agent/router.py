"""Query router that selects retrieval strategy based on intent."""

import logging
import math

from langchain_core.runnables import Runnable

from src.models import IntentType, GenerationResponse
from src.agent.intent_classifier import IntentClassifier
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


class QueryRouter:
    """Routes queries to appropriate retrieval and generation pipelines."""

    def __init__(
        self,
        intent_classifier: IntentClassifier,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        generator: Runnable,
        *,
        translate_query: bool = True,
    ) -> None:
        """Initialize the query router.

        Args:
            intent_classifier: IntentClassifier instance.
            hybrid_retriever: HybridRetriever instance.
            reranker: Reranker instance.
            generator: LLM generation chain.
            translate_query: Whether to translate non-Danish queries to Danish
                before retrieval. When False, language detection still runs for
                the answer-language rule but no translation is performed.
        """
        self._intent_classifier = intent_classifier
        self._hybrid_retriever = hybrid_retriever
        self._reranker = reranker
        self._generator = generator
        self._translate_query = translate_query

    def _detect_and_translate_query(self, query: str) -> tuple[str, str]:
        """Detect the query language and optionally translate to Danish.

        Translation is only performed when ``self._translate_query`` is True and
        the detected language is not Danish.

        Args:
            query: The user's original query.

        Returns:
            Tuple of (retrieval_query, detected_language).
            retrieval_query is Danish when translation is enabled; otherwise the
            original query. detected_language is e.g. "English", "Danish".
        """
        prompt = (
            "Detect the language of the following text. "
            "Reply with ONLY the language name in English (e.g. 'Danish', 'English', 'German'). "
            "Nothing else.\n\n"
            f"Text: {query}"
        )
        detected = str(self._generator.invoke(prompt)).strip().strip(".")
        logger.info("Detected query language: %s", detected)

        if detected.lower() in ("danish", "dansk"):
            return query, "Danish"

        if not self._translate_query:
            logger.info("Query translation disabled; using original query for retrieval")
            return query, detected

        translate_prompt = (
            "Translate the following text to Danish. "
            "Reply with ONLY the translated text, nothing else.\n\n"
            f"Text: {query}"
        )
        translated = str(self._generator.invoke(translate_prompt)).strip()
        logger.info("Translated query to Danish: %s", translated)
        return translated, detected

    @staticmethod
    def _sigmoid_normalize(score: float) -> float:
        """Normalize a raw cross-encoder score to 0-1 via sigmoid.

        Clamps the input to [-500, 500] to avoid math overflow.
        """
        score = max(-500.0, min(500.0, score))
        return 1.0 / (1.0 + math.exp(-score))

    def route(self, query: str, top_k: int) -> GenerationResponse:
        """Route a query through the full RAG pipeline.

        Args:
            query: The user's natural language query.
            top_k: Number of top documents to retrieve.

        Returns:
            GenerationResponse with answer, sources, and metadata.
        """
        logger.info("Routing query: %s", query)

        # Detect language and translate to Danish for retrieval if needed
        retrieval_query, user_language = self._detect_and_translate_query(query)

        intent = self._intent_classifier.classify(query)
        logger.info("Classified intent: %s", intent.value)
        logger.debug("[DEBUG] Intent classification result: %s for query='%s'", intent.value, query)

        should_retrieve = intent != IntentType.UNKNOWN
        logger.debug("[DEBUG] Retrieval executed: %s (intent=%s)", should_retrieve, intent.value)

        results = self._hybrid_retriever.search(retrieval_query, top_k=top_k) if should_retrieve else []
        logger.info("Retrieved %d results from hybrid search", len(results))
        logger.debug("[DEBUG] Retrieval returned %d results", len(results))

        reranked = self._reranker.rerank(retrieval_query, results, top_k=top_k) if results else []
        logger.info("Reranked to %d results", len(reranked))

        if reranked and intent == IntentType.FACTUAL:
            intent = IntentType.RAG
            logger.info("Overriding intent to RAG (sources retrieved)")

        context = "\n\n".join(r.chunk.text for r in reranked)
        prompt = self._build_prompt(query, intent, context, user_language)

        answer = self._generator.invoke(prompt)
        logger.info("Generated answer for intent=%s", intent.value)

        if reranked:
            raw_max = max(r.score for r in reranked)
            confidence = self._sigmoid_normalize(raw_max)
            logger.info(
                "Confidence: raw_max=%.4f, sigmoid=%.4f", raw_max, confidence
            )
        else:
            confidence = 0.0

        return GenerationResponse(
            answer=str(answer),
            sources=reranked,
            intent=intent,
            confidence=confidence,
        )

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
                "Answer the question as helpfully as possible based on the provided context."
            ),
        }

        instruction = intent_instructions[intent]

        language_rule = (
            f"IMPORTANT: You MUST answer in {user_language}. "
            f"The user asked in {user_language}, so your entire response must be in {user_language}. "
            f"The context documents may be in Danish — use them as reference but always reply in {user_language}."
        )

        return (
            f"You are a helpful assistant for administrative staff at the University of Copenhagen (KU).\n\n"
            f"{language_rule}\n\n"
            f"Instruction: {instruction}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
