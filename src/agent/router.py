"""Query router that selects retrieval strategy based on intent."""

import logging
import unicodedata

from langchain_core.runnables import Runnable

from src.models import IntentType, GenerationResponse, PipelineDetails
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

    @staticmethod
    def _detect_script(text: str) -> str | None:
        """Detect language from Unicode script for non-Latin text.

        Returns a language name (e.g. "Chinese") if the script is
        unambiguously identifiable, or None to fall back to LLM detection.
        """
        script_counts: dict[str, int] = {}
        for ch in text:
            if ch.isspace() or ch in ".,!?;:\"'()[]{}":
                continue
            try:
                name = unicodedata.name(ch, "")
            except ValueError:
                continue
            if name.startswith("CJK") or name.startswith("KANGXI"):
                script_counts["CJK"] = script_counts.get("CJK", 0) + 1
            elif name.startswith("HIRAGANA") or name.startswith("KATAKANA"):
                script_counts["Japanese"] = script_counts.get("Japanese", 0) + 1
            elif name.startswith("HANGUL"):
                script_counts["Korean"] = script_counts.get("Korean", 0) + 1
            elif name.startswith("ARABIC"):
                script_counts["Arabic"] = script_counts.get("Arabic", 0) + 1
            elif name.startswith("DEVANAGARI"):
                script_counts["Hindi"] = script_counts.get("Hindi", 0) + 1
            elif name.startswith("THAI"):
                script_counts["Thai"] = script_counts.get("Thai", 0) + 1
            elif name.startswith("CYRILLIC"):
                script_counts["Russian"] = script_counts.get("Russian", 0) + 1

        if not script_counts:
            return None

        dominant = max(script_counts, key=lambda k: script_counts[k])
        # CJK characters alone -> Chinese; if mixed with Hiragana/Katakana -> Japanese
        if dominant == "CJK" and "Japanese" in script_counts:
            return "Japanese"
        if dominant == "CJK":
            return "Chinese"
        return dominant

    def _detect_and_translate_query(self, query: str) -> tuple[str, str]:
        """Detect the query language and optionally translate to Danish.

        Uses Unicode script detection first for non-Latin scripts (Chinese,
        Japanese, Korean, Arabic, etc.) which are reliably identifiable from
        character ranges. Falls back to LLM-based detection for Latin-script
        languages.

        Translation is only performed when ``self._translate_query`` is True and
        the detected language is not Danish.

        Args:
            query: The user's original query.

        Returns:
            Tuple of (retrieval_query, detected_language).
            retrieval_query is Danish when translation is enabled; otherwise the
            original query. detected_language is e.g. "English", "Danish".
        """
        # Fast path: detect non-Latin scripts via Unicode
        detected = self._detect_script(query)

        if detected is None:
            # Latin-script text — use LLM for detection
            prompt = (
                "Detect the language of the following text. "
                "Reply with ONLY the language name in English "
                "(e.g. 'Danish', 'English', 'German'). "
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
        translated = retrieval_query != query

        intent = self._intent_classifier.classify(query)
        logger.info("Classified intent: %s", intent.value)
        logger.debug("Intent classification result: %s for query='%s'", intent.value, query)

        should_retrieve = intent != IntentType.UNKNOWN
        logger.debug("Retrieval executed: %s (intent=%s)", should_retrieve, intent.value)

        # Use detailed search to capture intermediate results
        pipeline = PipelineDetails(
            original_query=query,
            retrieval_query=retrieval_query,
            detected_language=user_language,
            translated=translated,
        )

        if should_retrieve:
            hybrid_result = self._hybrid_retriever.search_detailed(retrieval_query, top_k=top_k)
            pipeline.dense_results = hybrid_result.dense_results
            pipeline.sparse_results = hybrid_result.sparse_results
            pipeline.fused_results = hybrid_result.fused_results
            results = hybrid_result.fused_results
        else:
            results = []

        logger.info("Retrieved %d results from hybrid search", len(results))
        logger.debug("Retrieval returned %d results", len(results))

        reranked = self._reranker.rerank(retrieval_query, results, top_k=top_k) if results else []
        pipeline.reranked_results = reranked
        logger.info("Reranked to %d results", len(reranked))

        if reranked and intent == IntentType.FACTUAL:
            intent = IntentType.RAG
            logger.info("Overriding intent to RAG (sources retrieved)")

        context = "\n\n".join(r.chunk.text for r in reranked)
        prompt = self._build_prompt(query, intent, context, user_language)

        answer = self._generator.invoke(prompt)
        logger.info("Generated answer for intent=%s", intent.value)

        if reranked:
            confidence = max(r.score for r in reranked)
            logger.info("Confidence: %.4f (sigmoid-normalized by reranker)", confidence)
        else:
            confidence = 0.0

        return GenerationResponse(
            answer=str(answer),
            sources=reranked,
            intent=intent,
            confidence=confidence,
            pipeline_details=pipeline,
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
