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
        self._translate_query_enabled = translate_query

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

    def _detect_language_and_intent(self, query: str) -> tuple[str, IntentType]:
        """Detect the query language and classify intent in a single LLM call.

        Uses Unicode script detection first for non-Latin scripts.  For
        Latin-script text, a single LLM call returns both language and intent,
        saving one full round-trip compared to two separate calls.

        Args:
            query: The user's original query.

        Returns:
            Tuple of (detected_language, intent).
        """
        # Fast path: detect non-Latin scripts via Unicode
        script_language = self._detect_script(query)

        if script_language is not None:
            # Language is known; still need intent from LLM
            intent = self._intent_classifier.classify(query)
            logger.info("Detected query language: %s", script_language)
            logger.info("Classified intent: %s", intent.value)
            return script_language, intent

        # Latin-script text — combine language detection + intent classification
        valid_intents = "factual, summary, comparison, procedural, unknown"
        prompt = (
            "You are given a user query. Do TWO things:\n"
            "1. Detect the language of the query (reply with the language name in English, "
            "e.g. 'Danish', 'English', 'German').\n"
            "2. Classify the intent into exactly one of: "
            f"{valid_intents}.\n\n"
            "Reply with EXACTLY two lines, nothing else:\n"
            "language: <language>\n"
            "intent: <intent>\n\n"
            f"Query: {query}"
        )
        raw = str(self._generator.invoke(prompt)).strip()
        logger.debug("Combined detection raw response: %s", raw)

        # Parse response
        detected = "Danish"
        intent = IntentType.UNKNOWN
        for line in raw.splitlines():
            line = line.strip().lower()
            if line.startswith("language:"):
                detected = line.split(":", 1)[1].strip().strip(".")
            elif line.startswith("intent:"):
                raw_intent = line.split(":", 1)[1].strip().strip(".")
                if raw_intent in {i.value for i in IntentType}:
                    intent = IntentType(raw_intent)
                else:
                    logger.warning("Unrecognized intent '%s' from combined call, falling back to UNKNOWN", raw_intent)

        # Capitalize language name for display
        detected = detected.capitalize()
        logger.info("Detected query language: %s", detected)
        logger.info("Classified intent: %s", intent.value)
        return detected, intent

    def _translate_query(self, query: str, detected_language: str) -> str:
        """Translate the query to Danish if needed.

        Args:
            query: The user's original query.
            detected_language: Detected language of the query.

        Returns:
            The Danish retrieval query, or the original if already Danish.
        """
        if detected_language.lower() in ("danish", "dansk"):
            return query

        if not self._translate_query_enabled:
            logger.info("Query translation disabled; using original query for retrieval")
            return query

        translate_prompt = (
            "Translate the following text to Danish. "
            "Reply with ONLY the translated text, nothing else.\n\n"
            f"Text: {query}"
        )
        translated = str(self._generator.invoke(translate_prompt)).strip()
        logger.info("Translated query to Danish: %s", translated)
        return translated

    def route(self, query: str, top_k: int) -> GenerationResponse:
        """Route a query through the full RAG pipeline.

        Args:
            query: The user's natural language query.
            top_k: Number of top documents to retrieve.

        Returns:
            GenerationResponse with answer, sources, and metadata.
        """
        logger.info("Routing query: %s", query)

        # Single LLM call for both language detection and intent classification
        user_language, intent = self._detect_language_and_intent(query)
        retrieval_query = self._translate_query(query, user_language)
        translated = retrieval_query != query

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
