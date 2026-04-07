"""Intent classification for incoming user queries."""

import logging
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.models import IntentType

logger = logging.getLogger(__name__)

_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)

_VALID_INTENTS = {intent.value for intent in IntentType}

_SYSTEM_PROMPT = (
    "You are an intent classifier. Given a user query, classify it into exactly "
    "one of the following categories: factual, summary, comparison, procedural, unknown.\n\n"
    "- factual: the user asks for a specific fact or piece of information.\n"
    "- summary: the user wants a summary or overview of a topic.\n"
    "- comparison: the user wants to compare two or more things.\n"
    "- procedural: the user asks how to do something step by step.\n"
    "- unknown: the query does not fit any of the above.\n\n"
    "Respond with ONLY the category name in lowercase, nothing else."
)


class IntentClassifier:
    """Classifies user queries into predefined intent categories."""

    def __init__(self, llm: BaseChatModel, *, model_name: str = "") -> None:
        """Initialize the intent classifier.

        Args:
            llm: A LangChain BaseChatModel instance from provider.py.
            model_name: Model identifier used to detect models that lack
                system-message support (e.g. Gemma via Ollama).
        """
        if "gemma3" in model_name.lower():
            prompt = ChatPromptTemplate.from_messages([
                ("human", _SYSTEM_PROMPT + "\n\nQuery: {query}"),
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", _SYSTEM_PROMPT),
                ("human", "{query}"),
            ])
        self._chain = prompt | llm | StrOutputParser()

    def classify(self, query: str) -> IntentType:
        """Classify a user query into an intent type.

        Args:
            query: The user's natural language query.

        Returns:
            The classified IntentType.
        """
        _raw_out = self._chain.invoke({"query": query})
        raw = _THINK_UNCLOSED_RE.sub("", _THINK_CLOSED_RE.sub("", _raw_out)).strip().lower()
        logger.debug("Raw classification result: %s", raw)

        if raw in _VALID_INTENTS:
            return IntentType(raw)

        logger.warning("Unrecognized intent '%s', falling back to UNKNOWN", raw)
        return IntentType.UNKNOWN
