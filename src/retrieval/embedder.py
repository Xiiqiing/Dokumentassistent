"""Embedding generation using a provider-agnostic LangChain Embeddings interface."""

import logging

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings using a LangChain Embeddings instance."""

    def __init__(self, embeddings: Embeddings) -> None:
        """Initialize the embedder.

        Args:
            embeddings: A LangChain Embeddings instance from provider.py.
        """
        self._embeddings = embeddings

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        logger.debug("Embedding single text of length %d", len(text))
        return self._embeddings.embed_query(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        logger.debug("Embedding batch of %d texts", len(texts))
        return self._embeddings.embed_documents(texts)
