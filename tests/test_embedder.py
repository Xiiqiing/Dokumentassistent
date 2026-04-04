"""Tests for the Embedder class."""

from unittest.mock import MagicMock

import pytest

from src.retrieval.embedder import Embedder

EMBEDDING_DIM = 384


def _make_embedding(dim: int = EMBEDDING_DIM) -> list[float]:
    """Return a deterministic fake embedding vector."""
    return [0.01 * i for i in range(dim)]


class TestEmbedText:
    """Tests for Embedder.embed_text."""

    def test_returns_embedding_vector(self) -> None:
        expected = _make_embedding()
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = expected

        embedder = Embedder(embeddings=mock_embeddings)
        result = embedder.embed_text("hello world")

        assert result == expected
        mock_embeddings.embed_query.assert_called_once_with("hello world")

    def test_output_dimension(self) -> None:
        expected = _make_embedding(EMBEDDING_DIM)
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = expected

        embedder = Embedder(embeddings=mock_embeddings)
        result = embedder.embed_text("test")

        assert len(result) == EMBEDDING_DIM


class TestEmbedBatch:
    """Tests for Embedder.embed_batch."""

    def test_returns_list_of_vectors(self) -> None:
        emb1 = _make_embedding()
        emb2 = _make_embedding()
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [emb1, emb2]

        embedder = Embedder(embeddings=mock_embeddings)
        results = embedder.embed_batch(["text one", "text two"])

        assert len(results) == 2
        assert results[0] == emb1
        assert results[1] == emb2
        mock_embeddings.embed_documents.assert_called_once_with(
            ["text one", "text two"]
        )

    def test_empty_input_returns_empty_list(self) -> None:
        mock_embeddings = MagicMock()

        embedder = Embedder(embeddings=mock_embeddings)
        results = embedder.embed_batch([])

        assert results == []
        mock_embeddings.embed_documents.assert_not_called()

    def test_each_vector_has_correct_dimension(self) -> None:
        embeddings_data = [_make_embedding(EMBEDDING_DIM) for _ in range(3)]
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = embeddings_data

        embedder = Embedder(embeddings=mock_embeddings)
        results = embedder.embed_batch(["a", "b", "c"])

        assert all(len(v) == EMBEDDING_DIM for v in results)
