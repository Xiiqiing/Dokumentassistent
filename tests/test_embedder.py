"""Tests for the Embedder class."""

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.embedder import Embedder

FAKE_API_KEY = "sk-test-key"
MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def _make_embedding(dim: int = EMBEDDING_DIM) -> list[float]:
    """Return a deterministic fake embedding vector."""
    return [0.01 * i for i in range(dim)]


def _mock_response(embeddings: list[list[float]]) -> MagicMock:
    """Build a mock OpenAI embeddings.create response."""
    items = []
    for emb in embeddings:
        item = MagicMock()
        item.embedding = emb
        items.append(item)
    resp = MagicMock()
    resp.data = items
    return resp


@patch("src.retrieval.embedder.OpenAI")
class TestEmbedText:
    """Tests for Embedder.embed_text."""

    def test_returns_embedding_vector(self, mock_openai_cls: MagicMock) -> None:
        expected = _make_embedding()
        mock_client = mock_openai_cls.return_value
        mock_client.embeddings.create.return_value = _mock_response([expected])

        embedder = Embedder(model=MODEL_NAME, api_key=FAKE_API_KEY)
        result = embedder.embed_text("hello world")

        assert result == expected
        mock_client.embeddings.create.assert_called_once_with(
            model=MODEL_NAME, input="hello world"
        )

    def test_output_dimension(self, mock_openai_cls: MagicMock) -> None:
        expected = _make_embedding(EMBEDDING_DIM)
        mock_client = mock_openai_cls.return_value
        mock_client.embeddings.create.return_value = _mock_response([expected])

        embedder = Embedder(model=MODEL_NAME, api_key=FAKE_API_KEY)
        result = embedder.embed_text("test")

        assert len(result) == EMBEDDING_DIM


@patch("src.retrieval.embedder.OpenAI")
class TestEmbedBatch:
    """Tests for Embedder.embed_batch."""

    def test_returns_list_of_vectors(self, mock_openai_cls: MagicMock) -> None:
        emb1 = _make_embedding()
        emb2 = _make_embedding()
        mock_client = mock_openai_cls.return_value
        mock_client.embeddings.create.return_value = _mock_response([emb1, emb2])

        embedder = Embedder(model=MODEL_NAME, api_key=FAKE_API_KEY)
        results = embedder.embed_batch(["text one", "text two"])

        assert len(results) == 2
        assert results[0] == emb1
        assert results[1] == emb2
        mock_client.embeddings.create.assert_called_once_with(
            model=MODEL_NAME, input=["text one", "text two"]
        )

    def test_empty_input_returns_empty_list(self, mock_openai_cls: MagicMock) -> None:
        mock_client = mock_openai_cls.return_value

        embedder = Embedder(model=MODEL_NAME, api_key=FAKE_API_KEY)
        results = embedder.embed_batch([])

        assert results == []
        mock_client.embeddings.create.assert_not_called()

    def test_each_vector_has_correct_dimension(self, mock_openai_cls: MagicMock) -> None:
        embeddings = [_make_embedding(EMBEDDING_DIM) for _ in range(3)]
        mock_client = mock_openai_cls.return_value
        mock_client.embeddings.create.return_value = _mock_response(embeddings)

        embedder = Embedder(model=MODEL_NAME, api_key=FAKE_API_KEY)
        results = embedder.embed_batch(["a", "b", "c"])

        assert all(len(v) == EMBEDDING_DIM for v in results)
