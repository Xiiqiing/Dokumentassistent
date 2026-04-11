"""Tests for the cross-encoder reranker."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.models import DocumentChunk, QueryResult
from src.retrieval.reranker import Reranker, _sigmoid


def _make_result(text: str, score: float) -> QueryResult:
    """Create a QueryResult with the given text and score."""
    chunk = DocumentChunk(chunk_id="c1", document_id="d1", text=text)
    return QueryResult(chunk=chunk, score=score, source="test")


@pytest.fixture
def reranker() -> Reranker:
    """Return a Reranker with a mocked model."""
    model = MagicMock()
    return Reranker(model=model)


class TestRerank:
    """Tests for Reranker.rerank."""

    def test_rerank_reorders_by_cross_encoder_score(self, reranker: Reranker) -> None:
        """Reranked order should follow cross-encoder scores, not original order."""
        results = [
            _make_result("low relevance", score=0.9),
            _make_result("high relevance", score=0.1),
            _make_result("mid relevance", score=0.5),
        ]
        # Cross-encoder assigns: low->0.1, high->0.9, mid->0.5
        reranker._model.predict = MagicMock(return_value=np.array([0.1, 0.9, 0.5]))

        reranked = reranker.rerank("test query", results, top_k=3)

        assert len(reranked) == 3
        assert reranked[0].chunk.text == "high relevance"
        assert reranked[1].chunk.text == "mid relevance"
        assert reranked[2].chunk.text == "low relevance"
        assert all(r.source == "reranked" for r in reranked)
        # Scores must be sigmoid-normalized to [0, 1]
        assert all(0.0 <= r.score <= 1.0 for r in reranked)

    def test_rerank_respects_top_k(self, reranker: Reranker) -> None:
        """Only top_k results should be returned."""
        results = [_make_result(f"doc{i}", score=0.5) for i in range(5)]
        reranker._model.predict = MagicMock(return_value=np.array([0.1, 0.5, 0.9, 0.3, 0.7]))

        reranked = reranker.rerank("query", results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].chunk.text == "doc2"
        assert reranked[1].chunk.text == "doc4"

    def test_rerank_empty_list(self, reranker: Reranker) -> None:
        """Empty input should return empty list without calling the model."""
        reranked = reranker.rerank("query", [], top_k=5)

        assert reranked == []
        reranker._model.predict.assert_not_called()

    def test_rerank_single_result(self, reranker: Reranker) -> None:
        """A single result should be returned with sigmoid-normalized score."""
        results = [_make_result("only doc", score=0.3)]
        reranker._model.predict = MagicMock(return_value=np.array([0.85]))

        reranked = reranker.rerank("query", results, top_k=1)

        assert len(reranked) == 1
        assert reranked[0].chunk.text == "only doc"
        assert reranked[0].score == pytest.approx(_sigmoid(0.85))
        assert 0.0 <= reranked[0].score <= 1.0
        assert reranked[0].source == "reranked"

    def test_rerank_negative_scores_normalized(self, reranker: Reranker) -> None:
        """Negative cross-encoder scores must be normalized to [0, 1]."""
        results = [
            _make_result("bad", score=0.5),
            _make_result("worse", score=0.5),
        ]
        reranker._model.predict = MagicMock(return_value=np.array([-2.0, -5.0]))

        reranked = reranker.rerank("query", results, top_k=2)

        assert all(0.0 <= r.score <= 1.0 for r in reranked)
        assert reranked[0].score == pytest.approx(_sigmoid(-2.0))
        assert reranked[1].score == pytest.approx(_sigmoid(-5.0))
