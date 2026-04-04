"""Tests for the Qdrant vector store."""

import tempfile

import pytest

from src.models import ChunkStrategy, DocumentChunk
from src.retrieval.vector_store import VectorStore

DIMENSION = 4


def _make_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        metadata={"page": 1},
        strategy=ChunkStrategy.FIXED_SIZE,
    )


def _fake_embedding(seed: float) -> list[float]:
    """Return a deterministic fake embedding of length DIMENSION."""
    return [seed, 1.0 - seed, seed * 0.5, 0.1]


@pytest.fixture()
def store(tmp_path: str) -> VectorStore:
    return VectorStore(
        path=str(tmp_path / "qdrant"),
        collection_name="test_collection",
        dimension=DIMENSION,
    )


class TestAddChunks:
    """Tests for inserting documents into the store."""

    def test_add_and_search_returns_inserted_chunks(self, store: VectorStore) -> None:
        chunks = [_make_chunk("c1", "hello world"), _make_chunk("c2", "foo bar")]
        embeddings = [_fake_embedding(0.1), _fake_embedding(0.9)]

        store.add_chunks(chunks, embeddings)
        results = store.search(query_embedding=_fake_embedding(0.1), top_k=10)

        assert len(results) == 2
        returned_ids = {r.chunk.chunk_id for r in results}
        assert returned_ids == {"c1", "c2"}

    def test_add_chunks_length_mismatch_raises(self, store: VectorStore) -> None:
        chunks = [_make_chunk("c1", "text")]
        embeddings = [_fake_embedding(0.1), _fake_embedding(0.2)]

        with pytest.raises(ValueError, match="mismatch"):
            store.add_chunks(chunks, embeddings)

    def test_add_empty_chunks_is_noop(self, store: VectorStore) -> None:
        store.add_chunks([], [])
        results = store.search(query_embedding=_fake_embedding(0.5), top_k=10)
        assert results == []


class TestSearch:
    """Tests for querying the vector store."""

    def test_top_k_limits_results(self, store: VectorStore) -> None:
        chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(5)]
        embeddings = [_fake_embedding(i * 0.1) for i in range(5)]
        store.add_chunks(chunks, embeddings)

        results = store.search(query_embedding=_fake_embedding(0.0), top_k=3)
        assert len(results) == 3

    def test_search_empty_collection(self, store: VectorStore) -> None:
        results = store.search(query_embedding=_fake_embedding(0.5), top_k=5)
        assert results == []

    def test_results_contain_correct_metadata(self, store: VectorStore) -> None:
        chunk = _make_chunk("c1", "some text")
        store.add_chunks([chunk], [_fake_embedding(0.3)])

        results = store.search(query_embedding=_fake_embedding(0.3), top_k=1)
        assert len(results) == 1
        r = results[0]
        assert r.chunk.chunk_id == "c1"
        assert r.chunk.document_id == "doc-1"
        assert r.chunk.text == "some text"
        assert r.chunk.metadata == {"page": 1}
        assert r.chunk.strategy == ChunkStrategy.FIXED_SIZE
        assert r.source == "dense"

    def test_results_sorted_by_relevance(self, store: VectorStore) -> None:
        chunks = [_make_chunk("c1", "a"), _make_chunk("c2", "b")]
        embeddings = [_fake_embedding(0.9), _fake_embedding(0.1)]
        store.add_chunks(chunks, embeddings)

        results = store.search(query_embedding=_fake_embedding(0.9), top_k=2)
        assert results[0].score >= results[1].score


class TestDuplicateInsert:
    """Tests for upserting (re-inserting) documents."""

    def test_upsert_overwrites_existing_chunk(self, store: VectorStore) -> None:
        chunk_v1 = _make_chunk("c1", "version 1")
        chunk_v2 = _make_chunk("c1", "version 2")

        store.add_chunks([chunk_v1], [_fake_embedding(0.5)])
        store.add_chunks([chunk_v2], [_fake_embedding(0.5)])

        results = store.search(query_embedding=_fake_embedding(0.5), top_k=10)
        # Upsert uses enumerate index as point id, so same index → overwrite
        assert len(results) == 1
        assert results[0].chunk.text == "version 2"
