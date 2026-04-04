"""Tests for text chunking strategies."""

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.chunker import (
    BaseChunker,
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    _make_chunk_id,
    create_chunker,
)
from src.models import ChunkStrategy, DocumentChunk

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DANISH_TEXT = (
    "Københavns Universitet har følgende regler for behandling af persondata. "
    "Alle ansatte skal overholde retningslinjerne i henhold til GDPR. "
    "Særlige bestemmelser gælder for håndtering af følsomme oplysninger. "
    "Ændringer træder i kraft den 1. januar. "
    "Spørgsmål kan rettes til databeskyttelsesrådgiveren på ældre@telefonlinje.dk."
)

DOC_ID = "doc-test-001"
META: dict[str, str | int] = {"source": "test.pdf", "page": 1}


# ---------------------------------------------------------------------------
# Helper – deterministic chunk ID
# ---------------------------------------------------------------------------

class TestMakeChunkId:
    def test_deterministic(self) -> None:
        assert _make_chunk_id("doc1", 0) == _make_chunk_id("doc1", 0)

    def test_different_inputs(self) -> None:
        assert _make_chunk_id("doc1", 0) != _make_chunk_id("doc1", 1)
        assert _make_chunk_id("doc1", 0) != _make_chunk_id("doc2", 0)

    def test_length(self) -> None:
        assert len(_make_chunk_id("x", 0)) == 16


# ---------------------------------------------------------------------------
# Output format helpers (shared assertions)
# ---------------------------------------------------------------------------

def _assert_valid_chunks(
    chunks: list[DocumentChunk],
    expected_strategy: ChunkStrategy,
    document_id: str = DOC_ID,
) -> None:
    """Assert that every chunk has the correct structure and strategy."""
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for idx, chunk in enumerate(chunks):
        assert isinstance(chunk, DocumentChunk)
        assert chunk.document_id == document_id
        assert isinstance(chunk.chunk_id, str) and len(chunk.chunk_id) == 16
        assert isinstance(chunk.text, str) and len(chunk.text) > 0
        assert chunk.strategy == expected_strategy
        assert chunk.metadata["chunk_index"] == idx


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------

class TestFixedSizeChunker:
    def test_output_format(self) -> None:
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(DANISH_TEXT, DOC_ID, META)
        _assert_valid_chunks(chunks, ChunkStrategy.FIXED_SIZE)

    def test_chunk_size_respected(self) -> None:
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(DANISH_TEXT, DOC_ID, META)
        for chunk in chunks:
            assert len(chunk.text) <= 50

    def test_overlap(self) -> None:
        chunker = FixedSizeChunker(chunk_size=60, chunk_overlap=20)
        chunks = chunker.chunk(DANISH_TEXT, DOC_ID, META)
        if len(chunks) >= 2:
            tail = chunks[0].text[-20:]
            assert chunks[1].text.startswith(tail)

    def test_empty_text(self) -> None:
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("", DOC_ID, META)
        assert chunks == []

    def test_short_text(self) -> None:
        chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk("Hej", DOC_ID, META)
        assert len(chunks) == 1
        assert chunks[0].text == "Hej"
        assert chunks[0].strategy == ChunkStrategy.FIXED_SIZE

    def test_danish_characters_preserved(self) -> None:
        text = "æble, ørred, åben"
        chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk(text, DOC_ID, META)
        assert chunks[0].text == text

    def test_metadata_propagated(self) -> None:
        chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk(DANISH_TEXT, DOC_ID, META)
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["page"] == 1


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------

class TestRecursiveChunker:
    def test_output_format(self) -> None:
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(DANISH_TEXT, DOC_ID, META)
        _assert_valid_chunks(chunks, ChunkStrategy.RECURSIVE)

    def test_empty_text(self) -> None:
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("", DOC_ID, META)
        assert chunks == []

    def test_short_text(self) -> None:
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk("Hej", DOC_ID, META)
        assert len(chunks) == 1
        assert chunks[0].text == "Hej"
        assert chunks[0].strategy == ChunkStrategy.RECURSIVE

    def test_danish_characters_preserved(self) -> None:
        text = "Håndtering af ældre dokumenter kræver særlig opmærksomhed fra ændringsledelsen."
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk(text, DOC_ID, META)
        assert chunks[0].text == text

    def test_splits_long_text(self) -> None:
        chunker = RecursiveChunker(chunk_size=80, chunk_overlap=10)
        chunks = chunker.chunk(DANISH_TEXT, DOC_ID, META)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# SemanticChunker (requires a mock embeddings instance)
# ---------------------------------------------------------------------------

class TestSemanticChunker:
    @patch("src.ingestion.chunker.LCSemanticChunker")
    def test_output_format(self, mock_lc_chunker_cls: MagicMock) -> None:
        fake_doc_1 = MagicMock()
        fake_doc_1.page_content = "Første del af teksten."
        fake_doc_2 = MagicMock()
        fake_doc_2.page_content = "Anden del af teksten."
        mock_lc_chunker_cls.return_value.create_documents.return_value = [
            fake_doc_1,
            fake_doc_2,
        ]

        mock_embeddings = MagicMock()
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20, embeddings=mock_embeddings)
        chunks = chunker.chunk(DANISH_TEXT, DOC_ID, META)

        _assert_valid_chunks(chunks, ChunkStrategy.SEMANTIC)
        assert len(chunks) == 2
        assert chunks[0].text == "Første del af teksten."
        assert chunks[1].text == "Anden del af teksten."

    @patch("src.ingestion.chunker.LCSemanticChunker")
    def test_empty_text(self, mock_lc_chunker_cls: MagicMock) -> None:
        mock_lc_chunker_cls.return_value.create_documents.return_value = []
        mock_embeddings = MagicMock()
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20, embeddings=mock_embeddings)
        chunks = chunker.chunk("", DOC_ID, META)
        assert chunks == []

    @patch("src.ingestion.chunker.LCSemanticChunker")
    def test_short_text(self, mock_lc_chunker_cls: MagicMock) -> None:
        fake_doc = MagicMock()
        fake_doc.page_content = "Hej"
        mock_lc_chunker_cls.return_value.create_documents.return_value = [fake_doc]

        mock_embeddings = MagicMock()
        chunker = SemanticChunker(chunk_size=500, chunk_overlap=50, embeddings=mock_embeddings)
        chunks = chunker.chunk("Hej", DOC_ID, META)
        assert len(chunks) == 1
        assert chunks[0].text == "Hej"
        assert chunks[0].strategy == ChunkStrategy.SEMANTIC

    @patch("src.ingestion.chunker.LCSemanticChunker")
    def test_danish_characters_preserved(self, mock_lc_chunker_cls: MagicMock) -> None:
        text = "Ændringsforslag vedrørende årsregnskabet"
        fake_doc = MagicMock()
        fake_doc.page_content = text
        mock_lc_chunker_cls.return_value.create_documents.return_value = [fake_doc]

        mock_embeddings = MagicMock()
        chunker = SemanticChunker(chunk_size=500, chunk_overlap=0, embeddings=mock_embeddings)
        chunks = chunker.chunk(text, DOC_ID, META)
        assert chunks[0].text == text


# ---------------------------------------------------------------------------
# Factory: create_chunker
# ---------------------------------------------------------------------------

class TestCreateChunker:
    def test_fixed_size(self) -> None:
        chunker = create_chunker(ChunkStrategy.FIXED_SIZE, 100, 20)
        assert isinstance(chunker, FixedSizeChunker)

    def test_recursive(self) -> None:
        chunker = create_chunker(ChunkStrategy.RECURSIVE, 100, 20)
        assert isinstance(chunker, RecursiveChunker)

    def test_semantic(self) -> None:
        mock_embeddings = MagicMock()
        chunker = create_chunker(ChunkStrategy.SEMANTIC, 100, 20, embeddings=mock_embeddings)
        assert isinstance(chunker, SemanticChunker)

    def test_semantic_without_embeddings_raises(self) -> None:
        with pytest.raises(ValueError, match="Embeddings instance is required"):
            create_chunker(ChunkStrategy.SEMANTIC, 100, 20)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            create_chunker("invalid", 100, 20)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BaseChunker – not implemented guard
# ---------------------------------------------------------------------------

class TestBaseChunker:
    def test_chunk_raises_not_implemented(self) -> None:
        base = BaseChunker(chunk_size=100, chunk_overlap=20)
        with pytest.raises(NotImplementedError):
            base.chunk("text", DOC_ID, META)
