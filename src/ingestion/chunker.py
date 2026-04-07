"""Text chunking strategies for document processing."""

import hashlib
import logging

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker

from src.models import ChunkStrategy, DocumentChunk

logger = logging.getLogger(__name__)


def _make_chunk_id(document_id: str, index: int) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{document_id}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class BaseChunker:
    """Base class for text chunking strategies."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """Initialize chunker with size parameters.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self, text: str, document_id: str, metadata: dict[str, str | int],
        start_index: int = 0,
    ) -> list[DocumentChunk]:
        """Split text into chunks.

        Args:
            text: The full text to chunk.
            document_id: Identifier of the source document.
            metadata: Metadata to attach to each chunk.
            start_index: Starting chunk index for globally unique IDs.

        Returns:
            List of DocumentChunk objects.
        """
        raise NotImplementedError("Subclasses must implement chunk()")


class FixedSizeChunker(BaseChunker):
    """Splits text into fixed-size character chunks with overlap."""

    def chunk(
        self, text: str, document_id: str, metadata: dict[str, str | int],
        start_index: int = 0,
    ) -> list[DocumentChunk]:
        """Split text into fixed-size chunks using LangChain CharacterTextSplitter.

        Args:
            text: The full text to chunk.
            document_id: Identifier of the source document.
            metadata: Metadata to attach to each chunk.
            start_index: Starting chunk index for globally unique IDs.

        Returns:
            List of DocumentChunk with strategy=FIXED_SIZE.
        """
        from langchain_text_splitters import CharacterTextSplitter

        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="",
        )
        texts = splitter.split_text(text)
        chunks = [
            DocumentChunk(
                chunk_id=_make_chunk_id(document_id, start_index + i),
                document_id=document_id,
                text=chunk_text,
                metadata={**metadata, "chunk_index": start_index + i},
                strategy=ChunkStrategy.FIXED_SIZE,
            )
            for i, chunk_text in enumerate(texts)
        ]
        logger.debug("FixedSizeChunker produced %d chunks for %s", len(chunks), document_id)
        return chunks


class RecursiveChunker(BaseChunker):
    """Recursively splits text using LangChain's RecursiveCharacterTextSplitter."""

    def chunk(
        self, text: str, document_id: str, metadata: dict[str, str | int],
        start_index: int = 0,
    ) -> list[DocumentChunk]:
        """Split text using recursive character splitting.

        Args:
            text: The full text to chunk.
            document_id: Identifier of the source document.
            metadata: Metadata to attach to each chunk.
            start_index: Starting chunk index for globally unique IDs.

        Returns:
            List of DocumentChunk with strategy=RECURSIVE.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        texts = splitter.split_text(text)
        chunks: list[DocumentChunk] = []
        for i, chunk_text in enumerate(texts):
            chunks.append(DocumentChunk(
                chunk_id=_make_chunk_id(document_id, start_index + i),
                document_id=document_id,
                text=chunk_text,
                metadata={**metadata, "chunk_index": start_index + i},
                strategy=ChunkStrategy.RECURSIVE,
            ))
        logger.debug("RecursiveChunker produced %d chunks for %s", len(chunks), document_id)
        return chunks


class SemanticChunker(BaseChunker):
    """Splits text at semantic boundaries using embeddings similarity."""

    def __init__(
        self, chunk_size: int, chunk_overlap: int, embeddings: Embeddings
    ) -> None:
        """Initialize semantic chunker with an embeddings instance.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            embeddings: A LangChain Embeddings instance from provider.py.
        """
        super().__init__(chunk_size, chunk_overlap)
        self._embeddings = embeddings

    def chunk(
        self, text: str, document_id: str, metadata: dict[str, str | int],
        start_index: int = 0,
    ) -> list[DocumentChunk]:
        """Split text at semantic boundaries.

        Args:
            text: The full text to chunk.
            document_id: Identifier of the source document.
            metadata: Metadata to attach to each chunk.
            start_index: Starting chunk index for globally unique IDs.

        Returns:
            List of DocumentChunk with strategy=SEMANTIC.
        """
        splitter = LCSemanticChunker(embeddings=self._embeddings)
        docs = splitter.create_documents([text])
        chunks: list[DocumentChunk] = []
        for i, doc in enumerate(docs):
            chunks.append(DocumentChunk(
                chunk_id=_make_chunk_id(document_id, start_index + i),
                document_id=document_id,
                text=doc.page_content,
                metadata={**metadata, "chunk_index": start_index + i},
                strategy=ChunkStrategy.SEMANTIC,
            ))
        logger.debug("SemanticChunker produced %d chunks for %s", len(chunks), document_id)
        return chunks


def create_chunker(
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
    embeddings: Embeddings | None = None,
) -> BaseChunker:
    """Factory function to create a chunker based on the selected strategy.

    Args:
        strategy: The chunking strategy to use.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        embeddings: LangChain Embeddings instance (required for SEMANTIC strategy).

    Returns:
        An instance of the appropriate chunker class.

    Raises:
        ValueError: If the strategy is not recognized or embeddings missing for semantic.
    """
    match strategy:
        case ChunkStrategy.FIXED_SIZE:
            return FixedSizeChunker(chunk_size, chunk_overlap)
        case ChunkStrategy.RECURSIVE:
            return RecursiveChunker(chunk_size, chunk_overlap)
        case ChunkStrategy.SEMANTIC:
            if embeddings is None:
                raise ValueError("Embeddings instance is required for SEMANTIC chunking strategy")
            return SemanticChunker(chunk_size, chunk_overlap, embeddings)
        case _:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
