"""Shared data models used across the project."""

from dataclasses import dataclass, field
from enum import Enum


class ChunkStrategy(Enum):
    """Available text chunking strategies."""

    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class IntentType(Enum):
    """Classified intent types for incoming queries."""

    FACTUAL = "factual"
    RAG = "rag"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    PROCEDURAL = "procedural"
    UNKNOWN = "unknown"


@dataclass
class DocumentChunk:
    """A single chunk of text extracted from a PDF document.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        document_id: Identifier of the source document.
        text: The chunk text content.
        metadata: Additional metadata (page number, source file, etc.).
        strategy: The chunking strategy used to produce this chunk.
    """

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, str | int] = field(default_factory=dict)
    strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE


@dataclass
class QueryResult:
    """A single retrieval result with relevance scoring.

    Attributes:
        chunk: The retrieved document chunk.
        score: Relevance score (higher is better).
        source: Which retrieval method produced this result.
    """

    chunk: DocumentChunk
    score: float
    source: str


@dataclass
class GenerationResponse:
    """Structured response from the generation pipeline.

    Attributes:
        answer: The generated answer text.
        sources: List of source chunks used for generation.
        intent: Classified intent of the original query.
        confidence: Model confidence in the answer (0.0-1.0).
    """

    answer: str
    sources: list[QueryResult]
    intent: IntentType
    confidence: float