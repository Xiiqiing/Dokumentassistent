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

    def to_dict(self, *, include_text: bool = True) -> dict:
        """Serialise to a JSON-safe dictionary.

        Args:
            include_text: Whether to include the chunk text (default True).

        Returns:
            Dictionary with chunk_id, document_id, score, source, metadata,
            and optionally text.
        """
        d: dict = {
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "score": self.score,
            "source": self.source,
            "metadata": self.chunk.metadata,
        }
        if include_text:
            d["text"] = self.chunk.text
        return d


@dataclass
class PipelineDetails:
    """Intermediate pipeline data for debugging and transparency.

    Attributes:
        original_query: The user's original query text.
        retrieval_query: The query used for retrieval (may be translated).
        detected_language: Detected language of the original query.
        translated: Whether the query was translated for retrieval.
        dense_results: Results from dense (vector) retrieval.
        sparse_results: Results from sparse (BM25) retrieval.
        fused_results: Results after reciprocal rank fusion.
        reranked_results: Results after cross-encoder reranking.
        plan_steps: Ordered descriptions of planned steps (Plan-and-Execute mode).
        tool_calls: Log of tool invocations as "tool_name: argument" strings.
    """

    original_query: str = ""
    retrieval_query: str = ""
    detected_language: str = ""
    translated: bool = False
    dense_results: list[QueryResult] = field(default_factory=list)
    sparse_results: list[QueryResult] = field(default_factory=list)
    fused_results: list[QueryResult] = field(default_factory=list)
    reranked_results: list[QueryResult] = field(default_factory=list)
    plan_steps: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class GenerationResponse:
    """Structured response from the generation pipeline.

    Attributes:
        answer: The generated answer text.
        sources: List of source chunks used for generation.
        intent: Classified intent of the original query.
        confidence: Model confidence in the answer (0.0-1.0).
        pipeline_details: Optional intermediate pipeline data.
    """

    answer: str
    sources: list[QueryResult]
    intent: IntentType
    confidence: float
    pipeline_details: PipelineDetails = field(default_factory=PipelineDetails)