"""Qdrant vector store for dense retrieval."""

import json
import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.models import ChunkStrategy, DocumentChunk, QueryResult

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages document storage and dense retrieval via Qdrant."""

    def __init__(self, path: str, collection_name: str, dimension: int, url: str = "") -> None:
        """Initialize the Qdrant vector store.

        Args:
            path: File system path for Qdrant local storage (used when *url* is empty).
            collection_name: Name of the Qdrant collection.
            dimension: Embedding vector dimension.
            url: Optional Qdrant server URL. When provided, connects over HTTP
                 instead of using local file storage.
        """
        self._collection_name = collection_name
        if url:
            self._client = QdrantClient(url=url)
        else:
            self._client = QdrantClient(path=path)

        existing = [c.name for c in self._client.get_collections().collections]
        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s' (dim=%d)", collection_name, dimension)
        else:
            logger.info("Using existing Qdrant collection '%s'", collection_name)

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Index document chunks with their embeddings.

        Args:
            chunks: List of document chunks to store.
            embeddings: Corresponding embedding vectors.

        Raises:
            ValueError: If chunks and embeddings have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        if not chunks:
            return

        points = [
            PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "metadata": json.dumps(chunk.metadata),
                    "strategy": chunk.strategy.value,
                },
            )
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        self._client.upsert(collection_name=self._collection_name, points=points)
        logger.info("Indexed %d chunks into '%s'", len(points), self._collection_name)

    def search(self, query_embedding: list[float], top_k: int) -> list[QueryResult]:
        """Search for the most similar chunks by vector similarity.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of top results to return.

        Returns:
            List of QueryResult objects sorted by relevance.
        """
        hits = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
        ).points

        results: list[QueryResult] = []
        for hit in hits:
            payload = hit.payload
            chunk = DocumentChunk(
                chunk_id=payload["chunk_id"],
                document_id=payload["document_id"],
                text=payload["text"],
                metadata=json.loads(payload["metadata"]),
                strategy=ChunkStrategy(payload["strategy"]),
            )
            results.append(QueryResult(chunk=chunk, score=hit.score, source="dense"))

        logger.debug("Dense search returned %d results", len(results))
        return results

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Retrieve all document chunks stored in the collection.

        Returns:
            List of all DocumentChunk objects in the collection.
        """
        collection_info = self._client.get_collection(self._collection_name)
        total = collection_info.points_count
        if not total:
            return []

        records, _offset = self._client.scroll(
            collection_name=self._collection_name,
            limit=total,
            with_payload=True,
            with_vectors=False,
        )

        chunks: list[DocumentChunk] = []
        for record in records:
            payload = record.payload
            chunks.append(
                DocumentChunk(
                    chunk_id=payload["chunk_id"],
                    document_id=payload["document_id"],
                    text=payload["text"],
                    metadata=json.loads(payload["metadata"]),
                    strategy=ChunkStrategy(payload["strategy"]),
                )
            )
        logger.info("Loaded %d chunks from collection '%s'", len(chunks), self._collection_name)
        return chunks

    def as_retriever(self, embedder: Any, top_k: int) -> BaseRetriever:
        """Return a LangChain BaseRetriever wrapping this vector store.

        Args:
            embedder: Embedder instance used to encode queries.
            top_k: Number of results to return per query.

        Returns:
            A BaseRetriever that calls search() and returns Documents.
        """
        return _VectorStoreRetrieverAdapter(
            vector_store=self, embedder=embedder, top_k=top_k
        )

    def delete_collection(self) -> None:
        """Delete the entire collection from the store."""
        self._client.delete_collection(collection_name=self._collection_name)
        logger.info("Deleted Qdrant collection '%s'", self._collection_name)


class _VectorStoreRetrieverAdapter(BaseRetriever):
    """LangChain BaseRetriever adapter over VectorStore."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: Any
    embedder: Any
    top_k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(query_embedding, self.top_k)
        return [
            Document(
                page_content=r.chunk.text,
                metadata={
                    "chunk_id": r.chunk.chunk_id,
                    "document_id": r.chunk.document_id,
                    "chunk_metadata": r.chunk.metadata,
                    "strategy": r.chunk.strategy.value,
                    "score": r.score,
                },
            )
            for r in results
        ]
