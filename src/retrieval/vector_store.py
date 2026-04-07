"""Qdrant vector store for dense retrieval."""

import hashlib
import json
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from src.models import ChunkStrategy, DocumentChunk, QueryResult

logger = logging.getLogger(__name__)


def _payload_to_chunk(payload: dict) -> DocumentChunk:
    """Convert a Qdrant payload dict to a DocumentChunk.

    Args:
        payload: Qdrant point payload.

    Returns:
        DocumentChunk reconstructed from the payload.
    """
    return DocumentChunk(
        chunk_id=payload["chunk_id"],
        document_id=payload["document_id"],
        text=payload["text"],
        metadata=json.loads(payload["metadata"]),
        strategy=ChunkStrategy(payload["strategy"]),
    )


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
                id=int(hashlib.sha256(chunk.chunk_id.encode()).hexdigest()[:15], 16),
                vector=embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "metadata": json.dumps(chunk.metadata),
                    "strategy": chunk.strategy.value,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
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

        results: list[QueryResult] = [
            QueryResult(chunk=_payload_to_chunk(hit.payload), score=hit.score, source="dense")
            for hit in hits
        ]
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

        chunks = [_payload_to_chunk(record.payload) for record in records]
        logger.info("Loaded %d chunks from collection '%s'", len(chunks), self._collection_name)
        return chunks

    def list_document_ids(self) -> list[str]:
        """Return a sorted list of unique document IDs in the collection.

        Returns:
            Sorted list of document ID strings.
        """
        all_chunks = self.get_all_chunks()
        ids = sorted({chunk.document_id for chunk in all_chunks})
        logger.debug("Found %d unique document IDs", len(ids))
        return ids

    def get_chunks_by_document_id(self, document_id: str) -> list[DocumentChunk]:
        """Retrieve all chunks belonging to a specific document.

        Uses a Qdrant payload filter to avoid loading the full collection.

        Args:
            document_id: The document identifier to filter by.

        Returns:
            List of DocumentChunk objects for that document, in storage order.
        """
        records, _offset = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )

        chunks = [_payload_to_chunk(record.payload) for record in records]
        logger.debug(
            "Fetched %d chunks for document '%s'", len(chunks), document_id
        )
        return chunks

    def delete_collection(self) -> None:
        """Delete the entire collection from the store."""
        self._client.delete_collection(collection_name=self._collection_name)
        logger.info("Deleted Qdrant collection '%s'", self._collection_name)
