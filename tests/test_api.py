"""Tests for the FastAPI API routes."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.routes import router, set_dependencies, QueryResponse, HealthResponse
from src.models import DocumentChunk, GenerationResponse, IntentType, QueryResult


@pytest.fixture()
def mock_deps() -> dict[str, MagicMock]:
    """Create mock dependencies and inject them into the routes module."""
    query_router = MagicMock()
    ingestion_pipeline = MagicMock()
    embedder = MagicMock()
    vector_store = MagicMock()
    bm25_search = MagicMock()
    settings = MagicMock()
    settings.llm_provider = "ollama"
    settings.embedding_provider = "local"
    settings.embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
    settings.ollama_model = "llama3"
    settings.generation_model = "llama3"

    set_dependencies(
        query_router=query_router,
        ingestion_pipeline=ingestion_pipeline,
        embedder=embedder,
        vector_store=vector_store,
        bm25_search=bm25_search,
        settings=settings,
    )

    return {
        "query_router": query_router,
        "ingestion_pipeline": ingestion_pipeline,
        "embedder": embedder,
        "vector_store": vector_store,
        "bm25_search": bm25_search,
        "settings": settings,
    }


@pytest.fixture()
def client(mock_deps: dict[str, MagicMock]) -> TestClient:
    """Create a TestClient with mocked dependencies."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestHealthCheck:
    """Tests for the /health endpoint."""

    def test_health_check_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"] == "0.1.0"


class TestQueryEndpoint:
    """Tests for the /query endpoint."""

    def test_query_returns_structured_response(
        self, client: TestClient, mock_deps: dict[str, MagicMock]
    ) -> None:
        chunk = DocumentChunk(
            chunk_id="c1",
            document_id="doc1",
            text="Some policy text.",
        )
        generation_response = GenerationResponse(
            answer="The policy states that...",
            sources=[QueryResult(chunk=chunk, score=0.95, source="hybrid")],
            intent=IntentType.FACTUAL,
            confidence=0.88,
        )
        mock_deps["query_router"].route.return_value = generation_response

        response = client.post("/query", json={"question": "What is the policy?"})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "The policy states that..."
        assert body["intent"] == "factual"
        assert body["confidence"] == 0.88
        assert len(body["sources"]) == 1
        assert body["sources"][0]["chunk_id"] == "c1"
        assert body["sources"][0]["document_id"] == "doc1"
        assert body["sources"][0]["score"] == 0.95

    def test_empty_question_returns_422(self, client: TestClient) -> None:
        """FastAPI returns 422 for missing required fields; empty string passes validation."""
        response = client.post("/query", json={})

        assert response.status_code == 422

    def test_missing_body_returns_422(self, client: TestClient) -> None:
        response = client.post("/query")

        assert response.status_code == 422

    def test_query_uses_default_top_k(
        self, client: TestClient, mock_deps: dict[str, MagicMock]
    ) -> None:
        mock_deps["query_router"].route.return_value = GenerationResponse(
            answer="answer",
            sources=[],
            intent=IntentType.UNKNOWN,
            confidence=0.5,
        )

        client.post("/query", json={"question": "test"})

        mock_deps["query_router"].route.assert_called_once_with(query="test", top_k=5)


class TestIngestEndpoint:
    """Tests for the /ingest endpoint."""

    @patch("src.api.routes.os.path.isfile", return_value=False)
    def test_ingest_missing_file_returns_404(
        self, _mock_isfile: MagicMock, client: TestClient
    ) -> None:
        response = client.post("/ingest", json={"file_path": "/nonexistent.pdf"})

        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]

    @patch("src.api.routes.os.path.isfile", return_value=True)
    def test_ingest_success(
        self,
        _mock_isfile: MagicMock,
        client: TestClient,
        mock_deps: dict[str, MagicMock],
    ) -> None:
        chunks = [
            DocumentChunk(chunk_id="c1", document_id="doc.pdf", text="chunk1"),
            DocumentChunk(chunk_id="c2", document_id="doc.pdf", text="chunk2"),
        ]
        mock_deps["ingestion_pipeline"].ingest_pdf.return_value = chunks
        mock_deps["embedder"].embed_batch.return_value = [[0.1] * 1536, [0.2] * 1536]

        response = client.post("/ingest", json={"file_path": "/tmp/doc.pdf"})

        assert response.status_code == 200
        body = response.json()
        assert body["document_id"] == "doc.pdf"
        assert body["chunks_created"] == 2
        mock_deps["vector_store"].add_chunks.assert_called_once()
        mock_deps["vector_store"].get_all_chunks.assert_called_once()
        mock_deps["bm25_search"].index.assert_called_once_with(
            mock_deps["vector_store"].get_all_chunks.return_value
        )
