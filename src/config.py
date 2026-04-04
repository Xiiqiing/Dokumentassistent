"""Centralized configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class Settings:
    """Application-wide settings populated from environment variables."""

    # Provider selection
    llm_provider: str
    embedding_provider: str

    # General
    qdrant_path: str
    qdrant_url: str
    collection_name: str
    embedding_model: str
    embedding_dimension: int
    generation_model: str
    reranker_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    bm25_weight: float
    dense_weight: float
    log_level: str

    # Ollama
    ollama_base_url: str
    ollama_model: str

    # OpenAI
    openai_api_key: str
    openai_model: str
    openai_embedding_model: str

    # Azure OpenAI
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_api_version: str
    azure_openai_deployment: str
    azure_openai_embedding_deployment: str

    # Anthropic
    anthropic_api_key: str
    anthropic_model: str

    # Google GenAI
    google_api_key: str
    google_model: str
    google_embedding_model: str

    # Local embeddings (HuggingFace)
    local_embedding_model: str

    # Query translation
    translate_query: bool


def _parse_bool(value: str, *, default: bool) -> bool:
    """Parse a boolean environment variable string.

    Args:
        value: Raw env var value (may be empty).
        default: Fallback when value is empty or unset.

    Returns:
        Parsed boolean.
    """
    if not value:
        return default
    return value.strip().lower() in ("1", "true", "yes")


def load_settings() -> Settings:
    """Load and return application settings from environment variables.

    Returns:
        Settings: Frozen dataclass with all configuration values.
    """
    return Settings(
        # Provider selection
        llm_provider=os.environ.get("LLM_PROVIDER", "ollama"),
        embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "local"),

        # General
        qdrant_path=os.environ.get("QDRANT_PATH", "./qdrant_data"),
        qdrant_url=os.environ.get("QDRANT_URL", ""),
        collection_name=os.environ.get("COLLECTION_NAME", "ku_documents"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
        embedding_dimension=int(os.environ.get("EMBEDDING_DIMENSION", "384")),
        generation_model=os.environ.get("GENERATION_MODEL", "gemma3:4b"),
        reranker_model=os.environ.get("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "64")),
        top_k=int(os.environ.get("TOP_K", "5")),
        bm25_weight=float(os.environ.get("BM25_WEIGHT", "0.3")),
        dense_weight=float(os.environ.get("DENSE_WEIGHT", "0.7")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),

        # Ollama
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "gemma3:4b"),

        # OpenAI
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        openai_embedding_model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),

        # Azure OpenAI
        azure_openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        azure_openai_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        azure_openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_openai_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", ""),
        azure_openai_embedding_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", ""),

        # Anthropic
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),

        # Google GenAI
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
        google_model=os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash"),
        google_embedding_model=os.environ.get("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"),

        # Local embeddings
        local_embedding_model=os.environ.get(
            "LOCAL_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        ),

        # Query translation — auto-detect default based on provider
        translate_query=_parse_bool(
            os.environ.get("TRANSLATE_QUERY", ""),
            default=os.environ.get("LLM_PROVIDER", "ollama") == "ollama",
        ),
    )
