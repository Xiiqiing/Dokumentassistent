"""Factory functions for creating LLM and embedding instances.

All provider-specific imports are isolated here. The rest of the codebase
interacts only with LangChain abstract interfaces returned by these factories.
"""

import logging
from dataclasses import replace

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings

logger = logging.getLogger(__name__)

_SUPPORTED_LLM_PROVIDERS = ["ollama", "azure_openai", "openai", "groq", "anthropic", "google_genai", "bedrock"]
_SUPPORTED_EMBEDDING_PROVIDERS = ["local", "azure_openai", "openai", "google_genai", "bedrock"]


def create_llm(settings: Settings) -> BaseChatModel:
    """Create an LLM instance based on the configured provider.

    Args:
        settings: Application settings with provider configuration.

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = settings.llm_provider.lower()
    logger.info("Creating LLM with provider: %s", provider)

    match provider:
        case "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.0,
            )

        case "azure_openai":
            from langchain_openai import AzureChatOpenAI

            return AzureChatOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                azure_deployment=settings.azure_openai_deployment,
                temperature=0.0,
            )

        case "openai":
            from langchain_openai import ChatOpenAI

            kwargs: dict = {
                "model": settings.openai_model,
                "api_key": settings.openai_api_key,
                "temperature": 0.0,
            }
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            return ChatOpenAI(**kwargs)

        case "groq":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.groq_model,
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0.0,
            )

        case "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=settings.anthropic_model,
                api_key=settings.anthropic_api_key,
                temperature=0.0,
            )

        case "google_genai":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=settings.google_model,
                google_api_key=settings.google_api_key,
                temperature=0.0,
            )

        case "bedrock":
            from langchain_aws import ChatBedrockConverse

            return ChatBedrockConverse(
                model=settings.aws_bedrock_model,
                region_name=settings.aws_region,
                temperature=0.0,
            )

        case _:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Supported providers: {_SUPPORTED_LLM_PROVIDERS}"
            )


_EVALUATOR_MODEL_FIELD: dict[str, str] = {
    "groq": "groq_model",
    "openai": "openai_model",
    "anthropic": "anthropic_model",
    "google_genai": "google_model",
    "azure_openai": "azure_openai_deployment",
    "bedrock": "aws_bedrock_model",
    "ollama": "ollama_model",
}


def create_evaluator_llm(settings: Settings) -> BaseChatModel:
    """Create the LLM used as a RAGAS judge.

    The judge LLM is independent of the generation LLM so a strong cloud
    model (e.g. Qwen3-32B via Groq) can score outputs produced by a small
    local generation model. If ``EVALUATOR_LLM_PROVIDER`` is unset, falls
    back to ``create_llm(settings)`` which reuses the generation LLM.

    Args:
        settings: Application settings with provider configuration.

    Returns:
        A LangChain BaseChatModel instance to use as the RAGAS judge.

    Raises:
        ValueError: If ``EVALUATOR_LLM_PROVIDER`` is set to an unknown value.
    """
    provider = settings.evaluator_llm_provider.lower().strip()
    if not provider:
        logger.info("EVALUATOR_LLM_PROVIDER unset; reusing generation LLM as judge")
        return create_llm(settings)

    overrides: dict[str, str] = {"llm_provider": provider}
    if settings.evaluator_llm_model:
        model_field = _EVALUATOR_MODEL_FIELD.get(provider)
        if model_field is None:
            raise ValueError(
                f"Cannot override evaluator model for unknown provider: '{provider}'"
            )
        overrides[model_field] = settings.evaluator_llm_model

    overridden = replace(settings, **overrides)
    logger.info(
        "Creating evaluator (judge) LLM with provider: %s | model override: %s",
        provider,
        settings.evaluator_llm_model or "(provider default)",
    )
    return create_llm(overridden)


def create_embeddings(settings: Settings) -> Embeddings:
    """Create an embeddings instance based on the configured provider.

    Args:
        settings: Application settings with provider configuration.

    Returns:
        A LangChain Embeddings instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = settings.embedding_provider.lower()
    logger.info("Creating embeddings with provider: %s", provider)

    match provider:
        case "local":
            from langchain_huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=settings.local_embedding_model,
            )

        case "azure_openai":
            from langchain_openai import AzureOpenAIEmbeddings

            return AzureOpenAIEmbeddings(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                azure_deployment=settings.azure_openai_embedding_deployment,
            )

        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key,
            )

        case "google_genai":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(
                model=settings.google_embedding_model,
                google_api_key=settings.google_api_key,
            )

        case "bedrock":
            from langchain_aws import BedrockEmbeddings

            return BedrockEmbeddings(
                model_id=settings.aws_bedrock_embedding_model,
                region_name=settings.aws_region,
            )

        case _:
            raise ValueError(
                f"Unknown embedding provider: '{provider}'. "
                f"Supported providers: {_SUPPORTED_EMBEDDING_PROVIDERS}"
            )


def create_reranker(model_name: str) -> object:
    """Create a cross-encoder reranker model instance.

    Args:
        model_name: HuggingFace model name for the cross-encoder.

    Returns:
        A CrossEncoder model instance.
    """
    from sentence_transformers import CrossEncoder

    logger.info("Creating cross-encoder reranker: %s", model_name)
    return CrossEncoder(model_name)
