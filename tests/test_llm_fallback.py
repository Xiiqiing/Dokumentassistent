"""Tests for create_llm_with_fallback and the fallback chain runtime behaviour."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

import src.provider as provider_module
from src.config import load_settings


def _base_settings():  # noqa: ANN202
    """Return a Settings instance with fallback fields overridable for tests."""
    return load_settings()


def test_fallback_disabled_returns_plain_llm() -> None:
    settings = replace(
        _base_settings(),
        llm_fallback_enabled=False,
        llm_fallback_providers=("openai",),
    )
    fake = FakeListChatModel(responses=["hello"])
    with patch.object(provider_module, "create_llm", return_value=fake) as m:
        result = provider_module.create_llm_with_fallback(settings)
    # Exactly one LLM constructed: the primary.
    assert m.call_count == 1
    # The returned object is the plain fake — no with_fallbacks wrapper.
    assert result is fake


def test_fallback_enabled_but_empty_list_returns_plain_llm() -> None:
    settings = replace(
        _base_settings(),
        llm_fallback_enabled=True,
        llm_fallback_providers=(),
    )
    fake = FakeListChatModel(responses=["hello"])
    with patch.object(provider_module, "create_llm", return_value=fake):
        result = provider_module.create_llm_with_fallback(settings)
    assert result is fake


def test_fallback_chain_invokes_fallback_on_transient_error(caplog) -> None:  # noqa: ANN001
    """Primary raises ConnectionError → fallback is used and success is returned."""
    settings = replace(
        _base_settings(),
        llm_provider="primary_stub",
        llm_fallback_enabled=True,
        llm_fallback_providers=("fallback_stub",),
    )

    primary = RunnableLambda(lambda _x: (_ for _ in ()).throw(ConnectionError("down")))
    fallback = FakeListChatModel(responses=["rescued"])

    def fake_create(s):  # noqa: ANN001, ANN202
        if s.llm_provider == "primary_stub":
            return primary
        return fallback

    import logging
    caplog.set_level(logging.WARNING, logger="src.provider")
    with patch.object(provider_module, "create_llm", side_effect=fake_create):
        chain = provider_module.create_llm_with_fallback(settings)

    # Startup warning about the chain must be emitted.
    assert any("LLM fallback chain is ACTIVE" in r.message for r in caplog.records)

    # Invoking the chain transparently recovers via the fallback.
    result = chain.invoke("hi")
    # FakeListChatModel returns an AIMessage whose content is the response.
    assert getattr(result, "content", result) == "rescued"

    # Trigger-time warning must have fired when the fallback was used.
    assert any("fallback activated" in r.message.lower() for r in caplog.records)


def test_broken_fallback_provider_is_skipped(caplog) -> None:  # noqa: ANN001
    settings = replace(
        _base_settings(),
        llm_provider="primary_stub",
        llm_fallback_enabled=True,
        llm_fallback_providers=("broken", "good"),
    )
    primary = FakeListChatModel(responses=["primary"])
    good_fallback = FakeListChatModel(responses=["good"])

    def fake_create(s):  # noqa: ANN001, ANN202
        if s.llm_provider == "primary_stub":
            return primary
        if s.llm_provider == "broken":
            raise RuntimeError("cannot construct")
        return good_fallback

    import logging
    caplog.set_level(logging.ERROR, logger="src.provider")
    with patch.object(provider_module, "create_llm", side_effect=fake_create):
        chain = provider_module.create_llm_with_fallback(settings)

    assert any("Skipping LLM fallback provider 'broken'" in r.message for r in caplog.records)
    # Chain should still be usable (wraps primary + good).
    assert chain is not primary  # with_fallbacks wrapped the result


def test_streaming_pre_stream_failure_engages_fallback() -> None:
    """When primary fails before yielding any tokens, fallback streams cleanly.

    This is the expected happy path: the streaming entry point (e.g. a
    connection refused at request time) raises before any token leaves the
    primary, so ``with_fallbacks`` transparently substitutes the fallback
    and the caller sees exactly one clean stream.
    """
    settings = replace(
        _base_settings(),
        llm_provider="primary_stub",
        llm_fallback_enabled=True,
        llm_fallback_providers=("fallback_stub",),
    )

    # Primary throws on both invoke and stream — simulates full outage.
    primary = RunnableLambda(
        lambda _x: (_ for _ in ()).throw(ConnectionError("primary down"))
    )
    fallback = FakeListChatModel(responses=["rescued"])

    def fake_create(s):  # noqa: ANN001, ANN202
        return primary if s.llm_provider == "primary_stub" else fallback

    with patch.object(provider_module, "create_llm", side_effect=fake_create):
        chain = provider_module.create_llm_with_fallback(settings)

    chunks = list(chain.stream("hi"))
    joined = "".join(getattr(c, "content", str(c)) for c in chunks)
    # The fallback's single response is streamed character-by-character by
    # FakeListChatModel, so the joined output must equal the response exactly
    # — no duplicated tokens from the primary.
    assert joined == "rescued"


def test_streaming_mid_stream_failure_is_not_caught_by_fallback() -> None:
    """Mid-stream failures propagate; fallback is not engaged.

    This documents a real limitation of ``RunnableWithFallbacks``: it only
    catches exceptions raised when the stream is OPENED, not exceptions
    raised DURING iteration. If the primary yields some tokens and then
    the connection dies, the caller receives those partial tokens followed
    by the original exception — NOT a seamless switch to the fallback.

    This guards against silently relying on fallback to cover mid-stream
    outages; it cannot.
    """
    from langchain_core.messages import AIMessageChunk
    from langchain_core.runnables import Runnable

    class PartialThenFail(Runnable):
        def invoke(self, input, config=None, **kwargs):  # noqa: ANN001, A002
            raise ConnectionError("primary has no invoke")

        def stream(self, input, config=None, **kwargs):  # noqa: ANN001, A002
            yield AIMessageChunk(content="partial-")
            raise ConnectionError("mid-stream outage")

    settings = replace(
        _base_settings(),
        llm_provider="primary_stub",
        llm_fallback_enabled=True,
        llm_fallback_providers=("fallback_stub",),
    )
    primary = PartialThenFail()
    fallback = FakeListChatModel(responses=["rescued"])

    def fake_create(s):  # noqa: ANN001, ANN202
        return primary if s.llm_provider == "primary_stub" else fallback

    with patch.object(provider_module, "create_llm", side_effect=fake_create):
        chain = provider_module.create_llm_with_fallback(settings)

    observed: list[str] = []
    with pytest.raises(ConnectionError):
        for chunk in chain.stream("hi"):
            observed.append(chunk.content)

    # Partial token was delivered to the caller before the failure bubbled up.
    assert observed == ["partial-"]


def test_streaming_integration_with_query_router_uses_fallback(monkeypatch) -> None:  # noqa: ANN001
    """End-to-end: QueryRouter.route_stream survives a primary LLM outage.

    Wires a fallback-wrapped LLM into a real QueryRouter and asserts the
    SSE event stream completes with a ``done`` event carrying the fallback's
    answer. This proves the fallback integrates with the generation node's
    downstream ``StrOutputParser`` and with the router's streaming path.
    """
    from langchain_core.messages import AIMessage
    from langchain_core.output_parsers import StrOutputParser

    from src.agent.intent_classifier import IntentClassifier
    from src.agent.router import QueryRouter
    from src.models import IntentType, QueryResult, DocumentChunk

    # Build the fallback-wrapped LLM.
    settings = replace(
        _base_settings(),
        llm_provider="primary_stub",
        llm_fallback_enabled=True,
        llm_fallback_providers=("fallback_stub",),
    )
    primary = RunnableLambda(
        lambda _x: (_ for _ in ()).throw(ConnectionError("primary down"))
    )
    fallback = RunnableLambda(lambda _x: AIMessage(content="rescued answer"))

    def fake_create(s):  # noqa: ANN001, ANN202
        return primary if s.llm_provider == "primary_stub" else fallback

    with patch.object(provider_module, "create_llm", side_effect=fake_create):
        llm = provider_module.create_llm_with_fallback(settings)

    # Stub the intent classifier so we don't need to drive language detection.
    class _StubClassifier:
        def classify(self, _query: str) -> IntentType:
            return IntentType.FACTUAL

    # Stub retriever + reranker so they return a single fake result.
    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="doc1.pdf",
        text="Sample context.",
        metadata={"page_number": 1, "chunk_index": 0},
    )
    fake_qr = QueryResult(chunk=chunk, score=0.9, source="dense")

    class _StubHybridResult:
        def __init__(self):  # noqa: ANN204
            self.dense_results = [fake_qr]
            self.sparse_results = [fake_qr]
            self.fused_results = [fake_qr]

    class _StubHybrid:
        def __init__(self):  # noqa: ANN204
            self.vector_store = _StubVectorStore()
        def search_detailed(self, _q: str, top_k: int = 5) -> _StubHybridResult:
            return _StubHybridResult()

    class _StubVectorStore:
        def list_document_ids(self) -> list[str]:
            return []

    class _StubReranker:
        def rerank(self, _q: str, results, top_k: int = 5):  # noqa: ANN001, ANN201
            return list(results)[:top_k]

    router = QueryRouter(
        intent_classifier=_StubClassifier(),
        hybrid_retriever=_StubHybrid(),
        reranker=_StubReranker(),
        llm_chain=llm | StrOutputParser(),
        translate_query=False,
        document_languages=["Danish"],
    )

    events = list(router.route_stream("Hvor mange feriedage?", top_k=3))
    done = [e for e in events if e["step"] == "done"]
    assert done, f"expected a 'done' event, got steps={[e['step'] for e in events]}"
    assert done[0]["result"]["answer"] == "rescued answer"


def test_fallback_engages_on_sdk_style_exception() -> None:
    """Fallback must engage on arbitrary Exception subclasses.

    Real-world LLM SDK exceptions (openai.RateLimitError, httpx.ConnectError,
    etc.) do not inherit from stdlib ConnectionError / TimeoutError / OSError.
    Using a narrow exception tuple would silently make the fallback chain a
    no-op for these cases. This test simulates one of them with a custom
    Exception subclass that has no relation to ConnectionError.
    """

    class FakeRateLimitError(Exception):
        """Stand-in for openai.RateLimitError / anthropic.APIError."""

    settings = replace(
        _base_settings(),
        llm_provider="primary_stub",
        llm_fallback_enabled=True,
        llm_fallback_providers=("fallback_stub",),
    )

    primary = RunnableLambda(
        lambda _x: (_ for _ in ()).throw(FakeRateLimitError("429 Too Many Requests"))
    )
    fallback = FakeListChatModel(responses=["rescued"])

    def fake_create(s):  # noqa: ANN001, ANN202
        return primary if s.llm_provider == "primary_stub" else fallback

    with patch.object(provider_module, "create_llm", side_effect=fake_create):
        chain = provider_module.create_llm_with_fallback(settings)

    result = chain.invoke("hi")
    assert getattr(result, "content", result) == "rescued"


def test_all_fallbacks_broken_returns_primary_only(caplog) -> None:  # noqa: ANN001
    settings = replace(
        _base_settings(),
        llm_provider="primary_stub",
        llm_fallback_enabled=True,
        llm_fallback_providers=("broken",),
    )
    primary = FakeListChatModel(responses=["primary"])

    def fake_create(s):  # noqa: ANN001, ANN202
        if s.llm_provider == "primary_stub":
            return primary
        raise RuntimeError("nope")

    import logging
    caplog.set_level(logging.WARNING, logger="src.provider")
    with patch.object(provider_module, "create_llm", side_effect=fake_create):
        result = provider_module.create_llm_with_fallback(settings)

    assert result is primary
    assert any(
        "no fallback providers could be constructed" in r.message
        for r in caplog.records
    )
