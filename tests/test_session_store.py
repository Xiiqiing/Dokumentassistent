"""Tests for the SQLite-backed session store."""

import os
import tempfile

import pytest

from src.agent.memory import ConversationMemory
from src.agent.session_store import SessionStore
from src.models import DocumentChunk, QueryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: str) -> SessionStore:
    db_path = os.path.join(tmp_path, "test_sessions.db")
    return SessionStore(db_path=db_path)


def _qr(chunk_id: str = "c1", doc_id: str = "doc.pdf", score: float = 0.8) -> QueryResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id, document_id=doc_id, text="sample text",
        metadata={"page_number": 1},
    )
    return QueryResult(chunk=chunk, score=score, source="test")


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------

class TestSessionStoreBasic:
    def test_new_session_returns_empty_memory(self, tmp_path: str) -> None:
        store = _make_store(str(tmp_path))
        mem = store.get_memory("session-1")
        assert mem.is_empty

    def test_save_turn_and_retrieve(self, tmp_path: str) -> None:
        store = _make_store(str(tmp_path))
        store.save_turn("s1", "What is X?", "X is Y.", [_qr()])
        mem = store.get_memory("s1")
        assert not mem.is_empty
        assert mem.last_query() == "What is X?"

    def test_persist_turn_db_only(self, tmp_path: str) -> None:
        store = _make_store(str(tmp_path))
        # First get_memory to populate cache
        mem = store.get_memory("s1")
        # Manually add to in-memory (simulating what router does)
        mem.add_turn("Q1", "A1")
        # Persist to DB only (no duplicate in-memory add)
        store.persist_turn("s1", "Q1", "A1")
        assert len(mem.turns) == 1  # still 1, not 2

    def test_different_sessions_isolated(self, tmp_path: str) -> None:
        store = _make_store(str(tmp_path))
        store.save_turn("s1", "Q1", "A1")
        store.save_turn("s2", "Q2", "A2")
        mem1 = store.get_memory("s1")
        mem2 = store.get_memory("s2")
        assert mem1.last_query() == "Q1"
        assert mem2.last_query() == "Q2"

    def test_clear_session(self, tmp_path: str) -> None:
        store = _make_store(str(tmp_path))
        store.save_turn("s1", "Q1", "A1")
        store.clear_session("s1")
        # After clear, new get_memory should be empty
        mem = store.get_memory("s1")
        assert mem.is_empty


# ---------------------------------------------------------------------------
# Persistence across store instances (simulates server restart)
# ---------------------------------------------------------------------------

class TestSessionStorePersistence:
    def test_survives_restart(self, tmp_path: str) -> None:
        db_path = os.path.join(str(tmp_path), "persist.db")
        store1 = SessionStore(db_path=db_path)
        store1.save_turn("s1", "Q1", "A1", [_qr()])

        # Create a new store (simulates server restart)
        store2 = SessionStore(db_path=db_path)
        mem = store2.get_memory("s1")
        assert not mem.is_empty
        assert mem.last_query() == "Q1"
        assert len(mem.turns) == 1

    def test_multiple_turns_persist(self, tmp_path: str) -> None:
        db_path = os.path.join(str(tmp_path), "multi.db")
        store1 = SessionStore(db_path=db_path)
        store1.save_turn("s1", "Q1", "A1")
        store1.save_turn("s1", "Q2", "A2")

        store2 = SessionStore(db_path=db_path)
        mem = store2.get_memory("s1")
        assert len(mem.turns) == 2
        assert mem.turns[0].query == "Q1"
        assert mem.turns[1].query == "Q2"

    def test_sources_serialization_roundtrip(self, tmp_path: str) -> None:
        db_path = os.path.join(str(tmp_path), "sources.db")
        sources = [_qr(chunk_id="c1", doc_id="policy.pdf", score=0.9)]

        store1 = SessionStore(db_path=db_path)
        store1.save_turn("s1", "Q1", "A1", sources)

        store2 = SessionStore(db_path=db_path)
        mem = store2.get_memory("s1")
        restored_sources = mem.turns[0].sources
        assert len(restored_sources) == 1
        assert restored_sources[0].chunk.document_id == "policy.pdf"
        assert restored_sources[0].score == 0.9
