"""SQLite-backed session store for per-user conversation memory.

Maps session IDs (UUIDs from browser cookies) to independent
ConversationMemory instances.  Turns are persisted to SQLite so
that conversations survive server restarts.
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

from src.agent.memory import ConversationMemory
from src.models import DocumentChunk, QueryResult

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "./data/sessions.db"
_SESSION_MAX_AGE_SECONDS = 7 * 24 * 3600  # 7 days


class SessionStore:
    """Thread-safe, SQLite-backed store for per-session conversation memory.

    Each session ID maps to its own ConversationMemory.  On first access
    the store loads persisted turns from SQLite; new turns are written
    back immediately.

    Args:
        db_path: Path to the SQLite database file.
        max_age_seconds: Sessions older than this are purged on startup.
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        max_age_seconds: int = _SESSION_MAX_AGE_SECONDS,
    ) -> None:
        self._db_path = db_path
        self._max_age = max_age_seconds
        self._lock = threading.Lock()
        self._cache: dict[str, ConversationMemory] = {}

        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        self._purge_old_sessions()

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Create a new connection (one per call — safe for threads)."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        conn = self._get_conn()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL REFERENCES sessions(session_id),
                    query       TEXT NOT NULL,
                    answer      TEXT NOT NULL,
                    sources     TEXT NOT NULL DEFAULT '[]',
                    created_at  REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_turns_session
                    ON turns(session_id);
                """
            )
            conn.commit()
        finally:
            conn.close()
        logger.info("Session store initialised at %s", self._db_path)

    def _purge_old_sessions(self) -> None:
        """Delete sessions older than *max_age* seconds."""
        cutoff = time.time() - self._max_age
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT session_id FROM sessions WHERE updated_at < ?",
                (cutoff,),
            )
            old_ids = [row[0] for row in cursor.fetchall()]
            if old_ids:
                placeholders = ",".join("?" for _ in old_ids)
                conn.execute(
                    f"DELETE FROM turns WHERE session_id IN ({placeholders})",
                    old_ids,
                )
                conn.execute(
                    f"DELETE FROM sessions WHERE session_id IN ({placeholders})",
                    old_ids,
                )
                conn.commit()
                logger.info("Purged %d expired sessions", len(old_ids))
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_memory(self, session_id: str) -> ConversationMemory:
        """Return the ConversationMemory for *session_id*, loading from DB if needed.

        Args:
            session_id: Unique session identifier (UUID string).

        Returns:
            The session's ConversationMemory instance.
        """
        with self._lock:
            if session_id in self._cache:
                return self._cache[session_id]

        memory = self._load_from_db(session_id)
        with self._lock:
            # Double-check after loading
            if session_id not in self._cache:
                self._cache[session_id] = memory
            return self._cache[session_id]

    def save_turn(
        self,
        session_id: str,
        query: str,
        answer: str,
        sources: list[QueryResult] | None = None,
    ) -> None:
        """Persist a new turn to SQLite and update in-memory cache.

        Args:
            session_id: Session identifier.
            query: User question.
            answer: Assistant answer.
            sources: Optional retrieval sources.
        """
        sources = sources or []
        self._persist_turn_to_db(session_id, query, answer, sources)

        # Update in-memory cache
        memory = self.get_memory(session_id)
        memory.add_turn(query, answer, sources)
        logger.debug("Saved turn for session %s", session_id[:8])

    def persist_turn(
        self,
        session_id: str,
        query: str,
        answer: str,
        sources: list[QueryResult] | None = None,
    ) -> None:
        """Persist a turn to SQLite only (without updating in-memory cache).

        Use this when the in-memory ConversationMemory has already been
        updated (e.g. by PlanAndExecuteRouter.route() internally).

        Args:
            session_id: Session identifier.
            query: User question.
            answer: Assistant answer.
            sources: Optional retrieval sources.
        """
        self._persist_turn_to_db(session_id, query, answer, sources or [])
        logger.debug("Persisted turn to DB for session %s", session_id[:8])

    def _persist_turn_to_db(
        self,
        session_id: str,
        query: str,
        answer: str,
        sources: list[QueryResult],
    ) -> None:
        """Write a single turn row to SQLite."""
        now = time.time()
        sources_json = json.dumps(
            [_query_result_to_dict(r) for r in sources],
            ensure_ascii=False,
        )

        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO sessions (session_id, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (session_id, now, now),
            )
            conn.execute(
                "INSERT INTO turns (session_id, query, answer, sources, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, query, answer, sources_json, now),
            )
            conn.commit()
        finally:
            conn.close()

    def clear_session(self, session_id: str) -> None:
        """Remove all data for a session.

        Args:
            session_id: Session to clear.
        """
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()

        with self._lock:
            if session_id in self._cache:
                self._cache[session_id].clear()
                del self._cache[session_id]
        logger.info("Cleared session %s", session_id[:8])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_from_db(self, session_id: str) -> ConversationMemory:
        """Load turns from SQLite into a fresh ConversationMemory."""
        memory = ConversationMemory()
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT query, answer, sources FROM turns "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
            for query, answer, sources_json in cursor.fetchall():
                sources = _parse_sources_json(sources_json)
                memory.add_turn(query, answer, sources)
        finally:
            conn.close()

        if not memory.is_empty:
            logger.debug(
                "Loaded %d turns for session %s", len(memory.turns), session_id[:8]
            )
        return memory


# ------------------------------------------------------------------
# Serialisation helpers
# ------------------------------------------------------------------

def _query_result_to_dict(r: QueryResult) -> dict:
    """Serialise a QueryResult to a JSON-safe dict."""
    return {
        "chunk_id": r.chunk.chunk_id,
        "document_id": r.chunk.document_id,
        "text": r.chunk.text,
        "metadata": r.chunk.metadata,
        "score": r.score,
        "source": r.source,
    }


def _parse_sources_json(raw: str) -> list[QueryResult]:
    """Deserialise a JSON string back into QueryResult objects."""
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []

    results: list[QueryResult] = []
    for item in items:
        chunk = DocumentChunk(
            chunk_id=item.get("chunk_id", ""),
            document_id=item.get("document_id", ""),
            text=item.get("text", ""),
            metadata=item.get("metadata", {}),
        )
        results.append(
            QueryResult(
                chunk=chunk,
                score=item.get("score", 0.0),
                source=item.get("source", ""),
            )
        )
    return results
