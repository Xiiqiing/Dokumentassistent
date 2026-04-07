"""Conversation memory for multi-turn interactions.

Stores message history and retrieved sources across turns so that:
- Follow-up questions can reference prior context ("what about the other one?")
- The planner/synthesizer can see what was already discussed
- Previously retrieved sources are available without re-searching
"""

import logging
from dataclasses import dataclass, field

from src.models import QueryResult

logger = logging.getLogger(__name__)

_MAX_TURNS = 20


@dataclass
class Turn:
    """A single conversation turn.

    Attributes:
        query: The user's question.
        answer: The assistant's response.
        sources: Retrieved sources used to generate the answer.
    """

    query: str
    answer: str
    sources: list[QueryResult] = field(default_factory=list)


class ConversationMemory:
    """Manages multi-turn conversation state.

    Stores a rolling window of recent turns and provides formatted
    context for the planner and synthesizer prompts.
    """

    def __init__(self, max_turns: int = _MAX_TURNS) -> None:
        """Initialize conversation memory.

        Args:
            max_turns: Maximum number of turns to retain.
        """
        self._max_turns = max_turns
        self._turns: list[Turn] = []

    @property
    def turns(self) -> list[Turn]:
        """Return the list of conversation turns (read-only copy)."""
        return list(self._turns)

    @property
    def is_empty(self) -> bool:
        """Return True if no conversation history exists."""
        return len(self._turns) == 0

    def add_turn(self, query: str, answer: str, sources: list[QueryResult] | None = None) -> None:
        """Record a completed conversation turn.

        Args:
            query: The user's question.
            answer: The assistant's response.
            sources: Retrieved sources (optional).
        """
        self._turns.append(Turn(query=query, answer=answer, sources=sources or []))
        if len(self._turns) > self._max_turns:
            removed = self._turns.pop(0)
            logger.debug("Evicted oldest turn: %s", removed.query[:50])
        logger.debug("Memory now has %d turns", len(self._turns))

    def clear(self) -> None:
        """Clear all conversation history."""
        self._turns.clear()
        logger.info("Conversation memory cleared")

    def format_history(self, max_recent: int = 5) -> str:
        """Format recent conversation history for inclusion in prompts.

        Args:
            max_recent: Maximum number of recent turns to include.

        Returns:
            Formatted string of recent Q&A pairs, or empty string if no history.
        """
        if not self._turns:
            return ""

        recent = self._turns[-max_recent:]
        parts: list[str] = []
        for i, turn in enumerate(recent, 1):
            source_note = ""
            if turn.sources:
                doc_ids = sorted({s.chunk.document_id for s in turn.sources})
                source_note = f" [sources: {', '.join(doc_ids)}]"
            parts.append(
                f"Turn {i}:\n"
                f"  User: {turn.query}\n"
                f"  Assistant: {turn.answer[:500]}{source_note}"
            )
        return "\n\n".join(parts)

    def get_prior_sources(self) -> list[QueryResult]:
        """Return all unique sources from prior turns, sorted by score.

        Returns:
            Deduplicated list of QueryResult from all past turns.
        """
        by_id: dict[str, QueryResult] = {}
        for turn in self._turns:
            for r in turn.sources:
                cid = r.chunk.chunk_id
                if cid not in by_id or r.score > by_id[cid].score:
                    by_id[cid] = r
        return sorted(by_id.values(), key=lambda r: r.score, reverse=True)

    def last_query(self) -> str:
        """Return the last user query, or empty string."""
        return self._turns[-1].query if self._turns else ""

    def last_sources(self) -> list[QueryResult]:
        """Return sources from the most recent turn."""
        return self._turns[-1].sources if self._turns else []
