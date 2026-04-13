"""Token counting and budget tracking for LLM prompts.

Stage 1 (current): measure-only — counts tokens at known prompt injection
points and logs them. No truncation is performed.

Token counts are estimates: tiktoken's cl100k tokenizer is used as a
provider-agnostic baseline, multiplied by a safety factor because non-OpenAI
multilingual tokenizers (Llama, Gemma, Mistral) typically tokenize Danish /
mixed-language text 20-40% more aggressively than cl100k.

Provider-specific tokenizers (Ollama's /api/tokenize, HuggingFace AutoTokenizer)
are intentionally not used here to keep this module dependency-free and
process-local. When real usage data exposes the gap, swap in a
provider-aware backend.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Conservative scaling: cl100k under-counts multilingual / Danish text.
# 1.5× keeps us on the safe side for budget decisions.
_DEFAULT_SAFETY_FACTOR = 1.5

# Fallback when tiktoken is unavailable: ~4 characters per token is the
# common rule of thumb for English; multiplied by safety factor it's
# usable as a coarse upper bound.
_CHARS_PER_TOKEN_FALLBACK = 4

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001 — any tiktoken failure → heuristic
    _ENCODER = None
    logger.warning("tiktoken unavailable; falling back to character-based token estimation")


def count_tokens(text: str, *, safety_factor: float = _DEFAULT_SAFETY_FACTOR) -> int:
    """Estimate the token count of ``text``.

    Args:
        text: Text to measure. Empty / None-ish input returns 0.
        safety_factor: Multiplier applied to the raw count to compensate
            for non-OpenAI tokenizers being more aggressive on multilingual
            text. Defaults to 1.5×.

    Returns:
        Estimated token count, rounded up to the nearest int.
    """
    if not text:
        return 0
    if _ENCODER is not None:
        raw = len(_ENCODER.encode(text, disallowed_special=()))
    else:
        raw = max(1, len(text) // _CHARS_PER_TOKEN_FALLBACK)
    return int(raw * safety_factor + 0.5)


def measure(
    prompt_name: str,
    text: str,
    *,
    enabled: bool = True,
    safety_factor: float = _DEFAULT_SAFETY_FACTOR,
) -> int:
    """Count tokens for ``text`` and log the result.

    Args:
        prompt_name: Logical name of the prompt being measured (used in
            log lines so different injection points are easy to grep).
        text: The fully-rendered prompt string.
        enabled: When False, returns 0 immediately and logs nothing —
            lets callers gate on the ``TOKEN_BUDGET_ENABLED`` flag without
            duplicating the check.
        safety_factor: See :func:`count_tokens`.

    Returns:
        Estimated token count, or 0 when ``enabled`` is False.
    """
    if not enabled:
        return 0
    count = count_tokens(text, safety_factor=safety_factor)
    logger.info("token_budget prompt=%s tokens~=%d", prompt_name, count)
    return count
