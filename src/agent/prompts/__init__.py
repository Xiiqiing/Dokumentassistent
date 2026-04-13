"""Centralised prompt registry.

All user-visible LLM prompts live in ``*.yaml`` files in this directory.
The registry loads them once at import time and returns raw template
strings that callers format with ``str.format(**kwargs)``.
"""

from src.agent.prompts.registry import PromptRegistry, get_prompt, render_prompt

__all__ = ["PromptRegistry", "get_prompt", "render_prompt"]
