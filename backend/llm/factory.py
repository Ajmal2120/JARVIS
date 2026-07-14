"""
LLM Provider Factory — returns the configured provider based on LLM_PROVIDER env var.

Usage:
    from llm.factory import get_provider
    provider = get_provider()
    response = await provider.chat(messages=[...])
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from config import settings

from llm.provider import LLMProvider

logger = logging.getLogger("jarvis.llm.factory")

# Singleton instance
_provider: Optional[LLMProvider] = None


def _resolve_provider_name() -> str:
    """Read the LLM_PROVIDER env var, defaulting to 'claude'."""
    return os.getenv("LLM_PROVIDER", "claude").strip().lower()


def get_provider() -> LLMProvider:
    """
    Return the configured LLM provider (singleton).

    Provider is selected via the LLM_PROVIDER environment variable:
      LLM_PROVIDER=claude  → ClaudeProvider (default)
      LLM_PROVIDER=gemini  → GeminiProvider
    """
    global _provider
    if _provider is not None:
        return _provider

    provider_name = _resolve_provider_name()
    logger.info(f"Initializing LLM provider: {provider_name}")

    if provider_name == "gemini":
        _load_gemini_provider()
    else:
        # Default to Claude
        _load_claude_provider()

    return _provider


def _load_claude_provider() -> None:
    """Instantiate the Claude provider."""
    from llm.claude_provider import ClaudeProvider

    global _provider
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key or api_key.startswith("sk-ant-your"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Edit backend/.env and add your key."
        )
    _provider = ClaudeProvider(
        api_key=api_key,
        model=settings.CLAUDE_MODEL,
        base_url=settings.ANTHROPIC_BASE_URL or None,
    )
    logger.info(f"Claude provider ready (model={settings.CLAUDE_MODEL})")


def _load_gemini_provider() -> None:
    """Instantiate the Gemini provider."""
    from llm.gemini_provider import GeminiProvider

    global _provider
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Edit backend/.env and add: GEMINI_API_KEY=your-key"
        )
    _provider = GeminiProvider(
        api_key=api_key,
        model=settings.GEMINI_MODEL,
    )
    logger.info(f"Gemini provider ready (model={settings.GEMINI_MODEL})")


def reset_provider() -> None:
    """Clear the cached provider (useful for testing or reconfiguration)."""
    global _provider
    _provider = None
