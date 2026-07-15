"""
JARVIS 3.0 — LLM Executor package.

The LLM Executor converts a PreparedRequest into a normalized LLM
response using the currently configured LLM provider.

Public surface:
    LLMExecutor      — one-shot executor
    LLMResponse      — normalized response dataclass
    LLMExecutionResult — request/response bundle for callers
"""
from __future__ import annotations

from .executor import LLMExecutor
from .models import LLMExecutionResult, LLMResponse

__all__ = [
    "LLMExecutor",
    "LLMResponse",
    "LLMExecutionResult",
]
