"""
JARVIS 3.0 — LLM Executor models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LLMResponse:
    """The normalized result of one LLM execution."""

    content: str = ""
    provider: str = ""
    model: str = ""
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Any] = None
    success: bool = True
    error: str = ""


@dataclass(frozen=True)
class LLMExecutionResult:
    """Bundles a request and its response for logging/testing."""

    prepared_request: Any = None
    response: LLMResponse = field(default_factory=LLMResponse)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
