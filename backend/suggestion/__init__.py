"""
JARVIS 3.0 — Suggestion Engine package.

The Suggestion Engine is the first proactive component.  It sits between
the Planner and the future Voice / UI layers:

    Planner -> Suggestion Engine -> Pending Suggestion -> Voice / UI

It never names or calls any LLM.  It only prepares structured
``PreparedRequest`` objects that a future LLM executor layer can
pick up.

Public surface:
    SuggestionEngine       — RuntimeService implementation
    SuggestionProvider     — Abstract base class for providers
    GeneralDevelopmentProvider — First built-in provider (dev context)
    Suggestion             — Dataclass for one Suggestion
    PreparedRequest        — Structured payload for future LLM execution
    SuggestionStatus       — Enum: PENDING / ASKED / ACCEPTED / DECLINED / EXPIRED
"""
from __future__ import annotations

from .engine import SuggestionEngine
from .events import (
    SuggestionAccepted,
    SuggestionAsked,
    SuggestionCleared,
    SuggestionCreated,
    SuggestionDeclined,
    SuggestionExpired,
)
from .models import (
    PreparedRequest,
    Suggestion,
    SuggestionStatus,
)
from .providers import (
    GeneralDevelopmentProvider,
    SuggestionProvider,
)

__all__ = [
    "SuggestionEngine",
    "SuggestionProvider",
    "GeneralDevelopmentProvider",
    "Suggestion",
    "PreparedRequest",
    "SuggestionStatus",
    "SuggestionCreated",
    "SuggestionAsked",
    "SuggestionAccepted",
    "SuggestionDeclined",
    "SuggestionExpired",
    "SuggestionCleared",
]
