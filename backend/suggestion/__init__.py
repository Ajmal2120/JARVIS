"""
JARVIS 3.0 — Suggestion Engine package.

The Suggestion Engine is the first proactive component.  It sits between
the Planner and the future Voice / UI layers:

    Planner -> Suggestion Engine -> Pending Suggestion -> Voice / UI

It never names or calls any LLM.  It only prepares structured
``PreparedRequest`` objects that a future LLM executor layer can
pick up.

Public surface:
    SuggestionEngine           — RuntimeService implementation
    SuggestionProvider         — Abstract base class for providers
    GeneralDevelopmentProvider — First built-in provider (dev context)
    CooldownManager            — Tracks suggestion cooldown periods
    ActivityMonitor            — Tracks user input activity
    SuggestionMessenger        — Builds contextual messages and reasons
    SuggestionMetricsTracker   — Session-level suggestion counters
    Suggestion                 — Dataclass for one Suggestion
    PreparedRequest            — Structured payload for future LLM execution
    SuggestionStatus           — Enum: PENDING / ASKED / ACCEPTED / DECLINED / EXPIRED
"""
from __future__ import annotations

from .activity import ActivityMonitor
from .cooldown import CooldownManager
from .engine import SuggestionEngine
from .events import (
    CooldownExpired,
    CooldownStarted,
    SuggestionAccepted,
    SuggestionAsked,
    SuggestionCleared,
    SuggestionCreated,
    SuggestionDeclined,
    SuggestionDelayed,
    SuggestionDisplayed,
    SuggestionExpired,
    SuggestionSuppressed,
)
from .messenger import build_message, build_reason, build_suggestion_text
from .metrics import SuggestionMetricsTracker
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
    # Engine
    "SuggestionEngine",
    # Providers
    "SuggestionProvider",
    "GeneralDevelopmentProvider",
    # Services
    "CooldownManager",
    "ActivityMonitor",
    "SuggestionMetricsTracker",
    # Messenger
    "build_message",
    "build_reason",
    "build_suggestion_text",
    # Models
    "Suggestion",
    "PreparedRequest",
    "SuggestionStatus",
    # Events
    "SuggestionCreated",
    "SuggestionAsked",
    "SuggestionAccepted",
    "SuggestionDeclined",
    "SuggestionExpired",
    "SuggestionCleared",
    "SuggestionSuppressed",
    "SuggestionDelayed",
    "SuggestionDisplayed",
    "CooldownStarted",
    "CooldownExpired",
]
