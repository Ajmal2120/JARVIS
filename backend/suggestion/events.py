"""
JARVIS 3.0 — Suggestion Engine events.

All events are simple frozen dataclasses that subclass
``RuntimeEvent``. They are published on the Runtime EventBus so:

  - The CognitiveLogger can write structured log entries.
  - A future Voice / UI layer can render the message to the user.

The events carry enough information for downstream consumers to act
without having to call back into the Suggestion Engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from runtime.events import RuntimeEvent


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuggestionCreated(RuntimeEvent):
    """Emitted when a new Suggestion is produced.

    Subscribers (CognitiveLogger, future Voice/UI) use this to log the
    creation or start rendering the message.
    """

    type: str = "suggestion_created"
    suggestion_id: str = ""
    provider: str = ""
    title: str = ""
    confidence: float = 0.0
    cycle_id: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class SuggestionAsked(RuntimeEvent):
    """Emitted when the engine has presented the Suggestion to the user.

    This is the moment the engine transitions PENDING -> ASKED. The
    Voice/UI layer announces the message after this event fires.
    """

    type: str = "suggestion_asked"
    suggestion_id: str = ""
    message: str = ""
    cycle_id: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class SuggestionAccepted(RuntimeEvent):
    """Emitted when the user accepts the Suggestion.

    Subscribers (future LLM executor) should fetch the
    ``PreparedRequest`` via ``SuggestionEngine.take_accepted_request()``
    or by calling accept() themselves.
    """

    type: str = "suggestion_accepted"
    suggestion_id: str = ""
    provider: str = ""
    cycle_id: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class SuggestionDeclined(RuntimeEvent):
    """Emitted when the user declines the Suggestion."""

    type: str = "suggestion_declined"
    suggestion_id: str = ""
    reason: str = ""
    cycle_id: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class SuggestionExpired(RuntimeEvent):
    """Emitted when a Suggestion times out without user response."""

    type: str = "suggestion_expired"
    suggestion_id: str = ""
    age_s: float = 0.0
    cycle_id: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class SuggestionCleared(RuntimeEvent):
    """Emitted when the pending Suggestion is cleared without a verdict."""

    type: str = "suggestion_cleared"
    suggestion_id: str = ""
    cycle_id: str = ""
    timestamp: float = 0.0