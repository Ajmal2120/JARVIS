"""
JARVIS 3.0 — Suggestion Engine models.

The Suggestion Engine produces one pending Suggestion at a time when
the Planner asks permission. Each Suggestion carries:

  id              Stable slug
  title            Short human-readable label
  message          What JARVIS would say
  reason           Why this suggestion exists
  provider         Which SuggestionProvider produced it
  confidence       0.0–1.0
  created_at       When it was generated
  status           Lifecycle: PENDING / ASKED / ACCEPTED / DECLINED / EXPIRED
  prepared_request Everything a future LLM executor would need

The engine never names or imports any LLM. It only structures data
so a separate LLM manager can pick it up later.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class SuggestionStatus(str, enum.Enum):
    """Lifecycle states for a Suggestion.

    State transitions:

        PENDING  --ask_user()-->  ASKED
        ASKED    --accept()----->  ACCEPTED
        ASKED    --decline()---->  DECLINED
        ASKED    --expire()----->  EXPIRED  (timeout)

    The engine guarantees PENDING is empty before creating a new
    Suggestion — there is at most ONE pending Suggestion at a time.
    """

    PENDING = "pending"
    ASKED = "asked"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------
# PreparedRequest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreparedRequest:
    """A self-contained bundle the future LLM executor consumes.

    This object is the *only* thing accept() returns. It contains
    everything an LLM layer would need — context, interpretation,
    hypotheses, plan, and provider-specific metadata — without naming
    or depending on any particular LLM provider.

    The structure is intentionally opaque (a nested dict) so a future
    Schema for an LLM can be added without changing this class.
    """

    context: Optional[Any] = None           # Context snapshot
    interpretation: Optional[Any] = None    # ContextInterpretation
    hypotheses: Tuple[Any, ...] = ()        # Ranked Hypotheses
    plan: Optional[Any] = None              # Plan
    provider_metadata: Dict[str, Any] = field(default_factory=dict)
    request_type: str = ""                  # e.g. "development_context"
    suggested_action: str = ""              # e.g. "ask_for_focus_topic"
    created_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Suggestion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Suggestion:
    """One proactive Suggestion JARVIS is holding for the user.

    Frozen dataclass: state transitions return a new instance via
    ``.with_status()`` to keep the lifecycle explicit and auditable.
    """

    id: str
    title: str
    message: str
    reason: str
    provider: str
    confidence: float
    created_at: datetime
    status: SuggestionStatus = SuggestionStatus.PENDING
    cycle_id: str = ""
    prepared_request: Optional[PreparedRequest] = None
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Suggestion.id must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Suggestion.confidence must be in [0,1], got {self.confidence}"
            )

    # ------------------------------------------------------------------ #
    # Lifecycle helpers                                                    #
    # ------------------------------------------------------------------ #

    def with_status(
        self,
        status: SuggestionStatus,
        now: Optional[datetime] = None,
    ) -> "Suggestion":
        """Return a new Suggestion in the given status."""
        if not isinstance(status, SuggestionStatus):
            raise TypeError(
                f"status must be a SuggestionStatus, got {type(status).__name__}"
            )
        return replace(self, status=status, updated_at=(now or datetime.now()))


def new_suggestion_id() -> str:
    """Generate a unique Suggestion id."""
    return str(uuid.uuid4())
