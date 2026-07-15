"""
JARVIS 3.0 — Suggestion Providers.

A Provider is a pluggable strategy that inspects the current cognitive
state (Context, Interpretation, Hypotheses, Plan) and, when it has
something worth saying, produces a ``Suggestion``.

The engine iterates providers in order and takes the first non-None
result.  Providers are pure functions — they do not mutate any input,
they do not write to disk, and they do not call any LLM.

Adding a new provider means:
  1. Subclass ``SuggestionProvider`` (or implement ``generate()``).
  2. Register it in ``SUGGESTION_PROVIDERS`` in the engine config.

No engine code changes are required.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .models import PreparedRequest, Suggestion, SuggestionStatus


logger = logging.getLogger("jarvis.suggestion")


class SuggestionProvider(ABC):
    """Abstract base class for suggestion providers.

    Providers are stateless factories: they receive the full cognitive
    picture and return a Suggestion or None.
    """

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        context: Any,
        interpretation: Any,
        hypotheses: list[Any],
        plan: Any,
    ) -> "Optional[Suggestion]":
        """Return a Suggestion or None if there is nothing to say.

        Args:
            context:        Current ``Context`` snapshot (from ContextEngine).
            interpretation: Current ``ContextInterpretation`` (from
                            ContextInterpreter).
            hypotheses:     Ranked list of ``Hypothesis`` objects (from
                            InferenceEngine).
            plan:           Current ``Plan`` (from Planner).

        Returns:
            A ``Suggestion`` with a prepared_request, or ``None`` if
            this provider has nothing to contribute.
        """


# ---------------------------------------------------------------------------
# Built-in provider
# ---------------------------------------------------------------------------

class GeneralDevelopmentProvider(SuggestionProvider):
    """The first suggestion provider.

    Activates when the Planner says ASK_PERMISSION for a
    software_development hypothesis.  It does NOT inspect code, read
    files, or call any LLM.  It produces a polite, structured request
    for the user.

    The prepared_request contains everything a future LLM executor
    would need to produce a concrete suggestion.
    """

    name: str = "general_development"

    # Minimum confidence a hypothesis must have before we bother the user.
    _MIN_CONFIDENCE: float = 0.5

    def generate(
        self,
        context: Any,
        interpretation: Any,
        hypotheses: list[Any],
        plan: Any,
    ) -> "Optional[Suggestion]":
        from .models import (
            PreparedRequest,
            Suggestion,
            SuggestionStatus,
            new_suggestion_id,
        )
        from planner import Decision

        # Guard: Planner must explicitly allow us to speak.
        if plan is None or not hasattr(plan, "decision"):
            return None
        if plan.decision != Decision.ASK_PERMISSION:
            return None

        # Guard: we only help with software development hypotheses.
        if not hypotheses:
            return None
        top = hypotheses[0]
        if top.id != "software_development":
            return None
        if top.confidence < self._MIN_CONFIDENCE:
            return None

        # Build a minimal, provider-specific message.  This is what
        # JARVIS would say (via Voice or UI) — it does NOT call an LLM.
        focus_topic = (
            getattr(context, "active_window", "your work") or "your work"
        )
        focus_app = (
            getattr(context, "active_application", "your application")
            or "your application"
        )

        title = "Focus support"
        message = (
            f"Sir, I notice you're working on '{focus_topic}' in "
            f"'{focus_app}'. Would you like me to help?"
        )
        reason = (
            f"software_development hypothesis at confidence "
            f"{top.confidence:.0%}; stable context with high interaction "
            f"level."
        )

        # Prepare the structured payload a future LLM executor would
        # consume.  We intentionally avoid importing or referencing any
        # LLM here — the structure is provider-neutral.
        prepared = PreparedRequest(
            context=context,
            interpretation=interpretation,
            hypotheses=tuple(hypotheses),
            plan=plan,
            provider_metadata={
                "focus_topic": focus_topic,
                "focus_app": focus_app,
                "hypothesis_confidence": top.confidence,
                "hypothesis_id": top.id,
                "suggested_topics": [
                    "current_focus",
                    "recent_errors",
                    "context_switch_reason",
                ],
            },
            request_type="development_context_assist",
            suggested_action="ask_for_focus_topic",
        )

        suggestion_id = new_suggestion_id()
        now = datetime.now()

        logger.info(
            "GeneralDevelopmentProvider: created suggestion %r "
            "for %s (confidence=%.2f)",
            suggestion_id,
            focus_app,
            top.confidence,
        )

        return Suggestion(
            id=suggestion_id,
            title=title,
            message=message,
            reason=reason,
            provider=self.name,
            confidence=top.confidence,
            created_at=now,
            updated_at=now,
            status=SuggestionStatus.PENDING,
            prepared_request=prepared,
        )
