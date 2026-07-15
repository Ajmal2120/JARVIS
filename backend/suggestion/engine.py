"""
JARVIS 3.0 — Suggestion Engine.

The Suggestion Engine is the first proactive component.  It sits between
the Planner and the future Voice / UI layers:

    Planner -> Suggestion Engine -> Pending Suggestion -> Voice / UI

Its only responsibility: *prepare* a Suggestion when the Planner says
it is OK.  It does NOT:

    - Call any LLM
    - Execute any skill
    - Access memory
    - Mutate Context or Inference
    - Name or reference any LLM provider

Architecture:
    - Subscribes to ``decision`` events on the Runtime EventBus.
    - Holds at most one pending Suggestion at a time.
    - Exposes a minimal public API: get_pending_suggestion, accept,
      decline, expire, clear.
    - Providers are pluggable: add a new provider class, register it
      in the default tuple — no engine changes.

Design rules:
    - Event driven: re-evaluates only on Planner decisions.
    - Thread safe: internal state behind a lock.
    - Idempotent: same decision in the same cycle is a no-op.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from datetime import datetime
from typing import Any, Callable, List, Optional, Tuple, Type

from runtime.runtime import get_runtime
from runtime.events import RuntimeEvent

from .events import (
    SuggestionAccepted,
    SuggestionAsked,
    SuggestionCleared,
    SuggestionCreated,
    SuggestionDeclined,
    SuggestionExpired,
)
from .models import (
    Suggestion,
    SuggestionStatus,
)
from .providers import SuggestionProvider, GeneralDevelopmentProvider

logger = logging.getLogger("jarvis.suggestion")


# ---------------------------------------------------------------------------
# Default providers
# ---------------------------------------------------------------------------

DEFAULT_PROVIDERS: Tuple[Type[SuggestionProvider], ...] = (
    GeneralDevelopmentProvider,
)


# ---------------------------------------------------------------------------
# Optional lookups (passed by main.py / wiring layer)
# ---------------------------------------------------------------------------

#: Optional callable to retrieve the current Context.  Wired at
#: construction time so the engine doesn't pull from the EventBus.
ContextLookup = Optional[Callable[[], Any]]
#: Optional callable for ContextInterpretation.
InterpretationLookup = Callable[[], Any]
#: Optional callable for ranked Hypotheses list.
HypothesesLookup = Callable[[], List[Any]]
#: Optional callable for current Plan.
PlanLookup = Callable[[], Any]


class SuggestionEngine:
    """Runtime service that generates Suggestions when the Planner allows.

    Lifecycle:
        __init__()   -> constructed (idle, not running)
        .start()     -> subscribes to EventBus
        .stop()      -> unsubscribes, clears pending suggestion

    Public API (thread safe, snapshot semantics):
        get_pending_suggestion()  -> Suggestion | None
        accept()                  -> Suggestion | None
        decline()                 -> None
        expire()                  -> None
        clear()                   -> None

    Wiring::

        engine = SuggestionEngine(
            lookup_context=context_engine.get_current_context,
            lookup_interpretation=context_engine.get_current_interpretation,
            lookup_hypotheses=inference_engine.get_current_hypotheses,
            lookup_plan=planner.get_current_plan,
        )
        engine.start()
    """

    name: str = "suggestion_engine"

    def __init__(
        self,
        lookup_context: ContextLookup = None,
        lookup_interpretation: Optional[InterpretationLookup] = None,
        lookup_hypotheses: Optional[HypothesesLookup] = None,
        lookup_plan: Optional[PlanLookup] = None,
        provider_classes: Optional[Tuple[Type[SuggestionProvider], ...]] = None,
        ttl_s: float = 120.0,
    ) -> None:
        """Construct the engine.

        Args:
            lookup_context:        Optional callable returning a Context.
            lookup_interpretation: Optional callable returning a
                                   ContextInterpretation.
            lookup_hypotheses:     Optional callable returning ranked
                                   Hypotheses (list of ``Hypothesis``).
            lookup_plan:           Optional callable returning the
                                   current ``Plan``.
            provider_classes:      Providers to use.  Defaults to
                                   ``DEFAULT_PROVIDERS``.  The engine
                                   iterates providers in order and
                                   takes the first non-None result.
            ttl_s:                 Time-to-live for a pending Suggestion
                                   in seconds.
        """
        self._lookup_context = lookup_context
        self._lookup_interpretation = lookup_interpretation
        self._lookup_hypotheses = lookup_hypotheses
        self._lookup_plan = lookup_plan

        self._provider_classes: Tuple[Type[SuggestionProvider], ...] = (
            provider_classes if provider_classes is not None else DEFAULT_PROVIDERS
        )
        self._ttl_s = max(0.1, float(ttl_s))

        self._lock = threading.Lock()
        self._providers: List[SuggestionProvider] = []
        self._pending: Optional[Suggestion] = None
        self._current_cycle_id: str = ""
        self._unsubscribe: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------------ #
    # RuntimeService interface                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Subscribe to the Runtime EventBus and instantiate providers."""
        runtime = get_runtime()
        self._unsubscribe = runtime.event_bus.subscribe(self._on_event)
        with self._lock:
            self._providers = [cls() for cls in self._provider_classes]
        logger.info(
            "suggestion_engine: started with %d provider(s): %s",
            len(self._providers),
            ", ".join(p.name for p in self._providers),
        )

    def stop(self) -> None:
        """Unsubscribe and clear pending suggestion."""
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        with self._lock:
            self._pending = None
            self._providers.clear()
            self._current_cycle_id = ""
        logger.info("suggestion_engine: stopped")

    # ------------------------------------------------------------------ #
    # EventBus subscriber                                                 #
    # ------------------------------------------------------------------ #

    def _on_event(self, event: RuntimeEvent) -> None:
        """React to Planner decisions.

        Only ``decision`` events (type == "decision") are relevant.
        When the Planner says ASK_PERMISSION, iterate providers to
        build a Suggestion.  Otherwise clear any pending suggestion.
        """
        event_type = getattr(event, "type", "")
        if event_type != "decision":
            return

        cycle_id = getattr(event, "cycle_id", "") or ""
        decision = getattr(event, "decision", "") or ""

        if decision == "ASK_PERMISSION":
            self._try_generate(cycle_id)
        else:
            self._clear_pending(cycle_id, reason="planner_decision_changed")

    def _try_generate(self, cycle_id: str) -> None:
        """Run providers in order; store the first non-None result."""
        with self._lock:
            if self._pending is not None:
                # Already have a pending suggestion — don't replace.
                return
            providers = list(self._providers)
            context = self._safe(self._lookup_context)
            interpretation = self._safe(self._lookup_interpretation)
            hypotheses = list(self._safe(self._lookup_hypotheses) or [])
            plan = self._safe(self._lookup_plan)

        suggestion: Optional[Suggestion] = None
        for provider in providers:
            try:
                suggestion = provider.generate(
                    context, interpretation, hypotheses, plan
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "provider %s raised: %s", provider.name, exc
                )
                continue
            if suggestion is not None:
                break

        if suggestion is None:
            return

        # Stamp the suggestion with the originating cycle_id.
        suggestion = replace(suggestion, cycle_id=cycle_id)

        with self._lock:
            if self._pending is not None:
                # Someone beat us to it.
                return
            self._pending = suggestion
            self._current_cycle_id = cycle_id
            old_id = suggestion.id
            provider_name = suggestion.provider
            confidence = suggestion.confidence

        self._publish_lifecycle(SuggestionCreated(
            suggestion_id=old_id,
            provider=provider_name,
            title=suggestion.title,
            confidence=confidence,
            cycle_id=cycle_id,
            timestamp=time.time(),
        ))
        logger.debug(
            "suggestion_engine: created %s via %s",
            old_id,
            provider_name,
        )

    @staticmethod
    def _safe(fn: Optional[Callable[[], Any]]) -> Any:
        """Call fn, swallowing exceptions. Returns None if fn is None."""
        if fn is None:
            return None
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            logger.debug("suggestion_engine: lookup failed: %s", exc)
            return None

    def _publish_lifecycle(self, event: RuntimeEvent) -> None:
        """Publish a lifecycle event to the EventBus. Errors are silent."""
        try:
            runtime = get_runtime()
            runtime.event_bus.publish(event)
        except Exception:
            pass

    def _clear_pending(
        self,
        cycle_id: str = "",
        reason: str = "planner_decision_changed",
    ) -> None:
        """Remove the pending suggestion without raising an event.

        Used internally when the Planner's decision no longer allows
        a Suggestion (e.g. changed from ASK_PERMISSION to WAIT/IGNORE).
        """
        with self._lock:
            self._pending = None
            self._current_cycle_id = ""

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_pending_suggestion(self) -> Optional[Suggestion]:
        """Return the current pending Suggestion, or None.

        Returns a defensive copy so callers can't mutate internal
        state. Thread safe.
        """
        with self._lock:
            if self._pending is None:
                return None
            return replace(self._pending)

    def ask(self) -> Optional[Suggestion]:
        """Mark the pending Suggestion as ASKED and return it.

        This is the moment JARVIS would speak the message aloud or
        render it in the UI.  Returns None if no Suggestion is
        pending.
        """
        with self._lock:
            if self._pending is None:
                return None
            asked = self._pending.with_status(SuggestionStatus.ASKED)
            self._pending = asked
            cycle_id = self._current_cycle_id
            snapshot = replace(asked)

        self._publish_lifecycle(SuggestionAsked(
            suggestion_id=snapshot.id,
            message=snapshot.message,
            cycle_id=cycle_id,
            timestamp=time.time(),
        ))
        return snapshot

    def accept(self) -> Optional[Suggestion]:
        """Transition the pending Suggestion to ACCEPTED.

        Returns the Suggestion so the caller can hand its
        ``prepared_request`` to the LLM layer.  Returns None if
        there is no pending Suggestion.
        """
        with self._lock:
            if self._pending is None:
                return None
            accepted = self._pending.with_status(SuggestionStatus.ACCEPTED)
            self._pending = None
            cycle_id = self._current_cycle_id
            self._current_cycle_id = ""

        self._publish_lifecycle(SuggestionAccepted(
            suggestion_id=accepted.id,
            provider=accepted.provider,
            cycle_id=cycle_id,
            timestamp=time.time(),
        ))
        logger.info(
            "suggestion_engine: accepted %s from %s",
            accepted.id,
            accepted.provider,
        )
        return accepted

    def decline(self) -> None:
        """Transition the pending Suggestion to DECLINED.

        No-op if there is no pending Suggestion.
        """
        with self._lock:
            if self._pending is None:
                return
            declined = self._pending.with_status(SuggestionStatus.DECLINED)
            cycle_id = self._current_cycle_id
            self._pending = None
            self._current_cycle_id = ""

        self._publish_lifecycle(SuggestionDeclined(
            suggestion_id=declined.id,
            reason="user_declined",
            cycle_id=cycle_id,
            timestamp=time.time(),
        ))
        logger.info("suggestion_engine: declined %s", declined.id)

    def expire(self) -> None:
        """Expire the pending Suggestion if its TTL has elapsed.

        Safe to call frequently — no-op if there is no pending
        Suggestion or if the Suggestion is still fresh.
        """
        with self._lock:
            if self._pending is None:
                return
            age_s = (datetime.now() - self._pending.created_at).total_seconds()
            if age_s < self._ttl_s:
                return
            # Snapshot what we need, then release the lock.
            cycle_id = self._current_cycle_id
            old_id = self._pending.id
            self._pending = None
            self._current_cycle_id = ""

        self._publish_lifecycle(SuggestionExpired(
            suggestion_id=old_id,
            age_s=age_s,
            cycle_id=cycle_id,
            timestamp=time.time(),
        ))
        logger.info("suggestion_engine: expired %s (age=%.1fs)", old_id, age_s)

    def clear(self) -> None:
        """Remove the pending Suggestion without a lifecycle event.

        Used during shutdown or when a new decision supersedes the
        current suggestion.
        """
        with self._lock:
            self._pending = None
            self._current_cycle_id = ""

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #

    @property
    def has_pending(self) -> bool:
        """True when a Suggestion is waiting for user response."""
        with self._lock:
            return self._pending is not None

    @property
    def pending_confidence(self) -> float:
        """Confidence of the pending Suggestion, or 0.0."""
        with self._lock:
            return self._pending.confidence if self._pending is not None else 0.0

    @property
    def current_cycle_id(self) -> str:
        """The cycle_id of the currently-pending Suggestion, or empty."""
        with self._lock:
            return self._current_cycle_id