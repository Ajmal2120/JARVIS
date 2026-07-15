"""
JARVIS 3.0 — Inference Engine.

The first cognitive component.  It produces a ranked list of Hypotheses
about what the user is most likely trying to accomplish.

The engine:
    - Subscribes to Context Engine events on the Runtime EventBus.
    - Pulls Context and ContextInterpretation via callbacks at evaluation time.
    - Runs each registered InferenceRule.
    - Merges, deduplicates, and sorts the resulting Hypotheses.
    - Exposes get_current_hypotheses() for downstream layers.

The engine does NOT:
    - Call the LLM.
    - Execute skills.
    - Persist Memory.
    - Make suggestions.
    - Decide actions.

It only infers.

Architecture:
    InferenceEngine
        ↓
    InferenceRule (interface)
        ↓
    SoftwareDevelopmentRule (and future rules)
        ↓
    Hypothesis list

Adding a new rule requires creating a single file in
backend/inference/rules/ and adding the class to DEFAULT_RULES.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Type

from runtime.runtime import Runtime, get_runtime
from runtime.events import RuntimeEvent

from context import Context, ContextInterpretation
from .models import Hypothesis
from .base import InferenceRule
from .rules.software_development import SoftwareDevelopmentRule

logger = logging.getLogger("jarvis.inference.engine")


# ---------------------------------------------------------------------------
# Hypothesis cycle event (used by Planner and CognitiveLogger)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HypothesisGenerated(RuntimeEvent):
    """Published after a successful evaluate() cycle.

    Subscribers (Planner, CognitiveLogger) receive one event per cycle.
    ``cycle_id`` lets idempotent consumers avoid re-processing the same
    cycle more than once.  ``top`` is included for convenience so
    subscribers don't need to call back into the engine to read it.
    """

    type: str = "hypothesis_generated"
    cycle_id: str = ""
    ranked: Tuple[Hypothesis, ...] = ()
    top: Optional[Hypothesis] = None
    timestamp: float = 0.0


class InferenceEngine:
    """Runtime service that evaluates registered inference rules.

    Rules are pure plugins.  The engine:
        - Provides them with fresh Context + ContextInterpretation.
        - Collects their Hypothesis or None return.
        - Sorts the result by confidence descending.
    """

    name: str = "inference_engine"

    # Default ruleset.  Adding a new rule: drop its class into this tuple.
    # The engine instantiates each one in start() — no other file changes.
    DEFAULT_RULES: Tuple[Type[InferenceRule], ...] = (
        SoftwareDevelopmentRule,
    )

    def __init__(
        self,
        lookup_context: Callable[[], Context],
        lookup_interpretation: Callable[[], ContextInterpretation],
        rules: Optional[Tuple[Type[InferenceRule], ...]] = None,
    ) -> None:
        self._lookup_context = lookup_context
        self._lookup_interpretation = lookup_interpretation
        self._rule_classes: Tuple[Type[InferenceRule], ...] = (
            rules if rules is not None else self.DEFAULT_RULES
        )

        self._lock = threading.Lock()
        self._rules: List[InferenceRule] = []
        self._last_hypotheses: List[Hypothesis] = []
        self._last_evaluated_at: Optional[datetime] = None
        self._unsubscribe: Optional[Callable[[], None]] = None

    # ── RuntimeService interface ───────────────────────────────────────

    def start(self) -> None:
        # Instantiate rules.
        with self._lock:
            self._rules = [cls() for cls in self._rule_classes]
            logger.info(
                "inference_engine: started with %d rule(s): %s",
                len(self._rules),
                ", ".join(r.name for r in self._rules),
            )

        # Subscribe to the Runtime EventBus.  We don't process the event
        # itself — we only use it as a trigger to re-evaluate.
        runtime = get_runtime()
        self._unsubscribe = runtime.event_bus.subscribe(self._on_event)

    def stop(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        with self._lock:
            self._rules.clear()
        logger.info("inference_engine: stopped")

    # ── Event hook (trigger) ───────────────────────────────────────────

    def _on_event(self, event: RuntimeEvent) -> None:
        # Only re-evaluate when a fresh Context or Interpretation has been
        # produced.  Both are observable via the EventBus.
        snapshot = getattr(event, "snapshot", None)
        if not isinstance(snapshot, dict):
            return

        # Cheap filter: only react to events that carry user-facing state.
        # (WindowChanged, ApplicationChanged, IdleStateChanged, LockStateChanged.)
        if not any(k in snapshot for k in (
            "window_title", "process_name", "is_locked", "idle_seconds"
        )):
            return

        self.evaluate()

    # ── Core evaluation ────────────────────────────────────────────────

    def evaluate(self) -> List[Hypothesis]:
        """Run all rules, merge and sort the resulting hypotheses.

        Returns the new sorted list.  The same list is also cached for
        callers that hit get_current_hypotheses() between evaluations.
        """
        context = self._lookup_context()
        interpretation = self._lookup_interpretation()

        new_hypotheses: List[Hypothesis] = []

        with self._lock:
            for rule in self._rules:
                try:
                    h = rule.evaluate(context, interpretation)
                except Exception as exc:  # noqa: BLE001
                    # A failing rule must never break inference.
                    logger.exception(
                        "inference rule %s raised: %s", rule.name, exc
                    )
                    continue
                if h is not None:
                    new_hypotheses.append(h)

        # Deduplicate by id, keeping the highest-confidence match.
        deduped = self._dedupe(new_hypotheses)

        # Sort by confidence descending.
        deduped.sort(key=lambda hyp: hyp.confidence, reverse=True)

        with self._lock:
            self._last_hypotheses = deduped
            self._last_evaluated_at = datetime.now()

        # Notify downstream layers (Planner, CognitiveLogger).  We do
        # this even when ``deduped`` is empty: an empty ranking is a
        # valid Plan input and the Planner needs to know.
        try:
            runtime = get_runtime()
            runtime.event_bus.publish(HypothesisGenerated(
                cycle_id=str(uuid.uuid4()),
                ranked=tuple(deduped),
                top=(deduped[0] if deduped else None),
                timestamp=time.time(),
            ))
        except RuntimeError:
            # No Runtime singleton (e.g. unit tests).  Quiet skip.
            pass

        if deduped:
            logger.debug(
                "inference_engine: %d hypothesis(es) — top=%s@%.2f",
                len(deduped),
                deduped[0].id,
                deduped[0].confidence,
            )

        return deduped

    # ── Public API ─────────────────────────────────────────────────────

    def get_current_hypotheses(self) -> List[Hypothesis]:
        """Return the most recent hypothesis list.

        Returns a defensive copy so callers can't mutate the cache.
        If evaluate() has never been called, returns [].
        """
        with self._lock:
            return list(self._last_hypotheses)

    @property
    def last_evaluated_at(self) -> Optional[datetime]:
        with self._lock:
            return self._last_evaluated_at

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _dedupe(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Keep the highest-confidence Hypothesis per id."""
        by_id: dict[str, Hypothesis] = {}
        for h in hypotheses:
            existing = by_id.get(h.id)
            if existing is None or h.confidence > existing.confidence:
                by_id[h.id] = h
        return list(by_id.values())