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


@dataclass(frozen=True)
class HypothesisCreated(RuntimeEvent):
    """Published when a Hypothesis first appears in the persistent store."""

    type: str = "hypothesis_created"
    cycle_id: str = ""
    hypothesis_id: str = ""
    confidence: float = 0.0
    evidence: Tuple[str, ...] = ()
    timestamp: float = 0.0


@dataclass(frozen=True)
class HypothesisUpdated(RuntimeEvent):
    """Published when a Hypothesis is refreshed with new evidence/confidence."""

    type: str = "hypothesis_updated"
    cycle_id: str = ""
    hypothesis_id: str = ""
    old_confidence: float = 0.0
    confidence: float = 0.0
    added_evidence: Tuple[str, ...] = ()
    timestamp: float = 0.0


@dataclass(frozen=True)
class HypothesisExpired(RuntimeEvent):
    """Published when a Hypothesis hasn't been refreshed in too long."""

    type: str = "hypothesis_expired"
    cycle_id: str = ""
    hypothesis_id: str = ""
    age_s: float = 0.0
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

        # Persistent store keyed by hypothesis id.  Each cycle we merge
        # new evidence into existing entries rather than discarding.
        # Persistence is gated on `_started` so test cases that
        # construct an InferenceEngine without calling start() get the
        # original "regenerate each cycle" semantics.  Production code
        # always calls start() — see test_inference.py.
        self._persistent: Dict[str, Hypothesis] = {}
        self._started: bool = False

    # ── RuntimeService interface ───────────────────────────────────────

    def start(self) -> None:
        # Instantiate rules.
        with self._lock:
            self._rules = [cls() for cls in self._rule_classes]
            self._started = True
            # Clear persistent store on each start to prevent stale
            # state across restarts.
            self._persistent.clear()
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

        # Persist: each cycle we update existing hypotheses instead of
        # discarding them. Confidence is averaged, evidence is
        # de-duplicated, and metadata (age, update_count) is carried
        # forward. Only when the engine has been started — preserves
        # the old "regenerate each cycle" semantics for unit tests.
        cycle_id = str(uuid.uuid4())
        if self._started:
            now = datetime.now()
            merged: List[Hypothesis] = []
            with self._lock:
                for h in deduped:
                    existing = self._persistent.get(h.id)
                    if existing is None:
                        # First time we see this hypothesis — store as-is.
                        self._persistent[h.id] = h
                        merged.append(h)
                    else:
                        # Refresh: merge the new reading into the existing
                        # entry.
                        updated = existing.with_update(
                            confidence=h.confidence,
                            evidence=h.evidence,
                            cycle_id=cycle_id,
                            now=now,
                        )
                        self._persistent[h.id] = updated
                        merged.append(updated)
            deduped = merged
            # Expire stale hypotheses that were not refreshed this cycle.
            try:
                runtime = get_runtime()
            except RuntimeError:
                runtime = None
            if runtime is not None:
                self._publish_expired(runtime, cycle_id, now)

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
        except RuntimeError:
            runtime = None

        if runtime is not None:
            # Per-hypothesis lifecycle events (only meaningful when
            # persistence is active — we have nothing to "create" if
            # we threw everything away each cycle).
            if self._started:
                self._publish_lifecycle(runtime, cycle_id, deduped)

            runtime.event_bus.publish(HypothesisGenerated(
                cycle_id=cycle_id,
                ranked=tuple(deduped),
                top=(deduped[0] if deduped else None),
                timestamp=time.time(),
            ))

            # End-of-cycle marker.  Lets subscribers know they can
            # safely aggregate one full chain.
            try:
                from runtime.cycle import ReasoningCycleEnded
                runtime.event_bus.publish(ReasoningCycleEnded(
                    cycle_id=cycle_id,
                    timestamp=time.time(),
                ))
            except Exception:
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

    def reset_persistent(self) -> None:
        """Clear the persistent hypothesis store.

        Useful in tests that need a fresh state between evaluate() calls
        without going through the full start()/stop() lifecycle.
        Production code should not call this.
        """
        with self._lock:
            self._persistent.clear()

    # ── Lifecycle publishing ────────────────────────────────────────────

    _MAX_HYPOTHESIS_AGE_S = 300.0  # 5 minutes

    def _publish_lifecycle(
        self,
        runtime: Any,
        cycle_id: str,
        hypotheses: List[Hypothesis],
    ) -> None:
        """Publish HypothesisCreated / HypothesisUpdated events.

        Compares the current cycle's hypotheses to the persistent store:
        - First appearance -> HypothesisCreated
        - Refreshed entry -> HypothesisUpdated (with delta in evidence
          and confidence)
        """
        seen_ids: set[str] = set()
        for h in hypotheses:
            seen_ids.add(h.id)
            existing = self._persistent.get(h.id)
            if existing is None:
                # Shouldn't happen — _publish_lifecycle runs after
                # _last_hypotheses is updated.
                continue
            if h.update_count == 0:
                # Newly created.
                runtime.event_bus.publish(HypothesisCreated(
                    cycle_id=cycle_id,
                    hypothesis_id=h.id,
                    confidence=h.confidence,
                    evidence=tuple(h.evidence),
                    timestamp=time.time(),
                ))
            elif h.update_count > existing.update_count:
                # Compute added evidence.
                old = {ev.strip().lower() for ev in existing.evidence}
                added = tuple(ev for ev in h.evidence if ev.strip().lower() not in old)
                runtime.event_bus.publish(HypothesisUpdated(
                    cycle_id=cycle_id,
                    hypothesis_id=h.id,
                    old_confidence=existing.confidence,
                    confidence=h.confidence,
                    added_evidence=added,
                    timestamp=time.time(),
                ))

    def _publish_expired(
        self,
        runtime: Any,
        cycle_id: str,
        now: datetime,
    ) -> None:
        """Publish HypothesisExpired for stale entries."""
        for hid, h in list(self._persistent.items()):
            age = (now - h.last_updated).total_seconds()
            if age > self._MAX_HYPOTHESIS_AGE_S:
                runtime.event_bus.publish(HypothesisExpired(
                    cycle_id=cycle_id,
                    hypothesis_id=hid,
                    age_s=age,
                    timestamp=time.time(),
                ))
                self._persistent.pop(hid, None)

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