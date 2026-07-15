"""
JARVIS 3.0 — Planner.

The Planner is the decision component that sits between Inference and
the (future) Suggestion Engine.  Its single responsibility:

    Given the current ranked Hypotheses, decide whether JARVIS 3.0
    should ask the user for permission to act.

The Planner does NOT:
    - poll any source
    - read Observer state directly
    - mutate Context
    - call the LLM
    - execute skills
    - generate suggestions

It does:
    - Subscribe to ``hypothesis_generated`` events on the Runtime
      EventBus (emitted by the Inference Engine).
    - Run a fixed set of deterministic PlanningRules.
    - Pick the verdict with the highest weight × confidence, breaking
      ties by Decision priority (ASK_PERMISSION > WAIT > IGNORE).
    - Expose ``get_current_plan()`` for downstream consumers.

Lifecycle:
    - Implements ``RuntimeService.start()`` / ``stop()``.
    - Thread-safe: all internal state behind a lock.
    - Idempotent: multiple events for the same cycle only update state
      once.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Type

from inference.models import Hypothesis

from runtime.runtime import Runtime, get_runtime
from runtime.events import RuntimeEvent

from .models import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    Decision,
    Plan,
)
from .rules import PlanningRule, PlanningVerdict


logger = logging.getLogger("jarvis.planner")


_DECISION_PRIORITY: Dict[Decision, int] = {
    Decision.ASK_PERMISSION: 3,
    Decision.WAIT: 2,
    Decision.IGNORE: 1,
}


@dataclass(frozen=True)
class PlanProduced(RuntimeEvent):
    """Emitted after every Plan evaluation.

    Subscribers (CognitiveLogger, future Suggestion Engine) use this
    as the natural end-of-pipeline marker for one thought: from
    observation, through context, interpretation, inference, and
    decision.

    The `cycle_id` ties this event back to the originating reasoning
    cycle.
    """

    type: str = "decision"
    cycle_id: str = ""
    decision: str = ""
    reason: str = ""
    confidence: float = 0.0
    hypothesis_ids: Tuple[str, ...] = ()
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Default ruleset (built-in)
# ---------------------------------------------------------------------------

class NoHypothesesRule(PlanningRule):
    """IGNORE when there are no hypotheses at all."""

    name = "no_hypotheses"

    def evaluate(self, context, interpretation, hypotheses):
        if hypotheses:
            return None
        return PlanningVerdict(
            decision=Decision.IGNORE,
            reason="No hypotheses — nothing to act on.",
            confidence=1.0,
            weight=2.0,
        )


class LowConfidenceRule(PlanningRule):
    """WAIT when the top hypothesis is below the confidence threshold."""

    name = "low_confidence"

    def __init__(self, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self._threshold = threshold

    def evaluate(self, context, interpretation, hypotheses):
        if not hypotheses:
            return None
        top = hypotheses[0]
        if top.confidence >= self._threshold:
            return None
        return PlanningVerdict(
            decision=Decision.WAIT,
            reason=(
                f"Top hypothesis confidence {top.confidence:.2f} below "
                f"threshold {self._threshold:.2f}."
            ),
            confidence=1.0 - top.confidence,
            weight=1.5,
        )


class UserIdleRule(PlanningRule):
    """IGNORE when the user is idle."""

    name = "user_idle"

    def evaluate(self, context, interpretation, hypotheses):
        if context.is_idle:
            return PlanningVerdict(
                decision=Decision.IGNORE,
                reason="User is idle — do not interrupt.",
                confidence=1.0,
                weight=3.0,
            )
        return None


class UserLockedRule(PlanningRule):
    """IGNORE when the screen is locked."""

    name = "user_locked"

    def evaluate(self, context, interpretation, hypotheses):
        if context.is_locked:
            return PlanningVerdict(
                decision=Decision.IGNORE,
                reason="Screen is locked — no opportunity to assist.",
                confidence=1.0,
                weight=3.0,
            )
        return None


class AppFlickeringRule(PlanningRule):
    """WAIT when the user is switching apps rapidly.

    "Rapid" = more than 5 switches in the last 5 minutes, expressed as
    switches per minute by the Interpreter's ``application_switch_frequency``.
    """

    name = "app_flickering"
    _MAX_SWITCH_RATE = 5.0  # switches/minute

    def evaluate(self, context, interpretation, hypotheses):
        rate = getattr(interpretation, "application_switch_frequency", 0.0) or 0.0
        if rate > self._MAX_SWITCH_RATE:
            return PlanningVerdict(
                decision=Decision.WAIT,
                reason=(
                    f"User is switching apps rapidly ({rate:.1f}/min). "
                    "Wait for focus to settle."
                ),
                confidence=min(1.0, rate / 10.0),
                weight=2.0,
            )
        return None


class OnlyAskIfStrong(PlanningRule):
    """Allow ASK_PERMISSION only when the top hypothesis is strong AND
    context is stable.

    Without this rule, the Planner would never emit ASK_PERMISSION.
    """

    name = "strong_signal_gate"

    def __init__(self, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self._threshold = threshold

    def evaluate(self, context, interpretation, hypotheses):
        if not hypotheses:
            return None
        top = hypotheses[0]
        # Require stable context too — no point ASK_PERMISSION on a flaky read.
        stable = (
            getattr(interpretation, "context_stability_s", 0.0) >= 30.0
        )
        if top.confidence >= self._threshold and stable:
            return PlanningVerdict(
                decision=Decision.ASK_PERMISSION,
                reason=(
                    f"Top hypothesis '{top.id}' at confidence "
                    f"{top.confidence:.2f} is strong enough."
                ),
                confidence=top.confidence,
                weight=1.0,
            )
        return None


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """Deterministic decision layer.

    Subscribes to ``hypothesis_generated`` events on the Runtime EventBus.
    Re-evaluates on each event using its registered PlanningRules.
    """

    name: str = "planner"

    #: Built-in rules.  All deterministic, no I/O, no LLM.
    DEFAULT_RULES: Tuple[Type[PlanningRule], ...] = (
        UserLockedRule,
        UserIdleRule,
        AppFlickeringRule,
        NoHypothesesRule,
        LowConfidenceRule,
        OnlyAskIfStrong,
    )

    def __init__(
        self,
        lookup_hypotheses: Optional[Callable[[], List[Hypothesis]]] = None,
        lookup_context: Optional[Callable[[], object]] = None,
        lookup_interpretation: Optional[Callable[[], object]] = None,
        rules: Optional[Tuple[Type[PlanningRule], ...]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._lookup_hypotheses = lookup_hypotheses
        self._lookup_context = lookup_context
        self._lookup_interpretation = lookup_interpretation
        self._confidence_threshold = confidence_threshold
        self._rule_classes: Tuple[Type[PlanningRule], ...] = (
            rules if rules is not None else self.DEFAULT_RULES
        )

        self._lock = threading.Lock()
        self._rules: List[PlanningRule] = []
        self._last_plan: Plan = Plan(
            decision=Decision.WAIT,
            reason="Planner not yet evaluated.",
            confidence=0.0,
        )
        self._unsubscribe: Optional[Callable[[], None]] = None
        self._last_cycle_id: str = ""

    # ------------------------------------------------------------------ #
    # RuntimeService interface                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        with self._lock:
            self._instantiate_rules_locked()
            logger.info(
                "planner: started with %d rule(s): %s",
                len(self._rules),
                ", ".join(r.name for r in self._rules),
            )

        try:
            runtime = get_runtime()
        except RuntimeError:
            logger.warning("planner: no Runtime singleton; not subscribing")
            return
        self._unsubscribe = runtime.event_bus.subscribe(self._on_event)

    def _instantiate_rules(self) -> None:
        """Lazy rule instantiation when start() wasn't called.

        Used by tests that construct a Planner directly.
        """
        with self._lock:
            if self._rules:
                return
            self._instantiate_rules_locked()

    def _instantiate_rules_locked(self) -> None:
        """Instantiate rule classes. Caller must hold the lock."""
        self._rules = [cls() for cls in self._rule_classes]
        for rule in self._rules:
            if hasattr(rule, "_threshold") and isinstance(
                getattr(rule, "_threshold", None), (int, float),
            ):
                try:
                    rule._threshold = self._confidence_threshold
                except Exception:
                    pass

    def stop(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        with self._lock:
            self._rules.clear()
        logger.info("planner: stopped")

    # ------------------------------------------------------------------ #
    # Event hook (trigger)                                                #
    # ------------------------------------------------------------------ #

    def _on_event(self, event: RuntimeEvent) -> None:
        """Trigger a re-evaluation on HypothesisGenerated events."""
        event_type = getattr(event, "type", None) or ""
        cycle_id = getattr(event, "cycle_id", None) or ""

        if cycle_id and self._is_cycle_processed(cycle_id):
            return

        if event_type not in {"hypothesis_generated", "context_updated",
                               "interpretation_updated"}:
            return

        self._record_cycle(cycle_id or str(uuid.uuid4()))
        self.evaluate()

    def _is_cycle_processed(self, cycle_id: str) -> bool:
        with self._lock:
            return cycle_id == self._last_cycle_id and bool(cycle_id)

    def _record_cycle(self, cycle_id: str) -> None:
        with self._lock:
            self._last_cycle_id = cycle_id

    # ------------------------------------------------------------------ #
    # Core evaluation                                                     #
    # ------------------------------------------------------------------ #

    def evaluate(self) -> Plan:
        """Run all rules and produce a single Plan.

        Lazy-initialises rules if start() hasn't been called (useful in
        unit tests that construct a Planner directly without a Runtime).
        """
        if not self._rules:
            self._instantiate_rules()

        if self._lookup_hypotheses is None or self._lookup_context is None \
                or self._lookup_interpretation is None:
            logger.debug("planner: no lookups wired; emitting WAIT")
            plan = Plan(
                decision=Decision.WAIT,
                reason="Planner not yet wired to live services.",
                confidence=0.0,
            )
            with self._lock:
                self._last_plan = plan
            return plan

        try:
            hypotheses: List[Hypothesis] = list(self._lookup_hypotheses())
            context = self._lookup_context()
            interpretation = self._lookup_interpretation()
        except Exception as exc:  # noqa: BLE001
            logger.exception("planner: lookup failed: %s", exc)
            plan = Plan(
                decision=Decision.WAIT,
                reason=f"Lookup failed: {exc}",
                confidence=0.0,
            )
            with self._lock:
                self._last_plan = plan
            return plan

        verdicts: List[Tuple[PlanningVerdict, str]] = []
        with self._lock:
            rules_snapshot = list(self._rules)

        for rule in rules_snapshot:
            try:
                verdict = rule.evaluate(context, interpretation, hypotheses)
            except Exception as exc:  # noqa: BLE001
                logger.exception("planner: rule %s raised: %s", rule.name, exc)
                continue
            if verdict is not None:
                verdicts.append((verdict, rule.name))

        chosen = self._merge_verdicts(verdicts, hypotheses)
        with self._lock:
            self._last_plan = chosen

        # Publish a PlanProduced event so the CognitiveLogger and
        # future Suggestion Engine can observe the decision.  The
        # cycle_id is the one from the most recent triggering event,
        # or a fresh one for direct evaluate() calls.
        try:
            from runtime.runtime import get_runtime
            runtime = get_runtime()
            import time as _time
            runtime.event_bus.publish(PlanProduced(
                cycle_id=self._last_cycle_id or str(uuid.uuid4()),
                decision=chosen.decision.value,
                reason=chosen.reason,
                confidence=chosen.confidence,
                hypothesis_ids=chosen.supporting_hypotheses,
                timestamp=_time.time(),
            ))
        except Exception:
            pass

        logger.debug(
            "planner: produced Plan decision=%s confidence=%.2f reason=%r",
            chosen.decision.value,
            chosen.confidence,
            chosen.reason,
        )
        return chosen

    @staticmethod
    def _merge_verdicts(
        verdicts: List[Tuple[PlanningVerdict, str]],
        hypotheses: List[Hypothesis],
    ) -> Plan:
        """Pick the verdict with the highest weight × confidence.

        Tie-break by Decision priority (ASK_PERMISSION > WAIT > IGNORE).
        """
        if not verdicts:
            if not hypotheses:
                return Plan(
                    decision=Decision.IGNORE,
                    reason="No rules fired and no hypotheses.",
                    confidence=0.0,
                )
            top_id = hypotheses[0].id
            return Plan(
                decision=Decision.WAIT,
                reason="No rules fired; awaiting more signal.",
                confidence=0.0,
                supporting_hypotheses=(top_id,),
            )

        scored = sorted(
            verdicts,
            key=lambda pair: (
                pair[0].weight * pair[0].confidence,
                _DECISION_PRIORITY[pair[0].decision],
            ),
            reverse=True,
        )
        winning_verdict = scored[0][0]

        # Build the Plan with supporting_hypotheses populated from the
        # start so that ASK_PERMISSION's validation in __post_init__
        # doesn't fail before with_hypotheses() runs.
        chosen = Plan(
            decision=winning_verdict.decision,
            reason=winning_verdict.reason,
            confidence=winning_verdict.confidence,
            supporting_hypotheses=(
                tuple(h.id for h in hypotheses) if hypotheses else ()
            ),
            supporting_evidence=(
                tuple(ev for h in hypotheses for ev in h.evidence)
                if hypotheses else ()
            ),
        )
        return chosen

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_current_plan(self) -> Plan:
        """Return the most recently produced Plan (fresh frozen copy)."""
        with self._lock:
            current = self._last_plan
            return Plan(
                plan_id=current.plan_id,
                timestamp=current.timestamp,
                decision=current.decision,
                reason=current.reason,
                confidence=current.confidence,
                supporting_hypotheses=current.supporting_hypotheses,
                supporting_evidence=current.supporting_evidence,
                inference_cycle_id=current.inference_cycle_id,
            )
