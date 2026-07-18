"""
JARVIS 3.0 — Cognitive Logger.

A structured, optional logging layer dedicated to JARVIS's reasoning process.
It subscribes to the Runtime EventBus and writes JSON-structured entries to
``logs/cognitive.log``, giving developers a complete picture of how JARVIS
observes, interprets, and infers.

When ENABLE_COGNITIVE_LOGGING is False (the default for production), every
method is a short-circuit no-op — no formatting, no I/O, no overhead.

Design:
    - Passively subscribes to the Runtime EventBus for Observer events.
    - On each Observer change, pulls current state from Context,
      Interpreter, and Inference via their public APIs.
    - Each log entry is one JSON line: { timestamp, component, event, details }.
    - Thread-safe: Python's logging module serialises writes internally.
    - No duplicate handlers: guards against the handler being added twice.
    - Extensible: new components (Planner, Suggestion, Memory) use the same
      call pattern without touching this file.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


logger = logging.getLogger("jarvis.observability")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

try:
    from config import settings as _settings
    ENABLED: bool = bool(getattr(_settings, "ENABLE_COGNITIVE_LOGGING", True))
    _log_dir: Path = getattr(
        _settings, "BACKEND_DIR", Path(__file__).resolve().parent.parent
    )
except Exception:
    ENABLED = True
    _log_dir = Path(__file__).resolve().parent.parent

LOG_PATH = _log_dir / "logs" / "cognitive.log"


# ---------------------------------------------------------------------------
# Structured handler
# ---------------------------------------------------------------------------

_COGNITIVE_LOGGER_NAME = "jarvis.cognitive"
_cognitive_handler: Optional[logging.FileHandler] = None
_cognitive_handler_lock = threading.Lock()


def _ensure_handler() -> None:
    """Attach a FileHandler to the cognitive logger if not already present.

    Safe to call from any thread.  The lock prevents duplicate handlers when
    multiple services call CognitiveLogger methods concurrently during boot.
    """
    global _cognitive_handler
    if _cognitive_handler is not None:
        return
    with _cognitive_handler_lock:
        if _cognitive_handler is not None:
            return
        # Make sure the logs directory exists.
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(LOG_PATH), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        cog_logger = logging.getLogger(_COGNITIVE_LOGGER_NAME)
        cog_logger.addHandler(fh)
        cog_logger.setLevel(logging.DEBUG)
        # Prevent propagation so structured lines don't also hit
        # the console via the root handler.
        cog_logger.propagate = False
        _cognitive_handler = fh


# ---------------------------------------------------------------------------
# CognitiveLogger
# ---------------------------------------------------------------------------

class CognitiveLogger:
    """Writes structured cognitive events to ``logs/cognitive.log``.

    Usage::

        cog = CognitiveLogger()
        cog.hypothesis_generated(
            hypothesis_id="software_development",
            confidence=0.82,
            evidence=["VS Code active", "High interaction"],
        )

    All methods short-circuit immediately when cognitive logging is disabled,
    so callers do not need to check the flag themselves.
    """

    name: str = "cognitive_logger"

    def __init__(self) -> None:
        self._unsubscribe: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------------ #
    # Optional bindings to other services                                  #
    # ------------------------------------------------------------------ #
    #
    # The CognitiveLogger never reaches into other services' internals.
    # Operators may wire lookups for richer logging:
    #
    #     cog.bind(
    #         get_context=context_engine.get_current_context,
    #         get_interpretation=context_engine.get_current_interpretation,
    #         get_hypotheses=inference_engine.get_current_hypotheses,
    #     )
    #
    # These are read on every Observer event.  When unbound, the
    # CognitiveLogger still logs the raw Observer event.

    def bind(
        self,
        get_context: Optional[Callable[[], Any]] = None,
        get_interpretation: Optional[Callable[[], Any]] = None,
        get_hypotheses: Optional[Callable[[], list]] = None,
    ) -> None:
        """Optionally bind read-side callbacks for richer logging.

        None of these are required.  When unset, only Observer events get
        logged in detail; downstream services log their state if bound.
        """
        if get_context is not None:
            self._get_context = get_context
        if get_interpretation is not None:
            self._get_interpretation = get_interpretation
        if get_hypotheses is not None:
            self._get_hypotheses = get_hypotheses

    # ------------------------------------------------------------------ #
    # RuntimeService interface                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Subscribe to the Runtime EventBus."""
        if not ENABLED:
            logger.debug("cognitive_logger: disabled by config")
            return
        # Imported lazily to avoid a circular import at module load time
        # (config.py is read at import of this module).
        from runtime.runtime import get_runtime
        runtime = get_runtime()
        self._unsubscribe = runtime.event_bus.subscribe(self._on_event)
        _ensure_handler()
        logger.info("cognitive_logger: started -> %s", LOG_PATH)

    def stop(self) -> None:
        """Unsubscribe from the EventBus."""
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        logger.info("cognitive_logger: stopped")

    # ------------------------------------------------------------------ #
    # EventBus subscriber                                                 #
    # ------------------------------------------------------------------ #

    def _on_event(self, event: Any) -> None:
        """Route incoming EventBus events to the appropriate log method.

        On Observer events, also pull current state from bound services
        (Context, Interpreter, Inference) so every cognitive step is
        captured in a single log cycle.

        Adding a new event type means adding one entry to the dispatch
        table — no other file changes.
        """
        event_type = getattr(event, "type", None)
        cycle_id = getattr(event, "cycle_id", "") or ""

        # Observer events — also enrich with current Context / Interpretation /
        # Inference state if those services have been bound.
        _OBSERVER_TYPES = {
            "window_changed", "application_changed",
            "idle_state_changed", "lock_state_changed",
        }
        if event_type in _OBSERVER_TYPES:
            self._log_observer(event, event_type, cycle_id)
            self._log_downstream(cycle_id)
            return

        # Consolidated snapshot (from EventCoalescer) — treat as Observer.
        if event_type == "consolidated_snapshot":
            self._log_consolidated(event, cycle_id)
            self._log_downstream(cycle_id)
            return

        # Context events
        if event_type == "context_updated":
            self.context_updated(
                active_application=getattr(event, "active_application", ""),
                active_window=getattr(event, "active_window", ""),
                is_idle=getattr(event, "is_idle", False),
                is_locked=getattr(event, "is_locked", False),
                confidence=getattr(event, "confidence", 0.0),
                cycle_id=cycle_id,
            )
            return

        # Interpreter events
        if event_type == "interpretation_updated":
            self.interpretation_updated(
                interaction_level=getattr(event, "interaction_level", "idle"),
                context_stability=getattr(event, "context_stability", "fluctuating"),
                application_switch_frequency=getattr(event, "application_switch_frequency", 0.0),
                focus_duration_s=getattr(event, "focus_duration_s", 0.0),
                cycle_id=cycle_id,
            )
            return

        # Inference: persistent hypotheses — log as created / updated / expired
        if event_type == "hypothesis_created":
            self.hypothesis_created(
                hypothesis_id=getattr(event, "hypothesis_id", ""),
                confidence=getattr(event, "confidence", 0.0),
                evidence=getattr(event, "evidence", []),
                cycle_id=cycle_id,
            )
            return
        if event_type == "hypothesis_updated":
            self.hypothesis_updated(
                hypothesis_id=getattr(event, "hypothesis_id", ""),
                old_confidence=getattr(event, "old_confidence", 0.0),
                new_confidence=getattr(event, "confidence", 0.0),
                added_evidence=getattr(event, "added_evidence", []),
                cycle_id=cycle_id,
            )
            return
        if event_type == "hypothesis_expired":
            self.hypothesis_expired(
                hypothesis_id=getattr(event, "hypothesis_id", ""),
                age_s=getattr(event, "age_s", 0.0),
                cycle_id=cycle_id,
            )
            return

        # Legacy single-event path (deprecated, kept for back-compat).
        if event_type == "hypothesis_generated":
            self.hypothesis_generated(
                hypothesis_id=getattr(event, "hypothesis_id", ""),
                confidence=getattr(event, "confidence", 0.0),
                evidence=getattr(event, "evidence", []),
                cycle_id=cycle_id,
            )
            return

        # Planner
        if event_type == "decision":
            self.decision(
                decision=getattr(event, "decision", ""),
                reason=getattr(event, "reason", ""),
                confidence=getattr(event, "confidence", 0.0),
                hypothesis_ids=getattr(event, "hypothesis_ids", []),
                cycle_id=cycle_id,
            )
            return

        # Reasoning cycle bookkeeping
        if event_type == "reasoning_cycle_ended":
            self._log("Pipeline", "Reasoning Cycle Ended", {
                "cycle_id": cycle_id,
                "timestamp": getattr(event, "timestamp", 0.0),
            })
            return

        # Suggestion lifecycle
        if event_type == "suggestion_created":
            self.suggestion_event(
                action="Created",
                suggestion_id=getattr(event, "suggestion_id", ""),
                provider=getattr(event, "provider", ""),
                confidence=getattr(event, "confidence", 0.0),
                cycle_id=cycle_id,
            )
            return
        if event_type == "suggestion_asked":
            self.suggestion_event(
                action="Asked",
                suggestion_id=getattr(event, "suggestion_id", ""),
                message=getattr(event, "message", ""),
                cycle_id=cycle_id,
            )
            return
        if event_type == "suggestion_accepted":
            self.suggestion_event(
                action="Accepted",
                suggestion_id=getattr(event, "suggestion_id", ""),
                provider=getattr(event, "provider", ""),
                cycle_id=cycle_id,
            )
            return
        if event_type == "suggestion_declined":
            self.suggestion_event(
                action="Declined",
                suggestion_id=getattr(event, "suggestion_id", ""),
                cycle_id=cycle_id,
            )
            return
        if event_type == "suggestion_expired":
            self.suggestion_event(
                action="Expired",
                suggestion_id=getattr(event, "suggestion_id", ""),
                age_s=getattr(event, "age_s", 0.0),
                cycle_id=cycle_id,
            )
            return

        # Sprint 2: suppression and delay
        if event_type == "suggestion_suppressed":
            self.suggestion_suppressed(
                suggestion_id=getattr(event, "suggestion_id", ""),
                reason=getattr(event, "reason", ""),
                cycle_id=cycle_id,
            )
            return
        if event_type == "suggestion_delayed":
            self.suggestion_delayed(
                suggestion_id=getattr(event, "suggestion_id", ""),
                reason=getattr(event, "reason", ""),
                cycle_id=cycle_id,
            )
            return
        if event_type == "suggestion_displayed":
            self.suggestion_displayed(
                suggestion_id=getattr(event, "suggestion_id", ""),
                message=getattr(event, "message", ""),
                cycle_id=cycle_id,
            )
            return

        # Sprint 2: cooldown events
        if event_type == "cooldown_started":
            self.cooldown_started(
                reason=getattr(event, "reason", ""),
                duration_s=getattr(event, "duration_s", 0.0),
                cycle_id=cycle_id,
            )
            return
        if event_type == "cooldown_expired":
            self.cooldown_expired(
                reason=getattr(event, "reason", ""),
                duration_s=getattr(event, "duration_s", 0.0),
                cycle_id=cycle_id,
            )
            return

    def _log_observer(self, event: Any, event_type: str, cycle_id: str = "") -> None:
        """Log the Observer event itself."""
        if event_type == "window_changed":
            self.window_changed(
                window_title=getattr(event, "window_title", ""),
                process_name=getattr(event, "snapshot", {}).get("process_name", ""),
                confidence=getattr(event, "confidence", 1.0),
                cycle_id=cycle_id,
            )
        elif event_type == "application_changed":
            self.application_changed(
                process_name=getattr(event, "process_name", ""),
                window_title=getattr(event, "snapshot", {}).get("window_title", ""),
                confidence=getattr(event, "confidence", 1.0),
                cycle_id=cycle_id,
            )
        elif event_type == "idle_state_changed":
            self.idle_state_changed(
                is_idle=getattr(event, "is_idle", False),
                idle_seconds=getattr(event, "idle_seconds", 0.0),
                cycle_id=cycle_id,
            )
        elif event_type == "lock_state_changed":
            self.lock_state_changed(
                is_locked=getattr(event, "is_locked", False),
                cycle_id=cycle_id,
            )

    def _log_consolidated(self, event: Any, cycle_id: str) -> None:
        """Log a ConsolidatedSnapshot as a coalesced Observer entry."""
        snapshot = getattr(event, "snapshot", {})
        primary_type = getattr(event, "primary_type", "") or ""
        self._log("Observer", "Coalesced Snapshot", {
            "primary_type": primary_type,
            "event_count": getattr(event, "event_count", 1),
            "window_ms": getattr(event, "window_ms", 150),
            "active_window": snapshot.get("window_title", ""),
            "active_application": snapshot.get("process_name", ""),
            "is_idle": snapshot.get("is_idle", False),
            "is_locked": snapshot.get("is_locked", False),
            "cycle_id": cycle_id,
        })

    def _log_downstream(self, cycle_id: str = "") -> None:
        """Pull state from bound services and log it.

        Called after every Observer event so the cognitive log captures
        the full chain: observation → context → interpretation → inference.
        Missing callbacks are silently skipped.
        """
        ctx_fn = getattr(self, "_get_context", None)
        interp_fn = getattr(self, "_get_interpretation", None)
        hyp_fn = getattr(self, "_get_hypotheses", None)

        # --- Context ---
        if ctx_fn is not None:
            try:
                ctx = ctx_fn()
                self.context_updated(
                    active_application=getattr(ctx, "active_application", ""),
                    active_window=getattr(ctx, "active_window", ""),
                    is_idle=getattr(ctx, "is_idle", False),
                    is_locked=getattr(ctx, "is_locked", False),
                    confidence=getattr(ctx, "confidence", 0.0),
                    cycle_id=cycle_id,
                )
            except Exception:
                pass

        # --- Interpreter ---
        if interp_fn is not None:
            try:
                interp = interp_fn()
                # Use ``.value`` so enum members render as plain strings
                # (e.g. "high") rather than "InteractionLevel.HIGH".
                il = getattr(interp, "interaction_level", None)
                cs = getattr(interp, "context_stability", None)
                self.interpretation_updated(
                    interaction_level=(
                        il.value if hasattr(il, "value") else str(il or "idle")
                    ),
                    context_stability=(
                        cs.value if hasattr(cs, "value") else str(cs or "fluctuating")
                    ),
                    application_switch_frequency=getattr(interp, "application_switch_frequency", 0.0),
                    focus_duration_s=getattr(interp, "current_focus_duration_s", 0.0),
                    cycle_id=cycle_id,
                )
            except Exception:
                pass

        # --- Inference ---
        if hyp_fn is not None:
            try:
                hypotheses = hyp_fn()
                for h in hypotheses:
                    self.hypothesis_generated(
                        hypothesis_id=h.id,
                        confidence=h.confidence,
                        evidence=h.evidence,
                        cycle_id=cycle_id,
                    )
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Observer                                                             #
    # ------------------------------------------------------------------ #

    def window_changed(
        self, window_title: str, process_name: str, confidence: float = 1.0,
        cycle_id: str = "",
    ) -> None:
        """Log a WindowChanged event."""
        self._log("Observer", "WindowChanged", {
            "window_title": window_title,
            "process_name": process_name,
            "confidence": confidence,
            "cycle_id": cycle_id,
        })

    def application_changed(
        self, process_name: str, window_title: str = "", confidence: float = 1.0,
        cycle_id: str = "",
    ) -> None:
        """Log an ApplicationChanged event."""
        self._log("Observer", "ApplicationChanged", {
            "process_name": process_name,
            "window_title": window_title,
            "confidence": confidence,
            "cycle_id": cycle_id,
        })

    def idle_state_changed(self, is_idle: bool, idle_seconds: float = 0.0,
                           cycle_id: str = "") -> None:
        """Log an IdleStateChanged event."""
        self._log("Observer", "IdleStateChanged", {
            "is_idle": is_idle,
            "idle_seconds": idle_seconds,
            "cycle_id": cycle_id,
        })

    def lock_state_changed(self, is_locked: bool, cycle_id: str = "") -> None:
        """Log a LockStateChanged event."""
        self._log("Observer", "LockStateChanged", {
            "is_locked": is_locked,
            "cycle_id": cycle_id,
        })

    # ------------------------------------------------------------------ #
    # Context                                                              #
    # ------------------------------------------------------------------ #

    def context_updated(
        self,
        active_application: str = "",
        active_window: str = "",
        is_idle: bool = False,
        is_locked: bool = False,
        confidence: float = 0.0,
        cycle_id: str = "",
    ) -> None:
        """Log a Context Engine update."""
        self._log("Context", "Context Updated", {
            "active_application": active_application,
            "active_window": active_window,
            "is_idle": is_idle,
            "is_locked": is_locked,
            "confidence": confidence,
            "cycle_id": cycle_id,
        })

    # ------------------------------------------------------------------ #
    # Interpreter                                                          #
    # ------------------------------------------------------------------ #

    def interpretation_updated(
        self,
        interaction_level: str = "idle",
        context_stability: str = "fluctuating",
        application_switch_frequency: float = 0.0,
        focus_duration_s: float = 0.0,
        cycle_id: str = "",
    ) -> None:
        """Log a ContextInterpreter update."""
        self._log("Interpreter", "Interpretation Updated", {
            "interaction_level": interaction_level,
            "context_stability": context_stability,
            "application_switch_frequency": application_switch_frequency,
            "focus_duration_s": focus_duration_s,
            "cycle_id": cycle_id,
        })

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def hypothesis_generated(
        self, hypothesis_id: str, confidence: float, evidence: list[str],
        cycle_id: str = "",
    ) -> None:
        """Log a Hypothesis generated by the Inference Engine (legacy)."""
        self._log("Inference", "Hypothesis Generated", {
            "id": hypothesis_id,
            "confidence": confidence,
            "evidence": evidence,
            "cycle_id": cycle_id,
        })

    def hypothesis_created(
        self, hypothesis_id: str, confidence: float, evidence: list[str],
        cycle_id: str = "",
    ) -> None:
        """Log a brand-new Hypothesis (first appearance in the persistent store)."""
        self._log("Inference", "Hypothesis Created", {
            "id": hypothesis_id,
            "confidence": confidence,
            "evidence": evidence,
            "cycle_id": cycle_id,
        })

    def hypothesis_updated(
        self, hypothesis_id: str, old_confidence: float, new_confidence: float,
        added_evidence: list[str] = (), cycle_id: str = "",
    ) -> None:
        """Log a Hypothesis confidence/evidence update."""
        self._log("Inference", "Hypothesis Updated", {
            "id": hypothesis_id,
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "added_evidence": added_evidence,
            "cycle_id": cycle_id,
        })

    def hypothesis_expired(
        self, hypothesis_id: str, age_s: float = 0.0, cycle_id: str = "",
    ) -> None:
        """Log a Hypothesis that has expired (stale for too long)."""
        self._log("Inference", "Hypothesis Expired", {
            "id": hypothesis_id,
            "age_s": age_s,
            "cycle_id": cycle_id,
        })

    # ------------------------------------------------------------------ #
    # Planner (future extension points)                                    #
    # ------------------------------------------------------------------ #

    def decision(self, decision: str, reason: str = "",
                 confidence: float = 0.0, hypothesis_ids: list[str] = (),
                 cycle_id: str = "") -> None:
        """Log a Planner decision (ASK_PERMISSION / WAIT / IGNORE)."""
        self._log("Planner", "Decision", {
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
            "hypothesis_ids": list(hypothesis_ids),
            "cycle_id": cycle_id,
        })

    # ------------------------------------------------------------------ #
    # Suggestion (future extension points)                                 #
    # ------------------------------------------------------------------ #

    def suggestion_event(
        self,
        action: str = "",
        suggestion_id: str = "",
        provider: str = "",
        confidence: float = 0.0,
        message: str = "",
        age_s: float = 0.0,
        cycle_id: str = "",
    ) -> None:
        """Log a Suggestion event (created / asked / accepted / declined / expired)."""
        details = {
            "suggestion_id": suggestion_id,
            "provider": provider,
            "confidence": confidence,
            "message": message,
            "cycle_id": cycle_id,
        }
        if age_s:
            details["age_s"] = age_s
        self._log("Suggestion", f"Suggestion {action}" if action else "Suggestion", details)

    # ------------------------------------------------------------------ #
    # Sprint 2: suppression, delay, display, cooldown                     #
    # ------------------------------------------------------------------ #

    def suggestion_suppressed(
        self, suggestion_id: str = "", reason: str = "", cycle_id: str = "",
    ) -> None:
        """Log a suppressed suggestion (cooldown / duplicate)."""
        self._log("Suggestion", "Suggestion Suppressed", {
            "suggestion_id": suggestion_id,
            "reason": reason,
            "cycle_id": cycle_id,
        })

    def suggestion_delayed(
        self, suggestion_id: str = "", reason: str = "", cycle_id: str = "",
    ) -> None:
        """Log a delayed suggestion (user activity)."""
        self._log("Suggestion", "Suggestion Delayed", {
            "suggestion_id": suggestion_id,
            "reason": reason,
            "cycle_id": cycle_id,
        })

    def suggestion_displayed(
        self, suggestion_id: str = "", message: str = "", cycle_id: str = "",
    ) -> None:
        """Log a displayed suggestion."""
        self._log("Suggestion", "Suggestion Displayed", {
            "suggestion_id": suggestion_id,
            "message": message,
            "cycle_id": cycle_id,
        })

    def cooldown_started(
        self, reason: str = "", duration_s: float = 0.0, cycle_id: str = "",
    ) -> None:
        """Log a cooldown start."""
        self._log("Suggestion", "Cooldown Started", {
            "reason": reason,
            "duration_s": duration_s,
            "cycle_id": cycle_id,
        })

    def cooldown_expired(
        self, reason: str = "", duration_s: float = 0.0, cycle_id: str = "",
    ) -> None:
        """Log a cooldown expiry."""
        self._log("Suggestion", "Cooldown Expired", {
            "reason": reason,
            "duration_s": duration_s,
            "cycle_id": cycle_id,
        })

    # ------------------------------------------------------------------ #
    # Memory (future extension points)                                     #
    # ------------------------------------------------------------------ #

    def memory_event(self, action: str, key: str = "") -> None:
        """Log a Memory event (stored / retrieved / updated)."""
        self._log("Memory", f"Memory {action}", {
            "key": key,
        })

    # ------------------------------------------------------------------ #
    # Core write                                                           #
    # ------------------------------------------------------------------ #

    def _log(self, component: str, event: str, details: dict[str, Any]) -> None:
        """Write one structured entry.

        Fast-path check before any work — when disabled this returns
        before the dict lookup, timestamp call, or JSON dump.
        """
        if not ENABLED:
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": component,
            "event": event,
            "details": details,
        }

        try:
            cog_logger = logging.getLogger(_COGNITIVE_LOGGER_NAME)
            cog_logger.info(json.dumps(entry, default=str))
        except Exception:
            # The cognitive logger must never crash the caller.
            pass
