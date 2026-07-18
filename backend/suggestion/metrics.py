"""
JARVIS 3.0 — Suggestion Metrics.

Tracks session-level counters for suggestion lifecycle events:

    - suggestions_created
    - suggestions_shown
    - accepted
    - declined
    - expired
    - suppressed_cooldown
    - suppressed_duplicate
    - suppressed_other

No persistence.  In-memory only.

Thread-safe.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("jarvis.suggestion")


@dataclass
class SuggestionMetrics:
    """Session-level suggestion counters.

    All counters start at zero and increment monotonically.
    Reset clears all counters.
    """

    suggestions_created: int = 0
    suggestions_shown: int = 0       # count of ask() calls
    accepted: int = 0
    declined: int = 0
    expired: int = 0
    suppressed_cooldown: int = 0
    suppressed_duplicate: int = 0
    suppressed_other: int = 0

    #: Human-readable label for the active cooldown, or "".
    cooldown_reason: str = ""

    def to_dict(self) -> dict:
        """Return all counters as a plain dict."""
        return {
            "suggestions_created": self.suggestions_created,
            "suggestions_shown": self.suggestions_shown,
            "accepted": self.accepted,
            "declined": self.declined,
            "expired": self.expired,
            "suppressed_cooldown": self.suppressed_cooldown,
            "suppressed_duplicate": self.suppressed_duplicate,
            "suppressed_other": self.suppressed_other,
            "cooldown_reason": self.cooldown_reason,
        }

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.suggestions_created = 0
        self.suggestions_shown = 0
        self.accepted = 0
        self.declined = 0
        self.expired = 0
        self.suppressed_cooldown = 0
        self.suppressed_duplicate = 0
        self.suppressed_other = 0
        self.cooldown_reason = ""


class SuggestionMetricsTracker:
    """Thread-safe wrapper around SuggestionMetrics.

    Usage::

        tracker = SuggestionMetricsTracker()

        tracker.on_created()
        tracker.on_shown()
        tracker.on_declined()
        tracker.on_suppressed_cooldown("decline")

        print(tracker.get_session_metrics())
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics = SuggestionMetrics()

    # ------------------------------------------------------------------ #
    # Event recording                                                    #
    # ------------------------------------------------------------------ #

    def on_created(self) -> None:
        with self._lock:
            self._metrics.suggestions_created += 1

    def on_shown(self) -> None:
        with self._lock:
            self._metrics.suggestions_shown += 1

    def on_accepted(self) -> None:
        with self._lock:
            self._metrics.accepted += 1

    def on_declined(self) -> None:
        with self._lock:
            self._metrics.declined += 1

    def on_expired(self) -> None:
        with self._lock:
            self._metrics.expired += 1

    def on_suppressed_cooldown(self, reason: str = "") -> None:
        with self._lock:
            self._metrics.suppressed_cooldown += 1
            if reason:
                self._metrics.cooldown_reason = reason

    def on_suppressed_duplicate(self) -> None:
        with self._lock:
            self._metrics.suppressed_duplicate += 1

    def on_suppressed_other(self) -> None:
        with self._lock:
            self._metrics.suppressed_other += 1

    def set_cooldown_reason(self, reason: str) -> None:
        with self._lock:
            self._metrics.cooldown_reason = reason

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #

    def get_session_metrics(self) -> dict:
        """Return a snapshot of all session metrics."""
        with self._lock:
            return self._metrics.to_dict()

    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            self._metrics.reset()

    # ------------------------------------------------------------------ #
    # Logging                                                             #
    # ------------------------------------------------------------------ #

    def log_summary(self) -> None:
        """Log a one-line summary of session metrics."""
        m = self.get_session_metrics()
        logger.info(
            "session_metrics: created=%d shown=%d accepted=%d declined=%d "
            "expired=%d suppressed_cooldown=%d suppressed_duplicate=%d "
            "suppressed_other=%d",
            m["suggestions_created"],
            m["suggestions_shown"],
            m["accepted"],
            m["declined"],
            m["expired"],
            m["suppressed_cooldown"],
            m["suppressed_duplicate"],
            m["suppressed_other"],
        )
