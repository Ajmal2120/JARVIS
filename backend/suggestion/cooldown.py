"""
JARVIS 3.0 — Suggestion Cooldown Manager.

Manages per-user cooldown periods so JARVIS does not become annoying.

Rules:
    - After a decline: no suggestions for 10 minutes.
    - After an accept: no suggestions for 20 minutes.
    - Same suggestion text: suppressed for 5 minutes (duplicate window).

No persistence.  In-memory only.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("jarvis.suggestion")


# ---------------------------------------------------------------------------
# Configuration (values come from config.py, with safe defaults)
# ---------------------------------------------------------------------------

def _get_config():
    try:
        from config import settings as _s
        return _s
    except Exception:
        return None


def _cooldown_decline_s() -> int:
    cfg = _get_config()
    return getattr(cfg, "SUGGESTION_COOLDOWN_AFTER_DECLINE_S", 600)


def _cooldown_accept_s() -> int:
    cfg = _get_config()
    return getattr(cfg, "SUGGESTION_COOLDOWN_AFTER_ACCEPT_S", 1200)


def _duplicate_window_s() -> int:
    cfg = _get_config()
    return getattr(cfg, "SUGGESTION_DUPLICATE_WINDOW_S", 300)


# ---------------------------------------------------------------------------
# CooldownManager
# ---------------------------------------------------------------------------

@dataclass
class CooldownState:
    """Tracks the current cooldown state."""
    until: float = 0.0
    active: bool = False
    reason: str = ""
    started_at: float = 0.0

    @property
    def remaining_s(self) -> float:
        if not self.active:
            return 0.0
        return max(0.0, self.until - time.monotonic())


class CooldownManager:
    """Tracks cooldown periods and duplicate suppression.

    Thread-safe.  All methods are non-blocking.

    Usage::

        cm = CooldownManager()

        if cm.is_active():
            print(f"Cooldown for {cm.remaining_s:.0f}s — suppress")
            return

        if cm.is_duplicate(suggestion_text):
            print("Duplicate suppressed")
            return

        # ... show the suggestion ...

        cm.on_declined()   # starts decline cooldown
        cm.on_accepted()   # starts accept cooldown
        cm.on_displayed("suggestion text")  # track for duplicate detection
    """

    def __init__(
        self,
        decline_s: Optional[int] = None,
        accept_s: Optional[int] = None,
        duplicate_window_s: Optional[int] = None,
        on_expired: Optional[Callable] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._decline_s = decline_s if decline_s is not None else _cooldown_decline_s()
        self._accept_s = accept_s if accept_s is not None else _cooldown_accept_s()
        self._duplicate_window_s = duplicate_window_s if duplicate_window_s is not None else _duplicate_window_s()
        self._on_expired = on_expired

        self._state = CooldownState()
        self._last_displayed_text: str = ""
        self._last_displayed_at: float = 0.0

    # ------------------------------------------------------------------ #
    # Cooldown checks                                                    #
    # ------------------------------------------------------------------ #

    def is_active(self) -> bool:
        """True when a cooldown is currently in effect."""
        with self._lock:
            if not self._state.active:
                return False
            if time.monotonic() >= self._state.until:
                # Expired — clear it.
                expired_reason = self._state.reason
                expired_duration = time.monotonic() - self._state.started_at
                self._state = CooldownState()
                logger.debug(
                    "cooldown expired (reason=%s, duration=%.0fs)",
                    expired_reason,
                    expired_duration,
                )
                if self._on_expired:
                    try:
                        self._on_expired(expired_reason, expired_duration)
                    except Exception:
                        pass
                return False
            return True

    def remaining_s(self) -> float:
        """Seconds until cooldown expires, or 0."""
        with self._lock:
            return self._state.remaining_s

    def get_cooldown_info(self) -> dict:
        """Snapshot of cooldown state for introspection."""
        with self._lock:
            return {
                "active": self._state.active,
                "remaining_s": self._state.remaining_s,
                "reason": self._state.reason,
            }

    # ------------------------------------------------------------------ #
    # Duplicate detection                                                #
    # ------------------------------------------------------------------ #

    def is_duplicate(self, text: str) -> bool:
        """True if `text` was displayed within the duplicate window.

        Uses simple string equality on the message text.
        """
        if not text:
            return False
        with self._lock:
            if not self._last_displayed_text:
                return False
            if text != self._last_displayed_text:
                return False
            if time.monotonic() - self._last_displayed_at > self._duplicate_window_s:
                return False
            return True

    # ------------------------------------------------------------------ #
    # Cooldown triggers                                                  #
    # ------------------------------------------------------------------ #

    def on_declined(self) -> None:
        """Start a decline cooldown (10 minutes)."""
        with self._lock:
            self._state = CooldownState(
                until=time.monotonic() + self._decline_s,
                active=True,
                reason="decline",
                started_at=time.monotonic(),
            )
        logger.info(
            "cooldown started: decline, %ds", self._decline_s
        )

    def on_accepted(self) -> None:
        """Start an accept cooldown (20 minutes)."""
        with self._lock:
            self._state = CooldownState(
                until=time.monotonic() + self._accept_s,
                active=True,
                reason="accept",
                started_at=time.monotonic(),
            )
        logger.info(
            "cooldown started: accept, %ds", self._accept_s
        )

    def on_displayed(self, text: str) -> None:
        """Record that a suggestion was shown (for duplicate tracking)."""
        now = time.monotonic()
        with self._lock:
            self._last_displayed_text = text
            self._last_displayed_at = now

    def reset(self) -> None:
        """Clear all cooldown and duplicate state."""
        with self._lock:
            self._state = CooldownState()
            self._last_displayed_text = ""
            self._last_displayed_at = 0.0
        logger.debug("cooldown: reset")
