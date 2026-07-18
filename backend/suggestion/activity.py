"""
JARVIS 3.0 — Activity Monitor.

Tracks user input activity (keyboard and mouse) to avoid interrupting
the user while they are actively typing or working.

The monitor exposes:
    - record_activity()  — call when input is detected
    - is_quiet(threshold_s) — True if no activity for `threshold_s` seconds
    - last_activity_monotonic() — timestamp of last activity

No external dependencies.  On Windows, uses GetLastInputInfo via ctypes
for system-wide activity detection.  Falls back to programmatic tracking
when the OS hook is unavailable.

Thread-safe.
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from typing import Optional

logger = logging.getLogger("jarvis.suggestion")

# ---------------------------------------------------------------------------
# Windows system-wide activity via GetLastInputInfo
# ---------------------------------------------------------------------------

def _windows_last_input_s() -> Optional[float]:
    """Return seconds since last user input on Windows, or None."""
    if platform.system() != "Windows":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.UINT),
                ("dwTime", wintypes.DWORD),
            ]

        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            tick = lii.dwTime
            # GetTickCount wraps at ~49.7 days — handle that.
            current = ctypes.windll.kernel32.GetTickCount()
            ms_diff = (current - tick) & 0xFFFFFFFF
            return ms_diff / 1000.0
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# ActivityMonitor
# ---------------------------------------------------------------------------

class ActivityMonitor:
    """Tracks user activity and reports whether the user is "quiet".

    The monitor can be used in two modes:

    1. **External (recommended)**: The Observer or another component calls
       ``record_activity()`` whenever it detects input.  This is the most
       accurate mode.

    2. **Polling (fallback)**: A background thread polls the OS for last
       input time.  Used when no external feed is wired up.

    Args:
        poll_interval_s: How often the polling thread checks (default 1.0s).
        quiet_threshold_s: Minimum quiet time to consider the user idle.
    """

    def __init__(
        self,
        poll_interval_s: float = 1.0,
        quiet_threshold_s: float = 2.0,
    ) -> None:
        self._lock = threading.Lock()
        self._last_activity_monotonic: float = time.monotonic()
        self._quiet_threshold_s: float = quiet_threshold_s
        self._poll_interval_s: float = poll_interval_s

        self._external_mode: bool = False
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the background polling thread."""
        if self._poll_thread is not None:
            return
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="activity_monitor",
            daemon=True,
        )
        self._poll_thread.start()
        logger.debug("activity_monitor: started (poll_interval=%.1fs)", self._poll_interval_s)

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=3.0)
            self._poll_thread = None
        logger.debug("activity_monitor: stopped")

    # ------------------------------------------------------------------ #
    # External mode (Observer-driven)                                    #
    # ------------------------------------------------------------------ #

    def set_external_mode(self, enabled: bool = True) -> None:
        """Enable external mode: the caller is responsible for record_activity()."""
        self._external_mode = enabled

    def record_activity(self) -> None:
        """Record that user input was detected.

        Called by the Observer (or any component) when keyboard or mouse
        activity occurs.
        """
        with self._lock:
            self._last_activity_monotonic = time.monotonic()

    # ------------------------------------------------------------------ #
    # Queries                                                           #
    # ------------------------------------------------------------------ #

    def last_activity_monotonic(self) -> float:
        """Monotonic timestamp of last detected activity."""
        with self._lock:
            return self._last_activity_monotonic

    def seconds_since_last_activity(self) -> float:
        """Seconds since last detected activity."""
        return time.monotonic() - self.last_activity_monotonic()

    def is_quiet(self, threshold_s: Optional[float] = None) -> bool:
        """True if the user has been quiet for at least `threshold_s` seconds.

        Args:
            threshold_s: Override the default quiet threshold.

        Returns:
            True when the user is quiet enough to receive a suggestion.
        """
        thresh = threshold_s if threshold_s is not None else self._quiet_threshold_s
        return self.seconds_since_last_activity() >= thresh

    # ------------------------------------------------------------------ #
    # Background polling                                                 #
    # ------------------------------------------------------------------ #

    def _poll_loop(self) -> None:
        """Poll OS for last input time when no external feed is available."""
        while not self._stop_event.is_set():
            try:
                if not self._external_mode:
                    self._poll_once()
            except Exception:
                pass
            self._stop_event.wait(self._poll_interval_s)

    def _poll_once(self) -> None:
        """Single OS poll for last input time."""
        win_s = _windows_last_input_s()
        if win_s is not None:
            # On Windows, use GetLastInputInfo for system-wide tracking.
            with self._lock:
                self._last_activity_monotonic = time.monotonic() - win_s
        else:
            # Fallback: if we're in a test or non-Windows, just note
            # that we have no external activity source.
            pass
