"""
System data sources for the Observer.

Each function returns a raw value from the OS or None if the read fails.
None means "couldn't determine" — the Observer treats it as "no change"
so a transient read failure doesn't generate a spurious event.

No classification, no intent inference, no history. Just values.
"""
from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("jarvis.observer.sources")

# ---- Active window title ---------------------------------------------------

def get_active_window_title() -> str | None:
    """Return the title of the foreground window, or None."""
    try:
        import win32gui  # type: ignore[import-untyped]
        hwnd = win32gui.GetForegroundWindow()
        if hwnd == 0:
            return ""
        title = win32gui.GetWindowText(hwnd)
        return title if title else ""
    except Exception:
        return None


# ---- Active process name ---------------------------------------------------

def get_active_process_name() -> str | None:
    """Return the process name of the foreground window, or None."""
    try:
        import win32gui  # type: ignore[import-untyped]
        import win32process  # type: ignore[import-untyped]
        import psutil  # type: ignore[import-untyped]

        hwnd = win32gui.GetForegroundWindow()
        if hwnd == 0:
            return ""
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(pid)
        return process.name()
    except Exception:
        return None


# ---- Idle time -------------------------------------------------------------

def get_idle_seconds() -> float | None:
    """Return seconds since last user input, or None."""
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
        ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
        tick_count = ctypes.windll.kernel32.GetTickCount64()
        idle_ms = tick_count - lii.dwTime
        return idle_ms / 1000.0
    except Exception:
        return None


# ---- Screen lock state -----------------------------------------------------

# Cached value. On Windows, GetForegroundWindow() returns 0 when the
# screen is locked (Winlogon desktop is active). We detect transitions
# by comparing against the previous value.
_prev_foreground_hwnd: int = -1
_prev_locked: bool = False


def get_locked() -> bool | None:
    """Return True if the screen is locked, False if unlocked, None if unknown.

    Detection: GetForegroundWindow() returns 0 when the workstation is
    locked. This is a polling-based heuristic, not a notification-based
    one. It's reliable enough for a change-detection loop.
    """
    try:
        import win32gui  # type: ignore[import-untyped]

        hwnd = win32gui.GetForegroundWindow()
        return hwnd == 0
    except Exception:
        return None


def reset_lock_state() -> None:
    """Reset internal lock-state tracking. Call on startup so the first
    poll doesn't falsely report a LockStateChanged event."""
    global _prev_foreground_hwnd, _prev_locked
    _prev_foreground_hwnd = -1
    _prev_locked = False


def read_lock_state() -> tuple[bool | None, bool]:
    """Read the current lock state and detect changes.

    Returns (current_state, changed) where:
        current_state is True/False/None (locked/unlocked/unknown)
        changed is True only on the first call after reset, or when the
        value flips from the previous poll.

    The first call after reset_lock_state() reports a change so the
    Observer can initialize its snapshot.
    """
    global _prev_foreground_hwnd, _prev_locked

    current = get_locked()

    if current is None:
        return None, False

    # On the very first poll, seed the previous value and report no change.
    # The Observer's own comparison logic handles snapshot initialization.
    if _prev_foreground_hwnd == -1:
        _prev_foreground_hwnd = 0 if current else 1
        _prev_locked = current
        return current, False

    changed = current != _prev_locked
    _prev_locked = current
    return current, changed


# ---- System uptime ---------------------------------------------------------

def get_system_uptime_s() -> float | None:
    """Return seconds since the system booted, or None."""
    try:
        import ctypes

        # GetTickCount64 wraps around after ~585,000 years, so no overflow
        # concern for practical purposes.
        uptime_ms = ctypes.windll.kernel32.GetTickCount64()
        return uptime_ms / 1000.0
    except Exception:
        return None
