"""
Windows toast notifications for J.A.R.V.I.S.

Uses `winotify` to send native Windows 10/11 toasts. Falls back silently on
non-Windows platforms (so dev/testing on Mac/Linux doesn't break).
"""
from __future__ import annotations

import logging
import platform
import threading

logger = logging.getLogger("jarvis.toast")


def show_toast(title: str, message: str, duration: str = "short") -> bool:
    """
    Show a Windows toast notification. Returns True on success.

    `duration` is 'short' (~5s) or 'long' (~25s).

    Runs in a daemon thread so the event loop is never blocked.
    """
    if platform.system() != "Windows":
        logger.debug("Toast only supported on Windows (running on %s)", platform.system())
        return False

    def _do():
        try:
            from winotify import Notification, audio
            toast = Notification(
                app_id="J.A.R.V.I.S.",
                title=title,
                msg=message,
                duration=duration,
            )
            toast.set_audio(audio.Default, loop=False)
            toast.show()
        except Exception as e:
            logger.warning("Toast failed: %s", e)

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    return True


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    ok = show_toast("J.A.R.V.I.S.", "At your service, Sir.")
    print("toast dispatched:", ok)