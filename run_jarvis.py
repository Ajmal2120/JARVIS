"""
JARVIS — Complete System Launcher
===================================
Single entry point that:
  1. Selects the correct Python (venv preferred)
  2. Ensures the FastAPI backend is running
  3. Launches embedded mode (system tray, voice, hotkeys)
  4. Shows clear status output
  5. Handles graceful shutdown

Usage:
    python run_jarvis.py              # normal launch
    python run_jarvis.py --startup     # background mode (no tray popup)
    python run_jarvis.py --no-backend # skip backend auto-start
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR / "backend"
VENV_PYTHON = BACKEND_DIR / ".venv" / "Scripts" / "python.exe"
EMBEDDED_ENTRY = BACKEND_DIR / "jarvis_embedded.py"


# ---------------------------------------------------------------------------
# Python selection
# ---------------------------------------------------------------------------

def pick_python() -> Path:
    """Return the best available Python interpreter."""
    if VENV_PYTHON.exists():
        return VENV_PYTHON
    import shutil
    py = shutil.which("python")
    if py:
        return Path(py)
    return Path(sys.executable)


# ---------------------------------------------------------------------------
# Backend management
# ---------------------------------------------------------------------------

_BACKEND_STARTED_BY_US = False


def is_backend_up() -> bool:
    """Check whether the FastAPI backend is responding."""
    import urllib.request
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def wait_for_backend(timeout: float = 30.0) -> bool:
    """Poll until backend responds or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_backend_up():
            return True
        time.sleep(0.5)
    return False


def ensure_backend_running(python: Path, skip: bool = False) -> bool:
    """Start the FastAPI backend if not already running."""
    global _BACKEND_STARTED_BY_US

    if skip:
        print("[launcher] Skipping backend auto-start (--no-backend)")
        return True

    if is_backend_up():
        print("[launcher] Backend already running at http://127.0.0.1:8000")
        return True

    print("[launcher] Starting JARVIS backend...")
    log_file = BACKEND_DIR / "logs" / "backend_launcher.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.Popen(
            [str(python), str(BACKEND_DIR / "main.py")],
            cwd=str(BACKEND_DIR),
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            stdout=open(log_file, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        print(f"[launcher] Backend PID {proc.pid}")
        _BACKEND_STARTED_BY_US = True

        print("[launcher] Waiting for backend to become ready...")
        if wait_for_backend(timeout=30.0):
            print("[launcher] Backend is ready.")
            return True
        print("[launcher] Backend did not respond within 30s — continuing anyway")
        return True
    except Exception as exc:
        print(f"[launcher] Failed to start backend: {exc}")
        return False


# ---------------------------------------------------------------------------
# Embedded launcher
# ---------------------------------------------------------------------------

def launch_embedded(python: Path, extra_args: list[str]) -> None:
    """Launch jarvis_embedded.py (tray + voice + hotkeys)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(
            [str(python), str(EMBEDDED_ENTRY)] + extra_args,
            cwd=str(BACKEND_DIR),
            env=env,
            check=False,
        )
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def shutdown_backend_if_we_started_it() -> None:
    """If we started the backend, ask it to shut down gracefully."""
    if not _BACKEND_STARTED_BY_US:
        return
    import urllib.request
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8000/admin/shutdown",
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        print("[launcher] Backend shutdown signal sent.")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="JARVIS — Complete System Launcher",
    )
    parser.add_argument(
        "--startup",
        action="store_true",
        help="Background mode: no tray interaction prompt",
    )
    parser.add_argument(
        "--no-backend",
        action="store_true",
        help="Skip automatic backend startup",
    )
    args = parser.parse_args()

    python = pick_python()

    print("=" * 52)
    print("  J.A.R.V.I.S.  —  Complete System Launcher")
    print("=" * 52)
    print(f"  Python  : {python}")
    print(f"  Backend : {BACKEND_DIR}")
    print("=" * 52)
    print()

    # Phase 1: ensure backend is running
    if not ensure_backend_running(python, skip=args.no_backend):
        print("[launcher] Cannot proceed without backend. Exiting.")
        return 1

    print()
    print("[launcher] Backend ready. Launching JARVIS...")
    print()

    # Phase 2: embedded experience (tray, voice, hotkeys)
    try:
        embedded_args = ["--startup"] if args.startup else []
        launch_embedded(python, embedded_args)
    finally:
        shutdown_backend_if_we_started_it()

    return 0


if __name__ == "__main__":
    sys.exit(main())
