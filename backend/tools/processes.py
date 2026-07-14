"""
Process utilities: list/lookup PIDs by name, run blocking I/O in a thread.

Kept separate from system.py so the heavy imports (PIL, pyautogui, pycaw)
only load when those tools are actually called.
"""
from __future__ import annotations

import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Single shared executor for short blocking I/O calls (screenshot capture,
# clipboard, pycaw).  max_workers=4 is enough — we never queue many at once.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jarvis-io")


async def run_in_thread(fn, *args, **kwargs) -> Any:
    """Run a blocking function in the shared thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


def find_pids_by_name(name: str) -> list[int]:
    """
    Return PIDs whose process name matches `name`.

    Uses tasklist on Windows, pgrep elsewhere.  Partial match — "chrome"
    will match "chrome.exe".
    """
    name = name.strip()
    if not name:
        return []
    try:
        if os.name == "nt":
            import subprocess
            r = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10,
            )
            out = []
            target = name.lower()
            if not target.endswith(".exe"):
                target += ".exe"
            for line in r.stdout.splitlines():
                parts = line.strip().strip('"').split('","')
                if len(parts) >= 2:
                    proc = parts[0].strip('"').lower()
                    if proc == target or proc.startswith(target.replace(".exe", "")):
                        try:
                            out.append(int(parts[1].strip('"')))
                        except ValueError:
                            pass
            return out
        else:
            r = subprocess.run(["pgrep", "-i", "-f", name],
                               capture_output=True, text=True, timeout=10)
            return [int(x) for x in r.stdout.split() if x.strip().isdigit()]
    except Exception:
        return []