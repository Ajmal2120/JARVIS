"""
Short-term in-memory conversation buffer.

Keeps the most recent N user/assistant turns in process memory for fast access
without hitting SQLite on every chat. The conversation history is also written
to SQLite (long_term.add_conversation) for durability and reflection.
"""
from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Optional


class ShortTermBuffer:
    """Per-user bounded FIFO of recent conversation turns."""

    def __init__(self, maxlen: int = 30):
        self._buffers: dict[str, deque] = {}
        self._maxlen = maxlen
        self._lock = Lock()

    def add(self, user_id: str, role: str, content: str, mood: Optional[str] = None) -> None:
        with self._lock:
            buf = self._buffers.setdefault(user_id, deque(maxlen=self._maxlen))
            buf.append({"role": role, "content": content, "mood": mood})

    def recent(self, user_id: str, limit: int = 20) -> list[dict]:
        with self._lock:
            buf = self._buffers.get(user_id, deque())
            return list(buf)[-limit:]

    def clear(self, user_id: str) -> None:
        with self._lock:
            self._buffers.pop(user_id, None)


# Module-level singleton
buffer = ShortTermBuffer(maxlen=40)
