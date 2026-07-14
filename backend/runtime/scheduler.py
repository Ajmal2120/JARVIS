"""
Lightweight scheduler hook for runtime-managed services.

This is NOT a replacement for `apscheduler.schedulers.asyncio.AsyncIOScheduler`,
which the backend already uses for reflection, proactivity, and screen
memory. Those jobs are owned by the existing Reactive Engine and must
keep working.

`Scheduler` here is a future hook for the Operating Intelligence
components (Observer, Context Engine, Planner) to register their own
periodic work without colliding with APScheduler. It runs cooperatively
inside the Runtime's heartbeat cycle, so its overhead is bounded by the
heartbeat tick.

The Scheduler submits due tasks to a thread pool so a slow task (screen
capture, file I/O) never blocks the heartbeat pulse. The pool size is
bounded: default max_workers=4. The caller can tune it but should keep
it small — this is not a general-purpose executor.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger("jarvis.runtime.scheduler")


@dataclass
class ScheduledTask:
    """A registered periodic or one-shot task.

    `interval_s` of 0 or negative means one-shot (fires once on the
    next tick after the Runtime enters IDLE).
    """

    name: str
    callback: Callable[[], None]
    interval_s: float
    last_run: float = 0.0
    enabled: bool = True
    runs: int = 0


class Scheduler:
    """Minimal task table for runtime-managed services.

    Tasks are submitted to an internal thread pool so the heartbeat
    thread is never blocked by slow callbacks. The pool is created
    lazily on the first call to `tick()` so the Scheduler stays cheap
    when no tasks are registered.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._tasks: dict[str, ScheduledTask] = {}
        self._max_workers = max(1, max_workers)
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()

    def register(
        self,
        name: str,
        callback: Callable[[], None],
        interval_s: float = 1.0,
    ) -> ScheduledTask:
        """Add a task. Overwrites an existing task with the same name."""
        if interval_s < 0:
            raise ValueError("interval_s must be >= 0")
        task = ScheduledTask(
            name=name,
            callback=callback,
            interval_s=float(interval_s),
        )
        self._tasks[name] = task
        logger.debug("scheduler: registered %s (every %.3fs)", name, interval_s)
        return task

    def unregister(self, name: str) -> bool:
        """Remove a task by name. Returns True if it existed."""
        existed = self._tasks.pop(name, None) is not None
        if existed:
            logger.debug("scheduler: unregistered %s", name)
        return existed

    def disable(self, name: str) -> None:
        if name in self._tasks:
            self._tasks[name].enabled = False

    def enable(self, name: str) -> None:
        if name in self._tasks:
            self._tasks[name].enabled = True

    def tick(self, now: float) -> None:
        """Submit any due tasks to the thread pool.

        Called by the Runtime's heartbeat. Returns immediately — tasks
        run in the pool. If a task raises, it's caught in the pool
        thread, logged, and the task is disabled. One slow task can't
        stall the heartbeat or other tasks.
        """
        # Lazily create the pool so an empty scheduler costs nothing.
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(
                        max_workers=self._max_workers,
                        thread_name_prefix="jarvis-scheduler",
                    )

        for task in list(self._tasks.values()):
            if not task.enabled:
                continue
            if task.interval_s <= 0 and task.runs > 0:
                continue  # one-shot already fired
            if (now - task.last_run) >= task.interval_s:
                task.last_run = now
                self._pool.submit(self._run_task, task)

    def names(self) -> list[str]:
        return list(self._tasks.keys())

    def clear(self) -> None:
        """Drop all tasks and shut down the pool. Used during shutdown."""
        self._tasks.clear()
        if self._pool is not None:
            with self._pool_lock:
                if self._pool is not None:
                    self._pool.shutdown(wait=False, cancel_futures=True)
                    self._pool = None

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _run_task(self, task: ScheduledTask) -> None:
        """Wrapper that catches exceptions so the pool doesn't log
        them as unhandled futures."""
        try:
            task.callback()
            task.runs += 1
        except Exception as exc:
            logger.exception("scheduled task %s failed: %s", task.name, exc)
            task.enabled = False

