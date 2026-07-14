"""
Context enrichment for J.A.R.V.I.S.

Builds a `context` dict that gets injected into every Claude call. Captures:
  - current time (local)
  - user state (first chat today / in session / returning after days)
  - last interaction
  - recent messages (from short-term buffer)
  - message stats (avg length)
  - detected mood
  - relevant facts (keyword-matched against current message)
  - relevant patterns
  - applicable proactivity rules (informational; proactivity.py does the firing)
  - OS context: active window, recent commands, current project, browser, files
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from config import settings
from memory import long_term
from memory.short_term import buffer
from brain.sentiment import infer_mood

logger = logging.getLogger(__name__)


def _local_now() -> datetime:
    """Return the current local time as an offset-aware datetime."""
    try:
        tz = ZoneInfo(settings.TIMEZONE)
        return datetime.now(tz)
    except Exception:
        return datetime.now(timezone.utc)


def _parse_sqlite_ts(s: str) -> datetime:
    """
    Parse a SQLite CURRENT_TIMESTAMP ('YYYY-MM-DD HH:MM:SS', UTC) into an
    offset-aware datetime in the configured local timezone.
    """
    try:
        naive = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        aware_utc = naive.replace(tzinfo=timezone.utc)
        try:
            return aware_utc.astimezone(ZoneInfo(settings.TIMEZONE))
        except Exception:
            return aware_utc
    except Exception:
        return _local_now()


async def _user_state(user_id: str, last_at: Optional[datetime]) -> str:
    """first_today / in_session / returning_after_days."""
    if not last_at:
        return "first_today"
    # last_at may be naive UTC; coerce to local-tz-aware for safe comparison.
    if last_at.tzinfo is None:
        last_at = last_at.replace(tzinfo=timezone.utc)
    try:
        last_at = last_at.astimezone(ZoneInfo(settings.TIMEZONE))
    except Exception:
        pass
    now = _local_now()
    days = (now.date() - last_at.date()).days
    if days >= 3:
        return "returning_after_days"
    if days >= 1:
        if last_at.date() < now.date():
            return "first_today"
    return "in_session"


async def _message_stats(user_id: str) -> dict[str, Any]:
    msgs = buffer.recent(user_id, limit=20)
    user_msgs = [m for m in msgs if m["role"] == "user"]
    if not user_msgs:
        return {"count": 0, "avg_length": 0, "last_mood": None}
    avg_len = sum(len(m["content"]) for m in user_msgs) / len(user_msgs)
    return {
        "count": len(user_msgs),
        "avg_length": round(avg_len, 1),
        "last_mood": user_msgs[-1].get("mood"),
    }


async def _get_os_context() -> dict[str, Any]:
    """
    Get lightweight OS context for intelligence layer.
    This is the 'Perception' layer feeding into the Understanding layer.
    Uses the Context Manager for continuous tracking.
    """
    try:
        # Use new Context Manager for richer OS context
        from brain.os_context import get_os_context as get_os_ctx

        return get_os_ctx()

    except ImportError:
        # Fallback to simple window detection
        try:
            from tools.screen_control import get_active_window
            window = get_active_window()
            window_lower = window.lower() if window else ""

            return {
                "active_window": window,
                "active_app": None,
                "app_type": "unknown",
                "recent_commands": [],
                "recent_files": [],
                "working_directory": "",
                "clipboard_preview": None,
            }
        except Exception as e:
            logger.debug(f"OS context gathering failed: {e}")
            return {}
    except Exception as e:
        logger.debug(f"Context manager failed: {e}")
        return {}


async def build_context(user_id: str, current_message: str) -> dict[str, Any]:
    """Build the full context dict for one chat turn."""
    now = _local_now()
    
    # Get last interaction with error handling
    last_at = None
    try:
        last_at = await long_term.last_interaction_at()
    except:
        pass
    
    user_state = await _user_state(user_id, last_at)

    # Mood: infer once, reuse for prompt + storage
    mood = None
    try:
        mood = await infer_mood(user_id)
    except:
        pass

    facts = []
    patterns = []
    rules = []
    
    try:
        facts = await long_term.relevant_facts(current_message, limit=8)
        patterns = await long_term.relevant_patterns(current_message, limit=5)
        rules = await long_term.list_rules(enabled_only=True)
    except:
        pass

    msg_stats = await _message_stats(user_id)
    recent_msgs = buffer.recent(user_id, limit=8)

    # Get OS context (active window, recent commands, etc.)
    os_ctx = await _get_os_context()

    minutes_since_last = None
    if last_at:
        # Normalize to offset-aware UTC before subtraction.
        if last_at.tzinfo is None:
            last_at = last_at.replace(tzinfo=timezone.utc)
        delta = _local_now() - last_at
        minutes_since_last = int(delta.total_seconds() // 60)

    # Get active window info for screen context
    active_window = os_ctx.get("active_window")

    return {
        "now": now,
        "hour": now.hour,
        "weekday": now.weekday(),
        "is_weekend": now.weekday() >= 5,
        "user_state": user_state,
        "mood": mood,
        "last_interaction_at": last_at.isoformat() if last_at else None,
        "minutes_since_last": minutes_since_last,
        "facts": facts,
        "patterns": patterns,
        "rules": rules,
        "recent_messages": recent_msgs,
        "message_stats": msg_stats,
        "current_message": current_message,
        "active_window": active_window,
        # OS context
        "os": os_ctx,
    }


async def _get_active_window() -> str:
    """Get the currently active window title."""
    try:
        # Try to use screen_control module
        from tools.screen_control import get_active_window
        return get_active_window()
    except Exception:
        return "Unknown"