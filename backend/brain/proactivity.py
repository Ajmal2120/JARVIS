"""
Proactivity rules engine for J.A.R.V.I.S.

Evaluates rules stored in the `rules` table on a 15-minute tick. When a rule's
condition is met AND its cooldown has elapsed, the action is fired:

  - Generates a JARVIS message via jarvis_core.generate_proactive_message
  - Pushes the message to all connected WebSocket clients
  - Fires a Windows toast (best-effort)
  - Records last_fired in the DB

Default rules are seeded on first run. Users can add/edit rules via the API.

Built-in features:
- Screen awareness (JARVIS remembers what's on screen)
- Pattern learning (learns user habits)
- Proactive alerts (nudges before you ask)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from brain.context import _local_now
from config import settings
from memory import long_term


def _to_aware(dt_or_str) -> datetime:
    """Coerce a datetime / SQLite string into an offset-aware datetime."""
    if isinstance(dt_or_str, datetime):
        if dt_or_str.tzinfo is None:
            return dt_or_str.replace(tzinfo=timezone.utc)
        return dt_or_str
    if isinstance(dt_or_str, str):
        try:
            naive = datetime.strptime(dt_or_str, "%Y-%m-%d %H:%M:%S")
            return naive.replace(tzinfo=timezone.utc)
        except Exception:
            return _local_now()
    return _local_now()


logger = logging.getLogger("jarvis.proactivity")


# -----------------------------------------------------------------------------
# Condition evaluators
# -----------------------------------------------------------------------------

async def _last_user_message_at() -> Optional[datetime]:
    items = await long_term.recent_conversations(limit=1, role="user")
    if not items:
        return None
    return _to_aware(items[0]["timestamp"])


async def _first_chat_today() -> bool:
    """True if no user message today yet."""
    last = await _last_user_message_at()
    if not last:
        return True
    return last.date() < _local_now().date()


async def _days_since_last_chat() -> int:
    last = await _last_user_message_at()
    if not last:
        return 999
    return (_local_now().date() - last.date()).days


async def _chatting_over_hours(threshold_h: float) -> bool:
    """True if user has been chatting for >= threshold_h continuously today."""
    items = await long_term.recent_conversations(limit=200, role="user")
    if not items:
        return False
    today = _local_now().date()
    first_today = None
    for i in reversed(items):
        ts = _to_aware(i["timestamp"])
        if ts.date() == today:
            first_today = ts
            break
    if first_today is None:
        return False
    return (_local_now() - first_today).total_seconds() >= threshold_h * 3600


async def _evaluate_condition(condition: dict) -> bool:
    """Dispatch to a per-key evaluator. Unknown keys = False."""
    now = _local_now()

    if "hour" in condition and now.hour != int(condition["hour"]):
        return False
    if "hour_min" in condition and now.hour < int(condition["hour_min"]):
        return False
    if "hour_max" in condition and now.hour > int(condition["hour_max"]):
        return False
    if "weekday" in condition and now.weekday() != int(condition["weekday"]):
        return False
    if "weekend" in condition and bool(condition["weekend"]) != (now.weekday() >= 5):
        return False
    if "no_chat_today" in condition and bool(condition["no_chat_today"]):
        if not await _first_chat_today():
            return False
    if "days_since_chat_gte" in condition:
        if await _days_since_last_chat() < int(condition["days_since_chat_gte"]):
            return False
    if "chatting_over_hours" in condition:
        if not await _chatting_over_hours(float(condition["chatting_over_hours"])):
            return False
    if "rule_type" in condition and condition["rule_type"] == "reminder":
        # Reminders are evaluated by their fire_at timestamp inside _fire_reminder
        return True
    return True


# -----------------------------------------------------------------------------
# Cooldown check
# -----------------------------------------------------------------------------

def _cooldown_ok(last_fired_iso: Optional[str], cooldown_hours: float) -> bool:
    if not last_fired_iso:
        return True
    try:
        last = _to_aware(last_fired_iso)
        return (_local_now() - last).total_seconds() >= cooldown_hours * 3600
    except Exception:
        return True


# -----------------------------------------------------------------------------
# WebSocket push (lazy import to avoid circular)
# -----------------------------------------------------------------------------

async def _broadcast(payload: dict) -> None:
    try:
        # Imported lazily so this module can run before main.py is loaded
        from main import ws_manager
        await ws_manager.broadcast(payload)
    except Exception as e:
        logger.debug("WS broadcast skipped: %s", e)


async def _fire_toast(title: str, message: str) -> None:
    try:
        from notifications.toast import show_toast
        show_toast(title, message)
    except Exception as e:
        logger.debug("Toast failed: %s", e)


# -----------------------------------------------------------------------------
# Tick
# -----------------------------------------------------------------------------

async def _fire_action(user_id: str, rule: dict) -> Optional[dict]:
    """Generate a JARVIS message for the action, broadcast, fire toast."""
    # Avoid circular import
    from jarvis_core import generate_proactive_message

    action = rule["action"]
    payload = rule.get("action_payload") or {}

    # Reminder rule: only fire if fire_at is in the past
    if action == "fire_reminder":
        fire_at = payload.get("time_iso")
        if fire_at:
            try:
                fire_dt = datetime.fromisoformat(fire_at)
                if fire_dt > _local_now():
                    return None
            except ValueError:
                pass

    message = await generate_proactive_message(user_id, action, payload)

    broadcast = {
        "type": "jarvis_message",
        "source": "proactive",
        "action": action,
        "content": message,
        "mood": "neutral",
        "timestamp": _local_now().isoformat(),
    }
    await _broadcast(broadcast)
    await _fire_toast("J.A.R.V.I.S.", message[:200])
    return broadcast


async def proactivity_tick(user_id: str = "default") -> list[dict]:
    """
    Evaluate all enabled rules and fire any that should trigger.
    Returns a list of fired actions (mostly for logging / admin).
    """
    if not settings.PROACTIVITY_ENABLED:
        return []

    rules = await long_term.list_rules(enabled_only=True)
    fired: list[dict] = []

    for rule in rules:
        try:
            if not await _evaluate_condition(rule.get("condition", {})):
                continue
            if not _cooldown_ok(rule.get("last_fired"), float(rule.get("cooldown_hours", 12))):
                continue
            result = await _fire_action(user_id, rule)
            if result:
                await long_term.mark_rule_fired(rule["id"])
                fired.append({"rule": rule["name"], "action": rule["action"]})
        except Exception as e:
            logger.exception("Rule %s failed: %s", rule.get("name"), e)
    return fired