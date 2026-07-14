"""
JARVIS Screen & Pattern Memory
Tracks what's on screen, learns user patterns, enables proactive assistance.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("jarvis.screen_memory")

SCREEN_DIR = Path(__file__).parent.parent.parent / "memory" / "screens"
SCREEN_DIR.mkdir(parents=True, exist_ok=True)

PATTERNS_FILE = Path(__file__).parent.parent.parent / "memory" / "patterns.json"

# ---------------------------------------------------------------------------
# Screen Memory - JARVIS remembers what's on screen
# ---------------------------------------------------------------------------

async def capture_and_describe() -> dict[str, Any]:
    """Capture screen and store for JARVIS to reference later."""
    from tools.system import capture_screen
    from brain.vision import describe_image

    result = await capture_screen()
    if not result.get("ok"):
        return result

    try:
        # Read image and describe it
        image_data = result.get("image_data", "")
        if image_data:
            description = await describe_image(image_data)

            # Store in memory
            memory_entry = {
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "path": result.get("path"),
            }

            # Keep last 50 screenshots in memory
            memory_file = SCREEN_DIR / "recent.json"
            history = []
            if memory_file.exists():
                with open(memory_file) as f:
                    history = json.load(f)
            history.append(memory_entry)
            history = history[-50:]  # Keep last 50

            with open(memory_file, "w") as f:
                json.dump(history, f, indent=2)

            return {"ok": True, "description": description, "memory": memory_entry}
    except Exception as e:
        logger.error(f"Screen capture/describe failed: {e}")
        return {"ok": False, "error": str(e)}

    return result


async def get_screen_memory(hours: int = 2) -> list[dict]:
    """Get recent screen memories from the last N hours."""
    memory_file = SCREEN_DIR / "recent.json"
    if not memory_file.exists():
        return []

    try:
        with open(memory_file) as f:
            history = json.load(f)

        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [
            h for h in history
            if datetime.fromisoformat(h["timestamp"]) > cutoff
        ]
        return recent
    except Exception:
        return []


async def what_was_on_screen() -> str:
    """Return a natural language summary of what was recently on screen."""
    recent = await get_screen_memory(hours=1)
    if not recent:
        return "I don't have recent screen memory. I can start remembering if you'd like."

    summaries = [r["description"] for r in recent[-5:] if r.get("description")]
    if summaries:
        return "Recently on your screen: " + "; ".join(summaries[-3:])
    return "I've been watching your screen but haven't noticed anything particularly noteworthy."


# ---------------------------------------------------------------------------
# Pattern Learning - JARVIS learns user habits
# ---------------------------------------------------------------------------

def load_patterns() -> dict[str, Any]:
    """Load learned patterns from disk."""
    if PATTERNS_FILE.exists():
        with open(PATTERNS_FILE) as f:
            return json.load(f)
    return {
        "routines": {},  # time -> activity
        "forgets": [],   # things user repeatedly forgets
        "preferences": {},  # learned preferences
        "last_seen": None,
        "conversation_count": 0,
    }


def save_patterns(patterns: dict[str, Any]) -> None:
    """Save learned patterns to disk."""
    PATTERNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PATTERNS_FILE, "w") as f:
        json.dump(patterns, f, indent=2)


def learn_from_conversation(user_msg: str, jarvis_reply: str) -> None:
    """Learn patterns from conversation."""
    patterns = load_patterns()
    patterns["conversation_count"] = patterns.get("conversation_count", 0) + 1
    patterns["last_seen"] = datetime.now().isoformat()

    # Learn time-based routines
    hour = datetime.now().hour
    time_key = f"{hour:02d}:00"
    if time_key not in patterns["routines"]:
        patterns["routines"][time_key] = []
    patterns["routines"][time_key].append(user_msg[:50])

    # Learn what user forgets (patterns like "I forgot", "remind me")
    if any(word in user_msg.lower() for word in ["forgot", "remind me", "don't forget"]):
        patterns["forgets"].append(user_msg)
        patterns["forgets"] = patterns["forgets"][-20:]  # Keep last 20

    save_patterns(patterns)


def get_pattern_insights() -> str:
    """Get proactive insights based on learned patterns."""
    patterns = load_patterns()

    insights = []

    # Time-based routines
    if patterns.get("routines"):
        routines = patterns["routines"]
        current_hour = datetime.now().hour
        nearby_hours = [h for h in routines.keys() if abs(int(h.split(":")[0]) - current_hour) <= 2]
        if nearby_hours:
            common = routines.get(f"{current_hour:02d}:00", [])
            if common:
                insights.append(f"You usually ask about '{common[0]}' around this time.")

    # Forgotten things
    if patterns.get("forgets"):
        recent_forgets = patterns["forgets"][-3:]
        insights.append(f"You've mentioned forgetting: {recent_forgets[-1][:50]}")

    return " ".join(insights) if insights else ""


# ---------------------------------------------------------------------------
# Proactive Alert Checker
# ---------------------------------------------------------------------------

async def check_proactive_alerts() -> list[dict[str, str]]:
    """Check for things that might need user's attention."""
    alerts = []

    # Check patterns for reminders
    patterns = load_patterns()
    if patterns.get("forgets"):
        last_forget = patterns["forgets"][-1]
        if "tomorrow" in last_forget.lower():
            alerts.append({
                "type": "reminder",
                "message": f"Did you complete: {last_forget[:100]}?",
                "priority": "medium"
            })

    # Screen memory insights
    screen_memory = await get_screen_memory(hours=1)
    if screen_memory:
        last_screen = screen_memory[-1]
        if "email" in last_screen.get("description", "").lower():
            alerts.append({
                "type": "email",
                "message": "I noticed you were checking email. Any replies pending?",
                "priority": "low"
            })

    return alerts
