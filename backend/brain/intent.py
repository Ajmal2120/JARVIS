"""
Intent detection for JARVIS.

Fast, simple pattern matching for common commands that don't need Claude.
Returns an intent dict or None — the caller executes the async actions.

Handles: time, volume, apps, media, screenshots, screen lock, clipboard, shutdown/restart.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional


def detect_intent(command: str) -> Optional[dict[str, Any]]:
    """
    Detect a known intent from text and return what to do.

    Returns None when no intent matches (caller should fall through to Claude).

    Returned dict keys:
        type: str      — "time" | "volume" | "app" | "media" | "screenshot" |
                         "lock" | "shutdown" | "restart" | "clipboard"
        target: str    — the resolved target (app name, media query, volume level, etc.)
        response: str  — JARVIS-style confirmation text for TTS
    """
    cmd = command.strip()
    cmd_lower = cmd.lower()

    # --- Time ---
    if re.search(r'\b(time|what time is it|what\'s the time|current time)\b', cmd_lower):
        return {
            "type": "time",
            "target": "",
            "response": f"It is {datetime.now().strftime('%I:%M %p')}, Sir.",
        }

    # --- Volume ---
    vol = re.search(r'\b(volume|sound)\b', cmd_lower)
    if vol:
        # Pattern: "volume up" / "volume down" / "set volume to 50" / "turn volume 70"
        m = re.search(r'(?:set|turn|make)\s+(?:volume|sound)\s+(?:to\s+)?(\d+)', cmd_lower)
        if m:
            level = max(0, min(100, int(m.group(1))))
            return {"type": "volume", "target": str(level),
                    "response": f"Volume set to {level} percent, Sir."}
        if 'up' in cmd_lower or 'increase' in cmd_lower:
            return {"type": "volume", "target": "80",
                    "response": "Volume increased, Sir."}
        if 'down' in cmd_lower or 'lower' in cmd_lower or 'decrease' in cmd_lower:
            return {"type": "volume", "target": "30",
                    "response": "Volume lowered, Sir."}
        # bare "volume" — just say current volume isn't known here, ask Claude
        return {"type": "volume", "target": "50",
                "response": "Setting volume to 50 percent, Sir."}

    # --- Open app ---
    m = re.search(r'\b(open|launch|start|run)\s+(?:the\s+)?(.+)', cmd_lower)
    if m and len(m.group(2).strip()) < 40:
        app = m.group(2).strip()
        # Strip trailing junk
        app = re.sub(r'\s+(please|now|for me).*$', '', app).strip()
        if app:
            return {"type": "app", "target": app,
                    "response": f"Opening {app}, Sir."}

    # --- Media ---
    m = re.search(r'\b(play|play music|play song|listen to)\s+(?:the\s+)?(.+)', cmd_lower)
    if m:
        query = m.group(2).strip()
        query = re.sub(r'\s+(please|now|for me).*$', '', query).strip()
        if query:
            return {"type": "media", "target": query,
                    "response": f"Playing {query}, Sir."}
        return {"type": "media", "target": "",
                "response": "Playing music, Sir."}

    # --- Screenshot ---
    if re.search(r'\b(screenshot|screen shot|capture screen|take (a )?picture|snap)\b', cmd_lower):
        return {"type": "screenshot", "target": "0",
                "response": "Screenshot captured, Sir."}

    # --- Lock screen ---
    if re.search(r'\b(lock|lock screen|lock workstation|lock pc)\b', cmd_lower):
        return {"type": "lock", "target": "",
                "response": "Workstation locked, Sir."}

    # --- Shutdown ---
    if re.search(r'\b(shutdown|shut down|power off|turn off)\b', cmd_lower):
        return {"type": "shutdown", "target": "now",
                "response": "Shutting down the computer, Sir."}

    # --- Restart ---
    if re.search(r'\b(restart|reboot)\b', cmd_lower):
        return {"type": "restart", "target": "now",
                "response": "Restarting the computer, Sir."}

    # --- Clipboard copy ---
    m = re.search(r'\bcopy\s+(?:to\s+)?(?:clipboard\s+)?(.+)', cmd_lower)
    if m:
        text = m.group(1).strip()
        text = re.sub(r'\s+(please|now).*$', '', text).strip()
        if text:
            return {"type": "clipboard", "target": text,
                    "response": f"Copied to clipboard, Sir."}

    # --- Screen Control: Click element ---
    m = re.search(r'\b(click|tap|press)\s+(?:on\s+)?(?:the\s+)?(.+)', cmd_lower)
    if m:
        element = m.group(2).strip()
        if element and len(element) < 50:
            return {"type": "click", "target": element,
                    "response": f"Clicking on {element}, Sir."}

    # --- Screen Control: Type text ---
    m = re.search(r'\b(type|enter|input)\s+(?:the\s+)?(?:text\s+)?(.+)', cmd_lower)
    if m:
        text = m.group(2).strip()
        if text and len(text) < 100:
            return {"type": "type", "target": text,
                    "response": f"Typing: {text}"}

    # --- Screen Control: Scroll ---
    if re.search(r'\b(scroll|swipe)\s+(up|down)', cmd_lower):
        direction = "up" if "up" in cmd_lower else "down"
        return {"type": "scroll", "target": direction,
                "response": f"Scrolling {direction}, Sir."}

    # --- Screen Control: Press hotkey ---
    m = re.search(r'\bpress\s+((?:ctrl|alt|shift|win|super)\s*\+\s*)+(\w+)', cmd_lower)
    if m:
        keys = m.group(1).replace(" ", "").split("+") + [m.group(2).strip()]
        keys = [k for k in keys if k]
        return {"type": "hotkey", "target": keys,
                "response": f"Pressing {', '.join(keys)}, Sir."}

    # --- Screen Control: What's on screen ---
    if re.search(r'\b(what|what\'s)\s+(on|dear|showing)\s+(screen|display)', cmd_lower):
        return {"type": "read_screen", "target": "",
                "response": "Reading the screen, Sir."}

    # --- Screen Control: Screenshot ---
    if re.search(r'\b(screenshot|screen shot|take picture|snap)', cmd_lower):
        return {"type": "take_screenshot", "target": "",
                "response": "Taking screenshot, Sir."}

    # --- Screen Control: Open URL ---
    m = re.search(r'\b(open|go to|visit|navigate to)\s+(https?://\S+|www\.\S+)', cmd_lower)
    if not m:
        m = re.search(r'\b(open|go to|visit|navigate to)\s+(.+?)(?:\s|$)', cmd_lower)
    if m:
        url = m.group(2).strip()
        if url and not any(url.startswith(x) for x in ["the ", "a ", "an "]):
            if not url.startswith(('http://', 'https://', 'www.')):
                url = 'https://' + url
            return {"type": "open_url", "target": url,
                    "response": f"Opening {url}, Sir."}

    # --- Screen Control: Close window ---
    if re.search(r'\b(close|exit|quit)\s+(?:this\s+)?(window|app|application|program)', cmd_lower):
        return {"type": "quick_action", "target": "close_window",
                "response": "Closing window, Sir."}

    # --- Screen Control: Minimize ---
    if re.search(r'\b(minimize|iconify)', cmd_lower):
        return {"type": "quick_action", "target": "minimize",
                "response": "Minimizing, Sir."}

    # --- Screen Control: Maximize ---
    if re.search(r'\b(maximize|full screen|fullscreen)', cmd_lower):
        return {"type": "quick_action", "target": "maximize",
                "response": "Maximizing, Sir."}

    # --- Screen Control: Switch window ---
    if re.search(r'\b(switch|change)\s+(window|app|application)', cmd_lower):
        return {"type": "quick_action", "target": "switch_window",
                "response": "Switching window, Sir."}

    # --- Screen Control: Copy ---
    if re.search(r'\bcopy\b', cmd_lower) and not re.search(r'\bcopy\s+to\b', cmd_lower):
        return {"type": "quick_action", "target": "copy",
                "response": "Copying, Sir."}

    # --- Screen Control: Paste ---
    if re.search(r'\bpaste\b', cmd_lower):
        return {"type": "quick_action", "target": "paste",
                "response": "Pasting, Sir."}

    # --- Screen Control: Undo ---
    if re.search(r'\bundo\b', cmd_lower):
        return {"type": "quick_action", "target": "undo",
                "response": "Undoing, Sir."}

    # --- Screen Control: Select all ---
    if re.search(r'\bselect\s+all\b', cmd_lower):
        return {"type": "quick_action", "target": "select_all",
                "response": "Selecting all, Sir."}

    # --- Screen Control: Save ---
    if re.search(r'\bsave\b', cmd_lower):
        return {"type": "quick_action", "target": "save",
                "response": "Saving, Sir."}

    # --- Screen Control: New tab ---
    if re.search(r'\bnew\s+tab\b', cmd_lower):
        return {"type": "quick_action", "target": "new_tab",
                "response": "Opening new tab, Sir."}

    # --- Screen Control: Close tab ---
    if re.search(r'\bclose\s+tab\b', cmd_lower):
        return {"type": "quick_action", "target": "close_tab",
                "response": "Closing tab, Sir."}

    # --- Screen Control: Refresh ---
    if re.search(r'\b(refresh|reload)\b', cmd_lower):
        return {"type": "quick_action", "target": "refresh",
                "response": "Refreshing, Sir."}

    # --- Date ---
    if re.search(r'\b(date|what date is it|what\'s the date|today\'s date)\b', cmd_lower):
        return {
            "type": "date",
            "target": "",
            "response": f"Today's date is {datetime.now().strftime('%A, %B %d, %Y')}, Sir.",
        }

    return None