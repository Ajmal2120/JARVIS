"""
The J.A.R.V.I.S. core — the brain that ties everything together.

Wires together:
  - dynamic system prompt (brain/system_prompt.py)
  - context enrichment (brain/context.py)
  - sentiment (brain/sentiment.py)
  - long-term memory (memory/long_term.py)
  - short-term memory (memory/short_term.py)
  - tools (search, weather, files, system, memory ops)
  - Anthropic Claude API

Public API:
  chat(user_id, message) -> {reply, mood, tools_used}
  generate_proactive_message(user_id, action, payload) -> str
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

from brain.context import build_context
from brain.sentiment import infer_mood
from brain.system_prompt import build_system_prompt
from config import settings
from memory import long_term
from memory.short_term import buffer
from tools import apis, files, search, system as system_tool
from tools import whatsapp as whatsapp_tool
from tts import speaker

logger = logging.getLogger("jarvis.core")


def _speak_async(text: str) -> None:
    """Speak text asynchronously."""
    if text and text.strip():
        try:
            speaker.speak_async(text)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Retry logic with exponential backoff for rate limiting (429 errors)
# -----------------------------------------------------------------------------

async def _retry_with_backoff(coro_func, max_retries: int = 5, base_delay: float = 2.0, **kwargs):
    """
    Retry an async function with exponential backoff on 429 rate limit errors.

    Args:
        coro_func: Async function to call (e.g., client.messages.create)
        max_retries: Maximum number of retries (default 5)
        base_delay: Initial delay in seconds (default 2.0)
        **kwargs: Keyword args for the function (passed to coro_func)

    Returns:
        The result of the function call
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return await coro_func(**kwargs)
        except Exception as e:
            last_error = e

            # Check if it's a rate limit error (429)
            is_rate_limit = False
            if hasattr(e, 'status_code'):
                is_rate_limit = e.status_code == 429
            elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                is_rate_limit = e.response.status_code == 429
            elif "429" in str(e):
                is_rate_limit = True

            if is_rate_limit and attempt < max_retries:
                # Exponential backoff: 2s, 4s, 8s, 16s, 32s
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}), "
                             f"retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                # Not a rate limit or max retries reached
                raise

    raise last_error  # Shouldn't reach here, but just in case


# Lazy import helper for main.py (avoids circular import at module load).
def _broadcast(msg: dict) -> None:
    """Push a JSON message to all connected WebSocket clients."""
    try:
        from main import ws_manager
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(ws_manager.broadcast(msg))
    except Exception:
        # main.py not yet imported in some test contexts; ignore.
        pass


# -----------------------------------------------------------------------------
# Tool definitions (Anthropic tool-use schema)
# -----------------------------------------------------------------------------

# Tools whose effects are visible to the user / mutating — these run only
# after the user approves in the HUD.  Read-only / informational tools
# execute immediately.
APPROVAL_REQUIRED: set[str] = set()
# Sir gave JARVIS full system access — no approval gates.
# If you want to re-gate specific actions, add them back here.

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "web_search",
        "description": (
            "Search the public web for current information. Use when the user "
            "asks about news, facts you don't know, prices, etc. Returns a "
            "formatted text block of results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather and a 3-day forecast for coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "get_news",
        "description": "Pull current news headlines. Optionally filtered by topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Optional topic filter."},
                "max_per_feed": {"type": "integer", "default": 3},
            },
        },
    },
    {
        "name": "read_file",
        "description": "Read a local text file. Path is sandboxed to user-allowed directories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer", "default": 20000},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text to a local file. Refuses to overwrite unless overwrite=true.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "remember_fact",
        "description": (
            "Store a fact about the user for future sessions. Use when the user "
            "explicitly tells you something to remember, or when you deduce a "
            "stable preference (e.g. favorite color, name, timezone, dietary "
            "preference). Key is short identifier; value is the fact."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
                "category": {"type": "string", "default": "general"},
                "confidence": {"type": "number", "default": 0.9, "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "forget_fact",
        "description": "Remove a previously stored fact from memory.",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "read_pdf",
        "description": (
            "Read and extract text from a PDF file. Returns the text content "
            "of the PDF for summarisation or analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Full path to the PDF file."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "send_email",
        "description": (
            "Send an email. Requires EMAIL_USER and EMAIL_PASS set in backend/.env "
            "for Gmail SMTP (smtp.gmail.com:587)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Email subject line."},
                "body": {"type": "string", "description": "Email body text."},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "set_reminder",
        "description": (
            "Schedule a reminder that JARVIS will deliver at the given time. "
            "time_iso must be ISO 8601 in the user's timezone. e.g. 2026-06-25T20:30:00."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "time_iso": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["time_iso", "message"],
        },
    },
    {
        "name": "system_info",
        "description": "Get basic system information (OS, battery, python version).",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "open_app",
        "description": (
            "Launch a desktop application by friendly name (notepad, chrome, "
            "spotify, code, etc.) or by exe filename. JARVIS will PROPOSE the "
            "action and wait for user approval before launching."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Friendly name (e.g. 'notepad', 'chrome') or exe filename.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "kill_process",
        "description": (
            "Terminate a running process by name (e.g. 'notepad', 'chrome', "
            "'Spotify'). JARVIS will PROPOSE which processes will be killed "
            "and wait for user approval."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Process name (no .exe needed)."},
            },
            "required": ["name"],
        },
    },
    {
        "name": "screenshot",
        "description": (
            "Capture the current screen and return a path to the saved PNG. "
            "The image is also streamed inline to the HUD chat."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "default": 0,
                    "description": "0 = primary, 1+ = secondary monitor.",
                },
            },
        },
    },
    {
        "name": "capture_screen",
        "description": (
            "Capture the screen and return image data for AI vision analysis. "
            "Use this when the user asks about what's on screen, wants to explain "
            "an app, design, UI, or anything visible. JARVIS will see the screen "
            "and describe it intelligently."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "set_wallpaper",
        "description": "Change the Windows desktop wallpaper to an image file. Supports JPG, PNG, BMP. Use style='fill', 'stretch', 'fit', 'tile', or 'center'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Full path to image file, or relative to JARVIS/images/ folder."},
                "style": {"type": "string", "default": "fill", "description": "fill/stretch/fit/tile/center"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "android_answer_call",
        "description": "Answer the incoming call on the connected Android phone. Use when user says 'pick up' or 'answer that call'.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "android_reject_call",
        "description": "Reject the incoming call on the connected Android phone. Use when user says 'reject that call' or 'send to voicemail'.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "android_send_sms",
        "description": "Send an SMS text message from the connected Android phone to a phone number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "phone": {"type": "string", "description": "Recipient phone number, e.g. '+919876543210'"},
                "message": {"type": "string", "description": "Message text to send"},
            },
            "required": ["phone", "message"],
        },
    },
    {
        "name": "android_open_app",
        "description": "Open an app on the connected Android phone by package name. Common packages: com.android.dialer (phone), com.google.android.apps.messaging (messages), com.android.camera, com.google.android.youtube, com.instagram.android, com.whatsapp, com.spotify.music.",
        "input_schema": {
            "type": "object",
            "properties": {
                "package": {"type": "string", "description": "Android package name, e.g. 'com.google.android.youtube'"},
            },
            "required": ["package"],
        },
    },
    {
        "name": "android_close_app",
        "description": "Force-stop an app on the connected Android phone by package name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "package": {"type": "string"},
            },
            "required": ["package"],
        },
    },
    {
        "name": "android_notifications",
        "description": "Read recent notifications from the connected Android phone. Returns last few notifications.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "android_phone_info",
        "description": "Get the connected Android phone's model, OS version, and battery level.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "android_lock_screen",
        "description": "Lock the screen of the connected Android phone.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "android_volume",
        "description": "Set the media volume on the connected Android phone (0-100).",
        "input_schema": {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["level"],
        },
    },
    {
        "name": "android_wifi",
        "description": "Turn the WiFi on the connected Android phone on or off.",
        "input_schema": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "enum": ["on", "off"], "default": "on"},
            },
        },
    },
    {
        "name": "android_bluetooth",
        "description": "Turn Bluetooth on the connected Android phone on or off.",
        "input_schema": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "enum": ["on", "off"], "default": "on"},
            },
        },
    },
    {
        "name": "android_devices",
        "description": "List Android devices currently connected via ADB to this machine. Returns serial numbers and connection states.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "monitor_calls",
        "description": "Check if there is an incoming call ringing on the connected Android phone. Returns incoming_call: true/false and caller number if available.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "make_call",
        "description": "Make a phone call to a specific number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "number": {"type": "string"},
            },
            "required": ["number"],
        },
    },
    {
        "name": "get_call_history",
        "description": "Get recent call history from the Android phone. Returns list of last 10 calls with numbers and times.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_sms_history",
        "description": "Get recent SMS messages from the Android phone. Returns last 10 messages.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_clipboard",
        "description": "Read the current text contents of the system clipboard.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_clipboard",
        "description": "Write text to the system clipboard.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "set_volume",
        "description": "Set the system master volume to a level from 0 (mute) to 100.",
        "input_schema": {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["level"],
        },
    },
    {
        "name": "type_text",
        "description": (
            "Type text into the currently-focused window using the keyboard. "
            "Useful after opening an app and wanting to fill in a field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "interval": {"type": "number", "default": 0.0,
                             "description": "Seconds between keystrokes."},
            },
            "required": ["text"],
        },
    },
    {
        "name": "play_media",
        "description": (
            "Play music or media. Provide a song/artist query ('Believer by "
            "Imagine Dragons') or a URI (spotify:track:..., https://youtube.com/..."
            "). JARVIS searches for the URI and hands it to the OS."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Search query or full media URI."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "lock_workstation",
        "description": "Lock the workstation (returns user to the lock screen).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "shutdown",
        "description": "Shut down the computer immediately. Destructive.",
        "input_schema": {
            "type": "object",
            "properties": {
                "when": {"type": "string", "default": "now",
                         "description": "Human-readable hint ('now', 'in 5 min')."},
            },
            "required": [],
        },
    },
    {
        "name": "restart",
        "description": "Restart the computer. Destructive.",
        "input_schema": {
            "type": "object",
            "properties": {
                "when": {"type": "string", "default": "now"},
            },
            "required": [],
        },
    },
    {
        "name": "find_files",
        "description": (
            "Search the local filesystem for files matching a glob pattern. "
            "Searches the user's home folder and C:\\ drive (skips system "
            "directories). Returns paths, sizes, and modification times."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name_glob": {
                    "type": "string",
                    "description": "Glob like '*.pdf', 'resume*', '*2026*'.",
                },
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional filter: ['.pdf', '.docx'].",
                },
                "max_results": {"type": "integer", "default": 25, "maximum": 100},
            },
            "required": ["name_glob"],
        },
    },
    {
        "name": "grep_files",
        "description": (
            "Search the contents of files for a regex or substring pattern. "
            "Walks the user's home folder. Returns file paths, line numbers, "
            "and matching snippets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex or literal substring."},
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: only search .py, .txt, .md, etc.",
                },
                "max_results": {"type": "integer", "default": 50, "maximum": 200},
                "case_sensitive": {"type": "boolean", "default": False},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "send_whatsapp",
        "description": (
            "Send a WhatsApp message to a contact by name or phone number "
            "(E.164 format, e.g. +919876543210). "
            "Requires WhatsApp Web auth (QR scan on first use)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "phone": {"type": "string", "description": "Phone number or contact name."},
                "message": {"type": "string", "description": "Message text to send."},
            },
            "required": ["phone", "message"],
        },
    },
    {
        "name": "get_whatsapp_messages",
        "description": (
            "Read recent WhatsApp messages from a contact. "
            "Returns last N messages with sender, text, and time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "phone": {"type": "string", "description": "Phone number or contact name."},
                "limit": {"type": "integer", "default": 10, "maximum": 50},
            },
            "required": ["phone"],
        },
    },
    {
        "name": "capture_screen",
        "description": "Capture the current screen and return image data for analysis. Useful when user asks about what's on screen.",
        "input_schema": {
            "type": "object",
            "properties": {
                "monitor": {"type": "integer", "default": 0, "description": "Monitor index (0=primary, 1+=secondary)"},
            },
        },
    },
    {
        "name": "get_all_windows",
        "description": "Get a list of all open windows with titles and process IDs.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "activate_window",
        "description": "Bring a window to the front and activate it by title.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Partial or full title of the window"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "find_and_click",
        "description": "Find UI element by text and click it automatically.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text of the UI element to find (button, link, etc.)"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "type_in_field",
        "description": "Type text into a specific UI field (after clicking to focus).",
        "input_schema": {
            "type": "object",
            "properties": {
                "field_text": {"type": "string", "description": "Text of the field to find"},
                "text": {"type": "string", "description": "Text to type into the field"},
            },
            "required": ["field_text", "text"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate mathematical expressions and return the result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5', 'sqrt(16)', 'sin(90 degrees))"
                },
                "precision": {
                    "type": "integer",
                    "default": 2,
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Number of decimal places in the result"
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_time",
        "description": "Get the current date and time in various formats.",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "default": "readable",
                    "enum": ["readable", "iso", "time", "date", "unix"],
                    "description": "Format for the output: 'readable' (e.g., '2:30 PM, Monday June 25 2026'), 'iso' (ISO 8601), 'time' (just time), 'date' (just date), 'unix' (seconds since epoch)"
                },
                "timezone": {
                    "type": "string",
                    "default": "Asia/Kolkata",
                    "description": "Timezone (e.g., 'Asia/Kolkata', 'America/New_York', 'UTC')"
                }
            },
        },
    },
    {
        "name": "take_screenshot",
        "description": "Take a screenshot of the current screen.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [x, y, width, height] region to capture"
                }
            },
        },
    },
    {
        "name": "click_element",
        "description": "Click on a UI element described by the user. Takes a screenshot and uses Claude Vision to find the element.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Description of the element to click (e.g., 'the search button', 'the submit form')"
                }
            },
            "required": ["description"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text at the current cursor position.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to type"
                },
                "interval": {
                    "type": "number",
                    "default": 0.05,
                    "description": "Delay between keystrokes in seconds"
                }
            },
            "required": ["text"],
        },
    },
    {
        "name": "press_hotkey",
        "description": "Press a keyboard shortcut.",
        "input_schema": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keys to press (e.g., ['ctrl', 'c'] for Ctrl+C, ['alt', 'f4'] for Alt+F4)"
                }
            },
            "required": ["keys"],
        },
    },
    {
        "name": "scroll_screen",
        "description": "Scroll the screen up or down.",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Direction to scroll"
                },
                "amount": {
                    "type": "integer",
                    "default": 3,
                    "description": "Amount to scroll"
                }
            },
            "required": ["direction"],
        },
    },
    {
        "name": "read_screen",
        "description": "Read all visible text on the screen using OCR.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [x, y, width, height] region to read"
                }
            },
        },
    },
    {
        "name": "find_and_click",
        "description": "Find a UI element by description and click it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Description of the element to find and click"
                }
            },
            "required": ["description"],
        },
    },
    {
        "name": "open_url",
        "description": "Open a URL in the default browser.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to open"
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "get_active_window",
        "description": "Get the title of the currently active window.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


# -----------------------------------------------------------------------------
# Approval-gated tool execution (dry-run → user approves → run)
# -----------------------------------------------------------------------------

import asyncio
import base64
import logging
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from brain.context import _local_now
from tools import system as system_tool  # full PC-control module


# In-memory registry of pending approval actions.
#   action_id -> {tool, inputs, plan, future, created_at}
_PENDING: dict[str, dict[str, Any]] = {}


def _log_activity(kind: str, target: str, status: str, detail: str = "") -> None:
    """Append a row to the in-memory + DB activity log."""
    try:
        from memory.long_term import log_activity
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(log_activity(kind, target, status, detail))
    except Exception:
        pass


def _broadcast_toast(text: str, kind: str = "info") -> None:
    """Push a short toast notification to all WS clients.

    kind in: info | ok | warn | error
    """
    _broadcast({"type": "toast", "kind": kind, "text": text})


async def process_tool_call(name: str, inputs: dict) -> str:
    """
    Single-block backwards-compatible wrapper around process_compound_plan.
    """
    return await process_compound_plan([(name, inputs, uuid.uuid4().hex)])


async def process_compound_plan(blocks: list[tuple[str, dict, str]]) -> str:
    """
    Handle a Claude turn that may chain multiple tool calls.

    `blocks` is a list of (name, inputs, tool_use_id) — one per tool_use block.

    Two-tier behaviour:
      - If every tool is trivial/read-only: run all immediately, return the
        joined results so Claude can continue.
      - If ANY tool is destructive (in APPROVAL_REQUIRED): build a plan for
        each, emit ONE consolidated approval card, run all on approval,
        deny all on rejection.

    Each tool block's result string is appended as a separate tool_result
    entry in the order it ran, so Claude can correlate them with tool_use_id.
    """
    if not blocks:
        return ""

    # 1. Decide: any destructive in the batch?
    trust_destructive = await _trust_destructive()
    needs_approval = any(
        b[0] in APPROVAL_REQUIRED for b in blocks
    ) and not trust_destructive

    # 2. Build plans for destructive blocks up front (so the card is accurate)
    plans: list[tuple[str, dict, str, dict]] = []  # (name, inputs, id, plan)
    for name, inputs, tuid in blocks:
        if name in APPROVAL_REQUIRED:
            plan = await _build_plan(name, inputs)
            plans.append((name, inputs, tuid, plan))

    if not needs_approval:
        # 2a. Auto-run everything in order. Trivial blocks go through
        # _execute_tool directly; destructive blocks (when trust is on)
        # also use _run_approved.
        return await _run_blocks(blocks, plans)

    # 3. Build a single consolidated approval card
    action_id = uuid.uuid4().hex[:12]
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    _PENDING[action_id] = {
        "tool": "compound",
        "blocks": blocks,
        "plans": plans,
        "future": future,
        "created_at": _local_now(),
    }

    descriptions = []
    for name, inputs, _, plan in plans:
        if plan.get("ok") is False:
            descriptions.append(f"❌ {name}: {plan.get('error', 'plan failed')}")
        else:
            descriptions.append("• " + system_tool.describe_plan(plan))
    # Also include any trivial steps that will run alongside
    trivial_descs = []
    for name, inputs, _ in blocks:
        if name not in APPROVAL_REQUIRED:
            trivial_descs.append(f"• {name}({_summarize_inputs(name, inputs)})")
    if trivial_descs:
        descriptions.append("")
        descriptions.append("And these auto-runs:")
        descriptions.extend(trivial_descs)

    _broadcast({
        "type": "approval_request",
        "action_id": action_id,
        "tool": "compound",
        "plan": {"steps": [p for _, _, _, p in plans]},
        "description": "\n".join(descriptions),
    })

    # 4. Wait for verdict
    try:
        decision: dict = await asyncio.wait_for(future, timeout=60.0)
    except asyncio.TimeoutError:
        _PENDING.pop(action_id, None)
        _broadcast_toast("Approval timed out — nothing was run.", kind="warn")
        for name, inputs, _, plan in plans:
            _log_activity(name, _summarize_inputs(name, inputs),
                          "denied", "approval timed out (compound)")
        # Return error strings for each block
        out = []
        for name, _, tuid in blocks:
            out.append(json.dumps(
                {"ok": False, "error": "compound approval timed out"},
                ensure_ascii=False,
            ))
        return "\n".join(out)
    finally:
        _PENDING.pop(action_id, None)

    if not decision.get("approved"):
        _broadcast_toast("Denied — no actions were run.", kind="warn")
        for name, inputs, _, plan in plans:
            _log_activity(name, _summarize_inputs(name, inputs),
                          "denied", "compound denied by user")
        out = []
        for name, _, tuid in blocks:
            out.append(json.dumps(
                {"ok": False, "error": "compound denied by user"},
                ensure_ascii=False,
            ))
        return "\n".join(out)

    # 5. Run all blocks in order
    _broadcast_toast(
        f"Running {len(blocks)} step{'s' if len(blocks) != 1 else ''}…",
        kind="info",
    )
    return await _run_blocks(blocks, plans)


async def _run_blocks(blocks: list[tuple[str, dict, str]],
                      plans: list[tuple[str, dict, str, dict]]) -> str:
    """Execute each block in order; collect per-block JSON results.

    Returns the joined string that the chat loop will append as separate
    tool_result entries (each separated by a known delimiter).
    """
    results: list[str] = []
    for name, inputs, tuid in blocks:
        # Find matching plan if destructive
        plan = next((p[3] for p in plans if p[2] == tuid), None)
        if name in APPROVAL_REQUIRED and plan is not None:
            if not plan.get("ok"):
                # Plan failed (e.g. process not found) — surface that
                results.append(json.dumps(plan, ensure_ascii=False))
                _log_activity(name, _summarize_inputs(name, inputs),
                              "error", plan.get("error", ""))
                continue
            r = await _run_approved(name, inputs, plan)
        else:
            r = await _execute_tool(name, inputs)
        results.append(r)
    # Joined with a sentinel so the chat loop can split
    return "\n".join(results)


def _summarize_inputs(name: str, inputs: dict) -> str:
    """Short string for the activity log target column."""
    if name == "kill_process":
        return inputs.get("name", "?")
    if name == "write_file":
        return inputs.get("path", "?")
    if name == "lock_workstation":
        return "workstation"
    if name in ("shutdown", "restart"):
        return inputs.get("when", "now")
    if name == "play_media":
        return inputs.get("query", "?")
    if name == "open_app":
        return inputs.get("name", "?")
    if name == "set_volume":
        return str(inputs.get("level", "?"))
    if name == "type_text":
        return f"{len(inputs.get('text', ''))} chars"
    return json.dumps(inputs)[:60]


async def _build_plan(name: str, inputs: dict) -> dict:
    """Build a plan dict for a destructive tool (for the approval card)."""
    if name == "kill_process":
        return await system_tool.kill_process(inputs.get("name", ""))
    if name == "write_file":
        return {"ok": True, "action": "write_file",
                "path": inputs.get("path", ""),
                "bytes": len(inputs.get("content", "")),
                "preview": inputs.get("content", "")[:80]}
    if name == "lock_workstation":
        return {"ok": True, "action": "lock_workstation"}
    if name == "shutdown":
        return {"ok": True, "action": "shutdown",
                "when": inputs.get("when", "now")}
    if name == "restart":
        return {"ok": True, "action": "restart",
                "when": inputs.get("when", "now")}
    return {"ok": True, "action": name}


_TRUST_DESTRUCTIVE: Optional[bool] = None


async def _trust_destructive() -> bool:
    """Cached check on the 'trust_destructive_actions' setting."""
    global _TRUST_DESTRUCTIVE
    if _TRUST_DESTRUCTIVE is not None:
        return _TRUST_DESTRUCTIVE
    try:
        from memory.long_term import get_fact
        v = await get_fact("trust_destructive_actions")
        _TRUST_DESTRUCTIVE = bool(v and v.lower() in ("1", "true", "yes", "on"))
    except Exception:
        _TRUST_DESTRUCTIVE = False
    return _TRUST_DESTRUCTIVE


def set_trust_destructive(enabled: bool) -> None:
    """Invalidate cache and store the user's preference."""
    global _TRUST_DESTRUCTIVE
    _TRUST_DESTRUCTIVE = enabled
    # Persist via memory.long_term
    try:
        from memory import long_term
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(
                long_term.set_fact(
                    "trust_destructive_actions",
                    "1" if enabled else "0",
                    "settings", 1.0,
                )
            )
    except Exception:
        pass


async def _run_approved(name: str, inputs: dict, plan: dict) -> dict[str, Any]:
    """Actually execute an approved plan. Returns a result dict."""
    try:
        if name == "open_app":
            return system_tool.execute_open_app(plan["target"])
        if name == "kill_process":
            return system_tool.execute_kill_process(inputs["name"])
        if name == "set_clipboard":
            return await system_tool.set_clipboard(inputs["text"])
        if name == "set_volume":
            return await system_tool.set_volume(inputs["level"])
        if name == "type_text":
            return await system_tool.type_text(inputs["text"], inputs.get("interval", 0.0))
        if name == "write_file":
            from tools import files as files_tool
            return await files_tool.write_file(
                inputs["path"], inputs["content"], inputs.get("overwrite", False)
            )
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": f"unhandled approved tool: {name}"}


def resolve_approval(action_id: str, approved: bool, reason: str = "") -> bool:
    """Called by main.py when the user clicks Approve/Deny. Returns True if
    there was a pending action with that id."""
    entry = _PENDING.get(action_id)
    if not entry:
        return False
    if not entry["future"].done():
        entry["future"].set_result({"approved": approved, "reason": reason})
    return True


async def _stream_screenshot(result: dict) -> None:
    """Read the saved PNG and push it inline to all clients as a data URL."""
    path = result.get("path")
    if not path or not Path(path).exists():
        return
    try:
        data = Path(path).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        _broadcast({
            "type": "image",
            "src": f"data:image/png;base64,{b64}",
            "filename": result.get("filename"),
            "path": path,
        })
    except Exception:
        pass


from pathlib import Path


# -----------------------------------------------------------------------------
# Tool execution
# -----------------------------------------------------------------------------

async def _execute_tool(name: str, inputs: dict) -> str:
    """Run a tool and return a string result for Claude.

    Tools in APPROVAL_REQUIRED do NOT execute here — they return a plan and
    wait for the user to approve via /approve/{action_id}. The wrapper
    process_tool_call handles the plan/approve/run cycle.
    """
    try:
        if name == "web_search":
            results = search.web_search(
                inputs.get("query", ""),
                max_results=inputs.get("max_results", 5),
            )
            return search.format_results(results)
        if name == "get_weather":
            w = await apis.get_weather(inputs["lat"], inputs["lon"])
            return apis.format_weather(w)
        if name == "calculator":
            try:
                import math
                from typing import Any
                # Evaluate the expression safely
                expression = inputs["expression"]
                precision = inputs.get("precision", 2)

                # Replace common functions
                expression = expression.replace("sqrt(", "math.sqrt(")
                expression = expression.replace("sin(", "math.sin(")
                expression = expression.replace("cos(", "math.cos(")
                expression = expression.replace("tan(", "math.tan(")
                expression = expression.replace("log(", "math.log10(")
                expression = expression.replace("ln(", "math.log(")
                expression = expression.replace("pi", "math.pi")
                expression = expression.replace("e", "math.e")

                # Add degrees to radians conversion
                if "degrees" in expression:
                    expression = expression.replace("degrees(", "math.radians(")

                # Evaluate
                result = eval(expression, {"__builtins__": {}, "math": math})

                # Format with precision
                if isinstance(result, float):
                    result = round(result, precision)

                return f"The result is {result}"
            except Exception as e:
                return f"Error calculating: {str(e)}"
        if name == "get_time":
            from zoneinfo import ZoneInfo
            from datetime import datetime
            tz_name = inputs.get("timezone", "Asia/Kolkata")
            fmt = inputs.get("format", "readable")
            try:
                tz = ZoneInfo(tz_name)
                now = datetime.now(tz)
                if fmt == "readable":
                    result = now.strftime("%I:%M %p, %A %d %B %Y")
                elif fmt == "iso":
                    result = now.isoformat()
                elif fmt == "time":
                    result = now.strftime("%I:%M %p")
                elif fmt == "date":
                    result = now.strftime("%A %d %B %Y")
                elif fmt == "unix":
                    result = str(int(now.timestamp()))
                else:
                    result = now.strftime("%I:%M %p, %A %d %B %Y")
                return f"Current time: {result}"
            except Exception as e:
                return f"Error getting time: {str(e)}"
        if name == "take_screenshot":
            from tools.screen_control import take_screenshot
            region = inputs.get("region")
            b64 = take_screenshot(region)
            if b64:
                return f"Screenshot captured ({len(b64)} bytes)"
            return "Failed to capture screenshot"
        if name == "click_element":
            from tools.screen_control import click_element
            result = click_element(inputs["description"])
            return result.get("message", str(result))
        if name == "find_and_click":
            from tools.screen_control import find_and_click
            result = find_and_click(inputs["description"])
            return result.get("message", str(result))
        if name == "type_text":
            from tools.screen_control import type_text
            result = type_text(inputs["text"], inputs.get("interval", 0.05))
            return result.get("message", str(result))
        if name == "press_hotkey":
            from tools.screen_control import press_hotkey
            result = press_hotkey(*inputs["keys"])
            return result.get("message", str(result))
        if name == "scroll_screen":
            from tools.screen_control import scroll
            result = scroll(inputs["direction"], inputs.get("amount", 3))
            return result.get("message", str(result))
        if name == "read_screen":
            from tools.screen_control import read_screen_text
            region = inputs.get("region")
            text = read_screen_text(region)
            return text if text else "No text found on screen"
        if name == "open_url":
            from tools.screen_control import open_url
            result = open_url(inputs["url"])
            return result.get("message", str(result))
        if name == "get_active_window":
            from tools.screen_control import get_active_window
            return f"Active window: {get_active_window()}"
        if name == "get_news":
            items = await apis.get_news(inputs.get("topic"), inputs.get("max_per_feed", 3))
            return apis.format_news(items)
        if name == "read_file":
            r = await files.read_file(inputs["path"], inputs.get("max_chars", 20000))
            return json.dumps(r, ensure_ascii=False)
        if name == "write_file":
            r = await files.write_file(
                inputs["path"], inputs["content"], inputs.get("overwrite", False)
            )
            return json.dumps(r, ensure_ascii=False)
        if name == "remember_fact":
            await long_term.set_fact(
                inputs["key"], inputs["value"],
                inputs.get("category", "general"),
                inputs.get("confidence", 0.9),
            )
            return f"Remembered: {inputs['key']} = {inputs['value']}"
        if name == "forget_fact":
            ok = await long_term.delete_fact(inputs["key"])
            return f"Forgot: {inputs['key']}" if ok else f"No fact with key: {inputs['key']}"
        if name == "set_reminder":
            await long_term.add_rule(
                name=f"reminder_{inputs['time_iso']}",
                condition={"fire_at": inputs["time_iso"]},
                action="fire_reminder",
                action_payload={"message": inputs["message"], "time_iso": inputs["time_iso"]},
                cooldown_hours=0,
            )
            return f"Reminder set for {inputs['time_iso']}: {inputs['message']}"
        if name == "system_info":
            return json.dumps(system_tool.system_info(), ensure_ascii=False)
        if name == "find_files":
            r = files.find_files(
                inputs["name_glob"],
                extensions=inputs.get("extensions"),
                max_results=inputs.get("max_results", 25),
            )
            return json.dumps(r, ensure_ascii=False)
        if name == "grep_files":
            r = files.grep_in_files(
                inputs["pattern"],
                extensions=inputs.get("extensions"),
                max_results=inputs.get("max_results", 50),
                case_sensitive=inputs.get("case_sensitive", False),
            )
            return json.dumps(r, ensure_ascii=False)
        if name == "open_app":
            # Auto-run path (trivial tool, no approval card).
            plan = await system_tool.open_app(inputs["name"])
            if not plan.get("ok"):
                _broadcast_toast(
                    f"Couldn't find '{inputs['name']}'", kind="error"
                )
                _log_activity("open_app", inputs["name"], "error",
                              plan.get("error", ""))
                return json.dumps(plan, ensure_ascii=False)
            result = system_tool.execute_open_app(plan["target"])
            _broadcast_toast(
                f"Opened {plan.get('display', plan['target'])}", kind="ok"
            )
            _log_activity("open_app",
                          plan.get("display", plan["target"]),
                          "ok" if result.get("ok") else "error",
                          result.get("message") or result.get("error", ""))
            return json.dumps(result, ensure_ascii=False)
        if name == "set_volume":
            r = await system_tool.set_volume(inputs["level"])
            _broadcast_toast(
                f"🔊 Volume {max(0, min(100, int(inputs['level'])))}%",
                kind="ok" if r.get("ok") else "error"
            )
            _log_activity("set_volume", str(inputs["level"]),
                          "ok" if r.get("ok") else "error", "")
            return json.dumps(r, ensure_ascii=False)
        if name == "set_clipboard":
            r = await system_tool.set_clipboard(inputs["text"])
            _broadcast_toast(
                f"📋 Copied {len(inputs['text'])} chars", kind="ok"
            )
            _log_activity("set_clipboard", str(len(inputs["text"])),
                          "ok" if r.get("ok") else "error", "")
            return json.dumps(r, ensure_ascii=False)
        if name == "type_text":
            r = await system_tool.type_text(
                inputs["text"], inputs.get("interval", 0.0)
            )
            _broadcast_toast(
                f"⌨ Typed {len(inputs['text'])} chars", kind="ok" if r.get("ok") else "error"
            )
            _log_activity("type_text", str(len(inputs["text"])),
                          "ok" if r.get("ok") else "error", "")
            return json.dumps(r, ensure_ascii=False)
        if name == "play_media":
            _broadcast_toast(f"🎵 Searching for '{inputs['query']}'…", kind="info")
            r = await system_tool.play_media(inputs["query"])
            title = r.get("title") or inputs["query"]
            if r.get("ok"):
                _broadcast_toast(f"🎵 Playing {title}", kind="ok")
            else:
                _broadcast_toast(f"🎵 Couldn't find {inputs['query']}",
                                 kind="error")
            _log_activity("play_media", title,
                          "ok" if r.get("ok") else "error",
                          r.get("uri") or r.get("error", ""))
            return json.dumps(r, ensure_ascii=False)
        if name == "screenshot":
            r = await system_tool.screenshot(inputs.get("monitor", 0))
            if r.get("ok"):
                await _stream_screenshot(r)
                _broadcast_toast("📸 Screenshot saved", kind="ok")
            else:
                _broadcast_toast("📸 Screenshot failed", kind="error")
            _log_activity("screenshot", r.get("path", ""),
                          "ok" if r.get("ok") else "error",
                          str(r.get("size_bytes", "")))
            return json.dumps(r, ensure_ascii=False)
        if name == "capture_screen":
            r = await system_tool.capture_screen()
            # Handle result as both dict and tuple
            try:
                ok = r[0] if isinstance(r, tuple) else r.get("ok")
                path = r[1] if isinstance(r, tuple) else r.get("path", "")
            except Exception:
                ok = False
                path = ""
            if ok:
                # Convert to dict for _stream_screenshot
                result_dict = {"ok": True, "path": path} if not isinstance(r, dict) else r
                await _stream_screenshot(result_dict)
                _broadcast_toast("📸 Screen captured for analysis", kind="ok")
            else:
                _broadcast_toast("📸 Screen capture failed", kind="error")
            _log_activity("capture_screen", path,
                          "ok" if ok else "error",
                          "unknown")
            return json.dumps({"ok": ok, "path": path}, ensure_ascii=False)
        if name == "set_wallpaper":
            r = await system_tool.set_wallpaper(inputs.get("path", ""), inputs.get("style", "fill"))
            _broadcast_toast(f"🖼️ Wallpaper set to {inputs.get('path', '')}",
                            kind="ok" if r.get("ok") else "error")
            _log_activity("set_wallpaper", inputs.get("path", ""),
                          "ok" if r.get("ok") else "error",
                          r.get("error", ""))
            return json.dumps(r, ensure_ascii=False)
        if name == "android_devices":
            r = await system_tool.android_devices()
            return json.dumps(r, ensure_ascii=False)
        if name == "android_answer_call":
            r = await system_tool.android_answer_call()
            _broadcast_toast("📱 Answering call", kind="ok" if r.get("ok") else "error")
            _log_activity("android_answer_call", "",
                          "ok" if r.get("ok") else "error",
                          r.get("error", ""))
            return json.dumps(r, ensure_ascii=False)
        if name == "android_reject_call":
            r = await system_tool.android_reject_call()
            _broadcast_toast("📱 Rejected call", kind="ok" if r.get("ok") else "error")
            _log_activity("android_reject_call", "",
                          "ok" if r.get("ok") else "error",
                          r.get("error", ""))
            return json.dumps(r, ensure_ascii=False)
        if name == "android_send_sms":
            r = await system_tool.android_send_sms(inputs.get("phone", ""), inputs.get("message", ""))
            _broadcast_toast(f"📱 SMS → {inputs.get('phone', '')}", kind="ok" if r.get("ok") else "error")
            _log_activity("android_send_sms", inputs.get("phone", ""),
                          "ok" if r.get("ok") else "error",
                          inputs.get("message", "")[:50])
            return json.dumps(r, ensure_ascii=False)
        if name == "android_open_app":
            r = await system_tool.android_open_app(inputs.get("package", ""))
            _broadcast_toast(f"📱 Opened {inputs.get('package', '')}", kind="ok" if r.get("ok") else "error")
            _log_activity("android_open_app", inputs.get("package", ""),
                          "ok" if r.get("ok") else "error",
                          r.get("error", ""))
            return json.dumps(r, ensure_ascii=False)
        if name == "android_close_app":
            r = await system_tool.android_close_app(inputs.get("package", ""))
            _broadcast_toast(f"📱 Closed {inputs.get('package', '')}", kind="ok" if r.get("ok") else "error")
            _log_activity("android_close_app", inputs.get("package", ""),
                          "ok" if r.get("ok") else "error", "")
            return json.dumps(r, ensure_ascii=False)
        if name == "android_notifications":
            r = await system_tool.android_notifications()
            return json.dumps(r, ensure_ascii=False)
        if name == "android_phone_info":
            r = await system_tool.android_phone_info()
            return json.dumps(r, ensure_ascii=False)
        if name == "android_lock_screen":
            r = await system_tool.android_lock_screen()
            _broadcast_toast(" Locked phone", kind="ok" if r.get("ok") else "error")
            _log_activity("android_lock_screen", "",
                          "ok" if r.get("ok") else "error", "")
            return json.dumps(r, ensure_ascii=False)
        if name == "android_volume":
            r = await system_tool.android_volume(inputs.get("level", 50))
            _broadcast_toast(f" Phone volume {inputs.get('level', 50)}%",
                            kind="ok" if r.get("ok") else "error")
            _log_activity("android_volume", str(inputs.get("level", 50)),
                          "ok" if r.get("ok") else "error", "")
            return json.dumps(r, ensure_ascii=False)
        if name == "android_wifi":
            r = await system_tool.android_wifi(inputs.get("state", "on"))
            _broadcast_toast(f" WiFi {inputs.get('state', 'on')}",
                            kind="ok" if r.get("ok") else "error")
            _log_activity("android_wifi", inputs.get("state", "on"),
                          "ok" if r.get("ok") else "error", "")
            return json.dumps(r, ensure_ascii=False)
        if name == "android_bluetooth":
            r = await system_tool.android_bluetooth(inputs.get("state", "on"))
            _broadcast_toast(f" Bluetooth {inputs.get('state', 'on')}",
                            kind="ok" if r.get("ok") else "error")
            _log_activity("android_bluetooth", inputs.get("state", "on"),
                          "ok" if r.get("ok") else "error", "")
            return json.dumps(r, ensure_ascii=False)
        if name == "read_pdf":
            r = await system_tool.read_pdf(inputs.get("path", ""))
            return json.dumps(r, ensure_ascii=False)
        if name == "send_whatsapp":
            r = await whatsapp_tool.send_whatsapp(
                inputs.get("phone", ""),
                inputs.get("message", ""),
            )
            if r.get("ok"):
                return f"WhatsApp sent to {inputs.get('phone', '')}, Sir."
            return f"WhatsApp failed: {r.get('error', 'unknown error')}"
        if name == "get_whatsapp_messages":
            r = await whatsapp_tool.get_whatsapp_messages(
                inputs.get("phone", ""),
                limit=inputs.get("limit", 10),
            )
            if r.get("ok"):
                msgs = r.get("messages", [])
                lines = []
                for m in msgs[:10]:
                    sender = m.get("from", m.get("sender", "?"))
                    body = m.get("body", m.get("text", ""))
                    lines.append(f"[{sender}] {body}")
                return "\n".join(lines) if lines else "No messages found."
            return f"WhatsApp read failed: {r.get('error', 'unknown error')}"
        if name == "send_email":
            r = await system_tool.send_email(
                inputs.get("to", ""),
                inputs.get("subject", ""),
                inputs.get("body", ""),
            )
            _broadcast_toast(f" Email sent to {inputs.get('to', '')}",
                            kind="ok" if r.get("ok") else "error")
            return json.dumps(r, ensure_ascii=False)
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool {name} failed: {e}"


# -----------------------------------------------------------------------------
# Anthropic client (lazy)
# -----------------------------------------------------------------------------
from llm import get_provider
from llm.provider import ChatResponse

async def _get_chat_provider():
    """Get the configured LLM provider."""
    return get_provider()


# -----------------------------------------------------------------------------
# Main chat entry point
# -----------------------------------------------------------------------------

async def _history_messages(user_id: str, limit: int = 20) -> list[dict]:
    """Pull recent user/assistant turns as Anthropic message history."""
    items = buffer.recent(user_id, limit=limit)
    out = []
    for it in items:
        role = it["role"]
        if role not in ("user", "assistant"):
            continue
        out.append({"role": role, "content": it["content"]})
    return out


async def chat(user_id: str, message: str) -> dict:
    """
    Process a user message and return {reply, mood, tools_used, audio_b64?}.

    Uses streaming when no tool calls are needed — speaks the first sentence
    immediately while the rest generates (activated via streaming_patch).

    For complex goals, automatically activates Planning Layer to break them
    into executable task sequences.
    """
    # 1. Context (which also calls sentiment)
    ctx = await build_context(user_id, message)
    mood = ctx.get("mood") or "neutral"

    # 2. Planning Layer - Check if this is a complex goal
    # Keywords that indicate multi-step operations
    planning_keywords = ["organize", "sort", "clean up", "backup", "download all", "batch"]
    message_lower = message.lower()

    if any(kw in message_lower for kw in planning_keywords):
        try:
            from brain.planning import plan_and_execute
            plan = await plan_and_execute(message, ctx)

            if plan.status == "completed":
                plan_reply = f"Plan completed, Sir. {plan.result_summary}."
                await _remember_turn(user_id, message, plan_reply, mood)
                return {"reply": plan_reply, "mood": mood, "tools_used": ["planning"], "audio_b64": None}
            elif plan.status == "failed":
                plan_reply = f"Plan failed, Sir. {plan.result_summary}."
                await _remember_turn(user_id, message, plan_reply, mood)
                return {"reply": plan_reply, "mood": mood, "tools_used": ["planning"], "audio_b64": None}
        except Exception as e:
            logger.warning(f"Planning failed, falling through to normal processing: {e}")

    # 3. System prompt — inject current time so no tool call needed
    from datetime import datetime
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(settings.TIMEZONE)
    now = datetime.now(tz)
    time_str = now.strftime("%I:%M %p, %A %d %B %Y (%Z)")
    prompt = build_system_prompt(ctx)
    prompt = f"Current date and time: {time_str}\n\n" + prompt

    # 4. Conversation history
    history = await _history_messages(user_id)

    # 5. First call
    provider = await _get_chat_provider()
    messages = history + [{"role": "user", "content": message}]

    tools_used: list[str] = []

    try:
        # First call - enable tools for the LLM to request them
        response = await _retry_with_backoff(
            lambda: provider.chat(
                messages=messages,
                model=settings.CLAUDE_MODEL,
                max_tokens=1024,
                system=prompt,
                tools=TOOL_DEFINITIONS,
            ),
            max_retries=5,
            base_delay=2.0,
        )
    except Exception as e:
        # API failure — give a graceful, in-character reply
        err_reply = (
            "I seem to be having trouble reaching my cognitive systems at the moment, "
            "Sir. A moment of patience, if you would."
        )
        await _remember_turn(user_id, message, err_reply, mood)
        return {"reply": err_reply, "mood": mood, "tools_used": [], "audio_b64": None}

    # 5. Tool-use loop - execute tools LOCALLY, no 2nd API call needed
    while response.tool_calls:
        # Collect tool_use blocks
        tool_blocks = response.tool_calls
        if not tool_blocks:
            break

        # Execute each tool locally
        for tb in tool_blocks:
            tool_name = tb.get("name") or tb.get("function", {}).get("name", "unknown")
            tools_used.append(tool_name)
            tool_input = tb.get("input") or tb.get("function", {}).get("arguments", {})
            if isinstance(tool_input, str):
                import json as _json
                try:
                    tool_input = _json.loads(tool_input)
                except Exception:
                    tool_input = {}
            logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

            try:
                # Execute tool based on name
                if tool_name == "set_volume":
                    result = await system_tool.set_volume(tool_input.get("level", 50))
                    result_text = f"Volume set to {tool_input.get('level', 50)}, Sir." if result.get("ok") else f"Could not set volume: {result.get('error', 'unknown error')}"
                elif tool_name == "get_volume":
                    result = await system_tool.get_volume()
                    result_text = f"Current volume is {result.get('level', 'unknown')} percent."
                elif tool_name == "open_app":
                    result = await system_tool.open_app(tool_input.get("name", ""))
                    result_text = f"Opened {tool_input.get('name', 'app')}." if result.get("ok") else f"Could not open: {result.get('error', 'app not found')}"
                elif tool_name == "screenshot":
                    result = await system_tool.screenshot(tool_input.get("monitor", 0))
                    result_text = "Screenshot captured, Sir." if result.get("ok") else "Could not capture screenshot."
                elif tool_name == "lock_workstation":
                    result = await system_tool.lock_workstation()
                    result_text = "Workstation locked, Sir." if result.get("ok") else "Could not lock workstation."
                elif tool_name == "set_clipboard":
                    result = await system_tool.set_clipboard(tool_input.get("text", ""))
                    result_text = "Copied to clipboard, Sir." if result.get("ok") else "Could not copy to clipboard."
                elif tool_name == "get_clipboard":
                    result = await system_tool.get_clipboard()
                    result_text = result.get("text", "Clipboard is empty.")
                elif tool_name == "type_text":
                    result = await system_tool.type_text(tool_input.get("text", ""), tool_input.get("interval", 0.0))
                    result_text = "Text typed, Sir." if result.get("ok") else "Could not type text."
                elif tool_name == "play_media":
                    result = await system_tool.play_media(tool_input.get("query", ""))
                    result_text = f"Playing {tool_input.get('query', 'music')}." if result.get("ok") else "Could not play media."
                elif tool_name == "kill_process":
                    result = await system_tool.kill_process(tool_input.get("name", ""))
                    result_text = f"Stopped {tool_input.get('name', 'process')}." if result.get("ok") else "Could not stop process."
                elif tool_name == "web_search":
                    result = await search.search_web(tool_input.get("query", ""), tool_input.get("num_results", 5))
                    if result.get("ok"):
                        results = result.get("results", [])
                        result_text = " ".join([f"{r.get('title', '')}: {r.get('url', '')}" for r in results[:3]])
                    else:
                        result_text = "Search failed."
                elif tool_name == "get_weather":
                    result = await apis.get_weather(tool_input.get("city", ""))
                    if result.get("ok"):
                        result_text = f"Weather in {result.get('city', city)}: {result.get('temp', '')} degrees, {result.get('description', '')}"
                    else:
                        result_text = "Could not get weather."
                elif tool_name == "send_whatsapp":
                    result = await whatsapp_tool.send_whatsapp(tool_input.get("phone", ""), tool_input.get("message", ""))
                    result_text = "WhatsApp message sent, Sir." if result.get("ok") else f"WhatsApp failed: {result.get('error', '')}"
                elif tool_name == "get_whatsapp_messages":
                    result = await whatsapp_tool.get_whatsapp_messages(tool_input.get("phone", ""), tool_input.get("limit", 10))
                    if result.get("ok"):
                        msgs = result.get("messages", [])
                        result_text = "\n".join([f"{m.get('from', '?')}: {m.get('body', '')}" for m in msgs[:5]]) or "No messages."
                    else:
                        result_text = "Could not read WhatsApp messages."
                else:
                    result_text = f"Tool {tool_name} executed."

                # Add tool result to messages
                tool_id = tb.get("id", str(hash(tool_name)))
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_text,
                    }]
                })
                logger.info(f"Tool result: {result_text}")

            except Exception as e:
                logger.error(f"Tool execution failed: {tb.name} - {e}")
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tb.id,
                        "content": f"Error: {str(e)}",
                    }]
                })

        # Make a NEW API call with tool results (only one call per tool batch)
        # Using retry_with_backoff to handle 429 rate limits
        try:
            response = await _retry_with_backoff(
                lambda: provider.chat(
                    messages=messages,
                    model=settings.CLAUDE_MODEL,
                    max_tokens=1024,
                    system=prompt,
                    tools=[],
                ),
                max_retries=5,
                base_delay=2.0,
            )
        except Exception as e:
            logger.error(f"Follow-up API call failed: {e}")
            # If follow-up fails, just use the tool results directly
            response = None
            break

    # 6. Extract final text
    reply_text = ""
    if response and response.content:
        reply_text = response.content.strip()

    # Strip markdown JSON code fences if model wrapped response
    reply_text = re.sub(r'^```json\s*', '', reply_text)
    reply_text = re.sub(r'^```\s*', '', reply_text)
    reply_text = re.sub(r'\s*```$', '', reply_text).strip()

    # If entire reply is a JSON object, extract the "reply" field
    if reply_text.startswith('{'):
        try:
            parsed = json.loads(reply_text)
            if isinstance(parsed, dict):
                mood = parsed.get("mood", mood)
                reply_text = parsed.get("reply", reply_text)
        except json.JSONDecodeError:
            pass

    # Parse mood from inline JSON prefix if present
    mood_match = re.search(r'"mood"\s*:\s*"(\w+)"', reply_text)
    if mood_match:
        mood = mood_match.group(1)
        # Remove the JSON prefix from the spoken text
        reply_text = re.sub(r'\{[^}]*"mood"[^}]*\}', '', reply_text).strip()

    # Clean up any remaining tool call artifacts
    reply_text = re.sub(r'\[TOOL_CALL\].*?\[/TOOL_CALL\]', '', reply_text, flags=re.DOTALL).strip()
    reply_text = re.sub(r'<tool_call>.*?</tool_call>', '', reply_text, flags=re.DOTALL).strip()
    reply_text = re.sub(r'<function>.*?</function>', '', reply_text, flags=re.DOTALL).strip()
    reply_text = re.sub(r'```\w*\n?.*?\n?```', '', reply_text, flags=re.DOTALL).strip()
    reply_text = re.sub(r'\{[^}]*"tool"[^}]*\}', '', reply_text, flags=re.DOTALL).strip()
    reply_text = reply_text.strip()

    if not reply_text:
        reply_text = "I'm at a loss for words, Sir. A rare occurrence."

    # 7. Persist + buffer
    await _remember_turn(user_id, message, reply_text, mood)

    return {
        "reply": reply_text,
        "mood": mood,
        "tools_used": tools_used,
        "audio_b64": None,  # TTS is generated on-demand via /tts endpoint
    }


async def _remember_turn(user_id: str, user_msg: str, assistant_msg: str, mood: str) -> None:
    buffer.add(user_id, "user", user_msg, mood=mood)
    buffer.add(user_id, "assistant", assistant_msg)
    await long_term.add_conversation("user", user_msg, mood=mood)
    await long_term.add_conversation("assistant", assistant_msg)


# -----------------------------------------------------------------------------
# Proactive message generator
# -----------------------------------------------------------------------------

async def generate_proactive_message(user_id: str, action: str,
                                     payload: Optional[dict] = None) -> str:
    """
    Build a proactive JARVIS message (morning briefing, gentle nudge, etc.).
    Uses a lean prompt — no tool budget, just a short JARVIS-style line.
    """
    client = await _get_chat_provider()
    ctx = await build_context(user_id, current_message=f"[proactive:{action}]")
    base_prompt = build_system_prompt(ctx)

    action_instructions = {
        "morning_briefing": (
            "Compose Sir's morning briefing. Pull weather and any news that's relevant, "
            "and mention any reminders scheduled for today. Keep it under 90 seconds "
            "when spoken. Open with a short, warm greeting."
        ),
        "gentle_nudge": (
            "It's late and Sir has been at it for a while. Say something like: "
            "'Good evenin Sir, it's late — ready for some Batman work?' "
            "Keep it cool, short, and in character."
        ),
        "check_in": (
            "Sir has been away for several days. Write one or two warm sentences "
            "acknowledging the gap and gently checking in. No interrogation."
        ),
        "fire_reminder": (
            f"Sir set this reminder: {payload.get('message') if payload else ''}. "
            "Deliver it in JARVIS's voice, brief and clean."
        ),
    }
    instruction = action_instructions.get(action, "Deliver a brief JARVIS line.")

    messages = [{
        "role": "user",
        "content": (
            f"{instruction}\n\n"
            "Speak directly as JARVIS addressing the user. No meta-commentary. "
            "Just the line(s) to deliver."
        ),
    }]

    try:
        resp = await client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=400,
            system=base_prompt,
            tools=[],
            messages=messages,
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        return text or "At your service, Sir."
    except Exception:
        return "At your service, Sir."
    except Exception:
        # Local fallback if Claude API unreachable
        fallbacks = {
            "morning_briefing": "Good morning, Sir. A new day awaits.",
            "gentle_nudge": "Good evenin Sir, it's late — ready for some Batman work?",
            "check_in": "It has been a while, Sir. I trust all is in order.",
            "fire_reminder": (payload or {}).get("message", "You asked me to remind you of something, Sir."),
        }
        return fallbacks.get(action, "At your service, Sir.")


# -----------------------------------------------------------------------------
# One-shot TTS helper
# -----------------------------------------------------------------------------

async def speak(text: str, voice: Optional[str] = None) -> bytes:
    """Synthesize speech to MP3 bytes."""
    return await speaker.synthesize_bytes(text, voice=voice)


# -----------------------------------------------------------------------------


    reply_text = _re.sub(r'^```json\s*', '', reply_text)
    reply_text = _re.sub(r'^```\s*', '', reply_text)
    reply_text = _re.sub(r'\s*```$', '', reply_text).strip()
    reply_text = _re.sub(r'^#+\s*', '', reply_text)  # strip markdown headings

    await _remember_turn(user_id, message, reply_text, mood)
    return {
        "reply": reply_text,
        "mood": mood,
        "tools_used": result.tools_used,
        "audio_b64": None,
    }
