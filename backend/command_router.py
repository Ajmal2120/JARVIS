"""
Command Router for JARVIS — Single entry point for all user commands.

Flow:
    Voice/WebSocket → route_command() → detect_intent()
                                         ↓
                        If local intent → execute local skill → return
                                         ↓
                        Else → forward to jarvis_chat() → return

This keeps main.py thin and makes it easy to extend routing logic
without modifying the core pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("jarvis.router")


@dataclass
class CommandResult:
    """Result of processing a command."""
    reply: str
    mood: str = "neutral"
    tools_used: list[str] = None
    audio_b64: Optional[str] = None
    local_executed: bool = False  # True if executed locally (no API call)

    def __post_init__(self):
        if self.tools_used is None:
            self.tools_used = []


# ---------------------------------------------------------------------------
# Local Intent Executor
# ---------------------------------------------------------------------------

def _execute_intent(intent: dict) -> Optional[CommandResult]:
    """
    Execute a detected local intent and return the result.
    Returns None if execution failed (caller should fall through to Claude).
    """
    intent_type = intent.get('type', '')
    intent_target = intent.get('target', '')
    response_text = intent.get('response', 'Done, Sir.')

    try:
        # Screen control intents
        if intent_type == 'click':
            from tools.screen_control import click_element
            result = click_element(intent_target)
            response_text = result.get('message', 'Done, Sir.')
        elif intent_type == 'type':
            from tools.screen_control import type_text
            result = type_text(intent_target)
            response_text = result.get('message', 'Done, Sir.')
        elif intent_type == 'scroll':
            from tools.screen_control import scroll
            result = scroll(intent_target, 3)
            response_text = result.get('message', 'Done, Sir.')
        elif intent_type == 'hotkey':
            from tools.screen_control import press_hotkey
            keys = intent_target if isinstance(intent_target, list) else intent_target.split('+')
            result = press_hotkey(*keys)
            response_text = result.get('message', 'Done, Sir.')
        elif intent_type == 'read_screen':
            from tools.screen_control import read_screen_text
            text = read_screen_text()
            response_text = text[:200] if text else "Nothing on screen, Sir."
        elif intent_type == 'take_screenshot':
            from tools.screen_control import take_screenshot
            take_screenshot()
            response_text = "Screenshot captured, Sir."
        elif intent_type == 'open_url':
            from tools.screen_control import open_url
            result = open_url(intent_target)
            response_text = result.get('message', 'Done, Sir.')
        elif intent_type in ('quick_action', 'copy', 'paste', 'undo', 'select_all',
                             'save', 'new_tab', 'close_tab', 'refresh',
                             'close_window', 'minimize', 'maximize', 'switch_window'):
            from tools.screen_control import quick_action
            result = quick_action(intent_target)
            response_text = result.get('message', 'Done, Sir.')
        elif intent_type == 'volume':
            import asyncio
            from tools.system import set_volume
            level = int(intent_target) if intent_target else 50
            result = asyncio.run(set_volume(level))
            response_text = f"Volume set to {level}%, Sir." if result.get("ok") else "Volume change failed, Sir."
        elif intent_type == 'app':
            import asyncio
            from tools.system import open_app, execute_open_app
            plan = asyncio.run(open_app(intent_target))
            if plan.get("ok"):
                result = execute_open_app(plan["target"])
                response_text = result.get("message", f"Opened {intent_target}, Sir.")
            else:
                response_text = plan.get("error", f"Could not open {intent_target}")
        elif intent_type == 'media':
            import asyncio
            from tools.system import play_media
            result = asyncio.run(play_media(intent_target))
            response_text = result.get("title", intent_target) + ", Sir." if result.get("ok") else "Could not play media, Sir."
        elif intent_type == 'screenshot':
            import asyncio
            from tools.system import screenshot
            result = asyncio.run(screenshot(0))
            response_text = "Screenshot captured, Sir." if result.get("ok") else "Screenshot failed, Sir."
        elif intent_type == 'lock':
            import asyncio
            from tools.system import lock_workstation
            result = asyncio.run(lock_workstation())
            response_text = "Workstation locked, Sir." if result.get("ok") else "Could not lock, Sir."
        elif intent_type == 'shutdown':
            import asyncio
            from tools.system import shutdown
            result = asyncio.run(shutdown(intent_target or "now"))
            response_text = "Shutting down, Sir." if result.get("ok") else "Shutdown failed, Sir."
        elif intent_type == 'restart':
            import asyncio
            from tools.system import restart
            result = asyncio.run(restart(intent_target or "now"))
            response_text = "Restarting, Sir." if result.get("ok") else "Restart failed, Sir."
        elif intent_type == 'clipboard':
            import asyncio
            from tools.system import set_clipboard
            result = asyncio.run(set_clipboard(intent_target))
            response_text = "Copied to clipboard, Sir." if result.get("ok") else "Clipboard failed, Sir."
        elif intent_type == 'open_url':
            import asyncio
            from tools.system import open_app
            result = asyncio.run(open_app(intent_target))
            response_text = f"Opening {intent_target}, Sir." if result.get("ok") else "Could not open URL, Sir."
        # time, date, click, type, scroll, hotkey, read_screen, take_screenshot, quick_action
        # already handled above or have response from detect_intent()

        return CommandResult(
            reply=response_text,
            mood="neutral",
            tools_used=["local:" + intent_type],
            local_executed=True
        )

    except Exception as e:
        logger.error(f"Local intent execution failed: {e}")
        return CommandResult(
            reply="I encountered an error executing that command, Sir.",
            mood="neutral",
            tools_used=["local:" + intent_type],
            local_executed=True
        )


# ---------------------------------------------------------------------------
# Main Router
# ---------------------------------------------------------------------------

async def route_command(user_id: str = "default", command: str = "") -> CommandResult:
    """
    Route a command to local execution or Claude API.

    Args:
        user_id: The user identifier
        command: The raw command text

    Returns:
        CommandResult with reply, mood, tools_used, and local_executed flag
    """
    from brain.intent import detect_intent
    from jarvis_core import chat as jarvis_chat

    # Step 1: Detect local intent
    intent = detect_intent(command)

    # Step 2: If local intent found, try to execute it
    if intent and intent.get('response'):
        result = _execute_intent(intent)
        if result:
            logger.info(f"Local execution: {intent.get('type')} for '{command}'")
            return result
        # If _execute_intent returned None, fall through to Claude

    # Step 3: No local intent or execution failed — forward to Claude
    logger.info(f"Claude API call for: '{command}'")
    try:
        res = await jarvis_chat(user_id, command)
        return CommandResult(
            reply=res.get("reply", "I'm at a loss for words, Sir."),
            mood=res.get("mood", "neutral"),
            tools_used=res.get("tools_used", []),
            local_executed=False
        )
    except Exception as e:
        logger.exception("jarvis_chat failed")
        return CommandResult(
            reply=f"I encountered an error, Sir: {e}",
            mood="neutral",
            tools_used=[],
            local_executed=False
        )


# ---------------------------------------------------------------------------
# Synchronous wrapper for non-async contexts (e.g., jarvis_embedded.py)
# ---------------------------------------------------------------------------

def route_command_sync(user_id: str = "default", command: str = "") -> CommandResult:
    """
    Synchronous wrapper for route_command.
    Runs the async router in a new event loop.
    Use this from non-async contexts.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context — create a new loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    route_command(user_id, command)
                )
                return future.result(timeout=60)
        else:
            return asyncio.run(route_command(user_id, command))
    except RuntimeError as e:
        if "no current event loop" in str(e):
            # No event loop in this thread - create and run
            return asyncio.run(route_command(user_id, command))
        else:
            raise
    except Exception as e:
        logger.error(f"route_command_sync failed: {e}")
        return CommandResult(
            reply="I encountered an error processing your command, Sir.",
            mood="neutral",
            tools_used=[],
            local_executed=False
        )
