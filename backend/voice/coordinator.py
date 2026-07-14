"""
Voice Coordinator for JARVIS.

Manages the complete voice pipeline:
- Wake word detection
- Speech recognition
- Command processing (intent detection → Claude fallback)
- TTS response

Pipeline stages with logging:
  [Mic]      Listening → Speech detected → Speech ended
  [STT]      Transcribing → Recognized text
  [Wake]     Wake detected / Sleep detected / No match
  [Intent]   Matched locally / No match
  [Claude]   Request started → Response received
  [TTS]      Generating → Playing → Finished / Fallback
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from brain.intent import detect_intent

from .stt import listen_continuous, start_listening, stop_listening, is_listening
from .tts import speak_async
from .wake_word import WakeWordDetector

logger = logging.getLogger(__name__)


class VoiceCoordinator:
    """Coordinates the complete voice interaction pipeline."""

    def __init__(self, on_command: Optional[Callable] = None, speak_callback: Optional[Callable] = None):
        """Initialize voice coordinator."""
        self.on_command = on_command
        self.speak_callback = speak_callback  # Synchronous speak function for callbacks
        self.wake_detector = WakeWordDetector(
            on_wake=self._on_wake,
            on_sleep=self._on_sleep
        )
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_claude_latency: Optional[float] = None

    def _on_wake(self):
        """Called when wake word is detected."""
        logger.info("[Wake] JARVIS activated — Wake word detected")

    def _on_sleep(self):
        """Called when sleep command is detected."""
        logger.info("[Wake] JARVIS deactivated — Entering sleep mode")

    async def process_speech(self, text: str) -> None:
        """Process speech input and generate response."""
        logger.info("[Voice] Input received: \"%s\"", text)

        # Check for wake word or extract command
        command = self.wake_detector.process(text)

        if command is None:
            if not self.wake_detector.active:
                logger.info("[Wake] No wake word detected — ignoring")
                return
            # Sleep command handled
            return

        if not command:
            # Wake word only, no command
            logger.info("[Wake] Wake word only — waiting for command")
            return

        logger.info("[Voice] Command extracted: \"%s\"", command)

        # Try fast intent detection first (zero-latency, no Claude needed)
        logger.info("[Intent] Checking local intents...")
        intent = detect_intent(command)
        if intent:
            logger.info("[Intent] Matched: %s — executing locally", intent["type"])
            result = await self._execute_intent(intent)
            if result:
                logger.info("[TTS] Responding with intent result")
                await self._speak(result)
            else:
                logger.warning("[Intent] Execution returned no result")
            return

        # No intent matched — fall through to Claude via on_command
        logger.info("[Intent] No local match — routing to Claude")
        if self.on_command:
            await self._call_claude(command)
        else:
            logger.error("[Claude] No command handler configured — cannot respond")

    async def _call_claude(self, command: str) -> None:
        """Route command to Claude and speak the response."""
        logger.info("[Claude] Request started: \"%s\"", command)
        t0 = time.monotonic()

        try:
            response = await self.on_command(command)
            self._last_claude_latency = time.monotonic() - t0

            if response:
                logger.info("[Claude] Response received (%.1fs): \"%s\"",
                           self._last_claude_latency, response[:80])
                await self._speak(response)
            else:
                logger.warning("[Claude] Empty response received")

        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error("[Claude] Request failed (%.1fs): %s", elapsed, e, exc_info=True)
            await self._speak("I encountered an error processing that command, Sir.")

    async def _speak(self, text: str) -> None:
        """Speak text, using speak_callback if available, else speak_async."""
        if not text or not text.strip():
            logger.warning("[TTS] Empty text — nothing to speak")
            return

        logger.info("[TTS] Generating speech: \"%s\"", text[:80])

        try:
            if self.speak_callback:
                self.speak_callback(text)
            else:
                await speak_async(text)
        except Exception as e:
            logger.error("[TTS] Speech failed: %s", e, exc_info=True)

    async def _execute_intent(self, intent: dict) -> Optional[str]:
        """Execute a detected intent and return the response text."""
        from tools import system as system_tool
        import asyncio

        itype = intent.get("type")
        target = intent.get("target", "")
        fallback = intent.get("response", "")

        try:
            if itype == "time":
                return fallback

            elif itype == "volume":
                result = await system_tool.set_volume(int(target))
                if result.get("ok"):
                    return intent.get("response", "Volume adjusted, Sir.")
                return "Sorry, Sir, I couldn't adjust the volume."

            elif itype == "app":
                plan = await system_tool.open_app(target)
                if plan.get("ok"):
                    return intent.get("response", f"Opening {target}, Sir.")
                return f"Sorry, Sir, I couldn't find '{target}'."

            elif itype == "media":
                result = await system_tool.play_media(target)
                title = result.get("title") or target
                if result.get("ok"):
                    return intent.get("response", f"Playing {title}, Sir.")
                return f"Sorry, Sir, I couldn't find '{target}'."

            elif itype == "screenshot":
                result = await system_tool.screenshot(0)
                if result.get("ok"):
                    return intent.get("response", "Screenshot captured, Sir.")
                return "Sorry, Sir, I couldn't take the screenshot."

            elif itype == "lock":
                result = await system_tool.lock_workstation()
                if result.get("ok"):
                    return intent.get("response", "Workstation locked, Sir.")
                return "Sorry, Sir, I couldn't lock the workstation."

            elif itype in ("shutdown", "restart"):
                return fallback  # Safety: let the user confirm via HUD

            elif itype == "clipboard":
                result = await system_tool.set_clipboard(target)
                if result.get("ok"):
                    return intent.get("response", "Copied to clipboard, Sir.")
                return "Sorry, Sir, I couldn't copy that."

            elif itype == "click":
                from tools.screen_control import click_element
                result = click_element(target)
                return result.get("message", str(result))

            elif itype == "type":
                from tools.screen_control import type_text
                result = type_text(target)
                return result.get("message", str(result))

            elif itype == "scroll":
                from tools.screen_control import scroll
                result = scroll(target, 3)
                return result.get("message", str(result))

            elif itype == "hotkey":
                from tools.screen_control import press_hotkey
                keys = target if isinstance(target, list) else target.split("+")
                result = press_hotkey(*keys)
                return result.get("message", str(result))

            elif itype == "read_screen":
                from tools.screen_control import read_screen_text
                text = read_screen_text()
                return text if text else "No text found on screen"

            elif itype == "take_screenshot":
                from tools.screen_control import take_screenshot
                b64 = take_screenshot()
                return "Screenshot captured, Sir."

            elif itype == "open_url":
                from tools.screen_control import open_url
                result = open_url(target)
                return result.get("message", str(result))

            elif itype == "quick_action":
                from tools.screen_control import quick_action
                result = quick_action(target)
                return result.get("message", str(result))

            return fallback

        except Exception as e:
            logger.error("[Intent] Execution failed (%s): %s", itype, e)
            return f"Sorry, Sir, I encountered an error processing that."

    async def start(self) -> bool:
        """Start voice coordination."""
        if self._running:
            logger.warning("[Voice] Coordinator already running")
            return False

        logger.info("[Voice] Starting voice coordinator")

        try:
            # Start speech recognition (may already be running — that's fine)
            if not is_listening():
                if not start_listening():
                    logger.error("[Voice] Failed to start speech recognition")
                    return False

            # Start health watchdog
            try:
                from voice.health import get_watchdog
                watchdog = get_watchdog()
                watchdog.start()
                logger.info("[Health] Watchdog started")
            except Exception as e:
                logger.warning("[Health] Could not start watchdog: %s", e)

            # Start processing loop
            self._running = True
            self._task = asyncio.create_task(self._processing_loop())

            logger.info("[Voice] Coordinator started successfully")
            return True

        except Exception as e:
            logger.error("[Voice] Failed to start: %s", e)
            return False

    async def stop(self) -> None:
        """Stop voice coordination."""
        if not self._running:
            return

        logger.info("[Voice] Stopping coordinator")

        self._running = False

        # Stop health watchdog
        try:
            from voice.health import get_watchdog
            get_watchdog().stop()
        except Exception:
            pass

        # Stop speech recognition
        stop_listening()

        # Cancel processing task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("[Voice] Coordinator stopped")

    async def _processing_loop(self) -> None:
        """Main processing loop for speech recognition."""
        loop = asyncio.get_event_loop()
        speech_queue: asyncio.Queue = asyncio.Queue()
        _collector_restarts = 0  # Tracks how many times we've restarted the collector

        def _collect_speech():
            """Run blocking generator in thread, push results to async queue."""
            try:
                for text in listen_continuous():
                    if not self._running:
                        break
                    try:
                        loop.call_soon_threadsafe(speech_queue.put_nowait, text)
                    except RuntimeError:
                        break  # event loop closed, exit cleanly
            except Exception as e:
                logger.error("[Voice] Speech collection error: %s", e)
            finally:
                try:
                    loop.call_soon_threadsafe(speech_queue.put_nowait, None)
                except RuntimeError:
                    pass  # event loop already closed, nothing to signal

        import threading
        collector = threading.Thread(target=_collect_speech, daemon=True)
        collector.start()

        try:
            while self._running:
                text = await speech_queue.get()
                if text is None:
                    # Collector died — restart it
                    if self._running:
                        _collector_restarts += 1
                        backoff = min(_collector_restarts * 2, 10)  # Cap at 10s
                        logger.warning("[Voice] Speech collector stopped (restart #%d), restarting in %ds...",
                                      _collector_restarts, backoff)
                        await asyncio.sleep(backoff)
                        if not self._running:
                            break
                        collector = threading.Thread(target=_collect_speech, daemon=True)
                        collector.start()
                        logger.info("[Voice] Speech collector restarted")
                    continue
                try:
                    await self.process_speech(text)
                except Exception as e:
                    logger.error("[Voice] process_speech error (continuing): %s", e)
        except Exception as e:
            logger.error("[Voice] Processing loop error: %s", e)
            self._running = False

    @property
    def running(self) -> bool:
        """Check if voice coordinator is running."""
        return self._running

    @property
    def active(self) -> bool:
        """Check if JARVIS is active (awake)."""
        return self.wake_detector.active

    def activate(self) -> None:
        """Manually activate JARVIS."""
        self.wake_detector.activate()

    def deactivate(self) -> None:
        """Manually deactivate JARVIS."""
        self.wake_detector.deactivate()


# Global instance
_voice_coordinator: Optional[VoiceCoordinator] = None


def get_voice_coordinator(on_command: Optional[Callable] = None) -> VoiceCoordinator:
    """Get or create the global voice coordinator."""
    global _voice_coordinator

    if _voice_coordinator is None:
        _voice_coordinator = VoiceCoordinator(on_command=on_command)

    return _voice_coordinator


def start_voice_assistant(on_command: Callable) -> bool:
    """Start the voice assistant with command handler."""
    coordinator = get_voice_coordinator(on_command)
    return asyncio.run(coordinator.start())


def stop_voice_assistant() -> None:
    """Stop the voice assistant."""
    global _voice_coordinator

    if _voice_coordinator:
        asyncio.run(_voice_coordinator.stop())
        _voice_coordinator = None


def is_voice_assistant_running() -> bool:
    """Check if voice assistant is running."""
    return _voice_coordinator is not None and _voice_coordinator.running


def is_jarvis_awake() -> bool:
    """Check if JARVIS is awake (active after wake word)."""
    return _voice_coordinator is not None and _voice_coordinator.active


def activate_jarvis() -> None:
    """Manually activate JARVIS."""
    if _voice_coordinator:
        _voice_coordinator.activate()


def deactivate_jarvis() -> None:
    """Manually deactivate JARVIS."""
    if _voice_coordinator:
        _voice_coordinator.deactivate()
