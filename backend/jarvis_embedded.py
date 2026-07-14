"""
JARVIS Embedded Launcher — System Tray, Global Hotkey, Auto-Start

Makes JARVIS a true always-on PC companion:
  - Lives in the system tray (icon on taskbar)
  - Starts automatically on PC boot (optional)
  - Summoned with Ctrl+Shift+J global hotkey
  - Right-click menu: Open HUD / Speak / Activity / Settings / Start with Windows / Exit
  - Runs silently with no console window

Run: python jarvis_embedded.py
Or: python jarvis_embedded.py --startup  (for auto-start mode, no browser)
"""
from __future__ import annotations

import asyncio
import base64
import ctypes
import logging
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

try:
    import pystray
except ImportError:
    pystray = None

# Configure logging to file
log_path = Path(__file__).parent.parent / "logs" / "jarvis_embedded.log"
log_path.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("jarvis_embedded")


# ---------------------------------------------------------------------------
# Single-instance enforcement via Windows mutex
# ---------------------------------------------------------------------------
def _ensure_single_instance() -> bool:
    """Return True if this is the only instance, False if another is running."""
    try:
        mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "J.A.R.V.I.S._SingleInstance_Mutex")
        if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            logger.warning("JARVIS is already running! Check the system tray.")
            return False
        return True
    except Exception:
        return True  # Allow on non-Windows


# ---------------------------------------------------------------------------
# Backend health check & launch
# ---------------------------------------------------------------------------
def _is_backend_running() -> bool:
    """Check if the FastAPI backend is responding."""
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:8000/health")
        urllib.request.urlopen(req, timeout=3)
        return True
    except Exception:
        return False


def _get_python_exe() -> str:
    """Get the Python executable from venv."""
    venv_python = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable  # fallback to default

def _ensure_backend_running() -> None:
    """Start the backend if it's not already running."""
    if _is_backend_running():
        logger.info("Backend is already running.")
        return

    logger.info("Starting JARVIS backend...")
    backend_dir = Path(__file__).parent
    python_exe = _get_python_exe()
    # Start uvicorn in background, no console window
    subprocess.Popen(
        [python_exe, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(backend_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    # Wait for backend to come up
    for _ in range(30):
        if _is_backend_running():
            logger.info("Backend started successfully.")
            return
        time.sleep(1)
    logger.error("Backend failed to start!")


# ---------------------------------------------------------------------------
# Auto-start (Windows registry)
# ---------------------------------------------------------------------------
def _get_jarvis_path() -> str:
    """Get the path to the JARVIS embedded launcher."""
    return str(Path(__file__).resolve())


def _is_auto_start_enabled() -> bool:
    """Check if JARVIS is set to start with Windows."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                             winreg.KEY_READ)
        try:
            winreg.QueryValueEx(key, "JARVIS")
            winreg.CloseKey(key)
            return True
        except FileNotFoundError:
            winreg.CloseKey(key)
            return False
    except Exception:
        return False


def _set_auto_start(enable: bool) -> bool:
    """Enable or disable JARVIS auto-start on Windows login."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                             winreg.KEY_SET_VALUE)
        try:
            if enable:
                run_bat = Path(__file__).parent.parent / "run.bat"
                winreg.SetValueEx(key, "JARVIS", 0, winreg.REG_SZ, f'"{run_bat}" --startup')
                logger.info("Auto-start ENABLED.")
            else:
                try:
                    winreg.DeleteValue(key, "JARVIS")
                    logger.info("Auto-start DISABLED.")
                except FileNotFoundError:
                    pass
            winreg.CloseKey(key)
            return True
        except Exception:
            winreg.CloseKey(key)
            return False
    except Exception as e:
        logger.error(f"Failed to set auto-start: {e}")
        return False


# ---------------------------------------------------------------------------
# Monitor wake greeting system (via GetLastInputInfo polling)
# ---------------------------------------------------------------------------
class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


def get_idle_ms() -> int:
    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
    ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
    return ctypes.windll.kernel32.GetTickCount() - lii.dwTime


def lid_open_listener(speak_fn):
    SLEEP_THRESHOLD_MS = 30_000   # if idle > 30s, we consider it was asleep
    CHECK_INTERVAL_S   = 2         # poll every 2 seconds
    first_wake         = True
    was_asleep         = False

    while True:
        idle = get_idle_ms()

        if idle > SLEEP_THRESHOLD_MS:
            was_asleep = True

        elif was_asleep:
            # just woke up
            was_asleep = False
            if first_wake:
                first_wake = False
                hour = time.localtime().tm_hour
                if hour < 12:
                    greeting = "Good morning sir, JARVIS is online and ready."
                elif hour < 17:
                    greeting = "Good afternoon sir, systems are up."
                else:
                    greeting = "Good evening sir, JARVIS is ready."
            else:
                import random
                greeting = random.choice([
                    "Back already sir, shall we continue?",
                    "Welcome back sir. Picking up where we left off.",
                    "Systems resuming sir. Ready when you are.",
                    "Good to have you back sir.",
                ])
            speak_fn(greeting)

        time.sleep(CHECK_INTERVAL_S)


def _start_wake_monitor():
    """Start the wake event monitor as a daemon thread."""
    t = threading.Thread(
        target=lid_open_listener,
        args=(_speak_async,),
        daemon=True,
        name="WakeMonitor",
    )
    t.start()
    logger.info("Wake event monitor started")

# ---------------------------------------------------------------------------
# Tray icon generation
# ---------------------------------------------------------------------------
def _get_icon_image():
    """Generate a JARVIS icon (64x64 bright cyan on dark background)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (64, 64), (0, 50, 80))  # dark blue background
    draw = ImageDraw.Draw(img)
    # Draw a bright "J" in light cyan - very visible
    draw.rectangle([16, 8, 28, 52], fill=(100, 255, 255))  # vertical bar
    draw.rectangle([16, 44, 48, 52], fill=(100, 255, 255))  # bottom curve
    return img


# ---------------------------------------------------------------------------
# TTS for tray feedback
# ---------------------------------------------------------------------------
_async_loop: asyncio.AbstractEventLoop | None = None
_tts_lock = threading.Lock()


def _speak_async(text: str) -> None:
    """Speak text in a background thread using Kokoro TTS - silent playback."""
    logger.info(f"[TTS] Speaking: {text}")
    def _do():
        try:
            # Use the TTS module directly
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from voice.tts import speak_async as tts_speak
            tts_speak(text)
        except Exception as e:
            logger.warning(f"TTS failed: {e}")

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    logger.info(f"[TTS] Started background thread for: {text}")


# ---------------------------------------------------------------------------
# System tray
# ---------------------------------------------------------------------------
_tray_icon = None
_auto_start_var = None  # will hold the pystray MenuItem for toggle


def _build_menu():
    """Build the pystray menu dynamically."""
    import pystray

    if pystray is None:
        logger.error("pystray not installed: pip install pystray")
        return

    is_auto = _is_auto_start_enabled()

    def _open_hud(icon=None, item=None):
        logger.info("Opening JARVIS HUD...")
        subprocess.Popen(
            ["cmd", "/c", "start", "", str(Path(__file__).parent.parent / "frontend" / "index.html")],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        _speak_async("HUD opened, Sir.")

    def _on_speak(icon=None, item=None):
        logger.info("Speak activated from tray.")
        # Open HUD
        _open_hud()
        # Broadcast to frontend to start voice recognition
        _broadcast_voice_start()
        threading.Timer(1.5, _speak_async, args=("Listening, Sir. What would you like?",)).start()


def _broadcast_voice_start():
    """Tell the frontend to start voice recognition."""
    try:
        import urllib.request
        # Call internal endpoint to broadcast
        data = b'{"action": "start_voice"}'
        req = urllib.request.Request(
            "http://127.0.0.1:8000/internal/voice-start",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass  # Non-critical

    def _on_activity(icon=None, item=None):
        logger.info("Opening activity log...")
        subprocess.Popen(
            ["cmd", "/c", "start", "", "http://127.0.0.1:8000/activity"],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    def _on_settings(icon=None, item=None):
        logger.info("Opening settings...")
        subprocess.Popen(
            ["cmd", "/c", "start", "", "http://127.0.0.1:8000/settings"],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

    def _toggle_auto_start(icon=None, item=None):
        new_state = not _is_auto_start_enabled()
        if _set_auto_start(new_state):
            _speak_async(f"Auto-start {'enabled' if new_state else 'disabled'}, Sir.")
            # Rebuild menu to update the toggle label
            icon.menu = _build_menu()
        else:
            _speak_async("Failed to change auto-start setting.")

    def _on_exit(icon=None, item=None):
        logger.info("JARVIS exiting...")
        _speak_async("Goodbye, Sir.")
        time.sleep(1)
        icon.stop()

    auto_start_label = "✅ Start with Windows" if is_auto else "☐ Start with Windows"

    menu = pystray.Menu(
        pystray.MenuItem("🖥  Open JARVIS HUD", _open_hud),
        pystray.MenuItem("🎙  Speak to JARVIS", _on_speak),
        pystray.MenuItem("📋  Activity Log", _on_activity),
        pystray.MenuItem("⚙  Settings", _on_settings),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(auto_start_label, _toggle_auto_start, enabled=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("🚪  Exit JARVIS", _on_exit),
    )
    return menu


def _run_tray() -> None:
    """Run the system tray icon."""
    try:
        import pystray
    except ImportError:
        logger.error("pystray not installed: pip install pystray")
        return

    image = _get_icon_image()
    menu = _build_menu()
    icon = pystray.Icon("JARVIS", image, "J.A.R.V.I.S. — Always on", menu)

    # Add click handlers
    def on_left_click(icon, item):
        logger.info("Left click on tray - opening HUD")
        _open_hud()

    def on_right_click(icon, item):
        logger.info("Right click on tray - showing menu")
        icon.menu = _build_menu()
        icon._update_menu()

    icon.on_click = on_left_click
    icon.on_right_click = on_right_click

    global _tray_icon
    _tray_icon = icon

    logger.info("JARVIS system tray running.")
    icon.run()


# ---------------------------------------------------------------------------
# Global hotkey: Ctrl+Shift+J
# ---------------------------------------------------------------------------
def _register_hotkey(callback) -> None:
    """Register Ctrl+Shift+J global hotkey."""
    try:
        import keyboard
    except ImportError:
        logger.error("keyboard not installed: pip install keyboard")
        return

    keyboard.add_hotkey("ctrl+shift+j", callback, suppress=True)
    logger.info("Global hotkey registered: Ctrl+Shift+J")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS — Always on, always ready.")
    parser.add_argument("--startup", action="store_true",
                        help="Launch in background mode (no browser popup, for auto-start)")
    args = parser.parse_args()

    # Single instance check
    if not _ensure_single_instance():
        if not args.startup:
            input("\nPress Enter to exit...")
        return

    # Ensure backend is running
    _ensure_backend_running()

    # Start a persistent event loop in a dedicated thread
    global _async_loop
    _async_loop = asyncio.new_event_loop()

    def _run_event_loop():
        asyncio.set_event_loop(_async_loop)
        _async_loop.run_forever()

    _loop_thread = threading.Thread(target=_run_event_loop, daemon=True, name="AsyncLoop")
    _loop_thread.start()

    # Give the loop a moment to start
    import time
    time.sleep(0.1)

    # Initialize voice coordinator
    def _initialize_voice_coordinator():
        """Start voice assistant in background."""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from voice.wake_word import WakeWordDetector

            async def voice_command_handler(command: str) -> str:
                """Handle voice commands via HTTP to backend."""
                try:
                    import urllib.request
                    import json
                    data = json.dumps({"message": command}).encode('utf-8')
                    req = urllib.request.Request(
                        "http://127.0.0.1:8000/chat",
                        data=data,
                        headers={'Content-Type': 'application/json'},
                        method='POST'
                    )
                    with urllib.request.urlopen(req, timeout=60) as response:
                        result = json.loads(response.read().decode('utf-8'))
                        reply = result.get('reply', 'Command processed, Sir.')
                        # strip JSON wrapper if present
                        import re, json as json2
                        reply = re.sub(r'^```\w*\s*', '', reply)   # strip leading ``` or ```json
                        reply = re.sub(r'\s*```$', '', reply)       # strip trailing ```
                        reply = re.sub(r'\[TOOL_CALL\].*?\[/TOOL_CALL\]', '', reply, flags=re.DOTALL).strip()
                        reply = re.sub(r'<tool_call>.*?</tool_call>', '', reply, flags=re.DOTALL).strip()
                        reply = re.sub(r'<function.*?</function>', '', reply, flags=re.DOTALL).strip()
                        reply = re.sub(r'```.*?```', '', reply, flags=re.DOTALL).strip()
                        reply = re.sub(r'\{.*?"tool".*?\}', '', reply, flags=re.DOTALL).strip()
                        reply = reply.strip()
                        if not reply:
                            reply = "I'm working on that, Sir."
                        if reply.startswith('{'):
                            try:
                                parsed = json2.loads(reply)
                                reply = parsed.get('reply', reply)
                            except Exception:
                                pass
                        return reply
                except Exception as e:
                    logger.error(f"Voice command error: {e}")
                    return f"I encountered an error, Sir."

            # Simple wake word + listen loop running in its own thread
            wake_detector = WakeWordDetector()
            wake_detector.activate()  # start active so no wake word needed after hotkey

            # TEMPORARY: Test by bypassing wake detector
            _TEST_BYPASS_WAKE = False  # Set to True to process all speech as commands

            def _voice_loop():
                """Blocking voice loop — runs in its own thread."""
                import time
                from voice.stt import start_listening, _listening as _is_listening
                logger.info("Voice loop starting STT...")

                # Start STT if not already running
                from voice import stt as _stt_mod
                if not _stt_mod._listening:
                    if not start_listening():
                        logger.error("Failed to start STT, voice loop exiting")
                        return

                # Wait up to 5 seconds for STT to confirm running
                for _ in range(10):
                    if _stt_mod._listening:
                        break
                    time.sleep(0.5)
                else:
                    logger.error("STT never became ready, voice loop exiting")
                    return

                logger.info("STT ready, voice loop active")

                logger.info("Voice loop active, listening for commands...")
                from voice.stt import listen_continuous
                for text in listen_continuous():
                    logger.info(f"STT raw: '{text}'")
                    logger.info(f"Wake detector active: {wake_detector.active}")
                    if not text:
                        continue
                    logger.debug(f"Processing text with wake_detector (active={wake_detector.active}): '{text}'")

                    # TEMPORARY: Test bypass - treat all speech as commands
                    if _TEST_BYPASS_WAKE:
                        command = text
                        logger.info(f"[TEST MODE] Voice command (bypass): {command}")
                    else:
                        if wake_detector.active:
                            command = text  # already activated, use full text as command
                        else:
                            command = wake_detector.process(text)

                        if not command:
                            continue

                        logger.info(f"Voice command: {command}")

                    # Route through the command router for both local and API commands
                    from command_router import route_command_sync
                    result = route_command_sync(command=command)
                    if result.local_executed:
                        logger.info(f"Local execution: {result.tools_used[0]} for '{command}'")
                        _speak_async(result.reply)
                        continue

                    # Run the async handler in the persistent loop
                    future = asyncio.run_coroutine_threadsafe(
                        voice_command_handler(command), _async_loop
                    )
                    try:
                        reply = future.result(timeout=65)
                        if reply:
                            _speak_async(reply)
                    except Exception as e:
                        logger.error(f"Command failed: {e}")
                        _speak_async("I encountered an error, Sir.")

            voice_thread = threading.Thread(target=_voice_loop, daemon=True, name="VoiceLoop")
            voice_thread.start()

            # Return a simple object with activate() for compatibility
            class _VoiceHandle:
                def activate(self):
                    wake_detector.activate()
                def deactivate(self):
                    wake_detector.deactivate()

            return _VoiceHandle()

        except Exception as e:
            logger.error(f"Failed to initialize voice: {e}")
            return None

    # Start voice coordinator
    voice_coordinator = _initialize_voice_coordinator()

    # Start wake event monitor (monitors lid-open / resume)
    _start_wake_monitor()

    # Auto-activate voice assistant on startup
    if voice_coordinator:
        def auto_activate_voice():
            import time
            time.sleep(8)
            try:
                voice_coordinator.activate()
                logger.info("Voice assistant auto-activated")
            except Exception as e:
                logger.error(f"Failed to auto-activate voice: {e}")

        threading.Thread(target=auto_activate_voice, daemon=True).start()

    # Greeting based on time of day
    def get_greeting():
        hour = datetime.now().hour
        if hour < 5:
            return "Are we too late to work sir? or perhaps too early"
        elif hour < 12:
            return "Good morning, Sir. JARVIS is online and ready."
        elif hour < 17:
            return "Good afternoon, Sir. JARVIS is online and ready."
        elif hour < 21:
            return "Good evening, Sir. JARVIS is online and ready."
        else:
            return "Sir, it's quite late. JARVIS is online and ready."

    greeting = get_greeting()

    # Welcome message (only on non-startup launch)
    if not args.startup:
        print(r"""
+--------------------------------------------------------------+
|           J.A.R.V.I.S. -- Embedded Mode                    |
|                                                              |
|  * Ctrl+Shift+J  ->  Activate JARVIS from anywhere          |
|  * Tray icon     ->  Right-click for menu                   |
|  * Auto-starts   ->  Toggle via tray menu                   |
|                                                              |
|  JARVIS is running in the background.                       |
|  Close this window to hide -- JARVIS stays in tray.         |
+--------------------------------------------------------------+
""")
        _speak_async(greeting)
    else:
        # Startup mode: speak greeting after delay
        logger.info(f"JARVIS started in auto-start mode. Greeting: {greeting}")
        def speak_greeting():
            time.sleep(12)  # Wait for backend and voice to be ready
            _speak_async(greeting)
        threading.Thread(target=speak_greeting, daemon=True).start()

    def _hotkey_pressed():
        logger.info("Ctrl+Shift+J activated!")
        # Open HUD
        subprocess.Popen(
            ["cmd", "/c", "start", "", str(Path(__file__).parent.parent / "frontend" / "index.html")],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        _speak_async("At your service, Sir.")
        # Activate voice if available
        if voice_coordinator:
            voice_coordinator.activate()

    # Register hotkey in main thread
    _register_hotkey(_hotkey_pressed)

    # Start tray in a separate thread
    try:
        tray_thread = threading.Thread(target=_run_tray, daemon=False)
        tray_thread.start()
        logger.info("Tray thread started")
    except Exception as e:
        logger.error(f"Failed to start tray: {e}")

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("JARVIS shutting down.")


if __name__ == "__main__":
    main()
