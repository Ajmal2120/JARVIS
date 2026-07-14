"""
JARVIS Optimized Launcher - Fast, reliable startup with autostart support

This is the primary launcher for JARVIS - handles:
- Fast auto-start on Windows boot
- Single-instance enforcement
- Backend health check and auto-restart
- System tray integration (optional)
- Global hotkey Ctrl+Shift+J
- Silent background operation

Usage:
  python jarvis_launcher.py                # Normal launch (shows tray + popup HUD)
  python jarvis_launcher.py --silent       # Background launch (tray only, no popup)
  python jarvis_launcher.py --install      # Install auto-start (adds to Windows startup)
  python jarvis_launcher.py --uninstall    # Remove auto-start
"""
from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import winreg
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

# Configure logging
log_path = Path(__file__).parent.parent / "logs" / "jarvis_launcher.log"
log_path.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("jarvis_launcher")

# Constants
JARVIS_NAME = "J.A.R.V.I.S."
BACKEND_DIR = Path(__file__).parent
BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL = f"file://{BACKEND_DIR.parent / 'frontend' / 'index.html'}"
RUN_BATCH = BACKEND_DIR.parent / "run.bat"
PYTHON_EXE = BACKEND_DIR / ".venv" / "Scripts" / "python.exe"

# Ensure Python path includes backend
sys.path.insert(0, str(BACKEND_DIR))


class JarvisLauncher:
    """Main launcher class."""

    def __init__(self):
        self._single_instance = None
        self._backend_process = None
        self._tray_thread = None
        self._async_loop = None

    def ensure_single_instance(self) -> bool:
        """Ensure only one JARVIS instance is running."""
        try:
            # Use named mutex for Windows
            mutex_name = f"{JARVIS_NAME.replace(' ', '_')}_SingleInstance"
            self._single_instance = ctypes.windll.kernel32.CreateMutexW(
                None, False, mutex_name
            )
            if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
                logger.warning(f"{JARVIS_NAME} is already running!")
                return False
            return True
        except Exception:
            return True  # Fallback for non-Windows

    def install_autostart(self) -> bool:
        """Install JARVIS to start with Windows."""
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_SET_VALUE
            )
            winreg.SetValueEx(
                key,
                JARVIS_NAME,
                0,
                winreg.REG_SZ,
                f'"{sys.executable}" "{BACKEND_DIR}\\jarvis_launcher.py" --silent'
            )
            winreg.CloseKey(key)
            logger.info(f"{JARVIS_NAME} added to Windows startup")
            return True
        except Exception as e:
            logger.error(f"Failed to install autostart: {e}")
            return False

    def uninstall_autostart(self) -> bool:
        """Remove JARVIS from Windows startup."""
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0,
                winreg.KEY_SET_VALUE
            )
            try:
                winreg.DeleteValue(key, JARVIS_NAME)
                logger.info(f"{JARVIS_NAME} removed from Windows startup")
                winreg.CloseKey(key)
                return True
            except FileNotFoundError:
                winreg.CloseKey(key)
                return True
        except Exception as e:
            logger.error(f"Failed to uninstall autostart: {e}")
            return False

    def check_backend_health(self, timeout: float = 3.0) -> bool:
        """Check if backend is responding."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{BACKEND_URL}/health")
            req.timeout = timeout
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    logger.debug("Backend health check: OK")
                    return True
            return False
        except Exception:
            return False

    def start_backend(self) -> bool:
        """Start the FastAPI backend if not running."""
        if self.check_backend_health():
            logger.info("Backend already running")
            return True

        logger.info("Starting backend...")

        # Use the run.bat file if it exists, otherwise start uvicorn directly
        if RUN_BATCH.exists():
            subprocess.Popen(
                [str(RUN_BATCH)],
                creationflags=subprocess.CREATE_NO_WINDOW,
                shell=False
            )
        else:
            if PYTHON_EXE.exists():
                subprocess.Popen(
                    [str(PYTHON_EXE), "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", 8000],
                    cwd=str(BACKEND_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                # Fallback to system Python
                subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", 8000],
                    cwd=str(BACKEND_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )

        # Wait for backend to start
        for i in range(30):  # 30 seconds
            if self.check_backend_health():
                logger.info("Backend started successfully")
                return True
            time.sleep(1)

        logger.error("Backend failed to start!")
        return False

    def open_hud(self):
        """Open the JARVIS HUD in browser."""
        try:
            if sys.platform == "win32":
                subprocess.Popen(
                    ["cmd", "/c", "start", "", FRONTEND_URL],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    shell=False
                )
            else:
                # Linux/Mac - use webbrowser
                import webbrowser
                webbrowser.open(FRONTEND_URL)
        except Exception as e:
            logger.error(f"Failed to open HUD: {e}")

    def quick_speak(self, text: str):
        """Speak text using optimized TTS (background process)."""
        def _speak():
            try:
                from tts.speaker import speak as kokoro_speak
                kokoro_speak(text)
            except Exception as e:
                logger.warning(f"Quick speak failed: {e}")

        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()

    def setup_global_hotkey(self, callback):
        """Setup global hotkey Ctrl+Shift+J."""
        try:
            import keyboard
            keyboard.add_hotkey("ctrl+shift+j", callback, suppress=True)
            logger.info("Global hotkey Ctrl+Shift+J registered")
        except ImportError:
            logger.warning("keyboard module not installed - hotkey not available")

    def setup_system_tray(self, silent_mode: bool):
        """Setup system tray icon and menu."""
        try:
            import pystray
            from PIL import Image, ImageDraw

            # Create icon
            img = Image.new("RGB", (64, 64), (0, 50, 80))
            draw = ImageDraw.Draw(img)
            draw.rectangle([16, 8, 28, 52], fill=(100, 255, 255))
            draw.rectangle([16, 44, 48, 52], fill=(100, 255, 255))

            def on_open(icon, item):
                self.open_hud()
                self.quick_speak("At your service, Sir.")

            def on_speak(icon, item):
                self.open_hud()
                # Signal backend to start listening
                try:
                    import urllib.request
                    urllib.request.urlopen(f"{BACKEND_URL}/internal/voice-start", timeout=2)
                except:
                    pass
                self.quick_speak("Listening, Sir.")

            def on_exit(icon, item):
                icon.stop()
                logger.info(f"{JARVIS_NAME} exiting...")
                self.quick_speak("Goodbye, Sir.")
                time.sleep(1)
                sys.exit(0)

            menu = pystray.Menu(
                pystray.MenuItem("🖥  Open HUD", on_open),
                pystray.MenuItem("🎙  Speak to JARVIS", on_speak),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("🚪  Exit", on_exit),
            )

            icon = pystray.Icon(
                JARVIS_NAME.replace(" ", "_"),
                img,
                f"{JARVIS_NAME} — Online",
                menu
            )

            # Tray thread
            def run_tray():
                icon.run()

            self._tray_thread = threading.Thread(target=run_tray, daemon=True)
            self._tray_thread.start()

            if not silent_mode:
                self.open_hud()
                self.quick_speak("Hello, Sir. JARVIS is online and ready.")

        except ImportError:
            logger.warning("pystray not installed - tray icon not available")
            if not silent_mode:
                self.open_hud()

    def run(self, silent: bool = False):
        """Main launcher run method."""
        logger.info(f"Starting {JARVIS_NAME} launcher...")

        # Single instance check
        if not self.ensure_single_instance():
            if not silent:
                input("\nPress Enter to exit...")
            return

        # Start backend
        if not self.start_backend():
            logger.error("Backend failed to start. Exiting.")
            return

        # Setup async event loop for voice coordinator
        self._async_loop = asyncio.new_event_loop()

        def run_async_loop():
            asyncio.set_event_loop(self._async_loop)
            self._async_loop.run_forever()

        async_thread = threading.Thread(target=run_async_loop, daemon=True)
        async_thread.start()

        # Setup hotkey
        def on_hotkey():
            logger.info("Hotkey pressed!")
            self.open_hud()
            self.quick_speak("At your service, Sir.")
            # Activate voice if coordinator is running
            try:
                from voice.coordinator import activate_jarvis
                activate_jarvis()
            except:
                pass

        self.setup_global_hotkey(on_hotkey)

        # Setup tray
        self.setup_system_tray(silent)

        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Launcher shutting down...")
        finally:
            if self._async_loop:
                self._async_loop.call_soon_threadsafe(self._async_loop.stop)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=f"{JARVIS_NAME} Launcher")
    parser.add_argument("--silent", action="store_true", help="Run silently (no popup)")
    parser.add_argument("--install", action="store_true", help="Install auto-start")
    parser.add_argument("--uninstall", action="store_true", help="Remove auto-start")
    parser.add_argument("--status", action="store_true", help="Check if running")

    args = parser.parse_args()

    # Check status
    if args.status:
        try:
            import urllib.request
            req = urllib.request.Request(f"{BACKEND_URL}/health")
            req.timeout = 3
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    print(f"{JARVIS_NAME} is running")
                    sys.exit(0)
        except:
            print(f"{JARVIS_NAME} is not running")
            sys.exit(1)
        return

    # Install/uninstall autostart
    if args.install:
        if JarvisLauncher().install_autostart():
            print(f"{JARVIS_NAME} will now start automatically with Windows")
        else:
            print("Failed to install autostart")
        return

    if args.uninstall:
        if JarvisLauncher().uninstall_autostart():
            print(f"{JARVIS_NAME} removed from Windows startup")
        else:
            print("Failed to uninstall autostart")
        return

    # Run launcher
    launcher = JarvisLauncher()
    launcher.run(silent=args.silent)


if __name__ == "__main__":
    main()