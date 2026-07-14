"""
Operating Context Manager for JARVIS 2.5

Provides continuous OS context tracking for the Operating Intelligence layer.

Tracks:
- Active window/application
- Recent commands
- Current working directory
- Recent files
- Clipboard content
- System state

This feeds the "Perception" layer that enables JARVIS to understand
context before reasoning.
"""

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OSContext:
    """Lightweight OS context snapshot."""
    active_window: str = "Unknown"
    active_app: str = "Unknown"
    app_type: str = "unknown"  # browser, editor, file_manager, terminal, other
    working_directory: str = ""
    recent_commands: list = field(default_factory=list)
    recent_files: list = field(default_factory=list)
    clipboard_text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class ContextManager:
    """
    Singleton context tracker for Operating Intelligence.

    Maintains lightweight OS context that feeds into understanding
    and reasoning layers.
    """

    _instance: Optional['ContextManager'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        """Initialize context manager."""
        self._context = OSContext()
        self._update_lock = threading.Lock()
        self._recent_commands = deque(maxlen=10)
        self._recent_files = deque(maxlen=10)
        self._update_interval = 2.0  # seconds
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._last_clipboard = ""

        # App type keywords
        self._browser_kw = ['chrome', 'edge', 'firefox', 'brave', 'opera', 'vivaldi']
        self._editor_kw = ['code', 'visual studio', 'notepad++', 'sublime', 'atom', 'jetbrains']
        self._terminal_kw = ['powershell', 'cmd', 'terminal', 'bash', 'git-bash', 'windows terminal']
        self._file_mgr_kw = ['explorer', 'file explorer', 'dolphin', 'nautilus']

    def _classify_app(self, window_title: str, process_name: str = "") -> tuple[str, str]:
        """Classify the active application type."""
        combined = (window_title + " " + process_name).lower()

        if any(k in combined for k in self._browser_kw):
            return ("browser", "Browser")
        elif any(k in combined for k in self._editor_kw):
            return ("editor", "Editor")
        elif any(k in combined for k in self._terminal_kw):
            return ("terminal", "Terminal")
        elif any(k in combined for k in self._file_mgr_kw):
            return ("file_manager", "File Manager")
        else:
            return ("other", "Application")

    def _update_active_window(self):
        """Update the active window context."""
        try:
            import win32gui
            import win32process

            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd) or "Unknown"

            # Get process name
            process_name = "Unknown"
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                import psutil
                process = psutil.Process(pid)
                process_name = process.name()
            except:
                pass

            app_type, app_label = self._classify_app(title, process_name)

            with self._update_lock:
                self._context.active_window = title
                self._context.active_app = f"{process_name}"
                self._context.app_type = app_type

        except ImportError:
            logger.debug("win32gui not available for context tracking")
        except Exception as e:
            logger.debug(f"Failed to update active window: {e}")

    def _update_clipboard(self):
        """Track clipboard changes."""
        try:
            import pyperclip
            current = pyperclip.paste()
            if current != self._last_clipboard and current.strip():
                self._context.clipboard_text = current[:500]  # Limit size
                self._last_clipboard = current
        except:
            pass

    def _update_loop(self):
        """Background loop for periodic context updates."""
        while self._running:
            try:
                self._update_active_window()
                self._update_clipboard()

                with self._update_lock:
                    self._context.timestamp = datetime.now()
                    self._context.recent_commands = list(self._recent_commands)
                    self._context.recent_files = list(self._recent_files)

            except Exception as e:
                logger.debug(f"Context update error: {e}")

            time.sleep(self._update_interval)

    def start(self):
        """Start context tracking."""
        if self._running:
            return

        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        logger.info("Context manager started")

    def stop(self):
        """Stop context tracking."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
        logger.info("Context manager stopped")

    def get_context(self) -> OSContext:
        """Get current OS context snapshot."""
        with self._update_lock:
            return OSContext(
                active_window=self._context.active_window,
                active_app=self._context.active_app,
                app_type=self._context.app_type,
                working_directory=self._context.working_directory,
                recent_commands=list(self._recent_commands),
                recent_files=list(self._recent_files),
                clipboard_text=self._context.clipboard_text,
                timestamp=self._context.timestamp
            )

    def log_command(self, command: str, success: bool = True):
        """Log a command execution for context."""
        entry = {
            "command": command,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "window": self._context.active_window
        }
        self._recent_commands.appendleft(entry)
        if len(self._recent_commands) > 10:
            self._recent_commands.pop()

    def log_file_access(self, file_path: str):
        """Log a file access for context."""
        entry = {
            "path": file_path,
            "timestamp": datetime.now().isoformat()
        }
        self._recent_files.appendleft(entry)
        if len(self._recent_files) > 10:
            self._recent_files.pop()

    def set_working_directory(self, path: str):
        """Set current working directory context."""
        with self._update_lock:
            self._context.working_directory = path

    def get_active_app_type(self) -> str:
        """Get the type of the currently active application."""
        return self._context.app_type

    def is_browser_active(self) -> bool:
        """Check if a browser is currently active."""
        return self._context.app_type == "browser"

    def is_terminal_active(self) -> bool:
        """Check if a terminal is currently active."""
        return self._context.app_type == "terminal"


# Singleton instance
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get or create the context manager singleton."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


def start_context_tracking():
    """Start background context tracking."""
    get_context_manager().start()


def stop_context_tracking():
    """Stop background context tracking."""
    global _context_manager
    if _context_manager:
        _context_manager.stop()


def get_os_context() -> dict:
    """Get current OS context as dict for injection into prompts."""
    ctx = get_context_manager().get_context()
    return {
        "active_window": ctx.active_window,
        "active_app": ctx.active_app,
        "app_type": ctx.app_type,
        "working_directory": ctx.working_directory,
        "recent_commands": ctx.recent_commands,
        "recent_files": ctx.recent_files,
        "clipboard_preview": ctx.clipboard_text[:200] if ctx.clipboard_text else None,
    }
