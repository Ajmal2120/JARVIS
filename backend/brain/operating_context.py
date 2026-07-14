"""
Operating Context for JARVIS - Continuous OS awareness

Maintains lightweight operating system context for all reasoning:
- Active window classification and tracking
- Recent commands history
- Current browser/URL detection
- Project/workspace inference
- Recent files accessed
- Screen change tracking
- Clipboard monitoring (optional)

This context enables JARVIS to make intelligent decisions without
constantly querying the LLM for basic OS understanding.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import psutil

logger = logging.getLogger(__name__)


class AppType(Enum):
    """Classification of application types."""
    BROWSER = "browser"
    EDITOR = "editor"
    TERMINAL = "terminal"
    IDE = "ide"
    COMMUNICATION = "communication"
    MEDIA = "media"
    FILE_MANAGER = "file_manager"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class WindowContext:
    """Context about the current active window."""
    title: str
    process_name: str
    app_type: AppType
    url: Optional[str] = None
    executable_path: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CommandContext:
    """Context about a recently executed command."""
    command: str
    timestamp: datetime
    source: str  # 'voice', 'chat', 'automated'
    result: Optional[str] = None


@dataclass
class FileContext:
    """Context about a recently accessed file."""
    path: str
    timestamp: datetime
    action: str  # 'opened', 'saved', 'closed'


@dataclass
class OperatingContext:
    """Complete operating system context at a point in time."""
    active_window: Optional[WindowContext] = None
    recent_commands: List[CommandContext] = field(default_factory=list)
    recent_files: List[FileContext] = field(default_factory=list)
    current_project: Optional[str] = None
    browser_urls: Dict[str, str] = field(default_factory=dict)
    clipboard_content: Optional[str] = None
    last_screen_change: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.now)


class OperatingContextTracker:
    """
    Tracks and maintains operating system context.

    This is the foundation for Operating Intelligence. JARVIS uses this
    context to make intelligent decisions about user intent without
    constantly requiring LLM queries for basic OS understanding.
    """

    def __init__(self, max_commands: int = 50, max_files: int = 100):
        """
        Initialize the context tracker.

        Args:
            max_commands: Maximum number of recent commands to keep
            max_files: Maximum number of recent files to track
        """
        self._lock = threading.Lock()

        # History tracking
        self._commands: deque[CommandContext] = deque(maxlen=max_commands)
        self._files: deque[FileContext] = deque(maxlen=max_files)
        self._window_history: deque[WindowContext] = deque(maxlen=10)

        # Current state
        self._current_window: Optional[WindowContext] = None
        self._current_project: Optional[str] = None
        self._browser_urls: Dict[str, str] = {}  # process_name -> url

        # Change detection
        self._last_screen_hash: Optional[str] = None
        self._last_screen_change: Optional[datetime] = None

        # Monitoring
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 2.0  # Check every 2 seconds

        # Callbacks for context changes
        self._on_window_change: List[Callable[[WindowContext, WindowContext], None]] = []
        self._on_project_change: List[Callable[[str, str], None]] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def start_monitoring(self) -> None:
        """Start continuous context monitoring in background."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="OperatingContextMonitor"
        )
        self._monitor_thread.start()
        logger.info("Operating context monitoring started")

    def stop_monitoring(self) -> None:
        """Stop continuous context monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Operating context monitoring stopped")

    def get_context(self) -> OperatingContext:
        """
        Get the current operating context.

        Returns a snapshot of all tracked context information.
        """
        with self._lock:
            return OperatingContext(
                active_window=self._current_window,
                recent_commands=list(self._commands),
                recent_files=list(self._files),
                current_project=self._current_project,
                browser_urls=self._browser_urls.copy(),
                clipboard_content=self._get_clipboard(),
                last_screen_change=self._last_screen_change,
                timestamp=datetime.now()
            )

    def record_command(self, command: str, source: str = "voice", result: Optional[str] = None) -> None:
        """
        Record a command execution.

        Args:
            command: The command that was executed
            source: Where the command came from ('voice', 'chat', 'automated')
            result: Optional result of the command
        """
        with self._lock:
            cmd_ctx = CommandContext(
                command=command,
                timestamp=datetime.now(),
                source=source,
                result=result
            )
            self._commands.append(cmd_ctx)

    def record_file_access(self, path: str, action: str = "opened") -> None:
        """
        Record a file access event.

        Args:
            path: Path to the file
            action: Action performed ('opened', 'saved', 'closed')
        """
        with self._lock:
            file_ctx = FileContext(
                path=path,
                timestamp=datetime.now(),
                action=action
            )
            self._files.append(file_ctx)

            # Update project inference
            self._infer_project_from_file(path)

    def on_window_change(self, callback: Callable[[WindowContext, WindowContext], None]) -> None:
        """Register a callback for window changes."""
        self._on_window_change.append(callback)

    def on_project_change(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback for project changes."""
        self._on_project_change.append(callback)

    # -------------------------------------------------------------------------
    # Internal monitoring loop
    # -------------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Background thread that continuously updates context."""
        while self._monitoring:
            try:
                # Update active window
                new_window = self._get_active_window_context()
                if new_window and new_window != self._current_window:
                    self._handle_window_change(new_window)

                # Update browser URLs if needed
                if new_window and new_window.app_type == AppType.BROWSER:
                    self._update_browser_url(new_window)

                # Check for screen changes
                self._check_screen_change()

                # Clean up old data periodically
                self._cleanup_old_data()

            except Exception as e:
                logger.debug(f"Context monitoring error: {e}")

            time.sleep(self._monitor_interval)

    def _handle_window_change(self, new_window: WindowContext) -> None:
        """Handle when the active window changes."""
        old_window = self._current_window

        with self._lock:
            self._window_history.append(new_window)
            self._current_window = new_window

        # Notify callbacks
        for callback in self._on_window_change:
            try:
                callback(old_window, new_window)
            except Exception as e:
                logger.warning(f"Window change callback failed: {e}")

        # Infer project from new window
        self._infer_project_from_window(new_window)

    def _infer_project_from_window(self, window: WindowContext) -> None:
        """Infer the current project from the active window."""
        if not window:
            return

        old_project = self._current_project
        new_project = None

        # Infer from window title and path
        if window.title:
            # Check for common project indicators
            project_patterns = [
                r'(?:.*?[-–/\s])?(.+?)(?:\s*[-–/].*Visual Studio Code|\s*[-–/].*VS Code)',
                r'(?:.*?[-–/\s])?(.+?)(?:\s*[-–/].*Chrome)',
                r'(?:.*?[-–/\s])?(.+?)(?:\s*[-–/].*Firefox)',
                r'(.+?)[–/]\s*.*',  # Pattern: "ProjectName – File"
            ]

            for pattern in project_patterns:
                match = re.search(pattern, window.title, re.IGNORECASE)
                if match:
                    potential = match.group(1).strip()
                    # Filter out common non-project names
                    if potential and not potential.lower() in ['new tab', 'blank', 'untitled']:
                        new_project = potential
                        break

        # Also check executable path
        if not new_project and window.executable_path:
            path_str = str(window.executable_path)
            # Look for project folders in path
            for parent in Path(path_str).parents:
                if parent.name.lower() in ['src', 'app', 'backend', 'frontend']:
                    # One level up might be project
                    if parent.parent != parent:
                        new_project = parent.parent.name
                        break

        # Update and notify if changed
        if new_project and new_project != old_project:
            with self._lock:
                self._current_project = new_project

            for callback in self._on_project_change:
                try:
                    callback(old_project or "", new_project)
                except Exception as e:
                    logger.warning(f"Project change callback failed: {e}")

    def _infer_project_from_file(self, file_path: str) -> None:
        """Update project inference from a file access."""
        try:
            path = Path(file_path)

            # Look for project indicators in the path
            for parent in path.parents:
                # Common project root indicators
                if any((parent / marker).exists() for marker in [
                    'package.json', 'Cargo.toml', 'go.mod',
                    'pom.xml', 'requirements.txt', '.git'
                ]):
                    new_project = parent.name
                    if new_project != self._current_project:
                        with self._lock:
                            self._current_project = new_project
                    break
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Context gathering
    # -------------------------------------------------------------------------

    def _get_active_window_context(self) -> Optional[WindowContext]:
        """Get detailed context about the active window."""
        try:
            import win32gui
            import win32process

            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)

            if not title:
                return None

            # Get process info
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(pid)

            # Get executable path (with timeout)
            exe_path = None
            try:
                exe_path = process.exe()
            except:
                pass

            # Classify application
            app_type = self._classify_app(title, process.name(), exe_path)

            return WindowContext(
                title=title,
                process_name=process.name(),
                app_type=app_type,
                executable_path=exe_path,
                timestamp=datetime.now()
            )

        except ImportError:
            # Fallback without psutil/win32
            return None
        except Exception as e:
            logger.debug(f"Failed to get window context: {e}")
            return None

    def _classify_app(self, title: str, process_name: str, exe_path: Optional[str]) -> AppType:
        """Classify the application type."""
        process_lower = process_name.lower()
        title_lower = title.lower()

        # Browsers
        browsers = ['chrome.exe', 'msedge.exe', 'firefox.exe', 'brave.exe', 'opera.exe']
        if any(b in process_lower for b in browsers):
            return AppType.BROWSER

        # Editors/IDEs
        editors = ['code.exe', 'code_insiders.exe', 'notepad++.exe', 'sublime_text.exe']
        ides = ['idea64.exe', 'pycharm64.exe', 'code.exe', 'devenv.exe']
        if any(e in process_lower for e in editors):
            return AppType.EDITOR
        if any(i in process_lower for i in ides):
            return AppType.IDE

        # Terminals
        terminals = ['windowsterminal.exe', 'cmd.exe', 'powershell.exe', 'pwsh.exe']
        if any(t in process_lower for t in terminals):
            return AppType.TERMINAL

        # Communication
        comm = ['slack.exe', 'teams.exe', 'discord.exe', 'telegram.exe', 'whatsapp.exe']
        if any(c in process_lower for c in comm):
            return AppType.COMMUNICATION

        # Media
        media = ['spotify.exe', 'vlc.exe', 'wmplayer.exe', 'netflix.exe']
        if any(m in process_lower for m in media):
            return AppType.MEDIA

        # File managers
        if 'explorer.exe' in process_lower:
            return AppType.FILE_MANAGER

        return AppType.UNKNOWN

    def _update_browser_url(self, window: WindowContext) -> None:
        """Update the current browser URL if possible."""
        # This requires browser-specific handling
        # For now, we'll store that we have a browser active
        pass

    def _check_screen_change(self) -> None:
        """Check if the screen content has changed significantly."""
        try:
            from PIL import ImageGrab
            import hashlib

            # Capture a small sample for efficiency
            screenshot = ImageGrab.grab()
            # Resize to small thumbnail for hash calculation
            thumbnail = screenshot.resize((100, 100))

            # Calculate hash
            img_bytes = thumbnail.tobytes()
            current_hash = hashlib.md5(img_bytes).hexdigest()

            if self._last_screen_hash and current_hash != self._last_screen_hash:
                self._last_screen_change = datetime.now()

            self._last_screen_hash = current_hash

        except Exception as e:
            logger.debug(f"Screen change detection failed: {e}")

    def _get_clipboard(self) -> Optional[str]:
        """Get current clipboard content if available."""
        try:
            import pyperclip
            content = pyperclip.paste()
            # Only return if not too large
            if content and len(content) < 1000:
                return content
        except Exception:
            pass
        return None

    def _cleanup_old_data(self) -> None:
        """Clean up old context data."""
        cutoff = datetime.now() - timedelta(hours=1)

        with self._lock:
            # Clean old commands
            self._commands = deque(
                [c for c in self._commands if c.timestamp > cutoff],
                maxlen=self._commands.maxlen
            )

            # Clean old files
            self._files = deque(
                [f for f in self._files if f.timestamp > cutoff],
                maxlen=self._files.maxlen
            )


# Global singleton
_tracker: Optional[OperatingContextTracker] = None
_tracker_lock = threading.Lock()


def get_tracker(auto_start: bool = True) -> OperatingContextTracker:
    """
    Get the global context tracker instance.

    Args:
        auto_start: Whether to auto-start monitoring (default True for normal use)
    """
    global _tracker

    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = OperatingContextTracker()
                # Auto-start monitoring only when requested
                if auto_start:
                    try:
                        _tracker.start_monitoring()
                    except Exception as e:
                        logger.warning(f"Failed to start context monitoring: {e}")

    return _tracker


def get_context() -> OperatingContext:
    """Convenience function to get current operating context."""
    return get_tracker().get_context()


def record_command(command: str, source: str = "voice", result: Optional[str] = None) -> None:
    """Convenience function to record a command."""
    get_tracker().record_command(command, source, result)


def record_file_access(path: str, action: str = "opened") -> None:
    """Convenience function to record file access."""
    get_tracker().record_file_access(path, action)
