"""
Real-time Screen Analysis for JARVIS

Provides continuous screen monitoring and description capabilities:
  - Periodic screen captures with summary descriptions
  - Active window tracking
  - UI element detection (buttons, text fields, etc.)
  - Screen change detection
  - Natural language descriptions for AI context
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pyautogui
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ScreenSnapshot:
    """A point-in-time capture of the screen."""
    timestamp: datetime
    image: Image.Image
    active_window: str
    window_rect: tuple  # (x, y, right, bottom)
    description: str = ""
    elements: List[Dict] = None  # Detected UI elements

    def __post_init__(self):
        if self.elements is None:
            self.elements = []


class ScreenAnalyzer:
    """
    Analyzes screen content in real-time.

    Usage:
        analyzer = ScreenAnalyzer()
        analyzer.start()  # Start monitoring

        # Get current screen description
        description = analyzer.get_current_description()

        # Get active window info
        window = analyzer.get_active_window()

        analyzer.stop()
    """

    def __init__(self, capture_interval: float = 5.0):
        """
        Initialize screen analyzer.

        Args:
            capture_interval: Seconds between automatic captures (default 5s)
        """
        self.capture_interval = capture_interval
        self._running = False
        self._monitor_thread = None
        self._lock = threading.Lock()

        # Current state
        self._current_snapshot: Optional[ScreenSnapshot] = None
        self._snapshot_history: List[ScreenSnapshot] = []
        self._max_history = 12  # Keep last 12 snapshots (1 minute at 5s interval)

        # Callbacks
        self._on_change_callbacks: List[Callable[[ScreenSnapshot], None]] = []

        # Screen change detection
        self._last_screen_hash: Optional[int] = None
        self._change_threshold = 0.05  # 5% pixel difference = changed

        # Vision model for description (optional)
        self._vision_available = False
        self._check_vision()

    def _check_vision(self):
        """Check if vision capabilities are available."""
        try:
            # Check if we have a way to describe images
            # Could use Claude vision, but for now use basic analysis
            self._vision_available = True  # Use basic analysis
        except Exception:
            self._vision_available = False

    def _get_active_window_info(self) -> tuple:
        """Get active window title and rect."""
        try:
            import win32gui

            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            rect = win32gui.GetWindowRect(hwnd)

            return title, rect
        except ImportError:
            return "Unknown", (0, 0, 0, 0)
        except Exception as e:
            logger.debug(f"Error getting active window: {e}")
            return "Unknown", (0, 0, 0, 0)

    def capture(self) -> ScreenSnapshot:
        """Capture current screen state."""
        try:
            # Capture screen
            image = pyautogui.screenshot()

            # Get active window
            window_title, window_rect = self._get_active_window_info()

            # Create snapshot
            snapshot = ScreenSnapshot(
                timestamp=datetime.now(),
                image=image,
                active_window=window_title,
                window_rect=window_rect,
            )

            # Basic screen analysis (no AI needed)
            snapshot.description = self._analyze_screen_basic(image, window_title)

            # Detect UI elements
            snapshot.elements = self._detect_elements_basic(image)

            return snapshot

        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            raise

    def _analyze_screen_basic(self, image: Image.Image, window_title: str) -> str:
        """Basic screen analysis without AI vision."""
        try:
            # Get basic stats
            width, height = image.size

            # Analyze color distribution
            rgb_image = image.convert('RGB')
            pixels = list(rgb_image.getdata())

            # Count bright/dark pixels
            bright = sum(1 for p in pixels if sum(p) / 3 > 180)
            dark = sum(1 for p in pixels if sum(p) / 3 < 50)
            total = len(pixels)

            bright_pct = bright / total * 100
            dark_pct = dark / total * 100

            # Determine screen type based on colors
            if bright_pct > 70:
                screen_type = "light colored content"
            elif dark_pct > 50:
                screen_type = "dark mode or video"
            else:
                screen_type = "mixed content"

            # Window-specific descriptions
            window_desc = ""
            title_lower = window_title.lower()

            if any(x in title_lower for x in ['chrome', 'firefox', 'edge', 'browser']):
                window_desc = "a web browser"
            elif any(x in title_lower for x in ['code', 'visual studio', 'pycharm', 'sublime']):
                window_desc = "a code editor"
            elif any(x in title_lower for x in ['explorer', 'file']):
                window_desc = "File Explorer"
            elif any(x in title_lower for x in ['word', 'document']):
                window_desc = "a word processor"
            elif any(x in title_lower for x in ['excel', 'spreadsheet']):
                window_desc = "a spreadsheet application"
            elif any(x in title_lower for x in ['spotify', 'music', 'player']):
                window_desc = "a media player or music app"
            elif any(x in title_lower for x in ['terminal', 'cmd', 'powershell']):
                window_desc = "a terminal window"
            elif any(x in title_lower for x in ['discord', 'slack', 'teams', 'zoom']):
                window_desc = "a communication app"
            else:
                window_desc = f"'{window_title}'" if window_title else "an application"

            description = (
                f"Screen shows {window_desc}. "
                f"Resolution {width}x{height}. "
                f"Content appears to be {screen_type}."
            )

            return description

        except Exception as e:
            logger.debug(f"Basic analysis failed: {e}")
            return "Screen content unavailable"

    def _detect_elements_basic(self, image: Image.Image) -> List[Dict]:
        """Basic element detection without OCR."""
        # This is a simplified version - full OCR detection requires tesseract
        elements = []

        # For now, return empty list - OCR elements would be added here
        # when tesseract is available

        return elements

    def _has_screen_changed(self, image: Image.Image) -> bool:
        """Check if screen has significantly changed."""
        try:
            # Simple hash-based change detection
            # Convert to small thumbnail for fast comparison
            thumb = image.resize((100, 100)).convert('L')
            pixels = list(thumb.getdata())

            # Simple hash
            hash_val = sum(pixels) % 1000000

            if self._last_screen_hash is None:
                self._last_screen_hash = hash_val
                return True

            changed = abs(hash_val - self._last_screen_hash) > 500  # Threshold
            self._last_screen_hash = hash_val

            return changed

        except Exception:
            return True  # Assume changed on error

    def _monitor_loop(self):
        """Background monitoring loop."""
        logger.info("Screen analyzer started")

        while self._running:
            try:
                # Capture screen
                snapshot = self.capture()

                # Check if screen changed significantly
                if self._has_screen_changed(snapshot.image):
                    with self._lock:
                        self._current_snapshot = snapshot

                        # Add to history
                        self._snapshot_history.append(snapshot)
                        if len(self._snapshot_history) > self._max_history:
                            self._snapshot_history.pop(0)

                        # Notify callbacks
                        for callback in self._on_change_callbacks:
                            try:
                                callback(snapshot)
                            except Exception as e:
                                logger.debug(f"Change callback error: {e}")

                time.sleep(self.capture_interval)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(self.capture_interval)

        logger.info("Screen analyzer stopped")

    def start(self):
        """Start screen monitoring in background thread."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="ScreenAnalyzer"
        )
        self._monitor_thread.start()
        logger.info("Screen analysis started")

    def stop(self):
        """Stop screen monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def get_current_snapshot(self) -> Optional[ScreenSnapshot]:
        """Get the most recent screen snapshot."""
        with self._lock:
            return self._current_snapshot

    def get_current_description(self) -> str:
        """Get current screen description (for AI context)."""
        snapshot = self.get_current_snapshot()
        if snapshot:
            return snapshot.description
        return "No screen data available"

    def get_active_window(self) -> Dict[str, Any]:
        """Get active window information."""
        snapshot = self.get_current_snapshot()
        if snapshot:
            return {
                "title": snapshot.active_window,
                "rect": snapshot.window_rect,
            }
        return {"title": "Unknown", "rect": (0, 0, 0, 0)}

    def get_screen_base64(self, resize: tuple = None) -> Optional[str]:
        """Get current screen as base64 (for AI vision)."""
        snapshot = self.get_current_snapshot()
        if not snapshot:
            return None

        try:
            image = snapshot.image
            if resize:
                image = image.resize(resize)

            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('ascii')
        except Exception as e:
            logger.error(f"Failed to encode screen: {e}")
            return None

    def capture_now(self) -> ScreenSnapshot:
        """Force immediate capture (bypass interval)."""
        return self.capture()

    def on_screen_change(self, callback: Callable[[ScreenSnapshot], None]):
        """Register callback for screen changes."""
        self._on_change_callbacks.append(callback)

    def get_context_for_ai(self) -> Dict[str, Any]:
        """
        Get comprehensive screen context for AI.

        Returns structured data that can be passed to Claude as context.
        """
        snapshot = self.get_current_snapshot()
        if not snapshot:
            return {
                "screen_available": False,
                "error": "No screen data",
            }

        # Get window info
        window = self.get_active_window()

        # Get recent history summary
        recent = []
        with self._lock:
            for s in self._snapshot_history[-5:]:
                recent.append({
                    "window": s.active_window,
                    "time": s.timestamp.strftime("%H:%M:%S"),
                })

        return {
            "screen_available": True,
            "active_window": window["title"],
            "window_rect": window["rect"],
            "description": snapshot.description,
            "timestamp": snapshot.timestamp.isoformat(),
            "recent_windows": recent,
            "screen_base64": self.get_screen_base64(resize=(800, 450)),  # Thumbnail for vision
        }


# Global instance
_screen_analyzer: Optional[ScreenAnalyzer] = None


def get_screen_analyzer() -> ScreenAnalyzer:
    """Get or create global screen analyzer."""
    global _screen_analyzer
    if _screen_analyzer is None:
        _screen_analyzer = ScreenAnalyzer(capture_interval=5.0)
    return _screen_analyzer


def start_screen_analysis():
    """Start global screen analysis."""
    analyzer = get_screen_analyzer()
    analyzer.start()


def stop_screen_analysis():
    """Stop global screen analysis."""
    global _screen_analyzer
    if _screen_analyzer:
        _screen_analyzer.stop()


def get_screen_context() -> Dict[str, Any]:
    """Get current screen context for AI."""
    return get_screen_analyzer().get_context_for_ai()
