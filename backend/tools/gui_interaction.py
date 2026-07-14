"""
GUI Interaction Tools for JARVIS

Provides:
  - Screen capture with basic text detection (when pytesseract is available)
  - UI element coordinate detection
  - Automated mouse/keyboard control
  - Real-time screen analysis
  - Element clicking by text/description
  - Window detection and management

Uses only available packages: pyautogui, PIL, win32gui (Windows)
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pyautogui
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


@dataclass
class UIElement:
    """Represents a UI element on screen."""
    text: str
    x: int
    y: int
    width: int
    height: int
    window_title: str = ""
    element_type: str = "unknown"  # button, text_field, checkbox, etc.
    confidence: float = 0.0

    @property
    def center(self) -> Tuple[int, int]:
        """Get center coordinates."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Get bounding box (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class WindowInfo:
    """Window information."""
    title: str
    handle: int
    x: int
    y: int
    width: int
    height: int
    is_visible: bool = True

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside window."""
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)


class WindowManager:
    """Manage windows on Windows."""

    @staticmethod
    def get_window_by_title(title: str, partial: bool = True) -> Optional[int]:
        """Get window handle by title."""
        try:
            import win32gui

            windows = []

            def callback(hwnd, _windows):
                try:
                    window_title = win32gui.GetWindowText(hwnd)
                    if window_title:
                        if partial and title.lower() in window_title.lower():
                            _windows.append(hwnd)
                        elif window_title.lower() == title.lower():
                            _windows.append(hwnd)
                except Exception:
                    pass
                return True  # Continue enumeration

            win32gui.EnumWindows(callback, windows)
            return windows[0] if windows else None
        except ImportError:
            logger.warning("win32gui not available for window management")
            return None
        except Exception as e:
            logger.error(f"Error getting window: {e}")
            return None

    @staticmethod
    def get_window_rect(hwnd: int) -> Tuple[int, int, int, int]:
        """Get window coordinates."""
        try:
            import win32gui
            return win32gui.GetWindowRect(hwnd)
        except ImportError:
            logger.warning("win32gui not available")
            return (0, 0, 0, 0)
        except Exception as e:
            logger.error(f"Error getting window rect: {e}")
            return (0, 0, 0, 0)

    @staticmethod
    def activate_window(hwnd: int) -> bool:
        """Bring window to foreground."""
        try:
            import win32gui

            # Show and bring to foreground
            win32gui.ShowWindow(hwnd, 1)  # SW_SHOWNORMAL
            win32gui.SetForegroundWindow(hwnd)
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.error(f"Error activating window: {e}")
            return False

    @staticmethod
    def get_all_windows() -> List[WindowInfo]:
        """Get all visible windows."""
        try:
            import win32gui

            windows = []

            def callback(hwnd, windows):
                title = win32gui.GetWindowText(hwnd)
                if title and win32gui.IsWindowVisible(hwnd):
                    # Exclude some system windows
                    if not title.startswith("Default IME") and "Progman" not in title:
                        rect = win32gui.GetWindowRect(hwnd)
                        x, y, right, bottom = rect
                        width = right - x
                        height = bottom - y
                        windows.append(WindowInfo(
                            title=title,
                            handle=hwnd,
                            x=x,
                            y=y,
                            width=width,
                            height=height,
                            is_visible=True
                        ))

            win32gui.EnumWindows(callback, windows)
            return windows
        except ImportError:
            return []
        except Exception as e:
            logger.error(f"Error getting windows: {e}")
            return []


class ScreenReader:
    """Handles screen reading and OCR."""

    def __init__(self):
        self.tesseract_available = self._check_tesseract()

    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available."""
        try:
            import pytesseract
            # Try to find Tesseract in common locations
            tesseract_paths = [
                "C:/Program Files/Tesseract-OCR/tesseract.exe",
                "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe",
                r"C:\Tesseract-OCR\tesseract.exe",
            ]

            for path in tesseract_paths:
                if Path(path).exists():
                    pytesseract.pytesseract.tesseract_cmd = path
                    return True

            logger.warning("Tesseract not found - text detection will be limited")
            return False
        except:
            return False

    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """Capture screen or specific region."""
        try:
            return pyautogui.screenshot(region=region)
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            raise

    def capture_numpy(self, region: Optional[Tuple[int, int, int, int]] = None) -> "np.ndarray":
        """Capture screen as numpy array."""
        try:
            image = pyautogui.screenshot(region=region)
            return image.convert("RGB")
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            raise

    def capture_window(self, window_title: str) -> Optional[Image.Image]:
        """Capture a specific window."""
        hwnd = WindowManager.get_window_by_title(window_title)
        if hwnd:
            rect = WindowManager.get_window_rect(hwnd)
            x, y, right, bottom = rect
            return self.capture_screen((x, y, right - x, bottom - y))
        return None

    def ocr_text(self, image: Image.Image, lang: str = "eng") -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Extract text with bounding boxes from image (if Tesseract available)."""
        if not self.tesseract_available:
            return []

        try:
            import pytesseract

            # Convert to numpy for OpenCV compatibility if needed
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                output_type=pytesseract.Output.DICT
            )

            results = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 60:  # Confidence threshold
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    text = data['text'][i].strip()
                    if text:
                        results.append((text, (x, y, x + w, y + h)))

            return results
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return []

    def find_text_patterns(self, image: Image.Image, patterns: List[str]) -> Dict[str, List[Tuple[int, int]]]:
        """Find text patterns using OCR or simple image search."""
        results = {}

        if self.tesseract_available:
            texts_and_coords = self.ocr_text(image)
            for pattern in patterns:
                matches = []
                for text, bbox in texts_and_coords:
                    if pattern.lower() in text.lower():
                        matches.append(bbox)
                results[pattern] = matches
        else:
            # Fallback to simple image matching (less accurate)
            for pattern in patterns:
                # Convert to grayscale for better matching
                gray = image.convert('L')
                # Note: This is a simplified version - would need proper template matching
                # For now, just log
                results[pattern] = []

        return results


class MouseController:
    """Handles mouse automation."""

    @staticmethod
    def move_to(x: int, y: int, duration: float = 0.2):
        """Move mouse to position."""
        pyautogui.moveTo(x, y, duration=duration)

    @staticmethod
    def click(x: int, y: int, button: str = "left"):
        """Click at position."""
        pyautogui.click(x, y, button=button)

    @staticmethod
    def double_click(x: int, y: int):
        """Double-click at position."""
        pyautogui.doubleClick(x, y)

    @staticmethod
    def right_click(x: int, y: int):
        """Right-click at position."""
        pyautogui.rightClick(x, y)

    @staticmethod
    def drag(start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5):
        """Drag from one position to another."""
        pyautogui.drag(end_x, end_y, duration=duration, button='left')

    @staticmethod
    def scroll(amount: int):
        """Scroll mouse wheel."""
        pyautogui.scroll(amount)

    @staticmethod
    def position() -> Tuple[int, int]:
        """Get current mouse position."""
        return pyautogui.position()


class KeyboardController:
    """Handles keyboard automation."""

    @staticmethod
    def type_text(text: str, interval: float = 0.01):
        """Type text character by character."""
        pyautogui.write(text, interval=interval)

    @staticmethod
    def press(key: str):
        """Press a key."""
        pyautogui.press(key)

    @staticmethod
    def hotkey(*keys: str):
        """Press combination of keys."""
        pyautogui.hotkey(*keys)

    @staticmethod
    def enter():
        """Press Enter."""
        pyautogui.press('enter')

    @staticmethod
    def escape():
        """Press Escape."""
        pyautogui.press('escape')

    @staticmethod
    def tab():
        """Press Tab."""
        pyautogui.press('tab')

    @staticmethod
    def backspace():
        """Press Backspace."""
        pyautogui.press('backspace')

    @staticmethod
    def ctrl_a():
        """Ctrl+A (Select all)."""
        pyautogui.hotkey('ctrl', 'a')

    @staticmethod
    def ctrl_c():
        """Ctrl+C (Copy)."""
        pyautogui.hotkey('ctrl', 'c')

    @staticmethod
    def ctrl_v():
        """Ctrl+V (Paste)."""
        pyautogui.hotkey('ctrl', 'v')


class GUIAutomation:
    """Main GUI automation class."""

    def __init__(self):
        self.screen_reader = ScreenReader()
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.window_manager = WindowManager()

    def find_window_by_title(self, title: str, partial: bool = True) -> Optional[WindowInfo]:
        """Find window by title."""
        hwnd = WindowManager.get_window_by_title(title, partial)
        if hwnd:
            rect = WindowManager.get_window_rect(hwnd)
            x, y, right, bottom = rect
            return WindowInfo(
                title=win32gui.GetWindowText(hwnd),
                handle=hwnd,
                x=x,
                y=y,
                width=right - x,
                height=bottom - y
            )
        return None

    def activate_window(self, title: str, partial: bool = True) -> bool:
        """Activate window by title."""
        hwnd = WindowManager.get_window_by_title(title, partial)
        if hwnd:
            return WindowManager.activate_window(hwnd)
        return False

    def find_elements_by_text(self, text: str, window_title: str = None) -> List[UIElement]:
        """Find UI elements containing specific text."""
        try:
            # Capture screen or window
            if window_title:
                image = self.screen_reader.capture_window(window_title)
                if not image:
                    return []
            else:
                image = self.screen_reader.capture_screen()

            # OCR text
            if self.screen_reader.tesseract_available:
                texts_and_coords = self.screen_reader.ocr_text(image)
                elements = []

                for text_content, bbox in texts_and_coords:
                    if text.lower() in text_content.lower():
                        x1, y1, x2, y2 = bbox
                        elements.append(UIElement(
                            text=text_content,
                            x=x1,
                            y=y1,
                            width=x2 - x1,
                            height=y2 - y1,
                        ))

                return elements
            else:
                # Simple approach: look for image match
                # This would require creating templates for common buttons
                logger.warning("Tesseract not available - text search limited")
                return []

        except Exception as e:
            logger.error(f"Failed to find elements: {e}")
            return []

    def find_and_click(self, text: str, window_title: str = None, confidence: float = 0.8) -> bool:
        """Find element by text and click it."""
        try:
            elements = self.find_elements_by_text(text, window_title)
            if elements:
                # Take the first match
                element = elements[0]

                # If window specified, calculate absolute coordinates
                if window_title:
                    window = self.find_window_by_title(window_title)
                    if window:
                        x = window.x + element.x + element.width // 2
                        y = window.y + element.y + element.height // 2
                    else:
                        return False
                else:
                    x = element.x + element.width // 2
                    y = element.y + element.height // 2

                # Click with small delay to ensure accuracy
                time.sleep(0.1)
                self.mouse.click(x, y)
                time.sleep(0.5)  # Wait for action to register
                return True

            logger.warning(f"Element with text '{text}' not found")
            return False

        except Exception as e:
            logger.error(f"Failed to find and click: {e}")
            return False

    def click_at_position(self, x: int, y: int, window_title: str = None):
        """Click at specific screen coordinates."""
        try:
            # If window specified, calculate absolute coordinates
            if window_title:
                window = self.find_window_by_title(window_title)
                if window and window.contains_point(x, y):
                    x = window.x + x
                    y = window.y + y
                else:
                    logger.warning(f"Point ({x}, {y}) not in window {window_title}")
                    return

            self.mouse.click(x, y)
            time.sleep(0.5)
            logger.info(f"Clicked at position ({x}, {y})")

        except Exception as e:
            logger.error(f"Failed to click at position: {e}")

    def type_in_field(self, field_text: str, text: str, window_title: str = None) -> bool:
        """Find text field and type into it."""
        try:
            # Click on the field
            if self.find_and_click(field_text, window_title):
                time.sleep(0.3)
                self.keyboard.type_text(text)
                return True

            return False
        except Exception as e:
            logger.error(f"Failed to type in field: {e}")
            return False

    def wait_for_element(self, text: str, timeout: float = 10.0, check_interval: float = 0.5,
                        window_title: str = None) -> bool:
        """Wait for an element to appear on screen."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            elements = self.find_elements_by_text(text, window_title)
            if elements:
                return True
            time.sleep(check_interval)

        return False

    def highlight_element(self, element: UIElement, window_title: str = None,
                          color: Tuple[int, int, int] = (255, 0, 0), duration: float = 2.0):
        """Highlight an element on screen for debugging."""
        try:
            # Calculate absolute coordinates
            if window_title:
                window = self.find_window_by_title(window_title)
                if window:
                    x = window.x + element.x
                    y = window.y + element.y
                    width = element.width
                    height = element.height
                else:
                    return
            else:
                x = element.x
                y = element.y
                width = element.width
                height = element.height

            # Create temporary highlight
            screenshot = self.screen_reader.capture_screen()
            draw = ImageDraw.Draw(screenshot)

            # Draw rectangle
            draw.rectangle((x, y, x + width, y + height), outline=color, width=3)

            # Show screenshot
            screenshot.show()
            time.sleep(duration)

        except Exception as e:
            logger.error(f"Failed to highlight element: {e}")

    def get_active_window_info(self) -> Optional[WindowInfo]:
        """Get information about the active window."""
        try:
            import win32gui

            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            rect = win32gui.GetWindowRect(hwnd)

            x, y, right, bottom = rect
            return WindowInfo(
                title=title,
                handle=hwnd,
                x=x,
                y=y,
                width=right - x,
                height=bottom - y
            )
        except ImportError:
            return None
        except Exception as e:
            logger.error(f"Error getting active window: {e}")
            return None


# Global instance
_gui_automation = None


def get_gui_automation() -> GUIAutomation:
    """Get the global GUI automation instance."""
    global _gui_automation
    if _gui_automation is None:
        _gui_automation = GUIAutomation()
    return _gui_automation


# Tool definitions for JARVIS
TOOL_DEFINITIONS = [
    {
        "name": "find_and_click_element",
        "description": "Find UI element by text and click it. Searches entire screen or specific window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text of the element to find (case-insensitive)."
                },
                "window_title": {
                    "type": "string",
                    "description": "Optional: title of window to search within."
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "type_in_field",
        "description": "Find a text field and type text into it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "field_text": {
                    "type": "string",
                    "description": "Text identifying the text field."
                },
                "text": {
                    "type": "string",
                    "description": "Text to type into the field."
                },
                "window_title": {
                    "type": "string",
                    "description": "Optional: title of window containing the field."
                }
            },
            "required": ["field_text", "text"]
        }
    },
    {
        "name": "activate_window",
        "description": "Activate (bring to front) a window by title.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title or part of the window title."
                }
            },
            "required": ["title"]
        }
    },
    {
        "name": "capture_window",
        "description": "Capture screenshot of a specific window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title or part of the window title."
                }
            },
            "required": ["title"]
        }
    },
    {
        "name": "get_screen_text",
        "description": "Get all visible text on screen (requires Tesseract OCR).",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
]


# Convenience functions for jarvis_core.py integration
def capture_screen() -> Optional[Image.Image]:
    """Capture the entire screen. Returns PIL Image or None."""
    try:
        return ScreenReader().capture_screen()
    except Exception:
        return None


def get_all_windows() -> list:
    """Get all open windows."""
    try:
        return WindowManager().get_all_windows()
    except Exception:
        return []


def activate_window(title: str) -> bool:
    """Activate a window by title. Returns True if found."""
    try:
        return WindowManager().activate_window(title)
    except Exception:
        return False


def find_and_click(text: str) -> dict:
    """Find element by text and click it."""
    try:
        success = get_gui_automation().find_and_click(text)
        return {"success": bool(success), "message": "Clicked" if success else "Not found"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def type_in_field(field_text: str, text: str) -> dict:
    """Type text into a field identified by text."""
    try:
        result = get_gui_automation().type_in_field(field_text, text)
        return {"success": bool(result), "message": "Text entered" if result else "Field not found"}
    except Exception as e:
        return {"success": False, "message": str(e)}
