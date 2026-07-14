"""
Screen Control for JARVIS - Full GUI automation and vision capabilities.

Provides:
- Screenshot capture with base64 encoding
- Element finding via Claude Vision
- Click, type, scroll, hotkey operations
- OCR for reading screen text
- Active window detection
"""

import base64
import io
import logging
import re
import subprocess
import webbrowser
from typing import Optional


from config import settings

logger = logging.getLogger("jarvis.screen_control")

# Get LLM provider
async def _get_provider():
    """Get the configured LLM provider."""
    from llm import get_provider
    return get_provider()


# -----------------------------------------------------------------------------
# Screenshot
# -----------------------------------------------------------------------------

def take_screenshot(region: Optional[tuple] = None) -> str:
    """
    Take a screenshot and return as base64 PNG.

    Args:
        region: Optional (x, y, width, height) to capture specific area

    Returns:
        base64 encoded PNG image string
    """
    try:
        from PIL import Image, ImageGrab

        if region:
            screenshot = ImageGrab.grab(bbox=region)
        else:
            screenshot = ImageGrab.grab()

        # Resize to 1280x720 max for API efficiency
        max_width, max_height = 1280, 720
        if screenshot.width > max_width or screenshot.height > max_height:
            screenshot.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        # Convert to bytes
        buffer = io.BytesIO()
        screenshot.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()

        # Encode to base64
        return base64.b64encode(img_bytes).decode('utf-8')

    except Exception as e:
        logger.error(f"Screenshot failed: {e}")
        return ""


def save_screenshot(path: str, region: Optional[tuple] = None) -> bool:
    """Save screenshot to file."""
    try:
        from PIL import Image, ImageGrab

        if region:
            screenshot = ImageGrab.grab(bbox=region)
        else:
            screenshot = ImageGrab.grab()

        screenshot.save(path)
        return True
    except Exception as e:
        logger.error(f"Screenshot save failed: {e}")
        return False


# -----------------------------------------------------------------------------
# OCR - Read Screen Text
# -----------------------------------------------------------------------------

def read_screen_text(region: Optional[tuple] = None) -> str:
    """
    Read all visible text on screen using OCR.

    Args:
        region: Optional (x, y, width, height) to read specific area

    Returns:
        Extracted text as string
    """
    try:
        from PIL import Image, ImageGrab

        # Try to import pytesseract
        try:
            import pytesseract
            # Set tesseract path if specified
            tesseract_path = getattr(settings, 'TESSERACT_PATH', None)
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
        except ImportError:
            logger.warning("pytesseract not installed, OCR disabled")
            return "OCR not available. Please install pytesseract."

        if region:
            screenshot = ImageGrab.grab(bbox=region)
        else:
            screenshot = ImageGrab.grab()

        text = pytesseract.image_to_string(screenshot)
        return text.strip() if text else "No text found on screen."

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return f"Error reading screen: {str(e)}"


# -----------------------------------------------------------------------------
# Active Window
# -----------------------------------------------------------------------------

def get_active_window() -> str:
    """Get the title of the currently active window."""
    try:
        import win32gui
        import win32process

        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)

        if not title:
            return "Unknown window"

        # Get process name for additional context
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            import psutil
            process = psutil.Process(pid)
            process_name = process.name()
        except:
            process_name = None

        if process_name:
            return f"{title} ({process_name})"
        return title

    except ImportError:
        # Fallback without win32gui
        try:
            result = subprocess.run(
                ['powershell', '-Command',
                 '(Get-Process | Where-Object {$_.MainWindowHandle -ne 0} | '
                 'Where-Object {$_.Id -eq (GetForegroundWindow).MainWindowTitle} | '
                 'Select-Object -First 1 -ExpandProperty ProcessName)'],
                capture_output=True, text=True
            )
            return result.stdout.strip() or "Unknown window"
        except:
            return "Window detection unavailable"
    except Exception as e:
        logger.error(f"Get active window failed: {e}")
        return "Error getting active window"


# -----------------------------------------------------------------------------
# Mouse Control
# -----------------------------------------------------------------------------

def click_element(description: str) -> dict:
    """
    Find and click an element described by the user.
    Uses Claude Vision to locate the element on screen.

    Args:
        description: Description of element to click (e.g., "the search button")

    Returns:
        dict with success status and message
    """
    try:
        import pyautogui
        pyautogui.FAILSAFE = True

        # Take screenshot
        screenshot_b64 = take_screenshot()
        if not screenshot_b64:
            return {"success": False, "message": "Failed to capture screenshot"}

        # Use LLM Vision to find element coordinates
        provider = await _get_provider()

        response = await provider.chat(
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"""Look at this screenshot and find the element described as: "{description}"

If you find it, respond with ONLY the X and Y pixel coordinates of the CENTER of that element, formatted as:
X: [number], Y: [number]

If you cannot find it, respond with:
NOT FOUND

Do not include any other text in your response."""
                    }
                ]
            }],
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=None,
            tools=[],
        )

        result_text = response.text.strip()

        # Parse coordinates
        coord_match = re.search(r'X:\s*(\d+).*?Y:\s*(\d+)', result_text, re.IGNORECASE | re.DOTALL)

        if coord_match:
            x = int(coord_match.group(1))
            y = int(coord_match.group(2))

            # Scale coordinates back to actual screen size
            # (screenshot might be scaled down)
            from PIL import Image, ImageGrab
            actual_screen = ImageGrab.grab()
            screenshot = Image.open(io.BytesIO(base64.b64decode(screenshot_b64)))

            scale_x = actual_screen.width / screenshot.width
            scale_y = actual_screen.height / screenshot.height

            actual_x = int(x * scale_x)
            actual_y = int(y * scale_y)

            # Click
            pyautogui.click(actual_x, actual_y)
            logger.info(f"Clicked at ({actual_x}, {actual_y}): {description}")

            return {
                "success": True,
                "message": f"Clicked on {description} at ({actual_x}, {actual_y})"
            }
        else:
            return {
                "success": False,
                "message": f"Could not find '{description}' on screen"
            }

    except Exception as e:
        logger.error(f"Click element failed: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}


def find_and_click(description: str) -> dict:
    """Alias for click_element."""
    return click_element(description)


def scroll(direction: str, amount: int = 3) -> dict:
    """
    Scroll the screen.

    Args:
        direction: "up" or "down"
        amount: Number of scroll units (positive = up, negative = down)
    """
    try:
        import pyautogui
        pyautogui.FAILSAFE = True

        if direction.lower() == "up":
            pyautogui.scroll(amount)
        elif direction.lower() == "down":
            pyautogui.scroll(-amount)
        else:
            return {"success": False, "message": "Direction must be 'up' or 'down'"}

        return {"success": True, "message": f"Scrolled {direction} by {amount}"}
    except Exception as e:
        logger.error(f"Scroll failed: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}


# -----------------------------------------------------------------------------
# Keyboard Control
# -----------------------------------------------------------------------------

def type_text(text: str, interval: float = 0.05) -> dict:
    """
    Type text at current cursor position.

    Args:
        text: Text to type
        interval: Delay between keystrokes in seconds
    """
    try:
        import pyautogui
        pyautogui.FAILSAFE = True

        pyautogui.write(text, interval=interval)
        return {"success": True, "message": f"Typed: {text}"}
    except Exception as e:
        logger.error(f"Type text failed: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}


def press_hotkey(*keys) -> dict:
    """
    Press a keyboard shortcut.

    Args:
        *keys: Keys to press together (e.g., 'ctrl', 'c' for Ctrl+C)
    """
    try:
        import pyautogui
        pyautogui.FAILSAFE = True

        # Convert to list and ensure all are strings
        keys = [str(k).lower() for k in keys]
        pyautogui.hotkey(*keys)

        return {"success": True, "message": f"Pressed: {' + '.join(keys)}"}
    except Exception as e:
        logger.error(f"Hotkey failed: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}


# -----------------------------------------------------------------------------
# Browser Control
# -----------------------------------------------------------------------------

def open_url(url: str) -> dict:
    """Open a URL in the default browser."""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        webbrowser.open(url)
        return {"success": True, "message": f"Opened: {url}"}
    except Exception as e:
        logger.error(f"Open URL failed: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}


# -----------------------------------------------------------------------------
# Window Control
# -----------------------------------------------------------------------------

def close_window() -> dict:
    """Close the current window (Alt+F4)."""
    return press_hotkey('alt', 'f4')


def minimize_window() -> dict:
    """Minimize the current window."""
    return press_hotkey('win', 'down')


def maximize_window() -> dict:
    """Maximize/restore the current window."""
    return press_hotkey('win', 'up')


def switch_window() -> dict:
    """Switch to next window (Alt+Tab)."""
    return press_hotkey('alt', 'tab')


# -----------------------------------------------------------------------------
# Clipboard Operations
# -----------------------------------------------------------------------------

def copy_text() -> dict:
    """Copy selected text (Ctrl+C)."""
    return press_hotkey('ctrl', 'c')


def paste_text() -> dict:
    """Paste from clipboard (Ctrl+V)."""
    return press_hotkey('ctrl', 'v')


def select_all() -> dict:
    """Select all (Ctrl+A)."""
    return press_hotkey('ctrl', 'a')


def undo_action() -> dict:
    """Undo (Ctrl+Z)."""
    return press_hotkey('ctrl', 'z')


def save_file() -> dict:
    """Save file (Ctrl+S)."""
    return press_hotkey('ctrl', 's')


def new_tab() -> dict:
    """Open new tab (Ctrl+T)."""
    return press_hotkey('ctrl', 't')


def close_tab() -> dict:
    """Close tab (Ctrl+W)."""
    return press_hotkey('ctrl', 'w')


def refresh_page() -> dict:
    """Refresh page (F5)."""
    return press_hotkey('f5')


# -----------------------------------------------------------------------------
# Quick Actions
# -----------------------------------------------------------------------------

def quick_action(action: str) -> dict:
    """
    Execute a quick action by name.

    Actions: copy, paste, select_all, undo, save, new_tab, close_tab,
             refresh, close_window, minimize, maximize, switch_window
    """
    actions = {
        'copy': copy_text,
        'paste': paste_text,
        'select_all': select_all,
        'selectall': select_all,
        'undo': undo_action,
        'save': save_file,
        'new_tab': new_tab,
        'newtab': new_tab,
        'close_tab': close_tab,
        'closetab': close_tab,
        'refresh': refresh_page,
        'close_window': close_window,
        'closewindow': close_window,
        'minimize': minimize_window,
        'maximize': maximize_window,
        'switch_window': switch_window,
        'switchwindow': switch_window,
    }

    action_lower = action.lower().strip()

    if action_lower in actions:
        return actions[action_lower]()
    else:
        return {"success": False, "message": f"Unknown action: {action}"}
