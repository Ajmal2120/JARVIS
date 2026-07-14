"""
Browser Control Skill for JARVIS.

Provides comprehensive browser automation:
- Open URLs and search
- Tab management
- Page navigation
- Content extraction
- Form filling
"""

import asyncio
import logging
import subprocess
import webbrowser
from typing import Optional

import pyautogui

from skills.base_skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


class BrowserControlSkill(BaseSkill):
    """Skill for browser operations."""

    name = "browser_control"
    description = "Control web browser - open URLs, search, navigate tabs, fill forms"
    category = "browser"
    version = "1.0.0"

    # Browser paths (Windows)
    BROWSER_PATHS = {
        "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "firefox": r"C:\Program Files\Mozilla Firefox\firefox.exe",
    }

    async def execute(self, **kwargs) -> SkillResult:
        """Execute a browser operation."""
        operation = kwargs.get("operation", "")

        operations = {
            "open": self._open_url,
            "search": self._search,
            "new_tab": self._new_tab,
            "close_tab": self._close_tab,
            "refresh": self._refresh_page,
            "back": self._go_back,
            "forward": self._go_forward,
            "scroll": self._scroll_page,
            "click": self._click_element,
            "type": self._type_in_field,
            "submit": self._submit_form,
            "get_text": self._get_page_text,
            "screenshot": self._take_page_screenshot,
            "switch": self._switch_to_tab,
            "bookmark": self._add_bookmark,
        }

        if operation not in operations:
            return SkillResult(
                success=False,
                message=f"Unknown operation: {operation}. Available: {', '.join(operations.keys())}"
            )

        try:
            return await operations[operation](kwargs)
        except Exception as e:
            logger.error(f"Browser operation '{operation}' failed: {e}")
            return SkillResult(success=False, message=f"Error: {str(e)}")

    async def validate_params(self, **kwargs) -> bool:
        """Validate browser operation parameters."""
        operation = kwargs.get("operation", "")

        # Operations that don't need extra params
        no_param_ops = ["close_tab", "refresh_page", "go_back", "go_forward"]
        if operation.replace("_", "") in [op.replace("_", "") for op in no_param_ops]:
            return True

        # Operations that need specific params
        if operation in ["open", "new_tab", "switch"] and not kwargs.get("url"):
            return False
        if operation == "search" and not kwargs.get("query"):
            return False

        return True

    async def _open_url(self, kwargs) -> SkillResult:
        """Open a URL in the browser."""
        url = kwargs.get("url", "").strip()
        if not url:
            return SkillResult(success=False, message="No URL provided")

        # Add protocol if missing
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            # Try to use Chrome specifically for better automation
            try:
                subprocess.Popen([self.BROWSER_PATHS["chrome"], "--new-window", url])
            except (FileNotFoundError, subprocess.SubprocessError):
                # Fallback to default browser
                webbrowser.open(url, new=2)

            logger.info(f"Opened URL: {url}")
            return SkillResult(
                success=True,
                message=f"Opening {url}",
                data={"url": url}
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to open URL: {e}")

    async def _search(self, kwargs) -> SkillResult:
        """Perform a web search."""
        query = kwargs.get("query", "").strip()
        if not query:
            return SkillResult(success=False, message="No search query provided")

        try:
            # Use DuckDuckGo for privacy (no redirect)
            search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}&ia=web"

            try:
                subprocess.Popen([self.BROWSER_PATHS["chrome"], "--new-tab", search_url])
            except (FileNotFoundError, subprocess.SubprocessError):
                webbrowser.open(search_url, new=2)

            logger.info(f"Searching for: {query}")
            return SkillResult(
                success=True,
                message=f"Searching for '{query}'",
                data={"query": query, "url": search_url}
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Search failed: {e}")

    async def _new_tab(self, kwargs) -> SkillResult:
        """Open a new browser tab."""
        url = kwargs.get("url", "")

        try:
            if url:
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url

                try:
                    subprocess.Popen([self.BROWSER_PATHS["chrome"], "--new-tab", url])
                except (FileNotFoundError, subprocess.SubprocessError):
                    webbrowser.open(url, new=2)

                return SkillResult(success=True, message=f"Opened new tab: {url}")
            else:
                # Open blank tab
                pyautogui.hotkey('ctrl', 't')
                return SkillResult(success=True, message="Opened new tab")

        except Exception as e:
            return SkillResult(success=False, message=f"Failed to open new tab: {e}")

    async def _close_tab(self, kwargs=None) -> SkillResult:
        """Close current browser tab."""
        try:
            pyautogui.hotkey('ctrl', 'w')
            return SkillResult(success=True, message="Closed current tab")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to close tab: {e}")

    async def _refresh_page(self, kwargs=None) -> SkillResult:
        """Refresh current page."""
        try:
            pyautogui.hotkey('f5')
            return SkillResult(success=True, message="Page refreshed")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to refresh: {e}")

    async def _go_back(self, kwargs=None) -> SkillResult:
        """Go back to previous page."""
        try:
            pyautogui.hotkey('alt', 'left')
            return SkillResult(success=True, message="Navigated back")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to go back: {e}")

    async def _go_forward(self, kwargs=None) -> SkillResult:
        """Go forward to next page."""
        try:
            pyautogui.hotkey('alt', 'right')
            return SkillResult(success=True, message="Navigated forward")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to go forward: {e}")

    async def _scroll_page(self, kwargs) -> SkillResult:
        """Scroll the page."""
        try:
            direction = kwargs.get("direction", "down").lower()
            amount = int(kwargs.get("amount", 3))

            if direction == "up":
                pyautogui.scroll(amount * 100)  # Positive = up
            else:
                pyautogui.scroll(-amount * 100)  # Negative = down

            return SkillResult(success=True, message=f"Scrolled {direction}")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to scroll: {e}")

    async def _click_element(self, kwargs) -> SkillResult:
        """Click on an element by text."""
        try:
            from tools.screen_control import find_and_click
            text = kwargs.get("text", "")
            if not text:
                return SkillResult(success=False, message="No text specified to click")

            result = find_and_click(text)
            if result.get("success"):
                return SkillResult(success=True, message=f"Clicked: {text}")
            return SkillResult(success=False, message=f"Could not find: {text}")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to click: {e}")

    async def _type_in_field(self, kwargs) -> SkillResult:
        """Type text in a field."""
        try:
            from tools.screen_control import type_in_field
            field = kwargs.get("field", "")
            text = kwargs.get("text", "")

            if field:
                result = type_in_field(field, text)
            else:
                # Just type at current focus
                pyautogui.write(text, interval=0.05)
                result = {"success": True}

            if result.get("success"):
                return SkillResult(success=True, message=f"Typed: {text[:50]}")
            return SkillResult(success=False, message="Failed to type in field")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to type: {e}")

    async def _submit_form(self, kwargs) -> SkillResult:
        """Submit a form."""
        try:
            pyautogui.press('enter')
            return SkillResult(success=True, message="Form submitted")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to submit: {e}")

    async def _get_page_text(self, kwargs) -> SkillResult:
        """Get text content from the page."""
        try:
            from tools.screen_control import read_screen_text
            text = read_screen_text()

            if text:
                # Limit text length
                preview = text[:1000] + "..." if len(text) > 1000 else text
                return SkillResult(
                    success=True,
                    message=f"Page content ({len(text)} chars)",
                    data={"text": text, "preview": preview}
                )
            return SkillResult(success=False, message="No text found on page")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to read page: {e}")

    async def _take_page_screenshot(self, kwargs) -> SkillResult:
        """Take a screenshot of the browser."""
        try:
            from tools.screen_control import take_screenshot, save_screenshot
            import base64
            from datetime import datetime

            # Take screenshot
            b64 = take_screenshot()

            if b64:
                # Save to temp file
                import tempfile
                from pathlib import Path

                filename = f"browser_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                save_path = Path(tempfile.gettempdir()) / filename
                save_screenshot(str(save_path))

                return SkillResult(
                    success=True,
                    message=f"Screenshot saved: {filename}",
                    data={"path": str(save_path), "b64": b64[:100] + "..."}
                )
            return SkillResult(success=False, message="Failed to capture screenshot")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to screenshot: {e}")

    async def _switch_to_tab(self, kwargs) -> SkillResult:
        """Switch to a specific tab."""
        try:
            tab_index = kwargs.get("index", 0)
            current_tab = kwargs.get("current", 1)

            # Calculate how many Ctrl+Tab to press
            if tab_index > current_tab:
                presses = tab_index - current_tab
                for _ in range(presses):
                    pyautogui.hotkey('ctrl', 'tab')
            elif tab_index < current_tab:
                presses = current_tab - tab_index
                for _ in range(presses):
                    pyautogui.hotkey('ctrl', 'shift', 'tab')

            return SkillResult(success=True, message=f"Switched to tab {tab_index + 1}")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to switch tab: {e}")

    async def _add_bookmark(self, kwargs) -> SkillResult:
        """Add current page to bookmarks."""
        try:
            pyautogui.hotkey('ctrl', 'd')
            return SkillResult(success=True, message="Bookmark added")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to add bookmark: {e}")


# Instantiate to auto-register
_skill_instance = BrowserControlSkill()
