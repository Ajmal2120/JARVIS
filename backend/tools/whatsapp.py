"""
WhatsApp messaging for JARVIS via Selenium (WhatsApp Web).

Uses a real browser to control WhatsApp Web — no broken libraries needed.
First run opens a QR code window; scan it once and session persists.

Provides:
    send_whatsapp(phone, message) -> dict
    get_whatsapp_messages(phone, limit) -> dict
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_USER_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "whatsapp_profile")
_SESSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "whatsapp_session.json")

_driver = None
_lock = None


def _get_lock():
    """Lazy-init the lock."""
    global _lock
    if _lock is None:
        import threading
        _lock = threading.Lock()
    return _lock


def _ensure_data_dir():
    """Ensure data directories exist."""
    Path(os.path.dirname(_USER_DATA_DIR)).mkdir(parents=True, exist_ok=True)
    Path(_USER_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(_SESSION_FILE)).mkdir(parents=True, exist_ok=True)


def _get_driver(headless: bool = True):
    """Get or create the Selenium driver."""
    global _driver
    if _driver is not None:
        return _driver

    with _get_lock():
        if _driver is not None:
            return _driver

        try:
            from selenium import webdriver
            from selenium.webdriver.edge.service import Service as EdgeService
            from selenium.webdriver.edge.options import Options
            from webdriver_manager.microsoft import EdgeChromiumDriverManager

            _ensure_data_dir()
            options = Options()
            if headless:
                options.add_argument("--headless=new")
            options.add_argument(f"--user-data-dir={_USER_DATA_DIR}")
            options.add_argument("--no-first-run")
            options.add_argument("--no-default-browser-check")
            options.add_argument("--disable-background-timer-throttling")
            options.add_argument("--disable-backgrounding-occluded-windows")
            options.add_argument("--disable-renderer-backgrounding")
            options.add_argument("--window-size=1280,800")
            options.add_argument("--disable-gpu")
            options.add_argument("--remote-allow-origins=*")

            service = EdgeService(EdgeChromiumDriverManager().install())
            _driver = webdriver.Edge(service=service, options=options)
            _driver.set_window_size(1280, 800)
            logger.info("WhatsApp: Edge driver started")
            return _driver

        except Exception as e:
            logger.error(f"WhatsApp: Failed to start Edge driver: {e}")
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from webdriver_manager.chrome import ChromeDriverManager
                from selenium.webdriver.chrome.service import Service as ChromeService

                options = Options()
                if headless:
                    options.add_argument("--headless=new")
                options.add_argument(f"--user-data-dir={_USER_DATA_DIR}")
                options.add_argument("--no-first-run")
                options.add_argument("--window-size=1280,800")
                options.add_argument("--disable-gpu")

                service = ChromeService(ChromeDriverManager().install())
                _driver = webdriver.Chrome(service=service, options=options)
                _driver.set_window_size(1280, 800)
                logger.info("WhatsApp: Chrome driver started (fallback)")
                return _driver
            except Exception as e2:
                logger.error(f"WhatsApp: Chrome fallback also failed: {e2}")
                raise RuntimeError(f"WhatsApp browser not available. Install Edge or Chrome. Error: {e2}")


def _is_connected() -> bool:
    """Check if WhatsApp Web is logged in."""
    global _driver
    if _driver is None:
        return False
    try:
        _driver.get("https://web.whatsapp.com")
        time.sleep(2)
        # Check if we see the chat list (means logged in)
        # Look for the search box which appears when logged in
        try:
            _driver.find_element("css selector", "div[contenteditable='true'][data-tab='3']")
            return True
        except Exception:
            pass
        try:
            # Alternate selector for logged-in state
            _driver.find_element("css selector", "canvas")
            # If we see a canvas, it might be QR code - not connected
            return False
        except Exception:
            # No QR canvas found - might be logged in
            try:
                _driver.find_element("css selector", "div[title='Search input textbox']")
                return True
            except Exception:
                return False
    except Exception as e:
        logger.error(f"WhatsApp: connection check failed: {e}")
        return False


def _wait_for_login(timeout: int = 120) -> bool:
    """Wait for user to scan QR code."""
    global _driver
    logger.info(f"WhatsApp: Waiting for QR scan (timeout={timeout}s)")

    start = time.time()
    while time.time() - start < timeout:
        try:
            _driver.get("https://web.whatsapp.com")
            time.sleep(3)
            # Check if QR is present
            try:
                canvas = _driver.find_element("css selector", "canvas")
                logger.info("WhatsApp: QR code displayed - please scan within browser")
                # Poll until QR disappears (user scanned)
                for _ in range(timeout):
                    time.sleep(1)
                    try:
                        _driver.find_element("css selector", "canvas")
                    except Exception:
                        logger.info("WhatsApp: QR code gone - logged in!")
                        return True
            except Exception:
                # No QR code found - already logged in
                logger.info("WhatsApp: Already logged in (no QR code)")
                return True
        except Exception as e:
            logger.error(f"WhatsApp: Login wait error: {e}")
            time.sleep(2)

    logger.error("WhatsApp: Login timeout")
    return False


def _ensure_login():
    """Ensure we're logged into WhatsApp Web."""
    global _driver
    _driver = _get_driver(headless=True)

    # Quick check: are we already logged in?
    _driver.get("https://web.whatsapp.com")
    time.sleep(3)

    try:
        _driver.find_element("css selector", "canvas")
        logger.info("WhatsApp: QR code detected, opening browser for scan...")
        # QR is present - user needs to scan
        # Open a non-headless window for QR scanning
        _driver.quit()
        _driver = None
        _driver = _get_driver(headless=False)
        _driver.get("https://web.whatsapp.com")
        logger.info("WhatsApp: Browser window opened for QR scan. Please scan within 2 minutes.")
        if not _wait_for_login(timeout=120):
            logger.error("WhatsApp: QR scan timeout")
            _driver.quit()
            _driver = None
            raise RuntimeError("WhatsApp QR scan timeout. Please try again.")
        # Switch back to headless after login
        logger.info("WhatsApp: Login successful, switching to headless mode")
    except Exception:
        # No QR = already logged in
        pass


def _clean_number(phone: str) -> str:
    """Clean phone number to digits only."""
    return ''.join(c for c in str(phone) if c.isdigit())


def send_whatsapp(phone: str, message: str) -> dict[str, Any]:
    """
    Send a WhatsApp message to *phone*.

    phone can be:
      - E.164 format: "+919876543210"
      - Plain digits:   "919876543210"
      - WhatsApp link:  "https://wa.me/919876543210"

    Returns {"ok": True} or {"ok": False, "error": str}.
    """
    if not phone or not message:
        return {"ok": False, "error": "phone and message are required"}

    phone = str(phone).strip()
    message = str(message).strip()

    # Extract number from wa.me link
    if "wa.me/" in phone:
        phone = phone.split("wa.me/")[-1].split("?")[0]

    digits = _clean_number(phone)

    with _get_lock():
        try:
            _ensure_login()
            global _driver

            # Build URL with pre-filled message
            url = f"https://web.whatsapp.com/send?phone={digits}&text={message}"
            _driver.get(url)
            time.sleep(5)

            # Look for the send button
            try:
                send_btn = _driver.find_element("css selector", "button[data-testid='send']")
                send_btn.click()
                logger.info(f"WhatsApp: Message sent to {digits}")
                time.sleep(1)
                return {"ok": True}
            except Exception:
                pass

            # Try alternate send button selector
            try:
                send_btn = _driver.find_element("css selector", "span[data-icon='send']")
                send_btn.click()
                logger.info(f"WhatsApp: Message sent to {digits} (alt)")
                time.sleep(1)
                return {"ok": True}
            except Exception:
                pass

            # Try pressing Enter in the message box
            try:
                msg_box = _driver.find_element("css selector", "div[contenteditable='true']")
                msg_box.send_keys("\n")
                logger.info(f"WhatsApp: Message sent to {digits} (enter)")
                time.sleep(1)
                return {"ok": True}
            except Exception as e:
                logger.error(f"WhatsApp: Could not send message to {digits}: {e}")
                return {"ok": False, "error": f"Could not find send button for {digits}. Is the number valid and saved?"}

        except Exception as e:
            logger.error(f"WhatsApp: send failed: {e}")
            return {"ok": False, "error": str(e)}


def get_whatsapp_messages(phone: str, limit: int = 10) -> dict[str, Any]:
    """
    Read recent WhatsApp messages from a contact.

    Returns {"ok": True, "messages": [...]} or {"ok": False, "error": str}.
    """
    if not phone:
        return {"ok": False, "error": "phone is required"}

    phone = str(phone).strip()
    if "wa.me/" in phone:
        phone = phone.split("wa.me/")[-1].split("?")[0]
    digits = _clean_number(phone)

    with _get_lock():
        try:
            _ensure_login()
            global _driver

            url = f"https://web.whatsapp.com/send?phone={digits}"
            _driver.get(url)
            time.sleep(5)

            messages = []
            try:
                # Find all message elements
                msg_elements = _driver.find_elements("css selector", "div.message-in, div.message-out")
                for el in msg_elements[-limit:]:
                    try:
                        # Get text content
                        text_el = el.find_element("css selector", "span.selectable-text")
                        text = text_el.text

                        # Determine direction
                        is_in = "message-in" in el.get_attribute("class")
                        direction = "in" if is_in else "out"

                        messages.append({"direction": direction, "text": text})
                    except Exception:
                        continue
            except Exception:
                pass

            logger.info(f"WhatsApp: Read {len(messages)} messages from {digits}")
            return {"ok": True, "messages": messages}

        except Exception as e:
            logger.error(f"WhatsApp: read messages failed: {e}")
            return {"ok": False, "error": str(e)}


def close():
    """Close the WhatsApp browser."""
    global _driver
    if _driver is not None:
        try:
            _driver.quit()
        except Exception:
            pass
        _driver = None
        logger.info("WhatsApp: Browser closed")
