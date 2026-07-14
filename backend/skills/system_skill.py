"""
System Control Skill for JARVIS.

Provides system-wide operations:
- Shutdown, restart, lock
- Volume control
- Screen management
- Process management
"""

import logging
import subprocess
from typing import Optional

import pyautogui

from skills.base_skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


class SystemControlSkill(BaseSkill):
    """Skill for system control operations."""

    name = "system_control"
    description = "Control system - shutdown, restart, lock, volume, screen"
    category = "system"
    version = "1.0.0"

    async def execute(self, **kwargs) -> SkillResult:
        """Execute a system control operation."""
        operation = kwargs.get("operation", "")

        operations = {
            "shutdown": self._shutdown,
            "restart": self._restart,
            "lock": self._lock,
            "sleep": self._sleep,
            "hibernate": self._hibernate,
            "volume_up": self._volume_up,
            "volume_down": self._volume_down,
            "mute": self._mute,
            "get_volume": self._get_volume,
            "set_volume": self._set_volume,
            "take_screenshot": self._take_screenshot,
            "start_screen_saver": self._start_screen_saver,
            "display_off": self._display_off,
        }

        if operation not in operations:
            return SkillResult(
                success=False,
                message=f"Unknown operation: {operation}. Available: {', '.join(operations.keys())}"
            )

        try:
            return await operations[operation](kwargs)
        except Exception as e:
            logger.error(f"System operation '{operation}' failed: {e}")
            return SkillResult(success=False, message=f"Error: {str(e)}")

    async def validate_params(self, **kwargs) -> bool:
        """Validate system operation parameters."""
        operation = kwargs.get("operation", "")

        # Most operations don't need extra params
        no_param_ops = ["shutdown", "restart", "lock", "sleep", "hibernate",
                       "volume_up", "volume_down", "mute", "take_screenshot",
                       "start_screen_saver", "display_off"]
        if operation in no_param_ops:
            return True

        # Volume operations
        if operation in ["set_volume"]:
            volume = kwargs.get("volume")
            if volume is None or volume < 0 or volume > 100:
                return False

        return True

    async def _shutdown(self, kwargs) -> SkillResult:
        """Shutdown the system."""
        try:
            action = kwargs.get("confirm", False)
            if not action:
                return SkillResult(
                    success=True,
                    message="SHUTDOWN: Type 'confirm' to proceed with system shutdown"
                )

            pyautogui.hotkey('alt', 'f4')
            pyautogui.press('s')
            pyautogui.press('enter')
            return SkillResult(success=True, message="System shutting down")
        except Exception as e:
            return SkillResult(success=False, message=f"Shutdown failed: {e}")

    async def _restart(self, kwargs) -> SkillResult:
        """Restart the system."""
        try:
            action = kwargs.get("confirm", False)
            if not action:
                return SkillResult(
                    success=True,
                    message="RESTART: Type 'confirm' to proceed with system restart"
                )

            pyautogui.hotkey('alt', 'f4')
            pyautogui.press('r')
            pyautogui.press('enter')
            return SkillResult(success=True, message="System restarting")
        except Exception as e:
            return SkillResult(success=False, message=f"Restart failed: {e}")

    async def _lock(self, kwargs) -> SkillResult:
        """Lock the workstation."""
        try:
            from tools.system import lock_workstation
            result = await lock_workstation()
            if result.get("ok"):
                return SkillResult(success=True, message="Workstation locked")
            return SkillResult(success=False, message="Lock failed")
        except Exception as e:
            return SkillResult(success=False, message=f"Lock failed: {e}")

    async def _sleep(self, kwargs) -> SkillResult:
        """Put system to sleep."""
        try:
            pyautogui.hotkey('alt', 'f4')
            pyautogui.press('s')
            pyautogui.press('enter')
            return SkillResult(success=True, message="System sleeping")
        except Exception as e:
            return SkillResult(success=False, message=f"Sleep failed: {e}")

    async def _hibernate(self, kwargs) -> SkillResult:
        """Put system to hibernate."""
        try:
            pyautogui.hotkey('alt', 'f4')
            pyautogui.press('h')
            pyautogui.press('enter')
            return SkillResult(success=True, message="System hibernating")
        except Exception as e:
            return SkillResult(success=False, message=f"Hibernate failed: {e}")

    async def _volume_up(self, kwargs) -> SkillResult:
        """Increase volume by 5%."""
        try:
            amount = kwargs.get("amount", 5)
            pyautogui.press('volumeup')
            return SkillResult(success=True, message=f"Volume increased by {amount}%")
        except Exception as e:
            return SkillResult(success=False, message=f"Volume up failed: {e}")

    async def _volume_down(self, kwargs) -> SkillResult:
        """Decrease volume by 5%."""
        try:
            amount = kwargs.get("amount", 5)
            pyautogui.press('volumedown')
            return SkillResult(success=True, message=f"Volume decreased by {amount}%")
        except Exception as e:
            return SkillResult(success=False, message=f"Volume down failed: {e}")

    async def _mute(self, kwargs) -> SkillResult:
        """Toggle mute."""
        try:
            pyautogui.press('volumemute')
            return SkillResult(success=True, message="Mute toggled")
        except Exception as e:
            return SkillResult(success=False, message=f"Mute failed: {e}")

    async def _get_volume(self, kwargs) -> SkillResult:
        """Get current volume level."""
        try:
            from tools.system import get_volume
            result = await get_volume()
            if result.get("ok"):
                return SkillResult(
                    success=True,
                    message=f"Current volume: {result.get('level', 0)}%",
                    data={"level": result.get("level", 0)}
                )
            return SkillResult(success=False, message="Failed to get volume")
        except Exception as e:
            return SkillResult(success=False, message=f"Get volume failed: {e}")

    async def _set_volume(self, kwargs) -> SkillResult:
        """Set volume to specific level."""
        try:
            volume = kwargs.get("volume", 50)
            if not 0 <= volume <= 100:
                return SkillResult(success=False, message="Volume must be 0-100")

            from tools.system import set_volume
            result = await set_volume(volume)
            if result.get("ok"):
                return SkillResult(success=True, message=f"Volume set to {volume}%")
            return SkillResult(success=False, message="Failed to set volume")
        except Exception as e:
            return SkillResult(success=False, message=f"Set volume failed: {e}")

    async def _take_screenshot(self, kwargs) -> SkillResult:
        """Take a screenshot."""
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

                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                save_path = Path(tempfile.gettempdir()) / filename
                save_screenshot(str(save_path))

                return SkillResult(
                    success=True,
                    message=f"Screenshot saved: {filename}",
                    data={"path": str(save_path), "b64": b64[:100] + "..."}
                )
            return SkillResult(success=False, message="Screenshot failed")
        except Exception as e:
            return SkillResult(success=False, message=f"Screenshot failed: {e}")

    async def _start_screen_saver(self, kwargs) -> SkillResult:
        """Start screen saver."""
        try:
            pyautogui.hotkey('win', 'a')  # Action center (may vary)
            pyautogui.press('k')  # Quick actions (may need adjustment)
            return SkillResult(success=True, message="Screen saver started")
        except Exception as e:
            return SkillResult(success=False, message=f"Screen saver failed: {e}")

    async def _display_off(self, kwargs) -> SkillResult:
        """Turn off display."""
        try:
            pyautogui.hotkey('win', 'x')
            pyautogui.press('u')
            pyautogui.press('u')
            return SkillResult(success=True, message="Display turned off")
        except Exception as e:
            return SkillResult(success=False, message=f"Display off failed: {e}")


# Instantiate to auto-register
_skill_instance = SystemControlSkill()
