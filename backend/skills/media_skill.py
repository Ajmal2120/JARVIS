"""
Media Control Skill for JARVIS.

Provides media playback control:
- Play/pause
- Volume control
- Next/previous track
- Mute/unmute
- Launch media apps
"""

import logging
from typing import Optional

import pyautogui

from skills.base_skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


class MediaControlSkill(BaseSkill):
    """Skill for media playback control."""

    name = "media_control"
    description = "Control media playback - play, pause, volume, skip tracks"
    category = "media"
    version = "1.0.0"

    # Media applications paths
    MEDIA_APPS = {
        "spotify": r"C:\Users\LOQ\AppData\Local\Microsoft\WindowsApps\Spotify.exe",
        "vlc": r"C:\Program Files\VideoLAN\VLC\vlc.exe",
        "netflix": "https://www.netflix.com",
        "youtube": "https://www.youtube.com",
    }

    async def execute(self, **kwargs) -> SkillResult:
        """Execute a media control operation."""
        operation = kwargs.get("operation", "")

        operations = {
            "play_pause": self._play_pause,
            "play": self._play_pause,
            "pause": self._play_pause,
            "next": self._next_track,
            "previous": self._previous_track,
            "volume_up": self._volume_up,
            "volume_down": self._volume_down,
            "mute": self._mute,
            "stop": self._stop,
            "seek_forward": self._seek_forward,
            "seek_backward": self._seek_backward,
            "fullscreen": self._toggle_fullscreen,
            "launch": self._launch_app,
        }

        if operation not in operations:
            return SkillResult(
                success=False,
                message=f"Unknown operation: {operation}. Available: {', '.join(operations.keys())}"
            )

        try:
            return await operations[operation](kwargs)
        except Exception as e:
            logger.error(f"Media operation '{operation}' failed: {e}")
            return SkillResult(success=False, message=f"Error: {str(e)}")

    async def validate_params(self, **kwargs) -> bool:
        """Validate media operation parameters."""
        operation = kwargs.get("operation", "")

        # Most operations don't need extra params
        simple_ops = ["play_pause", "play", "pause", "next", "previous",
                     "volume_up", "volume_down", "mute", "stop", "fullscreen"]
        if operation in simple_ops:
            return True

        # Seek operations need amount
        if operation in ["seek_forward", "seek_backward"]:
            return True  # amount is optional

        # Launch needs app
        if operation == "launch":
            return kwargs.get("app") is not None

        return True

    async def _play_pause(self, kwargs) -> SkillResult:
        """Toggle play/pause."""
        try:
            pyautogui.press('playpause')
            return SkillResult(success=True, message="Play/Pause toggled")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to toggle: {e}")

    async def _next_track(self, kwargs) -> SkillResult:
        """Skip to next track."""
        try:
            pyautogui.press('nexttrack')
            return SkillResult(success=True, message="Next track")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to skip: {e}")

    async def _previous_track(self, kwargs) -> SkillResult:
        """Go to previous track."""
        try:
            pyautogui.press('prevtrack')
            return SkillResult(success=True, message="Previous track")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to go back: {e}")

    async def _volume_up(self, kwargs) -> SkillResult:
        """Increase volume."""
        try:
            amount = kwargs.get("amount", 5)
            # Use the system volume tool for better control
            from tools.system import set_volume
            current = await self._get_current_volume()
            new_vol = min(100, current + amount)
            result = await set_volume(new_vol)
            if result.get("ok"):
                return SkillResult(success=True, message=f"Volume: {new_vol}%")
            return SkillResult(success=False, message="Failed to adjust volume")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to increase volume: {e}")

    async def _volume_down(self, kwargs) -> SkillResult:
        """Decrease volume."""
        try:
            amount = kwargs.get("amount", 5)
            from tools.system import set_volume
            current = await self._get_current_volume()
            new_vol = max(0, current - amount)
            result = await set_volume(new_vol)
            if result.get("ok"):
                return SkillResult(success=True, message=f"Volume: {new_vol}%")
            return SkillResult(success=False, message="Failed to adjust volume")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to decrease volume: {e}")

    async def _mute(self, kwargs) -> SkillResult:
        """Toggle mute."""
        try:
            pyautogui.press('volumemute')
            return SkillResult(success=True, message="Mute toggled")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to toggle mute: {e}")

    async def _stop(self, kwargs) -> SkillResult:
        """Stop playback."""
        try:
            pyautogui.press('stop')
            return SkillResult(success=True, message="Playback stopped")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to stop: {e}")

    async def _seek_forward(self, kwargs) -> SkillResult:
        """Seek forward in media."""
        try:
            amount = kwargs.get("amount", 10)
            pyautogui.hotkey('shift', 'right')
            return SkillResult(success=True, message=f"Seeked forward {amount}s")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to seek: {e}")

    async def _seek_backward(self, kwargs) -> SkillResult:
        """Seek backward in media."""
        try:
            amount = kwargs.get("amount", 10)
            pyautogui.hotkey('shift', 'left')
            return SkillResult(success=True, message=f"Seeked backward {amount}s")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to seek: {e}")

    async def _toggle_fullscreen(self, kwargs) -> SkillResult:
        """Toggle fullscreen mode."""
        try:
            pyautogui.press('f')
            return SkillResult(success=True, message="Fullscreen toggled")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to toggle fullscreen: {e}")

    async def _launch_app(self, kwargs) -> SkillResult:
        """Launch a media application."""
        try:
            app = kwargs.get("app", "").lower()
            if not app:
                return SkillResult(success=False, message="No app specified")

            # Try to launch
            if app in self.MEDIA_APPS:
                path = self.MEDIA_APPS[app]
                if path.startswith("http"):
                    # It's a URL, open in browser
                    from skills.browser_skill import BrowserControlSkill
                    browser = BrowserControlSkill()
                    return await browser.execute(operation="open", url=path)
                else:
                    # It's an executable
                    import subprocess
                    subprocess.Popen([path], shell=True)
                    return SkillResult(success=True, message=f"Launched {app}")
            else:
                # Try to find and launch the app
                from tools.system import open_app
                result = await open_app(app)
                if result.get("ok"):
                    return SkillResult(success=True, message=f"Launched {app}")
                return SkillResult(success=False, message=f"Could not find: {app}")
        except Exception as e:
            return SkillResult(success=False, message=f"Failed to launch: {e}")

    async def _get_current_volume(self) -> int:
        """Get current system volume level."""
        try:
            from tools.system import get_volume
            result = await get_volume()
            if result.get("ok"):
                return result.get("level", 50)
            return 50
        except:
            return 50


# Instantiate to auto-register
_skill_instance = MediaControlSkill()
