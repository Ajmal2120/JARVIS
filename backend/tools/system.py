"""
PC-control tools: open apps, kill processes, screenshot, clipboard, volume.

All operations are sandboxed behind two safety layers:
  1. The dry-run engine (see main.py / jarvis_core.py) returns a "plan"
     first; only after the user approves does the real op run.
  2. pyautogui's FAILSAFE is on: dragging the mouse to a screen corner
     throws and aborts whatever was running.

These tools are OS-aware.  Each returns a small dict that the brain
turns into natural-language confirmation.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("jarvis.system")


# ---------------------------------------------------------------------------
# Path aliases: many apps live in C:\Program Files\...
# A small fuzzy lookup so "open discord" works.
# ---------------------------------------------------------------------------
COMMON_APPS = {
    "notepad":      ["notepad.exe"],
    "calculator":   ["calc.exe", "calculator.exe"],
    "paint":        ["mspaint.exe"],
    "explorer":     ["explorer.exe"],
    "file explorer":["explorer.exe"],
    "chrome":       ["chrome.exe"],
    "google chrome":["chrome.exe"],
    "firefox":      ["firefox.exe"],
    "edge":         ["msedge.exe"],
    "microsoft edge":["msedge.exe"],
    "word":         ["WINWORD.EXE"],
    "excel":        ["EXCEL.EXE"],
    "powerpoint":   ["POWERPNT.EXE"],
    "outlook":      ["OUTLOOK.EXE"],
    "spotify":      ["spotify.exe"],
    "discord":      ["discord.exe", "Discord.exe"],
    "vscode":       ["code.exe"],
    "vs code":      ["code.exe"],
    "code":         ["code.exe"],
    "visual studio code": ["code.exe"],
    "cmd":          ["cmd.exe"],
    "command prompt":["cmd.exe"],
    "terminal":     ["wt.exe", "cmd.exe"],
    "powershell":   ["powershell.exe"],
    "task manager": ["taskmgr.exe"],
    "settings":     ["ms-settings:"],
    "control panel":["control.exe"],
    "snipping tool":["snippingtool.exe"],
    "snip":         ["snippingtool.exe"],
    "vlc":          ["vlc.exe"],
    "steam":        ["steam.exe"],
    "cursor":       ["cursor.exe"],
    "pycharm":      ["pycharm64.exe", "pycharm.exe"],
    "intellij":     ["idea64.exe", "idea.exe"],
    "git bash":     ["C:/Program Files/Git/git-bash.exe"],
}


def _resolve_app_name(name: str) -> list[str]:
    """Return candidate exes for a user-friendly app name."""
    n = name.strip().lower()
    if n in COMMON_APPS:
        return COMMON_APPS[n]
    # Maybe it's a bare exe name
    if not n.endswith(".exe"):
        n_exe = n + ".exe"
        return [n_exe]
    return [n]


def _find_exe(candidates: list[str]) -> Optional[str]:
    """Pick the first candidate that exists in PATH or is an absolute path."""
    for c in candidates:
        if os.path.isabs(c) and os.path.exists(c):
            return c
        found = shutil.which(c)
        if found:
            return found
    return None


# ---------------------------------------------------------------------------
# open_app
# ---------------------------------------------------------------------------
async def open_app(name: str) -> dict[str, Any]:
    """
    Launch an application by friendly name or exe.  Returns a plan dict the
    brain uses to describe what it intends to do; the actual launch happens
    in the approval-flow wrapper.

    Returns {"ok": bool, "action": "open_app", "target": "...", "error": "..."}.
    """
    candidates = _resolve_app_name(name)
    exe = _find_exe_extended(candidates)
    if not exe:
        return {"ok": False, "action": "open_app", "target": name,
                "error": f"could not locate any of: {candidates}"}
    return {"ok": True, "action": "open_app", "target": exe,
            "display": name, "candidates": candidates}


def execute_open_app(target: str) -> dict[str, Any]:
    """Actually launch the .exe in a detached process."""
    try:
        if target.lower().endswith("ms-settings:"):
            os.startfile(target)
        else:
            subprocess.Popen(
                [target],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        return {"ok": True, "message": f"Launched {target}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# kill_process
# ---------------------------------------------------------------------------
async def kill_process(name: str) -> dict[str, Any]:
    """
    Plan to kill processes by name (e.g. "notepad" or "chrome").
    Returns a plan; doesn't actually kill anything until approved.
    """
    from tools.processes import find_pids_by_name  # local to keep import cold
    pids = find_pids_by_name(name)
    if not pids:
        return {"ok": False, "action": "kill_process", "target": name,
                "error": f"no running process matched '{name}'"}
    return {"ok": True, "action": "kill_process", "target": name,
            "pids": pids, "count": len(pids)}


def execute_kill_process(name: str) -> dict[str, Any]:
    """Terminate processes by name. Uses taskkill on Windows, pkill elsewhere."""
    try:
        if os.name == "nt":
            r = subprocess.run(
                ["taskkill", "/F", "/IM", name if name.lower().endswith(".exe") else name + ".exe"],
                capture_output=True, text=True, timeout=10,
            )
            return {"ok": r.returncode == 0,
                    "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
        else:
            r = subprocess.run(["pkill", "-f", name], capture_output=True, text=True, timeout=10)
            return {"ok": r.returncode == 0, "stdout": r.stdout.strip()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# wallpaper
# ---------------------------------------------------------------------------


async def set_wallpaper(path: str, style: str = "fill") -> dict[str, Any]:
    """Set Windows wallpaper using registry. Path can be full or relative to JARVIS."""
    from tools.processes import run_in_thread
    try:
        def _do():
            import winreg
            # Resolve path relative to JARVIS dir if needed
            if not os.path.isabs(path):
                base_dir = Path(os.path.dirname(__file__)).parent / "images"
                path_resolved = (base_dir / path).resolve()
                if not path_resolved.exists():
                    return False
            else:
                path_resolved = Path(path).resolve()

            # Set wallpaper style (fill, stretch, fit, tile, center)
            styles = {
                "fill": 0,
                "stretch": 2,
                "fit": 6,
                "tile": 1,
                "center": 0,
            }
            style_val = styles.get(style.lower(), 0)

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                              "Control Panel\\Desktop", 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "WallpaperStyle", 0, winreg.REG_SZ, str(style_val))
                winreg.SetValueEx(key, "TileWallpaper", 0, winreg.REG_SZ, "0")

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                              "Control Panel\\Desktop\\", 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "Wallpaper", 0, winreg.REG_SZ, str(path_resolved))

            # Refresh Windows shell
            import ctypes
            ctypes.windll.user32.SystemParametersInfoW(20, 0, str(path_resolved), 0)

            return True

        ok = await run_in_thread(_do)
        return {"ok": ok, "action": "set_wallpaper", "path": path, "style": style}
    except Exception as e:
        logger.warning("wallpaper set failed: %s", e)
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# screenshot
# ---------------------------------------------------------------------------
SCREENSHOTS_DIR = Path.home() / "Desktop" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


async def screenshot(monitor: int = 0) -> dict[str, Any]:
    """Capture the primary screen (or monitor N) and return its saved path."""
    from tools.processes import run_in_thread
    try:
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = SCREENSHOTS_DIR / f"jarvis-{ts}.png"
        def _capture():
            from PIL import ImageGrab  # type: ignore
            img = ImageGrab.grab(all_screens=(monitor != 0))
            img.save(path, "PNG")
        await run_in_thread(_capture)
        return {"ok": True, "action": "screenshot", "path": str(path),
                "filename": path.name, "size_bytes": path.stat().st_size}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def capture_screen() -> dict[str, Any]:
    """Capture the screen and return base64 image data for AI vision analysis.
    Use this when the user asks about what's on screen, wants to explain an app,
    design, or UI. Returns the image as base64 for Claude to analyze."""
    from tools.processes import run_in_thread
    import base64
    try:
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = SCREENSHOTS_DIR / f"jarvis-screen-{ts}.png"
        def _capture():
            from PIL import ImageGrab  # type: ignore
            img = ImageGrab.grab(all_screens=True)
            img.save(path, "PNG")
        await run_in_thread(_capture)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return {"ok": True, "image_data": b64, "path": str(path),
                "filename": path.name}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# clipboard
# ---------------------------------------------------------------------------
async def get_clipboard() -> dict[str, Any]:
    from tools.processes import run_in_thread
    def _read():
        try:
            import pyperclip  # type: ignore
            return pyperclip.paste()
        except ImportError:
            # Fall back to tkinter
            try:
                import tkinter as tk
                r = tk.Tk()
                r.withdraw()
                txt = r.clipboard_get()
                r.destroy()
                return txt
            except Exception as e:
                return f"[clipboard read failed: {e}]"
    text = await run_in_thread(_read)
    return {"ok": True, "action": "get_clipboard", "text": str(text)[:5000]}


async def set_clipboard(text: str) -> dict[str, Any]:
    from tools.processes import run_in_thread
    def _write():
        try:
            import pyperclip  # type: ignore
            pyperclip.copy(text)
            return True
        except ImportError:
            try:
                import tkinter as tk
                r = tk.Tk()
                r.withdraw()
                r.clipboard_clear()
                r.clipboard_append(text)
                r.update()
                r.destroy()
                return True
            except Exception:
                return False
    ok = await run_in_thread(_write)
    return {"ok": ok, "action": "set_clipboard", "length": len(text)}


# ---------------------------------------------------------------------------
# volume
# ---------------------------------------------------------------------------
async def set_volume(level: int) -> dict[str, Any]:
    """Set system volume to `level` (0-100). Uses pycaw on Windows."""
    level = max(0, min(100, int(level)))
    from tools.processes import run_in_thread
    def _do():
        try:
            if os.name == "nt":
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL  # type: ignore
                devices = AudioUtilities.GetSpeakers()
                # Newer pycaw (>=2024): exposes EndpointVolume directly;
                # older API: devices.Activate(...). Try both.
                if hasattr(devices, "EndpointVolume") and devices.EndpointVolume is not None:
                    vol = devices.EndpointVolume
                    vol.SetMasterVolumeLevelScalar(level / 100.0, None)
                else:
                    iface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                    vol = cast(iface, POINTER(IAudioEndpointVolume))
                    vol.SetMasterVolumeLevelScalar(level / 100.0, None)
            else:
                subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@",
                                f"{level}%"], check=False, capture_output=True)
            return True
        except Exception as e:
            logger.warning("volume set failed: %s", e)
            return False
    ok = await run_in_thread(_do)
    return {"ok": ok, "action": "set_volume", "level": level}


# ---------------------------------------------------------------------------
# type_text
# ---------------------------------------------------------------------------
async def type_text(text: str, interval: float = 0.0) -> dict[str, Any]:
    """Type text into the currently-focused window using pyautogui."""
    from tools.processes import run_in_thread
    def _do():
        try:
            import pyautogui  # type: ignore
            pyautogui.FAILSAFE = True
            pyautogui.write(text, interval=interval)
            return True
        except Exception as e:
            return f"err: {e}"
    res = await run_in_thread(_do)
    ok = res is True
    return {"ok": ok, "action": "type_text", "length": len(text),
            "error": None if ok else res}


# ---------------------------------------------------------------------------
# describe_plan
# ---------------------------------------------------------------------------
def describe_plan(plan: dict[str, Any]) -> str:
    """Turn a plan dict into a one-line human-readable proposal."""
    a = plan.get("action")
    if a == "open_app":
        return f"Open the application **{plan.get('display', plan.get('target'))}**."
    if a == "kill_process":
        c = plan.get("count", 0)
        return f"Kill {c} running process{'es' if c != 1 else ''} matching **{plan.get('target')}**."
    if a == "screenshot":
        return "Take a screenshot of your screen."
    if a == "get_clipboard":
        return "Read your clipboard contents."
    if a == "set_clipboard":
        return f"Write {plan.get('length', 0)} characters to your clipboard."
    if a == "set_volume":
        return f"Set system volume to **{plan.get('level')}%**."
    if a == "type_text":
        return f"Type {plan.get('length', 0)} characters into the focused window."
    if a == "play_media":
        return f"Play **{plan.get('query', '?')}**"
    if a == "lock_workstation":
        return "Lock the workstation."
    if a == "shutdown":
        return f"Shut down the computer ({plan.get('when', 'now')})."
    if a == "restart":
        return f"Restart the computer ({plan.get('when', 'now')})."
    return f"Run action `{a}`."


# ---------------------------------------------------------------------------
# On-the-fly app discovery
# ---------------------------------------------------------------------------
_PROGRAM_DIRS = [
    r"C:\Program Files",
    r"C:\Program Files (x86)",
]


def _scan_program_dirs(name: str) -> Optional[str]:
    """Look for `name.exe` one level under Program Files{, (x86)}."""
    target = name.lower()
    if not target.endswith(".exe"):
        target_exe = target + ".exe"
    else:
        target_exe = target
        target = target[:-4]
    for root in _PROGRAM_DIRS:
        if not os.path.isdir(root):
            continue
        try:
            for entry in os.listdir(root):
                full_dir = os.path.join(root, entry)
                if not os.path.isdir(full_dir):
                    continue
                # Direct match: <App>/<name>.exe
                cand = os.path.join(full_dir, target_exe)
                if os.path.isfile(cand):
                    return cand
                # Fuzzy: <App>/<App>.exe
                cand2 = os.path.join(full_dir, entry.lower() + ".exe")
                if target_exe == cand2.split(os.sep)[-1] and os.path.isfile(cand2):
                    return cand2
        except (PermissionError, OSError):
            continue
    return None


def _find_in_start_menu(name: str) -> Optional[str]:
    """Resolve a Start Menu .lnk shortcut by app name (Windows only)."""
    if os.name != "nt":
        return None
    try:
        import win32com.client  # type: ignore
    except ImportError:
        return None
    shell = win32com.client.Dispatch("WScript.Shell")
    target = name.lower().replace(".exe", "").strip()
    start_dirs = [
        os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"),
        r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs",
    ]
    for root in start_dirs:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                if not f.lower().endswith(".lnk"):
                    continue
                if target in f.lower():
                    shortcut = shell.CreateShortCut(os.path.join(dirpath, f))
                    target_path = shortcut.Targetpath
                    if target_path and os.path.isfile(target_path):
                        return target_path
    return None


# Patch _find_exe to fall back to Program Files + Start Menu
_orig_find_exe = _find_exe


def _find_exe_extended(candidates: list[str]) -> Optional[str]:
    exe = _orig_find_exe(candidates)
    if exe:
        return exe
    for c in candidates:
        exe = _scan_program_dirs(c)
        if exe:
            return exe
        exe = _find_in_start_menu(c)
        if exe:
            return exe
    return None


# ---------------------------------------------------------------------------
# play_media — open a Spotify track, YouTube URL, or any media URI
# ---------------------------------------------------------------------------
URI_SCHEMES = (
    "spotify:", "discord:", "steam://", "ms-settings:", "ms-windows-store:",
    "https://", "http://", "mailto:", "tel:",
)


def _looks_like_uri(q: str) -> bool:
    ql = q.strip().lower()
    return any(ql.startswith(s) for s in URI_SCHEMES)


async def play_media(query: str) -> dict[str, Any]:
    """
    Play a track, video, or open any URI.

    If `query` already looks like a URI, hand it straight to the OS.
    Otherwise call web_search to find a Spotify/YouTube URI, then open it.
    """
    query = (query or "").strip()
    if not query:
        return {"ok": False, "action": "play_media", "error": "empty query"}

    from tools.processes import run_in_thread

    # Direct URI path
    if _looks_like_uri(query):
        def _open_uri() -> tuple[bool, str]:
            try:
                if os.name == "nt":
                    os.startfile(query)  # type: ignore[attr-defined]
                else:
                    subprocess.Popen(["xdg-open", query],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                return True, query
            except Exception as e:
                return False, str(e)
        ok, info = await run_in_thread(_open_uri)
        return {"ok": ok, "action": "play_media", "uri": query,
                "query": query, "error": None if ok else info}

    # Search path — find a Spotify URI
    from tools import search as search_tool
    results: list = []
    try:
        # Step 1: try the literal site: search
        results = search_tool.web_search(
            f"{query} site:open.spotify.com/track", max_results=8
        ) or []
    except Exception:
        results = []

    # Step 2: if that returned nothing useful, try a plain query and pick
    # Spotify links out of the results.
    if not any("open.spotify.com/track/" in (r.get("href") or "") for r in results):
        try:
            results = search_tool.web_search(f"{query} spotify", max_results=8) or []
        except Exception:
            results = []

    uri = None
    title_hint = query
    for r in results:
        href = r.get("href") or r.get("url") or ""
        title = (r.get("title") or "").split("|")[0].split(" - ")[0].strip()
        if "open.spotify.com/track/" in href:
            track_id = href.split("/track/")[-1].split("?")[0].split("/")[0]
            if track_id:
                uri = f"spotify:track:{track_id}"
                title_hint = title or query
                break
        if not uri and ("youtube.com/watch" in href or "youtu.be/" in href):
            uri = href
            title_hint = title or query

    if not uri:
        # Last-ditch: search generic and open first YouTube link in browser
        try:
            results = search_tool.web_search(f"{query} youtube", max_results=5) or []
        except Exception:
            results = []
        for r in results:
            href = r.get("href") or ""
            if "youtube.com/watch" in href or "youtu.be/" in href:
                uri = href
                title = (r.get("title") or "").split("|")[0].split(" - ")[0].strip()
                title_hint = title or query
                break

    if not uri:
        return {"ok": False, "action": "play_media", "query": query,
                "error": "couldn't find a playable URI"}

    def _open() -> tuple[bool, str]:
        try:
            if os.name == "nt":
                os.startfile(uri)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", uri],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            return True, uri
        except Exception as e:
            return False, str(e)

    ok, info = await run_in_thread(_open)
    return {"ok": ok, "action": "play_media", "uri": uri,
            "query": query, "title": title_hint,
            "error": None if ok else info}


# ---------------------------------------------------------------------------
# wait_for_window — block until a window matching the name appears
# ---------------------------------------------------------------------------
async def wait_for_window(process_name: str, timeout: float = 5.0) -> bool:
    """Poll for a window whose title or process name matches `process_name`.
    Returns True if found within `timeout` seconds, False otherwise."""
    from tools.processes import run_in_thread
    name = (process_name or "").lower()
    if not name:
        return False

    def _check() -> bool:
        try:
            import pygetwindow  # type: ignore
            wins = pygetwindow.getAllWindows()
            for w in wins:
                if name in (w.title or "").lower():
                    try:
                        if not w.isMinimized:
                            w.activate()
                            return True
                    except Exception:
                        pass
            return False
        except Exception:
            return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        ok = await run_in_thread(_check)
        if ok:
            return True
        await asyncio.sleep(0.3)
    return False


# ---------------------------------------------------------------------------
# Destructive ops: lock, shutdown, restart
# ---------------------------------------------------------------------------
async def lock_workstation() -> dict[str, Any]:
    """Lock the Windows workstation (returns the user to the lock screen)."""
    from tools.processes import run_in_thread
    def _do() -> tuple[bool, str]:
        try:
            if os.name == "nt":
                import ctypes
                ctypes.windll.user32.LockWorkStation()
                return True, "locked"
            else:
                subprocess.run(["loginctl", "lock-session"], check=False,
                               capture_output=True)
                return True, "lock attempted (linux)"
        except Exception as e:
            return False, str(e)
    ok, info = await run_in_thread(_do)
    return {"ok": ok, "action": "lock_workstation", "message": info}


async def shutdown(when: str = "now") -> dict[str, Any]:
    """Shut down the computer. `when` is informational (now / +N min)."""
    from tools.processes import run_in_thread
    def _do() -> tuple[bool, str]:
        try:
            if os.name == "nt":
                subprocess.run(["shutdown", "/s", "/t", "0"],
                               check=False, capture_output=True)
                return True, "shutdown initiated"
            else:
                subprocess.run(["shutdown", "-h", "now"], check=False,
                               capture_output=True)
                return True, "shutdown initiated"
        except Exception as e:
            return False, str(e)
    ok, info = await run_in_thread(_do)
    return {"ok": ok, "action": "shutdown", "when": when, "message": info}


async def restart(when: str = "now") -> dict[str, Any]:
    """Restart the computer."""
    from tools.processes import run_in_thread
    def _do() -> tuple[bool, str]:
        try:
            if os.name == "nt":
                subprocess.run(["shutdown", "/r", "/t", "0"],
                               check=False, capture_output=True)
                return True, "restart initiated"
            else:
                subprocess.run(["shutdown", "-r", "now"], check=False,
                               capture_output=True)
                return True, "restart initiated"
        except Exception as e:
            return False, str(e)
    ok, info = await run_in_thread(_do)
    return {"ok": ok, "action": "restart", "when": when, "message": info}


# ---------------------------------------------------------------------------
# Android control (ADB)
# ---------------------------------------------------------------------------

def _find_adb() -> str:
    """Locate the ADB binary, checking common install paths."""
    # First, check PATH
    import shutil
    adb_path = shutil.which("adb")
    if adb_path:
        return adb_path
    # Then check common install locations
    candidates = [
        r"C:\Users\LOQ\AppData\Local\Android\Sdk\platform-tools\adb.exe",
        r"C:\Android\platform-tools\adb.exe",
        r"C:\platform-tools\adb.exe",
        r"C:\Program Files\Android\platform-tools\adb.exe",
        os.path.expanduser(r"~\AppData\Local\Android\Sdk\platform-tools\adb.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "adb"  # Fall back, will fail gracefully


async def adb_shell(cmd: str) -> dict[str, Any]:
    """Run an ADB shell command on a connected Android device."""
    from tools.processes import run_in_thread
    adb = _find_adb()
    try:
        def _do():
            r = subprocess.run(
                [adb, "shell", cmd],
                capture_output=True,
                timeout=30,  # First ADB call can be slow (starts daemon)
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            # Decode with replacement for Unicode chars that cp1252 can't handle
            stdout = r.stdout.decode("utf-8", errors="replace") if r.stdout else ""
            stderr = r.stderr.decode("utf-8", errors="replace") if r.stderr else ""
            return stdout, stderr, r.returncode
        stdout, stderr, code = await run_in_thread(_do)
        return {"ok": code == 0, "stdout": stdout.strip(),
                "stderr": stderr.strip(), "code": code}
    except FileNotFoundError:
        return {"ok": False, "error": "ADB not installed. Install platform-tools from https://developer.android.com/studio/releases/platform-tools"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def android_devices() -> dict[str, Any]:
    """List Android devices connected via ADB."""
    from tools.processes import run_in_thread
    adb = _find_adb()
    try:
        def _do():
            r = subprocess.run(
                [adb, "devices"],
                capture_output=True,
                text=True,
                timeout=15,  # First call starts daemon
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return r.stdout
        out = await run_in_thread(_do)
        devices = []
        for line in out.strip().split("\n")[1:]:
            if "\t" in line:
                serial, state = line.split("\t")
                devices.append({"serial": serial.strip(), "state": state.strip()})
        return {"ok": True, "devices": devices, "raw": out}
    except FileNotFoundError:
        return {"ok": False, "error": "ADB not installed"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def android_answer_call() -> dict[str, Any]:
    """Answer the incoming call on Android."""
    return await adb_shell("input keyevent KEYCODE_CALL")


async def android_reject_call() -> dict[str, Any]:
    """Reject the incoming call on Android (KEYCODE_ENDCALL)."""
    return await adb_shell("input keyevent KEYCODE_ENDCALL")


async def android_send_sms(phone: str, message: str) -> dict[str, Any]:
    """Send an SMS on Android via the Messages app."""
    # Try via am start with intent - works on most Android versions
    intent = (
        f"am start -a android.intent.action.SENDTO -d sms:{phone} "
        f"--es sms_body \"{message}\" --user 0"
    )
    result = await adb_shell(intent)
    # Press send button (usually at right edge, ~Y=1400)
    await adb_shell("input tap 1000 1400")
    return {"ok": result.get("ok"), "phone": phone, "message": message}


async def android_open_app(package: str) -> dict[str, Any]:
    """Open an Android app by package name (e.g. com.android.dialer)."""
    r = await adb_shell(f"monkey -p {package} -c android.intent.category.LAUNCHER 1")
    return {"ok": r.get("ok"), "package": package}


async def android_close_app(package: str) -> dict[str, Any]:
    """Force-stop an Android app."""
    r = await adb_shell(f"am force-stop {package}")
    return {"ok": r.get("ok"), "package": package}


async def android_notifications() -> dict[str, Any]:
    """Read recent Android notifications (latest 5 lines of dumpsys)."""
    r = await adb_shell("dumpsys notification --noredact | grep -E 'pkg|tag' | head -30")
    return {"ok": r.get("ok"), "raw": r.get("stdout", "")}


async def android_phone_info() -> dict[str, Any]:
    """Get basic Android device info."""
    model_r = await adb_shell("getprop ro.product.model")
    version_r = await adb_shell("getprop ro.build.version.release")
    battery_r = await adb_shell("dumpsys battery | grep level")
    return {
        "ok": model_r.get("ok"),
        "model": model_r.get("stdout", "?").strip(),
        "android_version": version_r.get("stdout", "?").strip(),
        "battery": battery_r.get("stdout", "?").strip(),
    }


async def android_lock_screen() -> dict[str, Any]:
    """Lock the Android screen."""
    return await adb_shell("input keyevent KEYCODE_POWER")


async def android_volume(level: int) -> dict[str, Any]:
    """Set Android media volume (0-100)."""
    level = max(0, min(100, int(level)))
    r = await adb_shell(f"settings put global volume_master {level}")
    if r.get("ok"):
        return {"ok": True, "level": level}
    return {"ok": False, "error": r.get("stderr", "failed")}


async def android_wifi(state: str = "on") -> dict[str, Any]:
    """Turn Android WiFi on/off."""
    cmd = "svc wifi enable" if state == "on" else "svc wifi disable"
    return await adb_shell(cmd)


async def android_bluetooth(state: str = "on") -> dict[str, Any]:
    """Turn Android Bluetooth on/off."""
    cmd = "svc bluetooth enable" if state == "on" else "svc bluetooth disable"
    return await adb_shell(cmd)



async def monitor_calls() -> dict[str, Any]:
    """Monitor for incoming calls on the Android device."""
    try:
        # Get telecom calls info
        result = await adb_shell("dumpsys telecom calls")
        if not result.get("ok"):
            return {"ok": True, "incoming_call": False, "caller": "Unknown", "details": ""}

        raw = result.get("stdout", "")
        if "Ringing" not in raw:
            return {"ok": True, "incoming_call": False, "caller": "Unknown", "details": ""}

        # Extract caller info
        caller = "Unknown"

        # Try multiple patterns to find the number
        import re

        # Pattern 1: number: +1234567890
        for line in raw.split("\n"):
            if "number:" in line.lower():
                numbers = re.findall(r'number[:\s]+(\d{10,}|\+?\d+)', line, re.IGNORECASE)
                if numbers:
                    num = numbers[0]
                    if len(num) == 10:
                        caller = f"Unknown contact ({num})"
                    elif num.startswith("+"):
                        caller = f"Unknown contact ({num})"
                    else:
                        caller = f"Unknown contact ({num})"
                    break

        # Pattern 2: display: Name
        if caller == "Unknown":
            for line in raw.split("\n"):
                if "display:" in line.lower() and "unknown" not in line.lower():
                    name_match = re.search(r'display[:\s]+([^\n]+)', line, re.IGNORECASE)
                    if name_match:
                        caller = name_match.group(1).strip()
                        break

        return {
            "ok": True,
            "incoming_call": True,
            "caller": caller,
            "details": raw
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "caller": "Unknown"}


async def make_call(number: str) -> dict[str, Any]:
    """Make a phone call to a number."""
    try:
        # Use CALL intent
        result = await adb_shell(f"am start -a android.intent.action.CALL -d tel://{number}")
        return {"ok": result.get("ok", False), "number": number}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def get_call_history() -> dict[str, Any]:
    """Get recent call history from the Android device."""
    try:
        # Use dumpsys call_log
        result = await adb_shell("dumpsys call_log | tail -20")
        calls = []
        if result.get("ok"):
            lines = result.get("stdout", "").split("\n")
            for line in lines:
                if "duration=" in line.lower() or "number=" in line.lower():
                    calls.append(line.strip())

        return {"ok": True, "calls": calls[:10]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def get_sms_history() -> dict[str, Any]:
    """Get recent SMS messages from the Android device."""
    try:
        # Use content resolver
        result = await adb_shell("content query --uri content://sms/inbox --limit 10")
        return {"ok": result.get("ok", False), "messages": result.get("stdout", "")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# PDF reading
# ---------------------------------------------------------------------------
async def read_pdf(path: str, max_pages: int = 5) -> dict[str, Any]:
    """Read and extract text from a PDF file. Returns first max_pages pages."""
    from tools.processes import run_in_thread
    try:
        def _do():
            import pypdf
            reader = pypdf.PdfReader(path)
            total = len(reader.pages)
            text_parts = []
            for i, page in enumerate(reader.pages[:max_pages]):
                text_parts.append(f"--- Page {i+1} ---\n{page.extract_text() or ''}")
            return {
                "ok": True,
                "total_pages": total,
                "pages_read": min(max_pages, total),
                "text": "\n\n".join(text_parts),
            }
        return await run_in_thread(_do)
    except FileNotFoundError:
        return {"ok": False, "error": f"PDF not found: {path}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
async def send_email(to: str, subject: str, body: str,
                    smtp_host: str = "smtp.gmail.com",
                    smtp_port: int = 587) -> dict[str, Any]:
    """Send an email. Configure SMTP credentials in .env (EMAIL_USER, EMAIL_PASS)."""
    from tools.processes import run_in_thread
    try:
        def _do():
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            # Read credentials from env
            email_user = os.environ.get("EMAIL_USER", "")
            email_pass = os.environ.get("EMAIL_PASS", "")
            if not email_user or not email_pass:
                return {"ok": False, "error": "EMAIL_USER and EMAIL_PASS not set in .env"}
            msg = MIMEMultipart()
            msg["From"] = email_user
            msg["To"] = to
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(email_user, email_pass)
                server.sendmail(email_user, to, msg.as_string())
            return {"ok": True, "to": to, "subject": subject}
        return await run_in_thread(_do)
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Auto-start on Windows login
# ---------------------------------------------------------------------------
def get_auto_start() -> dict[str, Any]:
    """Check if JARVIS is set to auto-start on Windows login."""
    if os.name != "nt":
        return {"ok": True, "enabled": False}
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                             winreg.KEY_READ)
        try:
            winreg.QueryValueEx(key, "JARVIS")
            winreg.CloseKey(key)
            return {"ok": True, "enabled": True}
        except FileNotFoundError:
            winreg.CloseKey(key)
            return {"ok": True, "enabled": False}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_auto_start(enable: bool) -> dict[str, Any]:
    """Enable or disable JARVIS auto-start on Windows login via registry."""
    if os.name != "nt":
        return {"ok": False, "error": "Not on Windows"}
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                             winreg.KEY_SET_VALUE)
        try:
            if enable:
                import sys as _sys
                jarvis_path = str(Path(__file__).parent.parent / "jarvis_embedded.py")
                winreg.SetValueEx(key, "JARVIS", 0, winreg.REG_SZ,
                                  f'"{_sys.executable}" "{jarvis_path}" --startup')
            else:
                try:
                    winreg.DeleteValue(key, "JARVIS")
                except FileNotFoundError:
                    pass
            winreg.CloseKey(key)
            return {"ok": True, "enabled": enable}
        except Exception as e:
            winreg.CloseKey(key)
            return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
