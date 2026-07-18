"""
Regression tests for app-launching reliability.

Verifies:
  - Web destinations resolve to URLs and execute via browser path.
  - Desktop applications resolve to .exe paths.
  - Nonexistent apps fail gracefully.
  - execute_open_app never claims success on failure.
  - command_router never says "Opened X" when execution failed.
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests: COMMON_APPS entries
# ---------------------------------------------------------------------------

class TestCommonApps(unittest.TestCase):
    """Web destinations must be in COMMON_APPS and resolve to URLs."""

    def test_youtube_is_url(self):
        from tools.system import COMMON_APPS
        self.assertEqual(COMMON_APPS["youtube"], ["https://www.youtube.com"])

    def test_github_is_url(self):
        from tools.system import COMMON_APPS
        self.assertEqual(COMMON_APPS["github"], ["https://github.com"])

    def test_chatgpt_is_url(self):
        from tools.system import COMMON_APPS
        self.assertEqual(COMMON_APPS["chatgpt"], ["https://chat.openai.com"])

    def test_gmail_is_url(self):
        from tools.system import COMMON_APPS
        self.assertEqual(COMMON_APPS["gmail"], ["https://mail.google.com"])

    def test_linkedin_is_url(self):
        from tools.system import COMMON_APPS
        self.assertEqual(COMMON_APPS["linkedin"], ["https://www.linkedin.com"])

    def test_reddit_is_url(self):
        from tools.system import COMMON_APPS
        self.assertEqual(COMMON_APPS["reddit"], ["https://www.reddit.com"])

    def test_stackoverflow_is_url(self):
        from tools.system import COMMON_APPS
        self.assertEqual(COMMON_APPS["stackoverflow"], ["https://stackoverflow.com"])

    def test_x_is_url(self):
        from tools.system import COMMON_APPS
        self.assertEqual(COMMON_APPS["x"], ["https://x.com"])

    def test_twitter_is_url(self):
        from tools.system import COMMON_APPS
        self.assertEqual(COMMON_APPS["twitter"], ["https://x.com"])

    def test_chrome_is_exe(self):
        from tools.system import COMMON_APPS
        self.assertEqual(COMMON_APPS["chrome"], ["chrome.exe"])

    def test_vscode_is_exe(self):
        from tools.system import COMMON_APPS
        self.assertEqual(COMMON_APPS["vscode"], ["code.exe"])

    def test_calculator_is_exe(self):
        from tools.system import COMMON_APPS
        self.assertEqual(COMMON_APPS["calculator"], ["calc.exe", "calculator.exe"])

    def test_notepad_is_exe(self):
        from tools.system import COMMON_APPS
        self.assertEqual(COMMON_APPS["notepad"], ["notepad.exe"])


# ---------------------------------------------------------------------------
# Tests: _find_exe URL passthrough
# ---------------------------------------------------------------------------

class TestFindExeURLPassthrough(unittest.TestCase):

    def test_https_url_passthrough(self):
        from tools.system import _find_exe
        self.assertEqual(_find_exe(["https://www.youtube.com"]), "https://www.youtube.com")

    def test_http_url_passthrough(self):
        from tools.system import _find_exe
        self.assertEqual(_find_exe(["http://example.com"]), "http://example.com")

    def test_mixed_candidates_url_first(self):
        from tools.system import _find_exe
        result = _find_exe(["https://github.com", "github.exe"])
        self.assertEqual(result, "https://github.com")


# ---------------------------------------------------------------------------
# Tests: open_app resolution
# ---------------------------------------------------------------------------

class TestOpenAppResolution(unittest.TestCase):

    def test_youtube_resolves(self):
        from tools.system import open_app
        plan = _run(open_app("youtube"))
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["target"], "https://www.youtube.com")

    def test_github_resolves(self):
        from tools.system import open_app
        plan = _run(open_app("github"))
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["target"], "https://github.com")

    def test_chatgpt_resolves(self):
        from tools.system import open_app
        plan = _run(open_app("chatgpt"))
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["target"], "https://chat.openai.com")

    def test_x_resolves(self):
        from tools.system import open_app
        plan = _run(open_app("x"))
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["target"], "https://x.com")

    def test_nonexistent_app_fails(self):
        from tools.system import open_app
        plan = _run(open_app("nonexistent_xyz_app"))
        self.assertFalse(plan["ok"])
        self.assertIn("error", plan)

    def test_chrome_resolves(self):
        from tools.system import open_app
        plan = _run(open_app("chrome"))
        self.assertTrue(plan["ok"])
        self.assertTrue(plan["target"].endswith("chrome.exe"))

    def test_vscode_resolves(self):
        from tools.system import open_app
        plan = _run(open_app("vscode"))
        self.assertTrue(plan["ok"])
        self.assertTrue(plan["target"].lower().endswith("code.exe"))

    def test_calculator_resolves(self):
        from tools.system import open_app
        plan = _run(open_app("calculator"))
        self.assertTrue(plan["ok"])
        self.assertTrue(plan["target"].endswith("calc.exe"))


# ---------------------------------------------------------------------------
# Tests: execute_open_app
# ---------------------------------------------------------------------------

class TestExecuteOpenApp(unittest.TestCase):

    def test_browser_url_succeeds(self):
        from tools.system import execute_open_app
        with patch("webbrowser.open"):
            result = execute_open_app("https://www.youtube.com")
        self.assertTrue(result["ok"])
        self.assertIn("message", result)
        self.assertIn("Opened", result["message"])

    def test_browser_url_method_is_browser(self):
        from tools.system import execute_open_app
        with patch("webbrowser.open"):
            result = execute_open_app("https://github.com")
        self.assertTrue(result["ok"])
        self.assertEqual(result.get("method"), "browser")

    def test_exe_crash_no_false_success(self):
        from tools.system import execute_open_app
        fake_proc = MagicMock()
        fake_proc.poll.return_value = 1
        fake_proc.returncode = 1
        with patch("tools.system.subprocess.Popen", return_value=fake_proc):
            with patch("tools.system.time.sleep"):  # speed up test
                result = execute_open_app(r"C:\fake\app.exe")
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_exe_exception_no_false_success(self):
        from tools.system import execute_open_app
        with patch("tools.system.subprocess.Popen", side_effect=OSError("file not found")):
            result = execute_open_app(r"C:\fake\app.exe")
        self.assertFalse(result["ok"])
        self.assertIn("error", result)
        self.assertEqual(result["error"], "file not found")

    def test_exe_success_returns_ok(self):
        from tools.system import execute_open_app
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        with patch("tools.system.subprocess.Popen", return_value=fake_proc):
            with patch("tools.system.time.sleep"):  # speed up test
                result = execute_open_app(r"C:\Windows\system32\notepad.exe")
        self.assertTrue(result["ok"])
        self.assertIn("message", result)
        self.assertEqual(result.get("method"), "exe")

    def test_ms_settings_returns_ok(self):
        from tools.system import execute_open_app
        with patch("os.startfile"):
            result = execute_open_app("ms-settings:")
        self.assertTrue(result["ok"])
        self.assertEqual(result.get("method"), "settings")


# ---------------------------------------------------------------------------
# Tests: command_router response correctness
# ---------------------------------------------------------------------------

class TestCommandRouterResponses(unittest.TestCase):

    def test_app_not_found_gives_error_message(self):
        from command_router import _execute_intent
        intent = {"type": "app", "target": "nonexistent_xyz_app",
                  "response": "Opening nonexistent_xyz_app, Sir."}
        result = _execute_intent(intent)
        self.assertIsNotNone(result)
        reply_lower = result.reply.lower()
        self.assertIn("could not", reply_lower)

    def test_app_found_but_launch_fails_gives_error_message(self):
        from command_router import _execute_intent
        from tools.system import execute_open_app
        intent = {"type": "app", "target": r"C:\fake\app.exe",
                  "response": "Opening app, Sir."}
        with patch("tools.system.open_app") as mock_plan:
            mock_plan.return_value = {"ok": True, "target": r"C:\fake\app.exe"}
            with patch("tools.system.execute_open_app") as mock_exec:
                mock_exec.return_value = {"ok": False, "error": "Process crashed"}
                result = _execute_intent(intent)
        self.assertIsNotNone(result)
        reply_lower = result.reply.lower()
        self.assertIn("crashed", reply_lower)

    def test_app_success_gives_opened_message(self):
        from command_router import _execute_intent
        intent = {"type": "app", "target": "chrome",
                  "response": "Opening chrome, Sir."}
        with patch("tools.system.open_app") as mock_plan:
            mock_plan.return_value = {"ok": True, "target": r"C:\fake\chrome.exe"}
            with patch("tools.system.execute_open_app") as mock_exec:
                mock_exec.return_value = {"ok": True, "message": "Opened chrome.exe"}
                result = _execute_intent(intent)
        self.assertIsNotNone(result)
        self.assertIn("Opened", result.reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
