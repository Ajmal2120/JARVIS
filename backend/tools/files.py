"""
File-system tools: find files, read content, search inside files.

Search scope: the user picked "whole C: drive" but we skip the well-known
junk that hangs Windows (System Volume Information, Recycle Bin, WinSxS).
"""
from __future__ import annotations

import fnmatch
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("jarvis.files")

# Paths to never recurse into.
SKIP_DIRS = {
    "System Volume Information",
    "$Recycle.Bin",
    "Config.Msi",
    "Recovery",
    "PerfLogs",
    "WindowsApps",
    "WinSxS",            # extremely deep, full of symlinks
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "AppData",
}

MAX_RESULTS_DEFAULT = 25
MAX_FILE_BYTES_READ = 200_000   # 200 KB; bigger files get truncated
MAX_GREP_LINES = 50


# ---------------------------------------------------------------------------
# find_files
# ---------------------------------------------------------------------------
def find_files(
    name_glob: str,
    roots: Optional[list[str]] = None,
    max_results: int = MAX_RESULTS_DEFAULT,
    extensions: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Walk the filesystem looking for files matching `name_glob`.

    Args:
        name_glob: shell-style pattern ("*.pdf", "resume*", "*.py")
        roots: starting directories; defaults to the user's known folders
        max_results: hard cap, walk stops once reached
        extensions: extra filter like [".py", ".txt"]

    Returns {"matches": [{path, size, modified}], "truncated": bool}.
    """
    if not roots:
        roots = _default_roots()
    matches: list[dict[str, Any]] = []
    truncated = False

    for root in roots:
        if len(matches) >= max_results:
            truncated = True
            break
        root_path = Path(root)
        if not root_path.exists():
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(root_path, topdown=True):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                               and not d.startswith(".")]
                try:
                    if os.path.islink(dirpath):
                        dirnames[:] = []
                        continue
                except OSError:
                    pass
                for fn in filenames:
                    if len(matches) >= max_results:
                        truncated = True
                        break
                    if not fnmatch.fnmatch(fn, name_glob):
                        continue
                    if extensions and not any(
                        fn.lower().endswith(e.lower()) for e in extensions
                    ):
                        continue
                    full = os.path.join(dirpath, fn)
                    try:
                        st = os.stat(full)
                        matches.append({
                            "path": full,
                            "size": st.st_size,
                            "modified": st.st_mtime,
                        })
                    except OSError:
                        continue
                if len(matches) >= max_results:
                    break
        except (PermissionError, OSError) as e:
            logger.debug("walk skipped %s: %s", root, e)
            continue

    return {"matches": matches, "truncated": truncated, "count": len(matches)}


def _default_roots() -> list[str]:
    """Sensible starting points: user profile + root of C:\\."""
    roots = [str(Path.home())]
    if os.name == "nt":
        roots += ["C:\\"]
    else:
        roots += ["/"]
    seen = set()
    return [r for r in roots if not (r in seen or seen.add(r))]


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------
def read_file(path: str, max_bytes: int = MAX_FILE_BYTES_READ) -> dict[str, Any]:
    """Read a text file (or first chunk of a binary)."""
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": f"file not found: {path}"}
    if not p.is_file():
        return {"ok": False, "error": f"not a regular file: {path}"}
    size = p.stat().st_size
    truncated = False
    try:
        if size > max_bytes:
            with p.open("rb") as f:
                content = f.read(max_bytes)
            truncated = True
            try:
                text = content.decode("utf-8", errors="replace")
            except Exception:
                text = f"[binary: {size} bytes, first {max_bytes} shown]"
        else:
            text = p.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return {"ok": False, "error": "permission denied"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "path": str(p),
        "size": size,
        "truncated": truncated,
        "text": text,
    }


# ---------------------------------------------------------------------------
# grep_in_files
# ---------------------------------------------------------------------------
def grep_in_files(
    pattern: str,
    roots: Optional[list[str]] = None,
    extensions: Optional[list[str]] = None,
    max_results: int = MAX_GREP_LINES,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """
    Walk files and return matching lines.

    `pattern` may be a literal substring or a regex; we'll try regex first,
    fall back to literal substring if it fails to compile.
    """
    if not roots:
        roots = _default_roots()

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        rx = re.compile(pattern, flags)
    except re.error:
        rx = re.compile(re.escape(pattern), flags)

    matches: list[dict[str, Any]] = []
    truncated = False

    for root in roots:
        if len(matches) >= max_results:
            truncated = True
            break
        rp = Path(root)
        if not rp.exists():
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(rp, topdown=True):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                               and not d.startswith(".")]
                if extensions:
                    filenames = [f for f in filenames
                                 if any(f.lower().endswith(e.lower()) for e in extensions)]
                for fn in filenames:
                    if len(matches) >= max_results:
                        truncated = True
                        break
                    full = os.path.join(dirpath, fn)
                    try:
                        with open(full, "r", encoding="utf-8", errors="ignore") as f:
                            for lineno, line in enumerate(f, 1):
                                if rx.search(line):
                                    matches.append({
                                        "path": full,
                                        "line": lineno,
                                        "snippet": line.rstrip("\n")[:300],
                                    })
                                    if len(matches) >= max_results:
                                        break
                    except (PermissionError, OSError):
                        continue
                if len(matches) >= max_results:
                    break
        except (PermissionError, OSError):
            continue

    return {"matches": matches, "truncated": truncated, "count": len(matches)}
