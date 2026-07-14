"""
File Management Skill for JARVIS.

Provides file and folder operations as a self-contained skill.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from skills.base_skill import BaseSkill, SkillResult


class FileManagementSkill(BaseSkill):
    """Skill for file and folder operations."""

    name = "file_management"
    description = "Manage files and folders - create, read, write, move, delete"
    category = "files"
    version = "1.0.0"

    async def execute(self, **kwargs) -> SkillResult:
        """Execute a file management operation."""
        operation = kwargs.get("operation", "")

        if operation == "list":
            return await self._list_files(kwargs.get("path", "."))
        elif operation == "read":
            return await self._read_file(kwargs.get("path"))
        elif operation == "write":
            return await self._write_file(kwargs.get("path"), kwargs.get("content", ""))
        elif operation == "create":
            return await self._create_file(kwargs.get("path"), kwargs.get("content", ""))
        elif operation == "delete":
            return await self._delete_path(kwargs.get("path"))
        elif operation == "move":
            return await self._move_path(kwargs.get("source"), kwargs.get("destination"))
        elif operation == "copy":
            return await self._copy_path(kwargs.get("source"), kwargs.get("destination"))
        elif operation == "search":
            return await self._search_files(kwargs.get("pattern"), kwargs.get("path", "."))
        else:
            return SkillResult(
                success=False,
                message=f"Unknown operation: {operation}"
            )

    async def validate_params(self, **kwargs) -> bool:
        """Validate file operation parameters."""
        operation = kwargs.get("operation", "")
        valid_ops = ["list", "read", "write", "create", "delete", "move", "copy", "search"]
        return operation in valid_ops and kwargs.get("path") is not None

    async def _list_files(self, path: str) -> SkillResult:
        """List files in a directory."""
        try:
            p = Path(path)
            if not p.exists():
                return SkillResult(success=False, message=f"Path not found: {path}")

            if not p.is_dir():
                return SkillResult(success=False, message=f"Not a directory: {path}")

            files = []
            for item in sorted(p.iterdir()):
                if item.is_dir():
                    files.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    files.append(f"📄 {item.name} ({self._format_size(size)})")

            return SkillResult(
                success=True,
                message=f"Contents of {path}:",
                data={"files": files, "count": len(files)}
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Error listing files: {e}")

    async def _read_file(self, path: str) -> SkillResult:
        """Read file contents."""
        try:
            p = Path(path)
            if not p.exists():
                return SkillResult(success=False, message=f"File not found: {path}")

            # Limit file size to 100KB for safety
            if p.stat().st_size > 100_000:
                return SkillResult(success=False, message="File too large to read (max 100KB)")

            content = p.read_text(encoding='utf-8', errors='replace')
            return SkillResult(
                success=True,
                message=f"Read {p.name} ({len(content)} chars)",
                data={"content": content, "path": str(p)}
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Error reading file: {e}")

    async def _write_file(self, path: str, content: str) -> SkillResult:
        """Write content to file."""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding='utf-8')
            return SkillResult(
                success=True,
                message=f"Written to {path} ({len(content)} chars)"
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Error writing file: {e}")

    async def _create_file(self, path: str, content: str = "") -> SkillResult:
        """Create a new file (fails if exists)."""
        try:
            p = Path(path)
            if p.exists():
                return SkillResult(success=False, message=f"File already exists: {path}")

            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding='utf-8')
            return SkillResult(
                success=True,
                message=f"Created file: {path}"
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Error creating file: {e}")

    async def _delete_path(self, path: str) -> SkillResult:
        """Delete file or directory."""
        try:
            p = Path(path)
            if not p.exists():
                return SkillResult(success=False, message=f"Path not found: {path}")

            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

            return SkillResult(success=True, message=f"Deleted: {path}")
        except Exception as e:
            return SkillResult(success=False, message=f"Error deleting: {e}")

    async def _move_path(self, source: str, destination: str) -> SkillResult:
        """Move/rename file or directory."""
        try:
            shutil.move(source, destination)
            return SkillResult(
                success=True,
                message=f"Moved {source} → {destination}"
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Error moving: {e}")

    async def _copy_path(self, source: str, destination: str) -> SkillResult:
        """Copy file or directory."""
        try:
            if os.path.isdir(source):
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            return SkillResult(
                success=True,
                message=f"Copied {source} → {destination}"
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Error copying: {e}")

    async def _search_files(self, pattern: str, path: str) -> SkillResult:
        """Search for files matching a pattern."""
        try:
            p = Path(path)
            if not p.exists():
                return SkillResult(success=False, message=f"Path not found: {path}")

            matches = list(p.rglob(pattern))
            if not matches:
                return SkillResult(success=True, message=f"No files found matching '{pattern}'")

            file_list = [str(m) for m in matches[:50]]  # Limit to 50 results
            return SkillResult(
                success=True,
                message=f"Found {len(matches)} files matching '{pattern}'",
                data={"files": file_list, "count": len(matches)}
            )
        except Exception as e:
            return SkillResult(success=False, message=f"Error searching: {e}")

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size for display."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"


# Instantiate to auto-register
_skill_instance = FileManagementSkill()
