"""
JARVIS 3.0 — SoftwareDevelopmentRule.

Infers "the user is doing software development" from generic signals:

    1. The foreground process is a known development tool
       (VS Code, Cursor, Visual Studio, PyCharm, IntelliJ, Terminal).
       These are hardcoded because the rule's *whole purpose* is to
       recognise development tools.  This is not Intent-level
       classification — it's a surface signal the rule is designed to
       detect, analogous to a motion sensor detecting a specific brand
       of door.

    2. Supporting signals raise confidence:
        - High interaction level (typing, clicking)
        - Stable context (long uninterrupted session)
        - Long focus duration on the same application

The rule returns None when confidence falls below the configurable
minimum (default 0.35).  This prevents low-evidence inferences from
cluttering the hypothesis list.

All scoring is deterministic.  No LLM.  No Memory.  No event
publication.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Set

from context import Context, ContextInterpretation
from ..base import InferenceRule
from ..models import Hypothesis

logger = logging.getLogger("jarvis.inference.rules.software_dev")

# Development-related process names (lowercase for case-insensitive matching).
# Kept in this rule because matching process names IS the rule's job —
# the engine knows nothing about processes.
_DEV_PROCESSES: Set[str] = {
    # Code editors
    "code",          # VS Code / Cursor
    "cursor",        # Cursor (separate binary)
    "devenv",        # Visual Studio
    "pycharm64",     # PyCharm
    "pycharm",       # PyCharm (32-bit)
    "idea64",        # IntelliJ IDEA
    "idea",          # IntelliJ (32-bit)
    "studio64",      # Android Studio
    "studio",        # Android Studio (32-bit)
    "rider64",       # Rider
    "rider",         # Rider (32-bit)
    "atom",          # Atom
    "sublime_text",  # Sublime Text
    "notepad++",     # Notepad++
    "notepadplusplus",
    "vim",           # GVim / Vim
    "nvim",          # Neovim GUI
    "emacs",         # Emacs
    "zed",           # Zed
    "helix",         # Helix
    # Terminals / shells
    "powershell",    # PowerShell
    "pwsh",          # PowerShell Core
    "cmd",           # cmd.exe
    "conhost",       # Windows Terminal (conhost.exe)
    "windows_terminal",
    "wt",            # Windows Terminal short name
    "terminal",      # Generic
    "bash",          # Git Bash, WSL bash
    "zsh",           # WSL zsh
    "tmux",          # tmux (WSL)
    "wsl",           # WSL host
    "wslhost",
    # Git / version control GUIs
    "gitkraken",     # GitKraken
    "sourcetree",    # SourceTree
    "tortoisegit",   # TortoiseGit
    # Database tools
    "dbeaver",       # DBeaver
    "datagrip64",    # DataGrip
    "datagrip",
    # API / network tools
    "postman",       # Postman
    "insomnia",      # Insomnia
    # Build / DevOps
    "docker desktop",
    "docker",
    "kubectl",
    # IDEs / editors on WSL
    "code",          # VS Code remote-WSL
    # Windows Subsystem
    "wsl",
    "wslhost",
}

# Windows processes that are always terminals (conhost.exe is the
# console host for both cmd and PowerShell).
_ALWAYS_DEV = {"conhost", "windows_terminal", "wt", "terminal"}


class SoftwareDevelopmentRule(InferenceRule):
    """Infer software development activity.

    Scoring model (max raw score = 6.0, normalized to 0.0–1.0):

        +2.0  foreground process matches a known dev tool
        +1.0  interaction level is HIGH
        +1.0  context stability is STEADY (≥ 60s)
        +1.0  current focus duration ≥ 60s
        +1.0  application switch frequency < 5 switches/min
             (low churn indicates sustained focus)
    """

    # Minimum confidence to produce a hypothesis (0.0–1.0)
    MIN_CONFIDENCE = 0.35

    # Maximum raw score across all evidence sources.
    _MAX_RAW_SCORE = 6.0

    # Evidence keys → point values
    _PROCESS_MATCH_PTS = 2.0
    _HIGH_INTERACTION_PTS = 1.0
    _STEADY_CONTEXT_PTS = 1.0
    _LONG_FOCUS_PTS = 1.0
    _LOW_SWITCH_PTS = 1.0

    @property
    def name(self) -> str:
        return "software_development"

    def evaluate(
        self, context: Context, interpretation: ContextInterpretation
    ) -> Hypothesis | None:
        evidence: List[str] = []
        raw_score = 0.0

        # ── Evidence 1: foreground process matches dev tool ─────────────
        if self._is_dev_process(context.active_application):
            raw_score += self._PROCESS_MATCH_PTS
            evidence.append(
                f"Foreground process '{context.active_application}' is a "
                "known development tool"
            )

        # ── Evidence 2: high interaction level ───────────────────────────
        if interpretation.interaction_level.name == "HIGH":
            raw_score += self._HIGH_INTERACTION_PTS
            evidence.append(
                "High interaction level (typing/clicking detected)"
            )

        # ── Evidence 3: stable context ───────────────────────────────────
        if interpretation.context_stability.name == "STEADY":
            raw_score += self._STEADY_CONTEXT_PTS
            evidence.append(
                f"Stable context ({interpretation.context_stability_s:.0f}s "
                "without window/app change)"
            )

        # ── Evidence 4: long focus duration ──────────────────────────────
        if interpretation.current_focus_duration_s >= 60.0:
            raw_score += self._LONG_FOCUS_PTS
            evidence.append(
                f"Focused on same application for "
                f"{interpretation.current_focus_duration_s:.0f}s"
            )

        # ── Evidence 5: low application-switch churn ─────────────────────
        if interpretation.application_switch_frequency < 5.0:
            raw_score += self._LOW_SWITCH_PTS
            evidence.append(
                f"Low app-switch rate "
                f"({interpretation.application_switch_frequency:.1f}/min)"
            )

        # ── Normalize and threshold ──────────────────────────────────────
        confidence = min(raw_score / self._MAX_RAW_SCORE, 1.0)

        if confidence < self.MIN_CONFIDENCE:
            logger.debug(
                "software_dev: confidence %.2f below threshold %.2f — no hypothesis",
                confidence, self.MIN_CONFIDENCE,
            )
            return None

        logger.info(
            "software_dev: hypothesis confidence=%.2f, evidence=%d items",
            confidence, len(evidence),
        )
        return Hypothesis(
            id=self.name,
            confidence=round(confidence, 4),
            evidence=evidence,
        )

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_process(name: str) -> str:
        """Lower-case and strip path/extension for matching."""
        import os
        base = os.path.basename(name.lower())
        base = re.sub(r"\.(exe|bat|cmd|com)$", "", base)
        return base.strip()

    @classmethod
    def _is_dev_process(cls, process_name: str) -> bool:
        if not process_name:
            return False
        normalised = cls._normalise_process(process_name)
        return normalised in _DEV_PROCESSES or normalised in _ALWAYS_DEV
