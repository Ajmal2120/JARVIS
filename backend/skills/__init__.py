"""
Skills package for JARVIS 2.5 Operating Intelligence.

All JARVIS capabilities are self-contained skills that register themselves.
Import this module to auto-load all skills.
"""

# Import base classes
from skills.base_skill import BaseSkill, SkillResult, SkillRegistry, execute_skill

# Import all skill modules to trigger auto-registration
from skills import file_skill
from skills import browser_skill
from skills import media_skill
from skills import system_skill

__all__ = [
    'BaseSkill',
    'SkillResult',
    'SkillRegistry',
    'execute_skill',
    'file_skill',
    'browser_skill',
    'media_skill',
    'system_skill',
]
