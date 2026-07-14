"""
Base Skill Interface for JARVIS Skills.

All JARVIS capabilities inherit from this abstract class.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    """Result of a skill execution."""
    success: bool
    message: str = ""
    data: Optional[Any] = None
    actions: Optional[List[Dict]] = None  # Follow-up actions for JARVIS


class BaseSkill(ABC):
    """
    Abstract base class for all JARVIS skills.

    Every capability (File Management, Browser Control, etc.) becomes a skill.
    Skills are self-contained, re-usable, and auto-register themselves.
    """

    # Skill metadata - override these
    name: str = ""
    description: str = ""
    category: str = ""  # 'system', 'files', 'browser', 'media', etc.
    version: str = "1.0.0"
    author: str = "JARVIS"

    # Enabled/disabled state
    _enabled: bool = True

    def __init__(self):
        """Initialize the skill."""
        if not self.name:
            self.name = self.__class__.__name__

        # Auto-register on instantiation
        SkillRegistry.register(self)

        logger.info(f"Skill registered: {self.name} v{self.version}")

    @abstractmethod
    async def execute(self, **kwargs) -> SkillResult:
        """
        Execute the skill with given parameters.

        Args:
            **kwargs: Skill-specific parameters

        Returns:
            SkillResult: Execution result with success/failure and message
        """
        pass

    async def validate_params(self, **kwargs) -> bool:
        """
        Validate input parameters before execution.

        Override this in subclasses for custom validation.

        Args:
            **kwargs: Parameters to validate

        Returns:
            bool: True if valid, False otherwise
        """
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        """Get what this skill can do."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "enabled": self._enabled
        }

    def enable(self):
        """Enable this skill."""
        self._enabled = True
        logger.info(f"Skill enabled: {self.name}")

    def disable(self):
        """Disable this skill."""
        self._enabled = False
        logger.info(f"Skill disabled: {self.name}")

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if this skill is enabled."""
        skill = SkillRegistry.get_skill_by_name(cls.__name__)
        return skill is not None and skill._enabled


class SkillRegistry:
    """Central registry for all JARVIS skills."""

    _skills: Dict[str, BaseSkill] = {}
    _skills_by_category: Dict[str, List[BaseSkill]] = {}

    @classmethod
    def register(cls, skill: BaseSkill):
        """Register a skill."""
        cls._skills[skill.name] = skill

        # Add to category index
        category = skill.category
        if category not in cls._skills_by_category:
            cls._skills_by_category[category] = []
        cls._skills_by_category[category].append(skill)

        logger.debug(f"Registered skill '{skill.name}' in category '{category}'")

    @classmethod
    def get_skill_by_name(cls, name: str) -> Optional[BaseSkill]:
        """Get a skill by name."""
        return cls._skills.get(name)

    @classmethod
    def get_skills_by_category(cls, category: str) -> List[BaseSkill]:
        """Get all skills in a category."""
        return cls._skills_by_category.get(category, [])

    @classmethod
    def list_all_skills(cls) -> Dict[str, Any]:
        """List all registered skills."""
        return {
            "skills": {name: skill.get_capabilities() for name, skill in cls._skills.items()},
            "categories": {cat: [s.name for s in skills] for cat, skills in cls._skills_by_category.items()}
        }

    @classmethod
    def list_skill_names(cls) -> List[str]:
        """Get list of all skill names."""
        return list(cls._skills.keys())


# Convenience functions for skill usage
async def execute_skill(skill_name: str, **kwargs) -> SkillResult:
    """Execute a skill by name."""
    skill = SkillRegistry.get_skill_by_name(skill_name)

    if not skill:
        return SkillResult(
            success=False,
            message=f"Skill '{skill_name}' not found"
        )

    if not skill._enabled:
        return SkillResult(
            success=False,
            message=f"Skill '{skill_name}' is disabled"
        )

    try:
        if not await skill.validate_params(**kwargs):
            return SkillResult(
                success=False,
                message="Invalid parameters for skill"
            )

        return await skill.execute(**kwargs)

    except Exception as e:
        logger.error(f"Skill '{skill_name}' execution failed: {e}")
        return SkillResult(
            success=False,
            message=f"Error executing skill: {str(e)}"
        )