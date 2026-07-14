"""
Skill Loader for JARVIS.

Auto-discovers and loads all skills from the skills directory.
"""

import importlib
import logging
import pkgutil
from pathlib import Path

logger = logging.getLogger(__name__)


def load_all_skills():
    """
    Auto-load all skills in the skills directory.
    Skills are discovered by finding Python files with a '_skill' suffix.
    """
    skills_dir = Path(__file__).parent
    loaded_count = 0

    for importer, modname, ispkg in pkgutil.iter_modules([str(skills_dir)]):
        if modname.endswith('_skill') and modname != 'base_skill':
            try:
                # Import the skill module
                module = importlib.import_module(f'skills.{modname}')

                # Find the skill class (class ending with 'Skill')
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        attr_name.endswith('Skill') and
                        attr_name != 'BaseSkill' and
                        attr_name != 'SkillResult' and
                        attr_name != 'SkillRegistry'):
                        # Instantiate to trigger registration
                        try:
                            attr()
                            loaded_count += 1
                            logger.info(f"Loaded skill: {attr_name}")
                        except Exception as e:
                            logger.warning(f"Failed to instantiate {attr_name}: {e}")

            except Exception as e:
                logger.warning(f"Failed to load skill module {modname}: {e}")

    logger.info(f"Skills loaded: {loaded_count}")
    return loaded_count


# Auto-load skills when this module is imported
def _ensure_skills_loaded():
    """Ensure skills are loaded (idempotent)."""
    from skills.base_skill import SkillRegistry
    if not SkillRegistry.list_skill_names():
        load_all_skills()
