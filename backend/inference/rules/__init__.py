"""
Built-in inference rules.

To add a new rule:
    1. Create a new file in this package (e.g. writing.py).
    2. Implement a class that subclasses InferenceRule.
    3. Add the class name to InferenceEngine.DEFAULT_RULES.
"""
from __future__ import annotations

from ..base import InferenceRule
from .software_development import SoftwareDevelopmentRule

__all__ = [
    "InferenceRule",
    "SoftwareDevelopmentRule",
]
