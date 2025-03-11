"""
fast-dedupe: Fast, Minimalist Text Deduplication Library for Python.

This module provides a simple, intuitive, ready-to-use deduplication wrapper
around RapidFuzz, minimizing setup effort while providing great speed and
accuracy out-of-the-box.
"""

__version__ = "0.1.0"

from .core import dedupe

__all__ = ["dedupe"] 