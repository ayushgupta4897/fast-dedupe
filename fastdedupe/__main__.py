"""
Main entry point for the fast-dedupe package.

This module allows the package to be run as a script using:
python -m fastdedupe
"""

from .cli import main

# This is the entry point when the module is run directly
if __name__ == "__main__":
    main()
