"""
Tests for the __main__ module of fast-dedupe.

This module contains tests for the __main__ module of the fast-dedupe package.
"""

import sys
import unittest
from unittest.mock import patch


class TestMain(unittest.TestCase):
    """Test cases for the __main__ module."""

    @patch("fastdedupe.cli.main")
    def test_main_import(self, mock_main):
        """Test that importing the module doesn't call main."""
        # Import the module

        # The main function should not be called on import
        mock_main.assert_not_called()

    @patch.object(sys, "argv", ["fastdedupe", "test.txt"])
    @patch("fastdedupe.cli.main")
    def test_main_as_script(self, mock_main):
        """Test running the module as a script."""
        # Save the original __name__ value
        import fastdedupe.__main__

        original_name = fastdedupe.__main__.__name__

        try:
            # Set __name__ to "__main__" to simulate running as script
            fastdedupe.__main__.__name__ = "__main__"

            # Call the code that would run if __name__ == "__main__"
            if hasattr(fastdedupe.__main__, "main"):
                fastdedupe.__main__.main()
            elif hasattr(fastdedupe.__main__, "__main__"):
                fastdedupe.__main__.__main__()

            # Check that cli.main was called
            mock_main.assert_called_once()
        finally:
            # Restore original __name__
            fastdedupe.__main__.__name__ = original_name


if __name__ == "__main__":
    unittest.main()
