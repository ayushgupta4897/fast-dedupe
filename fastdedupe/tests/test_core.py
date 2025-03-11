"""
Tests for the core functionality of fast-dedupe.

This module contains comprehensive tests for the dedupe function,
ensuring it works correctly with various inputs and edge cases.
"""

import unittest
from fastdedupe import dedupe
from fastdedupe.core import _dedupe_exact


class TestDedupe(unittest.TestCase):
    """Test cases for the dedupe function."""

    def test_empty_list(self):
        """Test deduplication of an empty list."""
        clean, dupes = dedupe([])
        self.assertEqual(clean, [])
        self.assertEqual(dupes, {})

    def test_single_item(self):
        """Test deduplication of a list with a single item."""
        data = ["Apple"]
        clean, dupes = dedupe(data)
        self.assertEqual(clean, ["Apple"])
        self.assertEqual(dupes, {})

    def test_no_duplicates(self):
        """Test deduplication of a list with no duplicates."""
        data = ["Apple", "Banana", "Cherry"]
        clean, dupes = dedupe(data)
        self.assertEqual(clean, data)
        self.assertEqual(dupes, {})

    def test_exact_duplicates(self):
        """Test deduplication of a list with exact duplicates."""
        data = ["Apple", "Apple", "Banana", "Cherry", "Cherry"]
        clean, dupes = dedupe(data)
        self.assertEqual(clean, ["Apple", "Banana", "Cherry"])
        self.assertEqual(dupes, {"Apple": ["Apple"], "Cherry": ["Cherry"]})

    def test_fuzzy_duplicates(self):
        """Test deduplication of a list with fuzzy duplicates."""
        data = ["Apple iPhone 12", "Apple iPhone12", "Samsung Galaxy", "Samsng Galaxy"]
        clean, dupes = dedupe(data)
        self.assertEqual(clean, ["Apple iPhone 12", "Samsung Galaxy"])
        self.assertEqual(dupes, {
            "Apple iPhone 12": ["Apple iPhone12"],
            "Samsung Galaxy": ["Samsng Galaxy"]
        })

    def test_case_sensitivity(self):
        """Test deduplication with case differences."""
        data = ["Apple", "apple", "APPLE", "Banana"]
        clean, dupes = dedupe(data, threshold=85)
        # RapidFuzz is case-sensitive by default, so these should be considered similar
        # but not exact matches
        self.assertEqual(len(clean), 2)
        self.assertTrue("Apple" in clean or "apple" in clean or "APPLE" in clean)
        self.assertTrue("Banana" in clean)

    def test_threshold_100(self):
        """Test deduplication with threshold=100 (exact matches only)."""
        data = ["Apple", "apple", "Apple iPhone", "Apple iPhone12"]
        clean, dupes = dedupe(data, threshold=100)
        # With threshold=100, only exact duplicates should be removed
        self.assertEqual(len(clean), 4)  # All items are different with exact matching
        self.assertTrue("Apple" in clean)
        self.assertTrue("apple" in clean)
        self.assertTrue("Apple iPhone" in clean)
        self.assertTrue("Apple iPhone12" in clean)
        self.assertEqual(dupes, {})  # No duplicates with exact matching

    def test_threshold_0(self):
        """Test deduplication with threshold=0 (everything matches)."""
        data = ["Apple", "Banana", "Cherry"]
        clean, dupes = dedupe(data, threshold=0)
        # With threshold=0, everything should match the first item
        self.assertEqual(clean, ["Apple"])
        self.assertEqual(dupes, {"Apple": ["Banana", "Cherry"]})

    def test_keep_first_true(self):
        """Test deduplication with keep_first=True."""
        data = ["short", "very long string", "another"]
        clean, dupes = dedupe(data, threshold=50, keep_first=True)
        # With keep_first=True, the first occurrence should be kept
        self.assertTrue("short" in clean)

    def test_keep_first_false(self):
        """Test deduplication with keep_first=False."""
        data = ["short", "very long string", "another"]
        clean, dupes = dedupe(data, threshold=50, keep_first=False)
        # With keep_first=False, the longest string should be kept
        self.assertTrue("very long string" in clean)

    def test_invalid_threshold_type(self):
        """Test deduplication with invalid threshold type."""
        data = ["Apple", "Banana"]
        with self.assertRaises(ValueError):
            dedupe(data, threshold="85")

    def test_invalid_threshold_value(self):
        """Test deduplication with invalid threshold value."""
        data = ["Apple", "Banana"]
        with self.assertRaises(ValueError):
            dedupe(data, threshold=101)
        with self.assertRaises(ValueError):
            dedupe(data, threshold=-1)

    def test_invalid_keep_first_type(self):
        """Test deduplication with invalid keep_first type."""
        data = ["Apple", "Banana"]
        with self.assertRaises(ValueError):
            dedupe(data, keep_first="True")

    def test_real_world_example(self):
        """Test deduplication with a real-world example."""
        data = [
            "Flipkart India", 
            "Flipkart-India",
            "Amazon", 
            "Amaz0n", 
            "Google LLC", 
            "Google LLC",
            "Meta Inc.", 
            "Meta Inc"
        ]
        clean, dupes = dedupe(data, threshold=80)
        self.assertEqual(len(clean), 4)
        self.assertTrue("Flipkart India" in clean)
        self.assertTrue("Amazon" in clean)
        self.assertTrue("Google LLC" in clean)
        self.assertTrue("Meta Inc." in clean or "Meta Inc" in clean)
        self.assertTrue("Flipkart-India" in dupes.get("Flipkart India", []))
        self.assertTrue("Amaz0n" in dupes.get("Amazon", []))
        self.assertTrue("Google LLC" in dupes.get("Google LLC", []))
        self.assertTrue("Meta Inc" in dupes.get("Meta Inc.", []) or "Meta Inc." in dupes.get("Meta Inc", []))

    def test_dedupe_exact_keep_first_false(self):
        """Test _dedupe_exact with keep_first=False."""
        data = ["short", "very long string", "short"]
        clean, dupes = _dedupe_exact(data, keep_first=False)
        # With keep_first=False, the function should still work but ignore the parameter
        self.assertEqual(len(clean), 2)
        self.assertTrue("short" in clean)
        self.assertTrue("very long string" in clean)
        self.assertEqual(dupes, {"short": ["short"]})


if __name__ == "__main__":
    unittest.main() 