"""
Tests for the similarity algorithms in fast-dedupe.

This module contains tests for the various similarity algorithms provided by
the fast-dedupe package.
"""

import os
import tempfile
import unittest

from fastdedupe.similarity import (
    SimilarityAlgorithm,
    compare_all_algorithms,
    cosine_ngram_similarity,
    get_similarity_function,
    jaccard_similarity,
    jaro_winkler_similarity,
    levenshtein_similarity,
    soundex_similarity,
    visualize_algorithm_comparison,
    visualize_similarity_matrix,
)


class TestSimilarityAlgorithms(unittest.TestCase):
    """Test cases for the similarity algorithms."""

    def test_levenshtein_similarity(self) -> None:
        """Test Levenshtein similarity."""
        # Identical strings should have 100% similarity
        self.assertEqual(levenshtein_similarity("apple", "apple"), 100)

        # Completely different strings should have low similarity
        self.assertLess(levenshtein_similarity("apple", "banana"), 50)

        # Strings with small differences should have high similarity
        self.assertGreater(levenshtein_similarity("apple", "aple"), 80)

        # Test with processor argument
        self.assertEqual(
            levenshtein_similarity("APPLE", "apple", processor=str.lower), 100
        )

        # Case sensitivity
        self.assertLess(levenshtein_similarity("Apple", "apple"), 100)

    def test_jaro_winkler_similarity(self) -> None:
        """Test Jaro-Winkler similarity."""
        # Identical strings should have 100% similarity
        self.assertEqual(jaro_winkler_similarity("john", "john"), 100)

        # Names with small differences should have high similarity
        self.assertGreater(jaro_winkler_similarity("john", "jon"), 90)

        # Jaro-Winkler gives higher scores to strings that match at the beginning
        score1 = jaro_winkler_similarity("martha", "marhta")
        score2 = jaro_winkler_similarity("dwayne", "duane")

        # Test with processor argument
        self.assertEqual(
            jaro_winkler_similarity("JOHN", "john", processor=str.lower), 100
        )
        self.assertGreater(score1, score2)

    def test_cosine_ngram_similarity(self) -> None:
        """Test cosine similarity with character n-grams."""
        # Identical strings should have 100% similarity
        self.assertEqual(cosine_ngram_similarity("document", "document"), 100)

        # Similar documents should have high similarity
        doc1 = "The quick brown fox jumps over the lazy dog"
        doc2 = "The quick brown fox jumps over the lazy cat"
        self.assertGreater(cosine_ngram_similarity(doc1, doc2), 90)

        # Different documents should have lower similarity
        doc3 = "Python is a programming language"
        self.assertLess(cosine_ngram_similarity(doc1, doc3), 50)

        # Test with very short strings (should fall back to Levenshtein)
        self.assertEqual(cosine_ngram_similarity("a", "a"), 100)
        self.assertEqual(cosine_ngram_similarity("a", "b"), 0)

        # Test with processor argument
        self.assertEqual(
            cosine_ngram_similarity("DOCUMENT", "document", processor=str.lower), 100
        )

    def test_jaccard_similarity(self) -> None:
        """Test Jaccard similarity."""
        # Identical strings should have 100% similarity
        self.assertEqual(jaccard_similarity("test", "test"), 100)

        # Test with word tokenization
        s1 = "the quick brown fox"
        s2 = "the brown quick fox"
        self.assertEqual(jaccard_similarity(s1, s2, tokenize=True), 100)

        # Test with character comparison
        self.assertLess(jaccard_similarity(s1, s2, tokenize=False), 100)

        # Test with different words
        s3 = "the quick brown fox"
        s4 = "the quick brown dog"
        self.assertLess(jaccard_similarity(s3, s4, tokenize=True), 100)
        self.assertGreater(jaccard_similarity(s3, s4, tokenize=True), 50)

        # Test with processor argument
        self.assertEqual(jaccard_similarity("TEST", "test", processor=str.lower), 100)

    def test_soundex_similarity(self) -> None:
        """Test Soundex similarity."""
        # Identical strings should have 100% similarity
        self.assertEqual(soundex_similarity("smith", "smith"), 100)

        # Names that sound similar should have high similarity
        self.assertGreater(soundex_similarity("smith", "smyth"), 50)
        self.assertGreater(soundex_similarity("catherine", "katherine"), 50)

        # Names that sound different should have low similarity
        self.assertLess(soundex_similarity("smith", "jones"), 50)

        # Test with multiple words
        self.assertEqual(soundex_similarity("john smith", "john smith"), 100)
        self.assertGreater(soundex_similarity("john smith", "jon smyth"), 50)

        # Test with processor argument
        self.assertEqual(soundex_similarity("SMITH", "smith", processor=str.lower), 100)

    def test_get_similarity_function(self) -> None:
        """Test getting similarity functions by name."""
        # Test with enum values
        self.assertEqual(
            get_similarity_function(SimilarityAlgorithm.LEVENSHTEIN),
            levenshtein_similarity,
        )
        self.assertEqual(
            get_similarity_function(SimilarityAlgorithm.JARO_WINKLER),
            jaro_winkler_similarity,
        )
        self.assertEqual(
            get_similarity_function(SimilarityAlgorithm.COSINE), cosine_ngram_similarity
        )
        self.assertEqual(
            get_similarity_function(SimilarityAlgorithm.JACCARD), jaccard_similarity
        )
        self.assertEqual(
            get_similarity_function(SimilarityAlgorithm.SOUNDEX), soundex_similarity
        )

        # Test with string values
        self.assertEqual(get_similarity_function("levenshtein"), levenshtein_similarity)
        self.assertEqual(
            get_similarity_function("jaro_winkler"), jaro_winkler_similarity
        )
        self.assertEqual(get_similarity_function("cosine"), cosine_ngram_similarity)
        self.assertEqual(get_similarity_function("jaccard"), jaccard_similarity)
        self.assertEqual(get_similarity_function("soundex"), soundex_similarity)

        # Test case insensitivity
        self.assertEqual(get_similarity_function("LEVENSHTEIN"), levenshtein_similarity)

        # Test invalid algorithm
        with self.assertRaises(ValueError):
            get_similarity_function("invalid_algorithm")

    def test_compare_all_algorithms(self) -> None:
        """Test comparing strings with all algorithms."""
        s1 = "catherine"
        s2 = "katherine"

        results = compare_all_algorithms(s1, s2)

        # Check that all algorithms are included
        self.assertEqual(len(results), len(SimilarityAlgorithm))

        for algorithm in SimilarityAlgorithm:
            self.assertIn(algorithm.value, results)
            self.assertIsInstance(results[algorithm.value], float)
            self.assertGreaterEqual(results[algorithm.value], 0)
            self.assertLessEqual(results[algorithm.value], 100)


class TestVisualization(unittest.TestCase):
    """Test cases for the visualization functions."""

    def test_visualize_similarity_matrix(self) -> None:
        """Test visualizing a similarity matrix."""
        strings = ["apple", "aple", "banana"]

        # Test without saving to file
        fig = visualize_similarity_matrix(strings, SimilarityAlgorithm.LEVENSHTEIN)
        self.assertIsNotNone(fig)

        # Test with saving to file
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "matrix.png")
            fig = visualize_similarity_matrix(
                strings, SimilarityAlgorithm.LEVENSHTEIN, output_file
            )
            self.assertTrue(os.path.exists(output_file))

    def test_visualize_algorithm_comparison(self) -> None:
        """Test visualizing algorithm comparison."""
        s1 = "catherine"
        s2 = "katherine"

        # Test without saving to file
        fig = visualize_algorithm_comparison(s1, s2)
        self.assertIsNotNone(fig)

        # Test with saving to file
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "comparison.png")
            fig = visualize_algorithm_comparison(s1, s2, output_file)
            self.assertTrue(os.path.exists(output_file))


if __name__ == "__main__":
    unittest.main()
