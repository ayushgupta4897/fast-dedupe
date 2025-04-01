"""
fast-dedupe: Fast, Minimalist Text Deduplication Library for Python.

This module provides a simple, intuitive, ready-to-use deduplication wrapper
with multiple string similarity algorithms, minimizing setup effort while
providing great speed and accuracy out-of-the-box.
"""

__version__ = "0.2.0"

from .core import dedupe
from .similarity import (
    SimilarityAlgorithm,
    levenshtein_similarity,
    jaro_winkler_similarity,
    cosine_ngram_similarity,
    jaccard_similarity,
    soundex_similarity,
    compare_all_algorithms,
    visualize_similarity_matrix,
    visualize_algorithm_comparison
)

__all__ = [
    "dedupe",
    "SimilarityAlgorithm",
    "levenshtein_similarity",
    "jaro_winkler_similarity",
    "cosine_ngram_similarity",
    "jaccard_similarity",
    "soundex_similarity",
    "compare_all_algorithms",
    "visualize_similarity_matrix",
    "visualize_algorithm_comparison"
]