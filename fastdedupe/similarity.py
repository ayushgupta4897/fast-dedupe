"""
Similarity algorithms for fast-dedupe.

This module provides various string similarity algorithms for use in the
fast-dedupe package. It includes implementations of:
- Levenshtein ratio (from RapidFuzz)
- Jaro-Winkler distance (from RapidFuzz)
- Cosine similarity with character n-grams
- Jaccard similarity
- Soundex/phonetic matching
"""

import re
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import jellyfish
import numpy as np

# Third-party imports
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityAlgorithm(str, Enum):
    """Enumeration of available similarity algorithms."""

    LEVENSHTEIN = "levenshtein"
    JARO_WINKLER = "jaro_winkler"
    COSINE = "cosine"
    JACCARD = "jaccard"
    SOUNDEX = "soundex"


# Type for similarity functions
SimilarityFunc = Callable[[str, str], float]


def get_similarity_function(
    algorithm: Union[str, SimilarityAlgorithm],
) -> SimilarityFunc:
    """
    Get the similarity function for the specified algorithm.

    Args:
        algorithm: The similarity algorithm to use. Can be a string or a
            SimilarityAlgorithm enum value.

    Returns:
        A function that takes two strings and returns a similarity score (0-100).

    Raises:
        ValueError: If the algorithm is not supported.
    """
    if isinstance(algorithm, str):
        try:
            algorithm = SimilarityAlgorithm(algorithm.lower())
        except ValueError:
            valid_algorithms = ", ".join([a.value for a in SimilarityAlgorithm])
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported algorithms: {valid_algorithms}"
            )

    if algorithm == SimilarityAlgorithm.LEVENSHTEIN:
        return levenshtein_similarity
    elif algorithm == SimilarityAlgorithm.JARO_WINKLER:
        return jaro_winkler_similarity
    elif algorithm == SimilarityAlgorithm.COSINE:
        return cosine_ngram_similarity
    elif algorithm == SimilarityAlgorithm.JACCARD:
        return jaccard_similarity
    elif algorithm == SimilarityAlgorithm.SOUNDEX:
        return soundex_similarity
    else:
        valid_algorithms = ", ".join([a.value for a in SimilarityAlgorithm])
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. "
            f"Supported algorithms: {valid_algorithms}"
        )


def levenshtein_similarity(s1: str, s2: str, **kwargs: Any) -> float:
    """
    Calculate the Levenshtein ratio between two strings.

    This is a wrapper around RapidFuzz's fuzz.ratio function.

    Args:
        s1: First string
        s2: Second string
        **kwargs: Additional keyword arguments passed to fuzz.ratio

    Returns:
        Similarity score (0-100)
    """
    # Pass through any additional kwargs that RapidFuzz might provide
    return float(fuzz.ratio(s1, s2, **kwargs))


def jaro_winkler_similarity(s1: str, s2: str, **kwargs: Any) -> float:
    """
    Calculate the Jaro-Winkler similarity between two strings.

    This is particularly effective for short strings like names.

    Args:
        s1: First string
        s2: Second string
        **kwargs: Additional keyword arguments

    Returns:
        Similarity score (0-100)
    """
    # Extract processor if present to handle it correctly
    processor = kwargs.pop("processor", None)

    # If processor is provided, apply it to the strings
    if processor:
        s1 = processor(s1)
        s2 = processor(s2)

    # For now, use a custom implementation based on Levenshtein distance
    # that gives higher weight to matching prefixes

    # First, get the basic Levenshtein similarity
    base_similarity = levenshtein_similarity(s1, s2)

    # Give bonus for matching prefixes (up to first 4 characters)
    prefix_len = 0
    max_prefix = min(4, min(len(s1), len(s2)))

    for i in range(max_prefix):
        if s1[i].lower() == s2[i].lower():
            prefix_len += 1
        else:
            break

    # Apply prefix bonus (0.1 per matching character, max 0.4 or 40%)
    prefix_bonus = prefix_len * 0.1 * (100 - base_similarity)

    return min(100, base_similarity + prefix_bonus)


def _create_character_ngrams(text: str, n: int = 3) -> List[str]:
    """
    Create character n-grams from a string.

    Args:
        text: Input string
        n: Size of n-grams

    Returns:
        List of n-grams
    """
    # Pad the text with spaces to include edge n-grams
    padded_text = " " * (n - 1) + text + " " * (n - 1)
    return [padded_text[i : i + n] for i in range(len(padded_text) - n + 1)]


def cosine_ngram_similarity(s1: str, s2: str, n: int = 3, **kwargs: Any) -> float:
    """
    Calculate cosine similarity between two strings using character n-grams.

    This is effective for comparing documents or longer text.

    Args:
        s1: First string
        s2: Second string
        n: Size of n-grams
        **kwargs: Additional keyword arguments

    Returns:
        Similarity score (0-100)
    """
    # Extract processor if present to handle it correctly
    processor = kwargs.pop("processor", None)

    # If processor is provided, apply it to the strings
    if processor:
        s1 = processor(s1)
        s2 = processor(s2)

    # For very short strings, fall back to Levenshtein
    if len(s1) < n or len(s2) < n:
        return levenshtein_similarity(s1, s2, **kwargs)

    # Create a vectorizer for character n-grams
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(n, n))

    # Fit and transform the strings
    try:
        X = vectorizer.fit_transform([s1, s2])
        # Calculate cosine similarity
        similarity = cosine_similarity(X[0], X[1])[0][0]
        # Convert to a 0-100 scale
        return float(similarity * 100)
    except ValueError:
        # If vectorization fails, fall back to Levenshtein
        return float(levenshtein_similarity(s1, s2, **kwargs))


def jaccard_similarity(s1: str, s2: str, tokenize: bool = True, **kwargs: Any) -> float:
    """
    Calculate Jaccard similarity between two strings.

    Jaccard similarity is the size of the intersection divided by the size of the union
    of the sample sets.

    Args:
        s1: First string
        s2: Second string
        tokenize: If True, tokenize the strings into words. If False, use characters.
        **kwargs: Additional keyword arguments

    Returns:
        Similarity score (0-100)
    """
    # Extract processor if present to handle it correctly
    processor = kwargs.pop("processor", None)

    # If processor is provided, apply it to the strings
    if processor:
        s1 = processor(s1)
        s2 = processor(s2)

    if not s1 and not s2:
        return 100.0
    if not s1 or not s2:
        return 0.0

    if tokenize:
        # Tokenize into words
        set1 = set(re.findall(r"\w+", s1.lower()))
        set2 = set(re.findall(r"\w+", s2.lower()))
    else:
        # Use characters
        set1 = set(s1.lower())
        set2 = set(s2.lower())

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 0.0

    return (intersection / union) * 100


def soundex_similarity(s1: str, s2: str, **kwargs: Any) -> float:
    """
    Calculate similarity based on Soundex phonetic algorithm.

    This is useful for matching names that sound similar but are spelled differently.

    Args:
        s1: First string
        s2: Second string
        **kwargs: Additional keyword arguments

    Returns:
        Similarity score (0-100)
    """
    # Extract processor if present to handle it correctly
    processor = kwargs.pop("processor", None)

    # If processor is provided, apply it to the strings
    if processor:
        s1 = processor(s1)
        s2 = processor(s2)

    # Split into words
    words1 = re.findall(r"\w+", s1.lower())
    words2 = re.findall(r"\w+", s2.lower())

    if not words1 or not words2:
        return 0.0

    # Calculate Soundex codes for each word
    soundex1 = [jellyfish.soundex(word) for word in words1]
    soundex2 = [jellyfish.soundex(word) for word in words2]

    # Find the best match for each word in the first string
    total_score = 0
    max_score = max(len(words1), len(words2))

    # Compare each word in the first string to each word in the second string
    for code1 in soundex1:
        best_match = 0
        for code2 in soundex2:
            if code1 == code2:
                best_match = 1
                break
        total_score += best_match

    # Normalize to 0-100
    if max_score == 0:
        return 0.0

    return (total_score / max_score) * 100


def compare_all_algorithms(s1: str, s2: str) -> Dict[str, float]:
    """
    Compare two strings using all available similarity algorithms.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Dictionary mapping algorithm names to similarity scores
    """
    results = {}
    for algorithm in SimilarityAlgorithm:
        similarity_func = get_similarity_function(algorithm)
        results[algorithm.value] = similarity_func(s1, s2)
    return results


# Visualization functions
def visualize_similarity_matrix(
    strings: List[str],
    algorithm: Union[str, SimilarityAlgorithm] = SimilarityAlgorithm.LEVENSHTEIN,
    output_file: Optional[str] = None,
) -> Any:
    """
    Visualize the similarity matrix for a list of strings.

    Args:
        strings: List of strings to compare
        algorithm: Similarity algorithm to use
        output_file: Path to save the visualization (if None, displays the plot)

    Returns:
        The matplotlib figure object
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Get the similarity function
    similarity_func = get_similarity_function(algorithm)

    # Calculate the similarity matrix
    n = len(strings)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = similarity_func(strings[i], strings[j])

    # Create a custom colormap (white to blue)
    cmap = LinearSegmentedColormap.from_list(
        "white_to_blue", ["#ffffff", "#0343df"], N=100
    )

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity_matrix, cmap=cmap, vmin=0, vmax=100)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Similarity Score", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    # Shorten long strings for display
    max_label_length = 20
    labels = [
        s if len(s) <= max_label_length else s[: max_label_length - 3] + "..."
        for s in strings
    ]

    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)

    # Add title
    algorithm_name = algorithm if isinstance(algorithm, str) else algorithm.value
    ax.set_title(f"String Similarity Matrix ({algorithm_name.capitalize()})")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{similarity_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="black" if similarity_matrix[i, j] < 70 else "white",
            )

    fig.tight_layout()

    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


def visualize_algorithm_comparison(
    s1: str, s2: str, output_file: Optional[str] = None
) -> Any:
    """
    Visualize the comparison of all algorithms for two strings.

    Args:
        s1: First string
        s2: Second string
        output_file: Path to save the visualization (if None, displays the plot)

    Returns:
        The matplotlib figure object
    """
    import matplotlib.pyplot as plt

    # Compare using all algorithms
    results = compare_all_algorithms(s1, s2)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the results
    algorithms = list(results.keys())
    scores = list(results.values())

    # Sort by score
    sorted_indices = np.argsort(scores)
    algorithms = [algorithms[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]

    # Create horizontal bar chart
    bars = ax.barh(algorithms, scores, color="skyblue")

    # Add labels
    ax.set_xlabel("Similarity Score (0-100)")
    ax.set_title(f'Algorithm Comparison\n"{s1}" vs "{s2}"')

    # Add text labels
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{scores[i]:.1f}",
            va="center",
        )

    # Set x-axis limits
    ax.set_xlim(0, 105)

    fig.tight_layout()

    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig
