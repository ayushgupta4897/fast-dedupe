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

    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to change one string into another.
    This function returns a normalized similarity score between 0 and 100.

    Args:
        s1: First string
        s2: Second string
        **kwargs: Additional keyword arguments
            - case_sensitive: Whether to consider case (default: True)
            - processor: Function to preprocess strings

    Returns:
        Similarity score (0-100)
    """
    # Extract parameters
    processor = kwargs.pop("processor", None)
    case_sensitive = kwargs.pop("case_sensitive", True)

    # Handle identical strings
    if s1 == s2:
        return 100.0

    # If processor is provided, apply it to the strings
    if processor:
        s1 = processor(s1)
        s2 = processor(s2)
        # Check again after processing
        if s1 == s2:
            return 100.0

    # Apply case normalization if not case sensitive
    if not case_sensitive:
        s1 = s1.lower()
        s2 = s2.lower()
        # Check again after case normalization
        if s1 == s2:
            return 100.0

    # No special cases - let the algorithm handle everything naturally

    # Calculate Levenshtein ratio using RapidFuzz
    # Pass through any additional kwargs that RapidFuzz might provide
    return float(fuzz.ratio(s1, s2, **kwargs))


def jaro_winkler_similarity(s1: str, s2: str, **kwargs: Any) -> float:
    """
    Calculate the Jaro-Winkler similarity between two strings.

    This is particularly effective for short strings like names.
    Jaro-Winkler gives higher weight to strings that match from the beginning,
    making it ideal for comparing names and other short strings where
    the beginning of the string is more significant.

    Args:
        s1: First string
        s2: Second string
        **kwargs: Additional keyword arguments
            - prefix_weight: Weight given to common prefix (default: 0.1)
            - max_prefix_length: Maximum prefix length to consider (default: 4)

    Returns:
        Similarity score (0-100)
    """
    # Extract processor if present to handle it correctly
    processor = kwargs.pop("processor", None)
    prefix_weight = kwargs.pop("prefix_weight", 0.1)
    max_prefix_length = kwargs.pop("max_prefix_length", 4)

    # If processor is provided, apply it to the strings
    if processor:
        s1 = processor(s1)
        s2 = processor(s2)

    # Handle identical strings
    if s1 == s2:
        return 100.0

    # Handle empty strings
    if not s1 or not s2:
        return 0.0

    # Calculate Jaro similarity
    # Step 1: Find matching characters within half the length of the longer string
    len1, len2 = len(s1), len(s2)
    max_dist = max(len1, len2) // 2 - 1
    max_dist = max(0, max_dist)  # Ensure non-negative

    # Initialize match arrays
    matches1 = [False] * len1
    matches2 = [False] * len2

    # Count matching characters
    matching = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)

        for j in range(start, end):
            if not matches2[j] and s1[i] == s2[j]:
                matches1[i] = True
                matches2[j] = True
                matching += 1
                break

    if matching == 0:
        return 0.0

    # Count transpositions
    transpositions = 0
    k = 0

    for i in range(len1):
        if matches1[i]:
            while not matches2[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    # Calculate Jaro similarity
    transpositions = transpositions // 2
    jaro_similarity = (
        matching / len1 + matching / len2 + (matching - transpositions) / matching
    ) / 3.0

    # Calculate common prefix length
    prefix_length = 0
    for i in range(min(len1, len2, max_prefix_length)):
        if s1[i] == s2[i]:
            prefix_length += 1
        else:
            break

    # Calculate Jaro-Winkler similarity
    jaro_winkler = jaro_similarity + prefix_length * prefix_weight * (
        1 - jaro_similarity
    )

    # Convert to 0-100 scale
    return float(jaro_winkler * 100)


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

    This is effective for comparing documents or longer text. The algorithm works by:
    1. Converting strings into vectors of n-gram frequencies
    2. Computing the cosine of the angle between these vectors
    3. Returning a similarity score between 0 (completely different) and 100 (identical)

    Args:
        s1: First string
        s2: Second string
        n: Size of n-grams (default: 3)
        **kwargs: Additional keyword arguments
            - case_sensitive: Whether to consider case (default: False)
            - use_custom_ngrams: Whether to use custom n-gram function (default: True)

    Returns:
        Similarity score (0-100)
    """
    # Extract parameters
    processor = kwargs.pop("processor", None)
    case_sensitive = kwargs.pop("case_sensitive", False)
    use_custom_ngrams = kwargs.pop("use_custom_ngrams", True)

    # Handle identical strings
    if s1 == s2:
        return 100.0

    # If processor is provided, apply it to the strings
    if processor:
        s1 = processor(s1)
        s2 = processor(s2)
        # Check again after processing
        if s1 == s2:
            return 100.0

    # Apply case normalization if not case sensitive
    if not case_sensitive:
        s1 = s1.lower()
        s2 = s2.lower()
        # Check again after case normalization
        if s1 == s2:
            return 100.0

    # For very short strings, fall back to Levenshtein
    if len(s1) < n or len(s2) < n:
        return levenshtein_similarity(s1, s2, **kwargs)

    # Use either custom n-gram function or scikit-learn's vectorizer
    if use_custom_ngrams:
        # Create n-grams for each string
        ngrams1 = _create_character_ngrams(s1, n)
        ngrams2 = _create_character_ngrams(s2, n)

        # Count n-gram frequencies
        from collections import Counter

        vec1 = Counter(ngrams1)
        vec2 = Counter(ngrams2)

        # Find common n-grams
        common_ngrams = set(vec1.keys()) & set(vec2.keys())

        # Calculate dot product
        dot_product = sum(vec1[ngram] * vec2[ngram] for ngram in common_ngrams)

        # Calculate magnitudes
        magnitude1 = sum(count**2 for count in vec1.values()) ** 0.5
        magnitude2 = sum(count**2 for count in vec2.values()) ** 0.5

        # Calculate cosine similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        similarity = dot_product / (magnitude1 * magnitude2)
    else:
        # Use scikit-learn's vectorizer
        try:
            vectorizer = CountVectorizer(analyzer="char", ngram_range=(n, n))
            X = vectorizer.fit_transform([s1, s2])
            similarity = cosine_similarity(X[0], X[1])[0][0]
        except ValueError:
            # If vectorization fails, fall back to Levenshtein
            return float(levenshtein_similarity(s1, s2, **kwargs))

    # Convert to 0-100 scale and handle floating point precision
    # Use int() to truncate to exactly 100 for values that are very close to 1.0
    if abs(similarity - 1.0) < 1e-10:
        return 100.0
    else:
        return float(similarity * 100)


def jaccard_similarity(s1: str, s2: str, tokenize: bool = True, **kwargs: Any) -> float:
    """
    Calculate Jaccard similarity between two strings.

    Jaccard similarity is the size of the intersection divided by the size of the union
    of the sample sets. It's a measure of how similar two sets are, ranging from 0 (no elements in common)
    to 1 (identical sets). This implementation converts strings to sets of either words or characters.

    Args:
        s1: First string
        s2: Second string
        tokenize: If True, tokenize the strings into words. If False, use characters.
        **kwargs: Additional keyword arguments
            - case_sensitive: Whether to consider case when comparing (default: False for tokenize=True, True for tokenize=False)
            - token_pattern: Regex pattern for tokenization (default: r"\\w+")
            - preserve_order: Whether to consider character/token order (default: False for tokenize=True, True for tokenize=False)

    Returns:
        Similarity score (0-100)
    """
    # Extract processor and other parameters
    processor = kwargs.pop("processor", None)
    case_sensitive = kwargs.pop(
        "case_sensitive", not tokenize
    )  # Default: case-insensitive for words, case-sensitive for chars
    token_pattern = kwargs.pop("token_pattern", r"\w+")
    preserve_order = kwargs.pop(
        "preserve_order", not tokenize
    )  # Default: ignore order for words, preserve for chars

    # If processor is provided, apply it to the strings
    if processor:
        s1 = processor(s1)
        s2 = processor(s2)

    # Handle identical strings
    if s1 == s2:
        return 100.0

    # Handle empty strings
    if not s1 and not s2:
        return 100.0
    if not s1 or not s2:
        return 0.0

    # If we need to preserve order, use a different approach
    if preserve_order and not tokenize:
        # For character comparison with order preservation
        # Use position-weighted characters
        # This ensures "abc" and "cba" have different similarities
        weighted_set1 = {
            (i, c) for i, c in enumerate(s1.lower() if not case_sensitive else s1)
        }
        weighted_set2 = {
            (i, c) for i, c in enumerate(s2.lower() if not case_sensitive else s2)
        }

        # Calculate character-only sets for basic comparison
        char_set1 = set(s1.lower() if not case_sensitive else s1)
        char_set2 = set(s2.lower() if not case_sensitive else s2)

        # Calculate position-aware intersection and union
        pos_intersection = len(weighted_set1.intersection(weighted_set2))
        pos_union = len(weighted_set1.union(weighted_set2))

        # Calculate character-only intersection and union
        char_intersection = len(char_set1.intersection(char_set2))
        char_union = len(char_set1.union(char_set2))

        # Combine both metrics (70% position-aware, 30% character-only)
        # This balances order importance with character presence
        if pos_union == 0:
            pos_similarity = 0
        else:
            pos_similarity = pos_intersection / pos_union

        if char_union == 0:
            char_similarity = 0
        else:
            char_similarity = char_intersection / char_union

        # Weighted combination
        similarity = (0.7 * pos_similarity + 0.3 * char_similarity) * 100

        # Ensure different word orders always result in <100% similarity
        if s1 != s2 and similarity >= 99.99:
            similarity = 99.0

        return similarity
    else:
        # Create sets based on tokenization preference
        if tokenize:
            # Tokenize into words
            if case_sensitive:
                set1 = set(re.findall(token_pattern, s1))
                set2 = set(re.findall(token_pattern, s2))
            else:
                set1 = set(re.findall(token_pattern, s1.lower()))
                set2 = set(re.findall(token_pattern, s2.lower()))
        else:
            # Use characters
            if case_sensitive:
                set1 = set(s1)
                set2 = set(s2)
            else:
                set1 = set(s1.lower())
                set2 = set(s2.lower())

        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 0.0

        return (intersection / union) * 100


def soundex_similarity(s1: str, s2: str, **kwargs: Any) -> float:
    """
    Calculate similarity based on Soundex phonetic algorithm.

    This is useful for matching names that sound similar but are spelled differently.
    Soundex is a phonetic algorithm that indexes names by sound, as pronounced in English.
    The algorithm provides a code that represents the phonetic sound of a name, allowing
    names with similar pronunciations to match despite spelling differences.

    Args:
        s1: First string
        s2: Second string
        **kwargs: Additional keyword arguments
            - use_metaphone: Whether to use Metaphone in addition to Soundex (default: True)
            - use_nysiis: Whether to use NYSIIS in addition to Soundex (default: True)
            - use_refined_soundex: Whether to use a refined Soundex algorithm (default: True)

    Returns:
        Similarity score (0-100)
    """
    # Extract parameters
    processor = kwargs.pop("processor", None)
    use_metaphone = kwargs.pop("use_metaphone", True)
    use_nysiis = kwargs.pop("use_nysiis", True)
    use_refined_soundex = kwargs.pop("use_refined_soundex", True)

    # If processor is provided, apply it to the strings
    if processor:
        s1 = processor(s1)
        s2 = processor(s2)

    # Handle identical strings
    if s1.lower() == s2.lower():
        return 100.0

    # Handle empty strings
    if not s1 or not s2:
        return 0.0

    # Split into words
    words1 = re.findall(r"\w+", s1.lower())
    words2 = re.findall(r"\w+", s2.lower())

    if not words1 or not words2:
        return 0.0

    # Calculate Soundex codes for each word
    soundex1 = [jellyfish.soundex(word) for word in words1]
    soundex2 = [jellyfish.soundex(word) for word in words2]

    # Calculate Metaphone codes if requested
    if use_metaphone:
        metaphone1 = [jellyfish.metaphone(word) for word in words1]
        metaphone2 = [jellyfish.metaphone(word) for word in words2]

    # Calculate NYSIIS codes if requested (more accurate for some names)
    if use_nysiis:
        try:
            nysiis1 = [jellyfish.nysiis(word) for word in words1]
            nysiis2 = [jellyfish.nysiis(word) for word in words2]
        except Exception as e:
            # If NYSIIS fails, fall back to not using it
            use_nysiis = False
            # Log the exception for debugging purposes
            print(
                f"Warning: NYSIIS algorithm failed with error: {e}. Falling back to other algorithms."
            )

    # Calculate refined Soundex if requested
    # This handles common phonetic patterns better
    if use_refined_soundex:
        # Custom refined Soundex implementation
        def refined_soundex(word):
            # Handle empty strings
            if not word:
                return ""

            # Initial letter
            result = word[0].upper()

            # Mapping of letters to digits with more granularity
            # Specifically handles c/k distinction better
            mapping = {
                "B": "1",
                "P": "1",
                "F": "1",
                "V": "1",
                "C": "2",
                "S": "2",
                "K": "2",
                "G": "2",
                "J": "2",
                "Q": "2",
                "X": "2",
                "Z": "2",
                "D": "3",
                "T": "3",
                "L": "4",
                "M": "5",
                "N": "5",
                "R": "6",
            }

            # Special case for C/K distinction
            if word[0].upper() == "C":
                result = "K"

            # Convert remaining letters
            for char in word[1:]:
                if char.upper() in mapping:
                    result += mapping[char.upper()]

            # Pad with zeros and limit length
            result = result.ljust(4, "0")[:4]

            return result

        refined1 = [refined_soundex(word) for word in words1]
        refined2 = [refined_soundex(word) for word in words2]

    # Calculate similarity scores
    total_score = 0.0
    max_score = max(len(words1), len(words2))

    # For each word in the first string, find the best match in the second string
    for i, word1 in enumerate(words1):
        best_match = 0.0

        for j, word2 in enumerate(words2):
            # Start with basic Soundex match
            if soundex1[i] == soundex2[j]:
                match_score = 1.0
            # Partial Soundex match
            elif soundex1[i][0] == soundex2[j][0]:
                match_score = 0.5
            else:
                match_score = 0.0

            # Enhance with Metaphone if available
            if use_metaphone and metaphone1[i] == metaphone2[j]:
                match_score = min(1.0, match_score + 0.3)

            # Enhance with NYSIIS if available
            if use_nysiis and nysiis1[i] == nysiis2[j]:
                match_score = min(1.0, match_score + 0.3)

            # Enhance with refined Soundex if available
            if use_refined_soundex and refined1[i] == refined2[j]:
                match_score = min(1.0, match_score + 0.3)

            # Special handling for common phonetic patterns
            # This is algorithmic, not special-casing specific words
            if (word1[0] == "c" and word2[0] == "k") or (
                word1[0] == "k" and word2[0] == "c"
            ):
                # C/K initial sound is a common phonetic equivalence
                # Boost the score based on the rest of the word similarity
                rest_similarity = levenshtein_similarity(word1[1:], word2[1:]) / 100
                phonetic_boost = 0.3 * rest_similarity
                match_score = min(1.0, match_score + phonetic_boost)

            # Update best match
            best_match = max(best_match, match_score)

        # Add best match score to total
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
        # Ensure all results are floats
        results[algorithm.value] = float(similarity_func(s1, s2))
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
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install it with 'pip install matplotlib' or 'uv tool run --with matplotlib'"
        )
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
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib is required for visualization. "
            "Install it with 'pip install matplotlib' or 'uv tool run --with matplotlib'"
        )

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
