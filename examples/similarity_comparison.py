"""
Example demonstrating the use of different similarity algorithms in fast-dedupe.

This example shows how to use the various similarity algorithms provided by
fast-dedupe, and how to visualize the results.
"""

import os

import matplotlib.pyplot as plt

from fastdedupe import (
    SimilarityAlgorithm,
    compare_all_algorithms,
    dedupe,
    visualize_algorithm_comparison,
    visualize_similarity_matrix,
)


def main():
    # Sample datasets for different use cases

    # 1. General text with typos and variations
    print("\n=== General Text Example ===")
    general_text = [
        "Apple iPhone 12 Pro Max",
        "Apple iPhone12 Pro Max",
        "Apple iPhone 12Pro Max",
        "iPhone 12 Pro Max by Apple",
        "Samsung Galaxy S21 Ultra",
        "Samsung Galaxy S21Ultra",
        "Samsung GalaxyS21 Ultra",
    ]

    # Compare deduplication with different algorithms
    for algorithm in SimilarityAlgorithm:
        clean, dupes = dedupe(
            general_text, threshold=85, similarity_algorithm=algorithm
        )
        print(f"\n{algorithm.value.capitalize()} Algorithm:")
        print(f"  Deduplicated items: {len(clean)}")
        print(f"  Kept items: {clean}")
        print(f"  Duplicates: {dupes}")

    # 2. Names with spelling variations
    print("\n=== Name Matching Example ===")
    names = [
        "John Smith",
        "Jon Smith",
        "John Smyth",
        "Johnny Smith",
        "J. Smith",
        "Mary Johnson",
        "Mary Jonson",
        "M. Johnson",
    ]

    # Compare Levenshtein vs Jaro-Winkler for names
    print("\nLevenshtein Algorithm (default):")
    clean_lev, dupes_lev = dedupe(names, threshold=75)
    print(f"  Deduplicated items: {len(clean_lev)}")
    print(f"  Kept items: {clean_lev}")
    print(f"  Duplicates: {dupes_lev}")

    print("\nJaro-Winkler Algorithm (better for names):")
    clean_jw, dupes_jw = dedupe(
        names, threshold=85, similarity_algorithm=SimilarityAlgorithm.JARO_WINKLER
    )
    print(f"  Deduplicated items: {len(clean_jw)}")
    print(f"  Kept items: {clean_jw}")
    print(f"  Duplicates: {dupes_jw}")

    # 3. Phonetic matching example
    print("\n=== Phonetic Matching Example ===")
    phonetic_examples = [
        "Catherine",
        "Katherine",
        "Kathryn",
        "Catharine",
        "Katarina",
        "Robert",
        "Rupert",
        "Roberto",
    ]

    print("\nLevenshtein Algorithm:")
    clean_lev, dupes_lev = dedupe(phonetic_examples, threshold=75)
    print(f"  Deduplicated items: {len(clean_lev)}")
    print(f"  Kept items: {clean_lev}")
    print(f"  Duplicates: {dupes_lev}")

    print("\nSoundex Algorithm (better for phonetic matching):")
    clean_soundex, dupes_soundex = dedupe(
        phonetic_examples,
        threshold=85,
        similarity_algorithm=SimilarityAlgorithm.SOUNDEX,
    )
    print(f"  Deduplicated items: {len(clean_soundex)}")
    print(f"  Kept items: {clean_soundex}")
    print(f"  Duplicates: {dupes_soundex}")

    # 4. Document similarity example
    print("\n=== Document Similarity Example ===")
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown fox jumps over a lazy dog",
        "The fast brown fox leaps over the sleepy dog",
        "The quick brown fox jumps over the lazy cat",
        "Python is a programming language with clean syntax",
        "Python programming has a clean and readable syntax",
    ]

    print("\nLevenshtein Algorithm:")
    clean_lev, dupes_lev = dedupe(documents, threshold=75)
    print(f"  Deduplicated items: {len(clean_lev)}")
    print(f"  Kept items: {clean_lev}")
    print(f"  Duplicates: {dupes_lev}")

    print("\nCosine Algorithm (better for documents):")
    clean_cosine, dupes_cosine = dedupe(
        documents, threshold=75, similarity_algorithm=SimilarityAlgorithm.COSINE
    )
    print(f"  Deduplicated items: {len(clean_cosine)}")
    print(f"  Kept items: {clean_cosine}")
    print(f"  Duplicates: {dupes_cosine}")

    # 5. Direct similarity comparison
    print("\n=== Direct Similarity Comparison ===")
    string1 = "The quick brown fox jumps over the lazy dog"
    string2 = "The fast brown fox leaps over the sleepy dog"

    # Compare using all algorithms
    results = compare_all_algorithms(string1, string2)
    print(f"Comparing:\n  1: '{string1}'\n  2: '{string2}'")
    for algorithm, score in results.items():
        print(f"  {algorithm.capitalize()}: {score:.2f}")

    # 6. Visualization examples
    print("\n=== Visualization Examples ===")

    # Create output directory
    os.makedirs("visualization_examples", exist_ok=True)

    # Visualize similarity matrix
    print("Generating similarity matrix visualization...")
    visualize_similarity_matrix(
        names,
        SimilarityAlgorithm.JARO_WINKLER,
        "visualization_examples/similarity_matrix.png",
    )

    # Visualize algorithm comparison
    print("Generating algorithm comparison visualization...")
    visualize_algorithm_comparison(
        string1, string2, "visualization_examples/algorithm_comparison.png"
    )

    print("\nVisualizations saved to 'visualization_examples' directory.")

    # Show the visualizations
    plt.show()


if __name__ == "__main__":
    main()
