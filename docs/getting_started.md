# Getting Started with fast-dedupe

This guide will help you get started with fast-dedupe, a fast and minimalist text deduplication library for Python with comprehensive string similarity comparison functionality.

## Installation

You can install fast-dedupe using pip:

```bash
pip install fast-dedupe
```

## Basic Usage

### Python API

The simplest way to use fast-dedupe is to import the `dedupe` function and pass it a list of strings:

```python
from fastdedupe import dedupe

# Sample data with duplicates
data = [
    "Apple iPhone 12", 
    "Apple iPhone12", 
    "Samsung Galaxy", 
    "Samsng Galaxy", 
    "MacBook Pro", 
    "Macbook-Pro"
]

# Deduplicate with default settings (threshold=85)
clean_data, duplicates = dedupe(data)

print("Deduplicated data:")
for item in clean_data:
    print(f"  - {item}")

print("\nDuplicates found:")
for key, dupes in duplicates.items():
    print(f"  - {key}: {dupes}")
```

Output:
```
Deduplicated data:
  - Apple iPhone 12
  - Samsung Galaxy
  - MacBook Pro

Duplicates found:
  - Apple iPhone 12: ['Apple iPhone12']
  - Samsung Galaxy: ['Samsng Galaxy']
  - MacBook Pro: ['Macbook-Pro']
```

### Command-line Interface

fast-dedupe also provides a command-line interface for deduplicating files:

```bash
# Basic usage
fastdedupe input.txt

# Save output to a file
fastdedupe input.txt -o deduplicated.txt

# Save duplicates mapping to a file
fastdedupe input.txt -o deduplicated.txt -d duplicates.json
```

## Choosing a Similarity Algorithm

fast-dedupe now supports multiple string similarity algorithms, each optimized for different use cases:

- **Levenshtein** (default): General-purpose edit distance, good for most text
- **Jaro-Winkler**: Optimized for short strings like names and personal information
- **Cosine**: Better for longer documents, using character n-gram comparison
- **Jaccard**: Good for set-based text analysis, comparing word overlap
- **Soundex**: Phonetic matching for names that sound similar but are spelled differently

```python
from fastdedupe import dedupe, SimilarityAlgorithm

# Using Jaro-Winkler for name matching
names = ["John Smith", "Jon Smith", "John Smyth"]
clean_data, duplicates = dedupe(names, similarity_algorithm=SimilarityAlgorithm.JARO_WINKLER)

# Or using a string to specify the algorithm
clean_data, duplicates = dedupe(names, similarity_algorithm="jaro_winkler")

# Using Soundex for phonetic matching
names = ["Catherine", "Katherine", "Kathryn"]
clean_data, duplicates = dedupe(names, similarity_algorithm=SimilarityAlgorithm.SOUNDEX)
```

In the command-line interface:

```bash
# Using Jaro-Winkler algorithm
fastdedupe names.txt -a jaro_winkler

# Using Cosine similarity for documents
fastdedupe documents.txt -a cosine
```

## Adjusting the Threshold

The `threshold` parameter controls how similar strings need to be to be considered duplicates. It ranges from 0 to 100:

- Lower values (e.g., 70) are more aggressive, matching more strings as duplicates
- Higher values (e.g., 95) are more conservative, only matching very similar strings

```python
# More aggressive deduplication
clean_data_70, duplicates_70 = dedupe(data, threshold=70)

# More conservative deduplication
clean_data_95, duplicates_95 = dedupe(data, threshold=95)
```

Note that different similarity algorithms may require different threshold values for optimal results:
- Levenshtein: 80-90 is typically a good range
- Jaro-Winkler: 85-95 works well for names
- Cosine: 70-80 is often good for documents
- Jaccard: 50-70 is typical for word-based comparison
- Soundex: 80-100 for phonetic matching

## Keeping First vs. Longest

By default, fast-dedupe keeps the first occurrence of a duplicate. You can change this to keep the longest string instead:

```python
# Keep the longest string instead of the first occurrence
clean_data, duplicates = dedupe(data, keep_first=False)
```

## Working with Different File Formats

The command-line interface supports different file formats:

```bash
# CSV files
fastdedupe data.csv -f csv --csv-column name

# JSON files
fastdedupe data.json -f json --json-key text
```

## Performance Considerations

- For small datasets (< 1000 items), fast-dedupe is very fast
- For larger datasets, consider adjusting the threshold to balance accuracy and performance
- If you need to deduplicate millions of strings, consider splitting the data into smaller batches
- Different algorithms have different performance characteristics:
  - Levenshtein and Jaro-Winkler are generally fastest
  - Cosine similarity with n-grams is efficient for longer texts
  - Soundex is very fast but less accurate for non-name data

## Visualizing Similarity Comparisons

fast-dedupe provides visualization tools to help you understand how different algorithms compare:

```python
from fastdedupe import (
    visualize_similarity_matrix,
    visualize_algorithm_comparison,
    SimilarityAlgorithm
)
import matplotlib.pyplot as plt

# Sample data
data = ["Apple iPhone 12", "Apple iPhone12", "Samsung Galaxy"]

# Visualize similarity matrix
fig1 = visualize_similarity_matrix(
    data,
    SimilarityAlgorithm.LEVENSHTEIN,
    "similarity_matrix.png"  # Optional: save to file
)

# Compare how different algorithms rate the similarity of two strings
s1 = "Catherine"
s2 = "Katherine"
fig2 = visualize_algorithm_comparison(
    s1,
    s2,
    "algorithm_comparison.png"  # Optional: save to file
)

# Display the visualizations
plt.show()
```

In the command-line interface:

```bash
# Generate a similarity matrix visualization
fastdedupe input.txt --visualize

# Compare all algorithms on the first two strings in the dataset
fastdedupe input.txt --compare-algorithms

# Specify output directory for visualizations
fastdedupe input.txt --visualize --viz-output ./visualizations/
```

## Direct Similarity Comparison

You can also use the similarity functions directly without deduplication:

```python
from fastdedupe import (
    levenshtein_similarity,
    jaro_winkler_similarity,
    cosine_ngram_similarity,
    jaccard_similarity,
    soundex_similarity,
    compare_all_algorithms
)

# Compare two strings using a specific algorithm
score = jaro_winkler_similarity("Catherine", "Katherine")
print(f"Jaro-Winkler similarity: {score}")

# Compare using all available algorithms
results = compare_all_algorithms("Catherine", "Katherine")
for algorithm, score in results.items():
    print(f"{algorithm}: {score}")
```

## Next Steps

- Check out the [examples](../examples/) directory for more usage examples, especially [similarity_comparison.py](../examples/similarity_comparison.py)
- Read the [API reference](../README.md#api-reference) for detailed documentation
- Explore the [benchmark results](../README.md#performance-benchmarks) to understand performance characteristics