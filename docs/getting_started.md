# Getting Started with fast-dedupe

This guide will help you get started with fast-dedupe, a fast and minimalist text deduplication library for Python.

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

## Next Steps

- Check out the [examples](../examples/) directory for more usage examples
- Read the [API reference](../README.md#api-reference) for detailed documentation
- Explore the [benchmark results](../README.md#performance-benchmarks) to understand performance characteristics 