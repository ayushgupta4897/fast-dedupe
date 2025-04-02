"""
Basic usage example for fast-dedupe.

This script demonstrates how to use the fast-dedupe library for
deduplicating a list of strings.
"""

import os
import sys
from fastdedupe import dedupe

# Add the parent directory to the path so we can import fastdedupe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    """Demonstrate basic usage of fast-dedupe."""
    # Example dataset
    data = [
        "Apple iPhone 12",
        "Apple iPhone12",
        "Samsung Galaxy",
        "Samsng Galaxy",
        "MacBook Pro",
        "Macbook-Pro",
    ]

    print("Original data:")
    for item in data:
        print(f"  - {item}")
    print()

    # Deduplicate with default settings
    clean_data, duplicates = dedupe(data)

    print("Deduplicated data (threshold=85):")
    for item in clean_data:
        print(f"  - {item}")
    print()

    print("Duplicates found:")
    for key, dupes in duplicates.items():
        print(f"  - {key}: {dupes}")
    print()

    # Deduplicate with higher threshold
    clean_data_strict, duplicates_strict = dedupe(data, threshold=95)

    print("Deduplicated data (threshold=95, stricter matching):")
    for item in clean_data_strict:
        print(f"  - {item}")
    print()

    print("Duplicates found (stricter matching):")
    for key, dupes in duplicates_strict.items():
        print(f"  - {key}: {dupes}")
    print()

    # Real-world example
    customer_names = [
        "Flipkart India",
        "Flipkart-India",
        "Amazon",
        "Amaz0n",
        "Google LLC",
        "Google LLC",
        "Meta Inc.",
        "Meta Inc",
    ]

    print("Customer data:")
    for item in customer_names:
        print(f"  - {item}")
    print()

    # Deduplicate with custom threshold
    clean_customers, duplicate_customers = dedupe(customer_names, threshold=80)

    print("Deduplicated customer data (threshold=80):")
    for item in clean_customers:
        print(f"  - {item}")
    print()

    print("Duplicate customers found:")
    for key, dupes in duplicate_customers.items():
        print(f"  - {key}: {dupes}")


if __name__ == "__main__":
    main()
