"""
Benchmark script for fast-dedupe.

This script benchmarks the performance of fast-dedupe against
different dataset sizes and similarity thresholds.
"""

import sys
import os
import time
import random
import string
from typing import List

# Add the parent directory to the path so we can import fastdedupe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastdedupe import dedupe


def generate_dataset(size: int, duplicate_ratio: float = 0.3, 
                     variation_level: int = 2) -> List[str]:
    """
    Generate a dataset of strings with controlled duplication.
    
    Args:
        size: Number of strings to generate
        duplicate_ratio: Ratio of duplicates to unique strings
        variation_level: Level of variation in duplicates (0-5)
            0: Exact duplicates
            1: Case variations
            2: Minor typos
            3: Word order changes
            4: Word additions/removals
            5: Major differences
            
    Returns:
        List of strings
    """
    # Generate base unique strings
    unique_count = int(size * (1 - duplicate_ratio))
    unique_strings = []
    
    # Generate product-like strings
    brands = ["Apple", "Samsung", "Google", "Sony", "Microsoft", "Dell", "HP", 
              "Lenovo", "Asus", "Acer", "LG", "Huawei", "Xiaomi", "OnePlus"]
    products = ["Phone", "Laptop", "Tablet", "TV", "Watch", "Headphones", 
                "Speaker", "Camera", "Monitor", "Keyboard", "Mouse", "Router"]
    models = ["Pro", "Max", "Ultra", "Lite", "Plus", "S", "X", "Z", "A", "E"]
    
    for _ in range(unique_count):
        brand = random.choice(brands)
        product = random.choice(products)
        model = random.choice(models)
        year = random.randint(2018, 2023)
        unique_strings.append(f"{brand} {product} {model} {year}")
    
    # Generate duplicates with variations
    duplicates = []
    duplicate_count = size - unique_count
    
    for _ in range(duplicate_count):
        original = random.choice(unique_strings)
        
        if variation_level == 0:
            # Exact duplicate
            duplicate = original
        elif variation_level == 1:
            # Case variation
            duplicate = ''.join(c.lower() if random.random() < 0.3 else c 
                               for c in original)
        elif variation_level == 2:
            # Minor typos
            chars = list(original)
            # Introduce 1-2 typos
            for _ in range(random.randint(1, 2)):
                if len(chars) > 3:
                    pos = random.randint(0, len(chars) - 1)
                    if random.random() < 0.5 and chars[pos] != ' ':
                        # Replace with a similar character
                        chars[pos] = random.choice(string.ascii_letters)
                    elif random.random() < 0.3:
                        # Delete a character
                        chars.pop(pos)
                    else:
                        # Insert a character
                        chars.insert(pos, random.choice(string.ascii_letters))
            duplicate = ''.join(chars)
        elif variation_level == 3:
            # Word order changes
            words = original.split()
            if len(words) > 2:
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
            duplicate = ' '.join(words)
        elif variation_level == 4:
            # Word additions/removals
            words = original.split()
            if random.random() < 0.5 and len(words) > 3:
                # Remove a word
                words.pop(random.randint(0, len(words) - 1))
            else:
                # Add a word
                extra_words = ["New", "Special", "Limited", "Edition", "Premium", 
                              "Basic", "Standard", "Advanced", "Gen", "Version"]
                words.insert(random.randint(0, len(words)), 
                            random.choice(extra_words))
            duplicate = ' '.join(words)
        else:  # variation_level == 5
            # Major differences
            words = original.split()
            # Replace 30-50% of words
            for i in range(len(words)):
                if random.random() < 0.4:
                    if i == 0:
                        words[i] = random.choice(brands)
                    elif i == 1:
                        words[i] = random.choice(products)
                    elif i == 2:
                        words[i] = random.choice(models)
                    else:
                        words[i] = str(random.randint(2018, 2023))
            duplicate = ' '.join(words)
        
        duplicates.append(duplicate)
    
    # Combine and shuffle
    dataset = unique_strings + duplicates
    random.shuffle(dataset)
    
    return dataset


def benchmark(dataset_sizes: List[int], thresholds: List[int], 
              variation_levels: List[int]) -> None:
    """
    Benchmark fast-dedupe with different dataset sizes and thresholds.
    
    Args:
        dataset_sizes: List of dataset sizes to benchmark
        thresholds: List of similarity thresholds to benchmark
        variation_levels: List of variation levels to benchmark
    """
    print("=== fast-dedupe Benchmark ===")
    print(f"{'Size':<10} {'Threshold':<10} {'Variation':<10} {'Time (s)':<10} "
          f"{'Unique':<10} {'Duplicates':<10}")
    print("-" * 60)
    
    for size in dataset_sizes:
        for variation in variation_levels:
            # Generate dataset
            dataset = generate_dataset(size, duplicate_ratio=0.3, 
                                      variation_level=variation)
            
            for threshold in thresholds:
                # Benchmark
                start_time = time.time()
                clean_data, duplicates = dedupe(dataset, threshold=threshold)
                end_time = time.time()
                
                # Calculate statistics
                execution_time = end_time - start_time
                unique_count = len(clean_data)
                duplicate_count = sum(len(dupes) for dupes in duplicates.values())
                
                print(f"{size:<10} {threshold:<10} {variation:<10} "
                      f"{execution_time:.4f}    {unique_count:<10} {duplicate_count:<10}")


def main():
    """Run the benchmark."""
    # Define benchmark parameters
    dataset_sizes = [100, 500, 1000, 5000]
    thresholds = [70, 85, 95]
    variation_levels = [1, 2, 3]
    
    # Run benchmark
    benchmark(dataset_sizes, thresholds, variation_levels)


if __name__ == "__main__":
    main() 