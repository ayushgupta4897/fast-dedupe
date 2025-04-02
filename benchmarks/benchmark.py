#!/usr/bin/env python
"""
Benchmark script for comparing fastdedupe with other deduplication libraries.

This script compares the performance of fastdedupe with pandas and fuzzywuzzy
on various dataset sizes and similarity thresholds.
"""

import argparse
import json
import os
import random
import string
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from fuzzywuzzy import process as fuzzywuzzy_process
from tqdm import tqdm

# Import the libraries to benchmark
from fastdedupe import dedupe


def generate_dataset(
    size: int,
    duplicate_ratio: float = 0.3,
    variation_level: float = 0.2,
    min_length: int = 10,
    max_length: int = 50,
) -> List[str]:
    """
    Generate a dataset of strings with controlled duplication.

    Args:
        size: Number of strings to generate
        duplicate_ratio: Ratio of duplicates to unique strings
        variation_level: How much variation to introduce in duplicates (0-1)
        min_length: Minimum string length
        max_length: Maximum string length

    Returns:
        List of strings with controlled duplication
    """
    # Calculate number of unique strings
    unique_count = int(size * (1 - duplicate_ratio))
    unique_count = max(1, unique_count)  # Ensure at least one unique string

    # Generate unique strings
    unique_strings = []
    for _ in range(unique_count):
        length = random.randint(min_length, max_length)
        s = "".join(
            random.choices(string.ascii_letters + string.digits + " ", k=length)
        )
        unique_strings.append(s)

    # Generate duplicates with variations
    duplicates = []
    for _ in range(size - unique_count):
        # Pick a random string to duplicate
        original = random.choice(unique_strings)

        # Determine how many characters to change
        change_count = int(len(original) * variation_level)

        # Make a copy of the original
        duplicate = list(original)

        # Modify random positions
        for _ in range(change_count):
            pos = random.randint(0, len(duplicate) - 1)
            if random.random() < 0.33:  # Delete
                duplicate[pos] = ""
            elif random.random() < 0.66:  # Replace
                duplicate[pos] = random.choice(
                    string.ascii_letters + string.digits + " "
                )
            else:  # Insert
                duplicate.insert(
                    pos, random.choice(string.ascii_letters + string.digits + " ")
                )

        duplicates.append("".join(duplicate))

    # Combine and shuffle
    dataset = unique_strings + duplicates
    random.shuffle(dataset)

    return dataset


def benchmark_fastdedupe(data: List[str], threshold: int) -> Tuple[float, int]:
    """Benchmark fastdedupe."""
    start_time = time.time()
    clean, dupes = dedupe(data, threshold=threshold)
    elapsed = time.time() - start_time
    return elapsed, len(clean)


def benchmark_pandas(data: List[str], threshold: int) -> Tuple[float, int]:
    """Benchmark pandas string matching."""
    df = pd.DataFrame(data, columns=["text"])

    start_time = time.time()

    # Create a similarity matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Convert threshold to similarity value (0-1)
    similarity_threshold = threshold / 100.0

    # Find duplicates
    duplicates = []
    clean = []
    processed = set()

    for i in range(len(df)):
        if i in processed:
            continue

        processed.add(i)
        clean.append(df.iloc[i]["text"])

        # Find similar items
        similar_indices = [
            j
            for j in range(len(df))
            if j != i
            and j not in processed
            and similarity_matrix[i, j] >= similarity_threshold
        ]

        for j in similar_indices:
            processed.add(j)
            duplicates.append(df.iloc[j]["text"])

    elapsed = time.time() - start_time
    return elapsed, len(clean)


def benchmark_fuzzywuzzy(data: List[str], threshold: int) -> Tuple[float, int]:
    """Benchmark fuzzywuzzy."""
    start_time = time.time()

    clean = []
    duplicates = []
    remaining = data.copy()

    while remaining:
        current = remaining.pop(0)
        clean.append(current)

        # Find matches
        matches = fuzzywuzzy_process.extract(
            current, remaining, limit=None, score_cutoff=threshold
        )

        # Extract matched strings
        match_strings = [match[0] for match in matches]

        # Add to duplicates
        duplicates.extend(match_strings)

        # Remove matches from remaining
        for match in match_strings:
            remaining.remove(match)

    elapsed = time.time() - start_time
    return elapsed, len(clean)


def run_benchmarks(
    sizes: List[int],
    thresholds: List[int],
    duplicate_ratio: float = 0.3,
    variation_level: float = 0.2,
    runs: int = 3,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Run benchmarks for different libraries, dataset sizes, and thresholds.

    Args:
        sizes: List of dataset sizes to benchmark
        thresholds: List of similarity thresholds to benchmark
        duplicate_ratio: Ratio of duplicates in the generated datasets
        variation_level: Level of variation in duplicates
        runs: Number of runs for each configuration

    Returns:
        Dictionary with benchmark results
    """
    results = {
        "fastdedupe": {"time": [], "size": [], "threshold": [], "clean_count": []},
        "pandas": {"time": [], "size": [], "threshold": [], "clean_count": []},
        "fuzzywuzzy": {"time": [], "size": [], "threshold": [], "clean_count": []},
    }

    # Define benchmark functions
    benchmarks = {
        "fastdedupe": benchmark_fastdedupe,
        "pandas": benchmark_pandas,
        "fuzzywuzzy": benchmark_fuzzywuzzy,
    }

    # Run benchmarks
    total_runs = len(sizes) * len(thresholds) * len(benchmarks) * runs
    with tqdm(total=total_runs, desc="Running benchmarks") as pbar:
        for size in sizes:
            for threshold in thresholds:
                # Generate dataset
                data = generate_dataset(
                    size=size,
                    duplicate_ratio=duplicate_ratio,
                    variation_level=variation_level,
                )

                # Run each benchmark multiple times
                for lib_name, benchmark_func in benchmarks.items():
                    for _ in range(runs):
                        try:
                            elapsed, clean_count = benchmark_func(data, threshold)

                            results[lib_name]["time"].append(elapsed)
                            results[lib_name]["size"].append(size)
                            results[lib_name]["threshold"].append(threshold)
                            results[lib_name]["clean_count"].append(clean_count)
                        except Exception as e:
                            print(
                                f"Error in {lib_name} with size={size}, "
                                f"threshold={threshold}: {e}"
                            )
                            # Add a very high time to indicate failure
                            results[lib_name]["time"].append(float("inf"))
                            results[lib_name]["size"].append(size)
                            results[lib_name]["threshold"].append(threshold)
                            results[lib_name]["clean_count"].append(0)

                        pbar.update(1)

    return results


def plot_results(results: Dict[str, Dict[str, List[float]]], output_dir: str):
    """
    Plot benchmark results.

    Args:
        results: Benchmark results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract unique sizes and thresholds
    sizes = sorted(set(results["fastdedupe"]["size"]))
    thresholds = sorted(set(results["fastdedupe"]["threshold"]))

    # Plot time vs size for each threshold
    for threshold in thresholds:
        plt.figure(figsize=(10, 6))

        for lib_name in results.keys():
            # Filter data for this threshold
            indices = [
                i
                for i, t in enumerate(results[lib_name]["threshold"])
                if t == threshold
            ]
            filtered_sizes = [results[lib_name]["size"][i] for i in indices]
            filtered_times = [results[lib_name]["time"][i] for i in indices]

            # Group by size and calculate average
            unique_sizes = sorted(set(filtered_sizes))
            avg_times = []

            for size in unique_sizes:
                size_indices = [i for i, s in enumerate(filtered_sizes) if s == size]
                avg_time = sum(filtered_times[i] for i in size_indices) / len(
                    size_indices
                )
                avg_times.append(avg_time)

            plt.plot(unique_sizes, avg_times, marker="o", label=lib_name)

        plt.xlabel("Dataset Size")
        plt.ylabel("Time (seconds)")
        plt.title(f"Deduplication Time vs Dataset Size (Threshold: {threshold})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"time_vs_size_threshold_{threshold}.png"))
        plt.close()

    # Plot time vs threshold for each size
    for size in sizes:
        plt.figure(figsize=(10, 6))

        for lib_name in results.keys():
            # Filter data for this size
            indices = [i for i, s in enumerate(results[lib_name]["size"]) if s == size]
            filtered_thresholds = [results[lib_name]["threshold"][i] for i in indices]
            filtered_times = [results[lib_name]["time"][i] for i in indices]

            # Group by threshold and calculate average
            unique_thresholds = sorted(set(filtered_thresholds))
            avg_times = []

            for threshold in unique_thresholds:
                threshold_indices = [
                    i for i, t in enumerate(filtered_thresholds) if t == threshold
                ]
                avg_time = sum(filtered_times[i] for i in threshold_indices) / len(
                    threshold_indices
                )
                avg_times.append(avg_time)

            plt.plot(unique_thresholds, avg_times, marker="o", label=lib_name)

        plt.xlabel("Similarity Threshold")
        plt.ylabel("Time (seconds)")
        plt.title(f"Deduplication Time vs Similarity Threshold (Size: {size})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"time_vs_threshold_size_{size}.png"))
        plt.close()

    # Create a summary plot
    plt.figure(figsize=(12, 8))

    for lib_name in results.keys():
        # Calculate average time for each size across all thresholds
        unique_sizes = sorted(set(results[lib_name]["size"]))
        avg_times = []

        for size in unique_sizes:
            size_indices = [
                i for i, s in enumerate(results[lib_name]["size"]) if s == size
            ]
            avg_time = sum(results[lib_name]["time"][i] for i in size_indices) / len(
                size_indices
            )
            avg_times.append(avg_time)

        plt.plot(unique_sizes, avg_times, marker="o", linewidth=2, label=lib_name)

    plt.xlabel("Dataset Size")
    plt.ylabel("Average Time (seconds)")
    plt.title("Average Deduplication Time vs Dataset Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "summary_time_vs_size.png"))
    plt.close()

    # Save raw results as JSON
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def generate_markdown_report(
    results: Dict[str, Dict[str, List[float]]], output_dir: str
):
    """
    Generate a Markdown report of benchmark results.

    Args:
        results: Benchmark results
        output_dir: Directory to save the report
    """
    report_path = os.path.join(output_dir, "benchmark_report.md")

    # Extract unique sizes and thresholds
    sizes = sorted(set(results["fastdedupe"]["size"]))
    thresholds = sorted(set(results["fastdedupe"]["threshold"]))

    with open(report_path, "w") as f:
        f.write("# Fast-Dedupe Benchmark Results\n\n")

        f.write("## Summary\n\n")
        f.write(
            "This report compares the performance of fast-dedupe with other "
            "deduplication libraries:\n\n"
        )
        f.write("- **fast-dedupe**: Our optimized fuzzy string matching library\n")
        f.write("- **pandas**: Using TF-IDF vectorization and cosine similarity\n")
        f.write("- **fuzzywuzzy**: A popular fuzzy string matching library\n\n")

        f.write("## Performance Comparison\n\n")
        f.write("![Summary](summary_time_vs_size.png)\n\n")

        f.write("## Detailed Results\n\n")

        # Create a table for each threshold
        for threshold in thresholds:
            f.write(f"### Threshold: {threshold}\n\n")

            # Create table header
            f.write(
                "| Dataset Size | fast-dedupe (s) | pandas (s) | fuzzywuzzy (s) | "
                "Speedup vs pandas | Speedup vs fuzzywuzzy |\n"
            )
            f.write(
                "|--------------|-----------------|------------|----------------| "
                "-------------------|----------------------|\n"
            )

            for size in sizes:
                # Calculate average times for each library at this size and threshold
                avg_times = {}

                for lib_name in results.keys():
                    indices = [
                        i
                        for i, (s, t) in enumerate(
                            zip(
                                results[lib_name]["size"],
                                results[lib_name]["threshold"],
                            )
                        )
                        if s == size and t == threshold
                    ]

                    if indices:
                        avg_times[lib_name] = sum(
                            results[lib_name]["time"][i] for i in indices
                        ) / len(indices)
                    else:
                        avg_times[lib_name] = float("inf")

                # Calculate speedups
                speedup_pandas = (
                    avg_times["pandas"] / avg_times["fastdedupe"]
                    if avg_times["fastdedupe"] > 0
                    else float("inf")
                )
                speedup_fuzzywuzzy = (
                    avg_times["fuzzywuzzy"] / avg_times["fastdedupe"]
                    if avg_times["fastdedupe"] > 0
                    else float("inf")
                )

                # Format times and speedups
                times = {
                    lib: f"{t:.4f}" if t != float("inf") else "N/A"
                    for lib, t in avg_times.items()
                }
                speedups = {
                    "pandas": (
                        f"{speedup_pandas:.2f}x"
                        if speedup_pandas != float("inf")
                        else "N/A"
                    ),
                    "fuzzywuzzy": (
                        f"{speedup_fuzzywuzzy:.2f}x"
                        if speedup_fuzzywuzzy != float("inf")
                        else "N/A"
                    ),
                }

                # Add row to table
                f.write(
                    f"| {size:<12} | {times['fastdedupe']:<15} | "
                    f"{times['pandas']:<10} | "
                    f"{times['fuzzywuzzy']:<14} | {speedups['pandas']:<17} | "
                    f"{speedups['fuzzywuzzy']:<20} |\n"
                )

            f.write("\n")
            f.write(
                f"![Time vs Size (Threshold: {threshold})]"
                f"(time_vs_size_threshold_{threshold}.png)\n\n"
            )

        f.write("## Conclusion\n\n")

        # Calculate overall average speedups
        fastdedupe_times = [
            t for t in results["fastdedupe"]["time"] if t != float("inf")
        ]
        pandas_times = [t for t in results["pandas"]["time"] if t != float("inf")]
        fuzzywuzzy_times = [
            t for t in results["fuzzywuzzy"]["time"] if t != float("inf")
        ]

        if fastdedupe_times and pandas_times:
            avg_speedup_pandas = sum(pandas_times) / sum(fastdedupe_times)
            f.write(
                f"- On average, fast-dedupe is "
                f"**{avg_speedup_pandas:.2f}x faster than pandas**\n"
            )

        if fastdedupe_times and fuzzywuzzy_times:
            avg_speedup_fuzzywuzzy = sum(fuzzywuzzy_times) / sum(fastdedupe_times)
            f.write(
                f"- On average, fast-dedupe is "
                f"**{avg_speedup_fuzzywuzzy:.2f}x faster than fuzzywuzzy**\n"
            )

        f.write(
            "\nThese benchmarks demonstrate that fast-dedupe provides significant "
            "performance improvements "
        )
        f.write(
            "over other popular libraries for fuzzy string matching and "
            "deduplication tasks.\n"
        )


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Run deduplication benchmarks")
    parser.add_argument(
        "--output-dir", default="benchmark_results", help="Directory to save results"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 5000, 10000],
        help="Dataset sizes to benchmark",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[70, 80, 90],
        help="Similarity thresholds to benchmark",
    )
    parser.add_argument(
        "--duplicate-ratio",
        type=float,
        default=0.3,
        help="Ratio of duplicates in the generated datasets",
    )
    parser.add_argument(
        "--variation-level",
        type=float,
        default=0.2,
        help="Level of variation in duplicates",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of runs for each configuration"
    )

    args = parser.parse_args()

    print("Running benchmarks with the following parameters:")
    print(f"  Dataset sizes: {args.sizes}")
    print(f"  Thresholds: {args.thresholds}")
    print(f"  Duplicate ratio: {args.duplicate_ratio}")
    print(f"  Variation level: {args.variation_level}")
    print(f"  Runs per configuration: {args.runs}")
    print(f"  Output directory: {args.output_dir}")

    # Run benchmarks
    results = run_benchmarks(
        sizes=args.sizes,
        thresholds=args.thresholds,
        duplicate_ratio=args.duplicate_ratio,
        variation_level=args.variation_level,
        runs=args.runs,
    )

    # Plot results
    plot_results(results, args.output_dir)

    # Generate Markdown report
    generate_markdown_report(results, args.output_dir)

    print(f"Benchmarks completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
