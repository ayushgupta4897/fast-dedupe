"""
Command-line interface for fast-dedupe.

This module provides a command-line interface for the fast-dedupe package,
allowing users to deduplicate text files from the command line.
"""

import argparse
import csv
import json
import sys
import os
from typing import List, Dict, Optional, Union

from .core import dedupe
from .similarity import (
    SimilarityAlgorithm,
    visualize_similarity_matrix,
    visualize_algorithm_comparison,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deduplicate text data using fuzzy matching."
    )
    parser.add_argument(
        "input_file",
        help="Input file containing text data (one item per line, or CSV/JSON)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file for deduplicated data (default: stdout)",
        default=None,
    )
    parser.add_argument(
        "-d",
        "--duplicates",
        help="Output file for duplicate mappings (default: none)",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        help="Similarity threshold (0-100, default: 85)",
        type=int,
        default=85,
    )
    parser.add_argument(
        "-f",
        "--format",
        help="Input file format (txt, csv, json, default: txt)",
        choices=["txt", "csv", "json"],
        default="txt",
    )
    parser.add_argument(
        "--csv-column",
        help="Column name/index to deduplicate in CSV (default: 0)",
        default=0,
        type=lambda x: int(x) if x.isdigit() else x,
    )
    parser.add_argument(
        "--json-key",
        help="Key to deduplicate in JSON (default: 'text')",
        default="text",
    )
    parser.add_argument(
        "--keep-first",
        help="Keep first occurrence of duplicates (default)",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--keep-longest",
        help="Keep longest occurrence of duplicates",
        action="store_false",
        dest="keep_first",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        help="Similarity algorithm to use (default: levenshtein)",
        choices=[algo.value for algo in SimilarityAlgorithm],
        default=SimilarityAlgorithm.LEVENSHTEIN.value,
    )
    parser.add_argument(
        "--visualize",
        help="Generate visualization of similarity comparisons",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--viz-output",
        help="Output directory for visualizations (default: current directory)",
        default=".",
    )
    parser.add_argument(
        "--compare-algorithms",
        help="Compare all algorithms on the dataset and visualize results",
        action="store_true",
        default=False,
    )

    return parser.parse_args()


def read_input_data(
    input_file: str, file_format: str, csv_column: Union[str, int], json_key: str
) -> List[str]:
    """
    Read input data from a file.

    Args:
        input_file: Path to the input file
        file_format: Format of the input file (txt, csv, json)
        csv_column: Column name/index to deduplicate in CSV
        json_key: Key to deduplicate in JSON

    Returns:
        List of strings to deduplicate
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            if file_format == "txt":
                # Read text file (one item per line)
                return [line.strip() for line in f if line.strip()]
            elif file_format == "csv":
                # Read CSV file
                data = []
                # First, read the entire file content
                csv_content = list(csv.reader(f))

                # If csv_column is a string (column name), find its index
                if isinstance(csv_column, str):
                    if not csv_content:
                        return []

                    # Get header row
                    header = csv_content[0]

                    # Find column index
                    try:
                        col_idx = header.index(csv_column)
                    except ValueError:
                        print(
                            f"Column '{csv_column}' not found in CSV header",
                            file=sys.stderr,
                        )
                        return []

                    # Extract data from that column (skip header)
                    for row in csv_content[1:]:
                        if len(row) > col_idx:
                            data.append(row[col_idx])
                else:
                    # Use column index directly
                    col_idx = int(csv_column)

                    # Skip header if present
                    start_idx = 1 if csv_content else 0

                    # Extract data from that column
                    for row in csv_content[start_idx:]:
                        if len(row) > col_idx:
                            data.append(row[col_idx])

                return data
            elif file_format == "json":
                # Read JSON file
                json_data = json.load(f)
                if isinstance(json_data, list):
                    # List of objects
                    return [
                        str(item[json_key]) for item in json_data if json_key in item
                    ]
                elif isinstance(json_data, dict):
                    # Dictionary
                    if json_key in json_data:
                        json_value = json_data[json_key]
                        if isinstance(json_value, list):
                            return [str(item) for item in json_value]
                        else:
                            return [str(json_value)]
                    else:
                        return [str(value) for value in json_data.values()]
                return []
        # Add explicit return for mypy
        return []
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)


def write_output_data(output_file: Optional[str], data: List[str]) -> None:
    """
    Write output data to a file or stdout.

    Args:
        output_file: Path to the output file, or None for stdout
        data: List of strings to write
    """
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(f"{item}\n")
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        for item in data:
            print(item)


def write_duplicates(duplicates_file: str, duplicates: Dict[str, List[str]]) -> None:
    """
    Write duplicate mappings to a file.

    Args:
        duplicates_file: Path to the duplicates file
        duplicates: Dictionary mapping each kept string to its duplicates
    """
    try:
        with open(duplicates_file, "w", encoding="utf-8") as f:
            json.dump(duplicates, f, indent=2)
    except Exception as e:
        print(f"Error writing duplicates file: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    # Read input data
    data = read_input_data(args.input_file, args.format, args.csv_column, args.json_key)

    # Deduplicate data
    clean_data, duplicates = dedupe(
        data,
        threshold=args.threshold,
        keep_first=args.keep_first,
        similarity_algorithm=args.algorithm,
    )

    # Generate visualizations if requested
    if args.visualize and data:
        # Create output directory if it doesn't exist
        os.makedirs(args.viz_output, exist_ok=True)

        # Limit to a reasonable number of items for visualization
        viz_data = data[:20] if len(data) > 20 else data

        # Generate similarity matrix visualization
        matrix_file = os.path.join(
            args.viz_output, f"similarity_matrix_{args.algorithm}.png"
        )
        visualize_similarity_matrix(viz_data, args.algorithm, matrix_file)
        print(
            f"Similarity matrix visualization saved to: {matrix_file}", file=sys.stderr
        )

    # Compare algorithms if requested
    if args.compare_algorithms and len(data) >= 2:
        # Create output directory if it doesn't exist
        os.makedirs(args.viz_output, exist_ok=True)

        # Select a pair of strings to compare
        s1 = data[0]
        s2 = data[1] if len(data) > 1 else data[0]

        # Generate algorithm comparison visualization
        comparison_file = os.path.join(args.viz_output, "algorithm_comparison.png")
        visualize_algorithm_comparison(s1, s2, comparison_file)
        print(
            f"Algorithm comparison visualization saved to: {comparison_file}",
            file=sys.stderr,
        )

    # Write output data
    write_output_data(args.output, clean_data)

    # Write duplicates if requested
    if args.duplicates:
        write_duplicates(args.duplicates, duplicates)


if __name__ == "__main__":
    main()
