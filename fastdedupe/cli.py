"""
Command-line interface for fast-dedupe.

This module provides a command-line interface for the fast-dedupe package,
allowing users to deduplicate text files from the command line.
"""

import argparse
import csv
import json
import sys
from typing import List, Dict, Any, Optional

from .core import dedupe


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deduplicate text data using fuzzy matching."
    )
    parser.add_argument(
        "input_file", 
        help="Input file containing text data (one item per line, or CSV/JSON)"
    )
    parser.add_argument(
        "-o", 
        "--output", 
        help="Output file for deduplicated data (default: stdout)",
        default=None
    )
    parser.add_argument(
        "-d", 
        "--duplicates", 
        help="Output file for duplicate mappings (default: none)",
        default=None
    )
    parser.add_argument(
        "-t", 
        "--threshold", 
        help="Similarity threshold (0-100, default: 85)",
        type=int, 
        default=85
    )
    parser.add_argument(
        "-f", 
        "--format", 
        help="Input file format (txt, csv, json, default: txt)",
        choices=["txt", "csv", "json"], 
        default="txt"
    )
    parser.add_argument(
        "--csv-column", 
        help="Column name/index to deduplicate in CSV (default: 0)",
        default=0
    )
    parser.add_argument(
        "--json-key", 
        help="Key to deduplicate in JSON (default: 'text')",
        default="text"
    )
    parser.add_argument(
        "--keep-first", 
        help="Keep first occurrence of duplicates (default)",
        action="store_true", 
        default=True
    )
    parser.add_argument(
        "--keep-longest", 
        help="Keep longest occurrence of duplicates",
        action="store_false", 
        dest="keep_first"
    )
    
    return parser.parse_args()


def read_input_data(
    input_file: str, 
    file_format: str, 
    csv_column: Any, 
    json_key: str
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
                reader = csv.DictReader(f) if isinstance(csv_column, str) else csv.reader(f)
                data = []
                for row in reader:
                    if isinstance(csv_column, str):
                        # Use column name
                        if csv_column in row:
                            data.append(row[csv_column])
                    else:
                        # Use column index
                        if len(row) > csv_column:
                            data.append(row[csv_column])
                return data
            elif file_format == "json":
                # Read JSON file
                json_data = json.load(f)
                if isinstance(json_data, list):
                    # List of objects
                    return [item[json_key] for item in json_data if json_key in item]
                elif isinstance(json_data, dict):
                    # Dictionary
                    if json_key in json_data:
                        return json_data[json_key]
                    else:
                        return list(json_data.values())
                return []
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)


def write_output_data(
    output_file: Optional[str], 
    data: List[str]
) -> None:
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


def write_duplicates(
    duplicates_file: str, 
    duplicates: Dict[str, List[str]]
) -> None:
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
    data = read_input_data(
        args.input_file, 
        args.format, 
        args.csv_column, 
        args.json_key
    )
    
    # Deduplicate data
    clean_data, duplicates = dedupe(
        data, 
        threshold=args.threshold, 
        keep_first=args.keep_first
    )
    
    # Write output data
    write_output_data(args.output, clean_data)
    
    # Write duplicates if requested
    if args.duplicates:
        write_duplicates(args.duplicates, duplicates)


if __name__ == "__main__":
    main() 