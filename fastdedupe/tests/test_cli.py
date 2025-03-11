"""
Tests for the CLI functionality of fast-dedupe.

This module contains tests for the CLI interface of the fast-dedupe package.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from fastdedupe import cli


class TestCLI(unittest.TestCase):
    """Test cases for the CLI interface."""

    def setUp(self) -> None:
        """Set up temporary files for testing."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test files
        self.txt_file = os.path.join(self.temp_dir.name, "test.txt")
        with open(self.txt_file, "w", encoding="utf-8") as f:
            f.write("Apple iPhone 12\n")
            f.write("Apple iPhone12\n")
            f.write("Samsung Galaxy\n")
            f.write("Samsng Galaxy\n")
        
        self.csv_file = os.path.join(self.temp_dir.name, "test.csv")
        with open(self.csv_file, "w", encoding="utf-8") as f:
            f.write("name,price\n")
            f.write("Apple iPhone 12,999\n")
            f.write("Apple iPhone12,999\n")
            f.write("Samsung Galaxy,899\n")
            f.write("Samsng Galaxy,899\n")
        
        self.json_file = os.path.join(self.temp_dir.name, "test.json")
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump([
                {"text": "Apple iPhone 12", "price": 999},
                {"text": "Apple iPhone12", "price": 999},
                {"text": "Samsung Galaxy", "price": 899},
                {"text": "Samsng Galaxy", "price": 899},
            ], f)
        
        # Output files
        self.output_file = os.path.join(self.temp_dir.name, "output.txt")
        self.duplicates_file = os.path.join(self.temp_dir.name, "duplicates.json")

    def tearDown(self) -> None:
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_read_input_data_txt(self) -> None:
        """Test reading input data from a text file."""
        data = cli.read_input_data(self.txt_file, "txt", 0, "text")
        self.assertEqual(len(data), 4)
        self.assertEqual(data[0], "Apple iPhone 12")
        self.assertEqual(data[1], "Apple iPhone12")
        self.assertEqual(data[2], "Samsung Galaxy")
        self.assertEqual(data[3], "Samsng Galaxy")

    def test_read_input_data_csv(self) -> None:
        """Test reading input data from a CSV file."""
        # Test with column index
        data = cli.read_input_data(self.csv_file, "csv", 0, "text")
        self.assertEqual(len(data), 5)  # Including header row
        self.assertEqual(data[1], "Apple iPhone 12")
        self.assertEqual(data[2], "Apple iPhone12")
        self.assertEqual(data[3], "Samsung Galaxy")
        self.assertEqual(data[4], "Samsng Galaxy")
        
        # Test with column name
        data = cli.read_input_data(self.csv_file, "csv", "name", "text")
        self.assertEqual(len(data), 4)
        self.assertEqual(data[0], "Apple iPhone 12")
        self.assertEqual(data[1], "Apple iPhone12")
        self.assertEqual(data[2], "Samsung Galaxy")
        self.assertEqual(data[3], "Samsng Galaxy")

    def test_read_input_data_json(self) -> None:
        """Test reading input data from a JSON file."""
        data = cli.read_input_data(self.json_file, "json", 0, "text")
        self.assertEqual(len(data), 4)
        self.assertEqual(data[0], "Apple iPhone 12")
        self.assertEqual(data[1], "Apple iPhone12")
        self.assertEqual(data[2], "Samsung Galaxy")
        self.assertEqual(data[3], "Samsng Galaxy")

    def test_write_output_data(self) -> None:
        """Test writing output data to a file."""
        data = ["Apple iPhone 12", "Samsung Galaxy"]
        cli.write_output_data(self.output_file, data)
        
        with open(self.output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0].strip(), "Apple iPhone 12")
            self.assertEqual(lines[1].strip(), "Samsung Galaxy")

    def test_write_duplicates(self) -> None:
        """Test writing duplicates to a file."""
        duplicates = {
            "Apple iPhone 12": ["Apple iPhone12"],
            "Samsung Galaxy": ["Samsng Galaxy"],
        }
        cli.write_duplicates(self.duplicates_file, duplicates)
        
        with open(self.duplicates_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(len(data), 2)
            self.assertEqual(data["Apple iPhone 12"], ["Apple iPhone12"])
            self.assertEqual(data["Samsung Galaxy"], ["Samsng Galaxy"])

    @patch("sys.argv", ["fastdedupe", "input.txt"])
    def test_parse_args_defaults(self) -> None:
        """Test parsing command-line arguments with defaults."""
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            cli.parse_args()
            mock_parse_args.assert_called_once()

    @patch("fastdedupe.cli.dedupe")
    @patch("fastdedupe.cli.read_input_data")
    @patch("fastdedupe.cli.write_output_data")
    @patch("fastdedupe.cli.write_duplicates")
    @patch("fastdedupe.cli.parse_args")
    def test_main(self, mock_parse_args: MagicMock, 
                 mock_write_duplicates: MagicMock, 
                 mock_write_output: MagicMock, 
                 mock_read_input: MagicMock, 
                 mock_dedupe: MagicMock) -> None:
        """Test the main function."""
        # Mock the parsed arguments
        mock_args = MagicMock()
        mock_args.input_file = self.txt_file
        mock_args.output = self.output_file
        mock_args.duplicates = self.duplicates_file
        mock_args.threshold = 85
        mock_args.format = "txt"
        mock_args.csv_column = 0
        mock_args.json_key = "text"
        mock_args.keep_first = True
        mock_parse_args.return_value = mock_args
        
        # Mock the input data
        mock_read_input.return_value = [
            "Apple iPhone 12", "Apple iPhone12", 
            "Samsung Galaxy", "Samsng Galaxy"
        ]
        
        # Mock the dedupe function
        mock_dedupe.return_value = (
            ["Apple iPhone 12", "Samsung Galaxy"],
            {
                "Apple iPhone 12": ["Apple iPhone12"],
                "Samsung Galaxy": ["Samsng Galaxy"],
            }
        )
        
        # Call the main function
        cli.main()
        
        # Check that the functions were called with the correct arguments
        mock_read_input.assert_called_once_with(
            self.txt_file, "txt", 0, "text"
        )
        mock_dedupe.assert_called_once_with(
            mock_read_input.return_value, threshold=85, keep_first=True
        )
        mock_write_output.assert_called_once_with(
            self.output_file, ["Apple iPhone 12", "Samsung Galaxy"]
        )
        mock_write_duplicates.assert_called_once_with(
            self.duplicates_file, 
            {
                "Apple iPhone 12": ["Apple iPhone12"],
                "Samsung Galaxy": ["Samsng Galaxy"],
            }
        )


if __name__ == "__main__":
    unittest.main() 