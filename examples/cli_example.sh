#!/bin/bash
# Example script demonstrating CLI usage of fast-dedupe

# Make sure we're in the examples directory
cd "$(dirname "$0")"

# Install the package in development mode if not already installed
if ! python -c "import fastdedupe" &> /dev/null; then
    echo "Installing fast-dedupe in development mode..."
    pip install -e ..
fi

echo "=== Basic Usage ==="
echo "Input data:"
cat sample_data.txt
echo

echo "=== Deduplicating with default settings (threshold=85) ==="
python -m fastdedupe sample_data.txt
echo

echo "=== Deduplicating with higher threshold (threshold=95) ==="
python -m fastdedupe sample_data.txt -t 95
echo

echo "=== Deduplicating with lower threshold (threshold=70) ==="
python -m fastdedupe sample_data.txt -t 70
echo

echo "=== Saving output to a file ==="
python -m fastdedupe sample_data.txt -o deduplicated.txt -t 85
echo "Output saved to deduplicated.txt"
echo

echo "=== Saving duplicates mapping to a file ==="
python -m fastdedupe sample_data.txt -o deduplicated.txt -d duplicates.json -t 85
echo "Duplicates mapping saved to duplicates.json"
echo "Contents of duplicates.json:"
cat duplicates.json
echo

echo "=== Using keep-longest option ==="
python -m fastdedupe sample_data.txt --keep-longest -t 85
echo

echo "=== Done ==="