#!/bin/bash
# Build script for fast-dedupe

set -e  # Exit on error

# Function to display help
show_help() {
    echo "Build script for fast-dedupe"
    echo ""
    echo "Usage: ./build.sh [command]"
    echo ""
    echo "Commands:"
    echo "  clean       - Clean build artifacts"
    echo "  test        - Run tests"
    echo "  coverage    - Run tests with coverage"
    echo "  lint        - Run linting checks"
    echo "  format      - Format code with black and isort"
    echo "  build       - Build package"
    echo "  install     - Install package in development mode"
    echo "  publish     - Build and publish package to PyPI"
    echo "  help        - Show this help message"
    echo ""
}

# Clean build artifacts
clean() {
    echo "Cleaning build artifacts..."
    rm -rf build/ dist/ *.egg-info/ .coverage coverage.xml .pytest_cache/ __pycache__/ fastdedupe/__pycache__/ fastdedupe/tests/__pycache__/
    find . -name "*.pyc" -delete
    echo "Done!"
}

# Run tests
run_tests() {
    echo "Running tests..."
    python -m pytest
    echo "Done!"
}

# Run tests with coverage
run_coverage() {
    echo "Running tests with coverage..."
    python -m pytest --cov=fastdedupe --cov-report=term --cov-report=xml
    echo "Done!"
}

# Run linting checks
run_lint() {
    echo "Running linting checks..."
    python -m flake8 fastdedupe
    python -m mypy fastdedupe
    echo "Done!"
}

# Format code
format_code() {
    echo "Formatting code..."
    python -m black fastdedupe
    python -m isort fastdedupe
    echo "Done!"
}

# Build package
build_package() {
    echo "Building package..."
    python -m build
    echo "Done!"
}

# Install package in development mode
install_dev() {
    echo "Installing package in development mode..."
    pip install -e .
    echo "Done!"
}

# Publish package to PyPI
publish_package() {
    echo "Building and publishing package to PyPI..."
    python -m build
    python -m twine upload dist/*
    echo "Done!"
}

# Main script
case "$1" in
    clean)
        clean
        ;;
    test)
        run_tests
        ;;
    coverage)
        run_coverage
        ;;
    lint)
        run_lint
        ;;
    format)
        format_code
        ;;
    build)
        build_package
        ;;
    install)
        install_dev
        ;;
    publish)
        publish_package
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac 